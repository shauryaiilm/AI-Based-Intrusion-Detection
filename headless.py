"""Headless batch runner for Intrusion Guard without GUI."""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from guard.alerts import RunRecorder
from guard.config import get_zones, load_config
from guard.detector import YoloDetector
from guard.pipeline import FramePipeline, compute_working_size, downscale_if_needed
from guard.rules import RuleEngine


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Intrusion Guard in batch mode without a graphical window."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Path(s) to video files to process.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=None,
        help="Override dwell time threshold in seconds.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override detection confidence threshold (0.05 - 0.95).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        choices=[320, 480, 640],
        default=None,
        help="Override YOLO inference image size (320, 480, 640).",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        default=False,
        help="Enable time-of-day schedule enforcement.",
    )
    parser.add_argument(
        "--sim-start",
        type=str,
        default=None,
        help="Simulated clock time at frame 0 (HH:MM).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=False,
        help="Save annotated MP4 video into the run output folder.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base directory for output runs (default from config).",
    )
    return parser.parse_args(args)


def process_video(
    video_path: str,
    cfg_path: str,
    dwell_override: float | None = None,
    conf_override: float | None = None,
    imgsz_override: int | None = None,
    schedule_override: bool = False,
    sim_start_override: str | None = None,
    save_video_override: bool = False,
    output_override: str | None = None,
    custom_detector: Any = None,
) -> dict[str, Any] | None:
    cfg = load_config(cfg_path)
    v_name = os.path.basename(video_path)
    zones = get_zones(cfg, v_name)

    if not zones:
        print(
            f"No zones saved for {v_name}. Open it in the GUI (python app.py) and draw zones first."
        )
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(fps) or fps < 1.0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0:
        total_frames = 0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        ok, probe = cap.read()
        if not ok or probe is None:
            print(f"Error: Could not read frames from '{video_path}'.")
            cap.release()
            return None
        orig_h, orig_w = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    max_w = cfg.settings.max_work_width
    work_w, work_h = compute_working_size(orig_w, orig_h, max_w)
    frame_size = (work_w, work_h)

    dwell = dwell_override if dwell_override is not None else cfg.settings.dwell_seconds
    conf = conf_override if conf_override is not None else cfg.settings.confidence
    imgsz = imgsz_override if imgsz_override is not None else cfg.settings.imgsz
    schedule_enabled = schedule_override or cfg.settings.schedule_enabled
    sim_start = sim_start_override if sim_start_override is not None else cfg.settings.sim_start
    save_video = save_video_override or cfg.settings.save_video
    out_dir = output_override if output_override is not None else cfg.settings.output_dir

    recorder = RunRecorder(
        run_dir=out_dir,
        video_name=v_name,
        fps=fps,
        frame_size=frame_size,
        save_video=save_video,
    )

    if custom_detector is not None:
        detector = custom_detector
    else:
        detector = YoloDetector(
            model_name=cfg.model,
            class_map=cfg.classes,
            conf=conf,
            imgsz=imgsz,
        )

    engine = RuleEngine(
        zones=zones,
        frame_size=frame_size,
        dwell_seconds=dwell,
        grace_seconds=cfg.settings.grace_seconds,
        schedule_enabled=schedule_enabled,
        sim_start=sim_start,
    )

    zones_px = [
        (z.name, [(pt[0] * work_w, pt[1] * work_h) for pt in z.points])
        for z in zones
    ]
    pipeline = FramePipeline(detector, engine, recorder, zones_px, list(cfg.classes.values()))

    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = downscale_if_needed(frame, max_w)
        video_t = frame_idx / fps
        pipeline.process(frame, frame_idx, video_t)
        frame_idx += 1

    cap.release()

    elapsed = time.perf_counter() - t0
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    video_dur = total_frames / fps if total_frames > 0 else (frame_idx / fps)

    unique_tracks = {cls: len(s) for cls, s in pipeline.seen_tracks.items()}
    final_stats = {
        "video_duration_s": round(video_dur, 2),
        "frames_total": total_frames if total_frames > 0 else frame_idx,
        "frames_processed": frame_idx,
        "frames_skipped": 0,
        "avg_processing_fps": round(avg_fps, 1),
        "settings": {
            "dwell_seconds": dwell,
            "grace_seconds": cfg.settings.grace_seconds,
            "confidence": conf,
            "imgsz": imgsz,
            "realtime": False,
            "save_video": save_video,
            "schedule_enabled": schedule_enabled,
            "sim_start": sim_start,
        },
        "zones": {z.name: z.allowed for z in zones},
        "unique_tracks_by_class": unique_tracks,
    }

    summary = recorder.finish(final_stats)
    return {
        "name": v_name,
        "duration_s": video_dur,
        "frames": frame_idx,
        "avg_fps": avg_fps,
        "alerts": summary["alerts_total"],
        "first_alert": summary["first_alert_video_time_s"],
        "run_dir": str(recorder.run_dir),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    results: list[dict[str, Any]] = []
    for video in args.videos:
        res = process_video(
            video_path=video,
            cfg_path=args.config,
            dwell_override=args.dwell,
            conf_override=args.conf,
            imgsz_override=args.imgsz,
            schedule_override=args.schedule,
            sim_start_override=args.sim_start,
            save_video_override=args.save_video,
            output_override=args.output,
        )
        if res:
            first_s = (
                f"{res['first_alert']:.1f}s"
                if res["first_alert"] is not None
                else "None"
            )
            print(
                f"{res['name']} | duration: {res['duration_s']:.1f}s | "
                f"frames: {res['frames']} | avg FPS: {res['avg_fps']:.1f} | "
                f"alerts: {res['alerts']} | first alert: {first_s}"
            )
            results.append(res)

    if results:
        print("\n### Batch Test Results Summary\n")
        print("| Video | Duration | Frames | Avg FPS | Alerts | First Alert | Run Directory |")
        print("|---|---|---|---|---|---|---|")
        for r in results:
            first_s = f"{r['first_alert']:.1f}s" if r["first_alert"] is not None else "None"
            print(
                f"| {r['name']} | {r['duration_s']:.1f}s | {r['frames']} | "
                f"{r['avg_fps']:.1f} | {r['alerts']} | {first_s} | `{r['run_dir']}` |"
            )


if __name__ == "__main__":
    main()

