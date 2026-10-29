from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from guard.alerts import RunRecorder
from guard.detector import DetectorProtocol, YoloDetector
from guard.draw import draw_info_bar, draw_tracks, draw_zones
from guard.rules import RuleEngine, format_hhmmss
from guard.schema import (
    AlertMsg,
    DoneMsg,
    ErrorMsg,
    FramePacket,
    FrameResult,
    SavedAlert,
    Settings,
    StatusMsg,
    Zone,
)

logger = logging.getLogger("guard")


def downscale_if_needed(frame: np.ndarray, max_work_width: int) -> np.ndarray:
    """Downscales frame keeping aspect ratio if its width exceeds max_work_width."""
    h, w = frame.shape[:2]
    if w > max_work_width:
        scale = max_work_width / float(w)
        new_w = max_work_width
        new_h = int(round(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def compute_working_size(orig_w: int, orig_h: int, max_work_width: int) -> tuple[int, int]:
    """Calculates the working frame size after downscaling."""
    if orig_w > max_work_width:
        scale = max_work_width / float(orig_w)
        return max_work_width, int(round(orig_h * scale))
    return orig_w, orig_h


class FramePipeline:
    def __init__(
        self,
        detector: DetectorProtocol,
        engine: RuleEngine,
        recorder: RunRecorder,
        zones_px: Sequence[tuple[str, Sequence[tuple[float, float]]]],
        class_names: Sequence[str],
    ) -> None:
        self.detector = detector
        self.engine = engine
        self.recorder = recorder
        self.zones_px = list(zones_px)
        self.class_names = list(class_names)
        self.seen_tracks: dict[str, set[int]] = {cls: set() for cls in self.class_names}

    def process(
        self, frame_bgr: np.ndarray, frame_idx: int, video_time_s: float
    ) -> FrameResult:
        # 1. Detect
        detections = self.detector.detect(frame_bgr)

        # 2. Update rule engine
        states, events = self.engine.update(detections, frame_idx, video_time_s)

        # 3. If events fired, copy raw frame for un-annotated crop
        if events:
            raw_frame = frame_bgr.copy()
        else:
            raw_frame = None

        # Draw zones
        zone_info_map = self.engine.zone_states(video_time_s)
        active_map = {name: info.is_active for name, info in zone_info_map.items()}
        violating_map = {name: info.has_unauthorised for name, info in zone_info_map.items()}
        draw_zones(frame_bgr, self.zones_px, active_map, violating_map)

        # Draw tracks
        draw_tracks(frame_bgr, states)

        # Draw info bar
        sim_str = ""
        if self.engine._schedule_enabled:
            sim_s = int(self.engine._sim_start_sec + video_time_s) % 86400
            sim_str = f" | sim {format_hhmmss(sim_s)}"
        info_text = f"t={video_time_s:.1f}s | frame {frame_idx}{sim_str}"
        draw_info_bar(frame_bgr, info_text)

        annotated_frame = frame_bgr

        # 4. Save alerts
        saved_alerts: list[SavedAlert] = []
        if events and raw_frame is not None:
            for ev in events:
                saved = self.recorder.save_alert(ev, annotated_frame, raw_frame)
                saved_alerts.append(saved)

        # 5. Write video frame if requested
        self.recorder.write_frame(annotated_frame)

        # 6. Update counts
        counts_now: dict[str, int] = {cls: 0 for cls in self.class_names}
        for st in states:
            cls_name = st.detection.cls_name
            counts_now[cls_name] = counts_now.get(cls_name, 0) + 1
            self.seen_tracks.setdefault(cls_name, set()).add(st.detection.track_id)

        return FrameResult(
            annotated_frame=annotated_frame,
            states=states,
            saved_alerts=saved_alerts,
            counts_now=counts_now,
            frame_idx=frame_idx,
            video_time_s=video_time_s,
        )


def _push_frame_packet(q: queue.Queue, packet: FramePacket) -> None:
    try:
        q.put_nowait(packet)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(packet)
        except queue.Full:
            pass


class VideoWorker(threading.Thread):
    def __init__(
        self,
        video_path: str,
        zones: list[Zone],
        settings: Settings,
        class_map: Mapping[int, str],
        frame_queue: queue.Queue,
        event_queue: queue.Queue,
        stop_event: threading.Event,
        pause_event: threading.Event,
        detector: DetectorProtocol | None = None,
        recorder: RunRecorder | None = None,
    ) -> None:
        super().__init__(name="VideoWorker", daemon=True)
        self.video_path = video_path
        self.zones = list(zones)
        self.settings = settings
        self.class_map = dict(class_map)
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self._custom_detector = detector
        self._custom_recorder = recorder

    def run(self) -> None:
        recorder: RunRecorder | None = None
        try:
            self.event_queue.put(StatusMsg("Opening video..."))
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                has_non_ascii = any(ord(c) >= 128 for c in self.video_path)
                msg = f"Cannot open video file: '{self.video_path}'."
                if has_non_ascii:
                    msg += " Note: The path contains non-ASCII characters."
                self.event_queue.put(ErrorMsg(msg))
                self.event_queue.put(DoneMsg({}))
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if math.isnan(fps) or fps < 1.0:
                fps = 25.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 0:
                total_frames = 0

            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if orig_w <= 0 or orig_h <= 0:
                # Read one frame to detect size
                ok, probe_frame = cap.read()
                if not ok or probe_frame is None:
                    self.event_queue.put(ErrorMsg("Failed to read frames from video."))
                    self.event_queue.put(DoneMsg({}))
                    cap.release()
                    return
                orig_h, orig_w = probe_frame.shape[:2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            work_w, work_h = compute_working_size(orig_w, orig_h, self.settings.max_work_width)
            frame_size = (work_w, work_h)

            video_basename = os.path.basename(self.video_path)

            # Initialize RunRecorder
            if self._custom_recorder is not None:
                recorder = self._custom_recorder
            else:
                recorder = RunRecorder(
                    run_dir=self.settings.output_dir,
                    video_name=video_basename,
                    fps=fps,
                    frame_size=frame_size,
                    save_video=self.settings.save_video,
                )

            # Initialize detector
            self.event_queue.put(StatusMsg("Loading detector..."))
            if self._custom_detector is not None:
                detector = self._custom_detector
            else:
                detector = YoloDetector(
                    model_name="yolo11n.pt",
                    class_map=self.class_map,
                    conf=self.settings.confidence,
                    imgsz=self.settings.imgsz,
                )

            # Initialize rule engine
            engine = RuleEngine(
                zones=self.zones,
                frame_size=frame_size,
                dwell_seconds=self.settings.dwell_seconds,
                grace_seconds=self.settings.grace_seconds,
                schedule_enabled=self.settings.schedule_enabled,
                sim_start=self.settings.sim_start,
            )

            # Convert zone points to working frame pixels for drawing
            zones_px = [
                (z.name, [(pt[0] * work_w, pt[1] * work_h) for pt in z.points])
                for z in self.zones
            ]

            class_names = list(self.class_map.values())
            pipeline = FramePipeline(detector, engine, recorder, zones_px, class_names)

            self.event_queue.put(StatusMsg("Running"))

            realtime = self.settings.realtime and not self.settings.save_video
            frame_idx = 0
            processed = 0
            skipped = 0
            t0 = time.perf_counter()
            last_dwell = self.settings.dwell_seconds
            last_conf = self.settings.confidence

            while not self.stop_event.is_set():
                # Handle pause
                if self.pause_event.is_set():
                    pause_start = time.perf_counter()
                    while self.pause_event.is_set() and not self.stop_event.is_set():
                        time.sleep(0.05)
                    # Shift reference time forward by paused duration
                    t0 += (time.perf_counter() - pause_start)

                if self.stop_event.is_set():
                    break

                # Live update dwell & confidence from shared settings
                if self.settings.dwell_seconds != last_dwell:
                    engine.set_dwell_seconds(self.settings.dwell_seconds)
                    last_dwell = self.settings.dwell_seconds

                if self.settings.confidence != last_conf:
                    detector.set_confidence(self.settings.confidence)
                    last_conf = self.settings.confidence

                if not cap.grab():
                    break  # End of video

                video_t = frame_idx / fps

                if realtime:
                    due = t0 + video_t
                    now = time.perf_counter()
                    if now < due:
                        # Sleep in small slices checking stop/pause
                        sleep_s = due - now
                        slice_end = now + sleep_s
                        while time.perf_counter() < slice_end:
                            if self.stop_event.is_set() or self.pause_event.is_set():
                                break
                            time.sleep(min(0.005, slice_end - time.perf_counter()))
                    elif now - due > 1.0 / fps:
                        # Behind: skip cheaply
                        frame_idx += 1
                        skipped += 1
                        continue

                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break

                frame = downscale_if_needed(frame, self.settings.max_work_width)
                result = pipeline.process(frame, frame_idx, video_t)

                # Send alerts to event queue
                for sa in result.saved_alerts:
                    self.event_queue.put(AlertMsg(sa))

                # Compute current FPS
                elapsed_so_far = time.perf_counter() - t0
                current_fps = (processed + 1) / elapsed_so_far if elapsed_so_far > 0 else fps

                # Push frame packet to frame queue
                stats = {
                    "fps": round(current_fps, 1),
                    "counts_now": result.counts_now,
                    "alerts_total": recorder.alert_no,
                    "skipped": skipped,
                }
                packet = FramePacket(
                    frame=result.annotated_frame,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    video_time_s=video_t,
                    stats=stats,
                )
                _push_frame_packet(self.frame_queue, packet)

                frame_idx += 1
                processed += 1

            cap.release()

            # Finalize run
            elapsed_total = time.perf_counter() - t0
            avg_fps = processed / elapsed_total if elapsed_total > 0 else 0.0
            video_duration_s = total_frames / fps if total_frames > 0 else (frame_idx / fps)

            zones_dict = {z.name: z.allowed for z in self.zones}
            unique_tracks = {cls: len(s) for cls, s in pipeline.seen_tracks.items()}

            final_stats = {
                "video_duration_s": round(video_duration_s, 2),
                "frames_total": total_frames if total_frames > 0 else frame_idx,
                "frames_processed": processed,
                "frames_skipped": skipped,
                "avg_processing_fps": round(avg_fps, 1),
                "settings": {
                    "dwell_seconds": self.settings.dwell_seconds,
                    "grace_seconds": self.settings.grace_seconds,
                    "confidence": self.settings.confidence,
                    "imgsz": self.settings.imgsz,
                    "realtime": self.settings.realtime,
                    "save_video": self.settings.save_video,
                    "schedule_enabled": self.settings.schedule_enabled,
                    "sim_start": self.settings.sim_start,
                },
                "zones": zones_dict,
                "unique_tracks_by_class": unique_tracks,
            }

            summary = recorder.finish(final_stats)
            self.event_queue.put(DoneMsg(summary))

        except Exception as exc:
            logger.error(f"Error in VideoWorker: {exc}", exc_info=True)
            self.event_queue.put(ErrorMsg(f"Worker error: {exc}"))
            if recorder is not None:
                try:
                    summary = recorder.finish({})
                except Exception:
                    summary = {}
            else:
                summary = {}
            self.event_queue.put(DoneMsg(summary))

