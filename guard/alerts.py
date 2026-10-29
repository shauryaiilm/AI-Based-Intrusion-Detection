from __future__ import annotations

import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from guard.draw import annotate_alert
from guard.schema import AlertEvent, SavedAlert

logger = logging.getLogger("guard")

CSV_HEADER = [
    "alert_no",
    "wall_time",
    "video_time_s",
    "sim_time",
    "track_id",
    "class",
    "zone",
    "dwell_s",
    "confidence",
    "frame_idx",
    "snapshot_file",
    "crop_file",
    "video_file",
]


def _write_image_safe(path: str | Path, img: np.ndarray) -> None:
    """Writes an image to disk safely on Windows even with non-ASCII characters in path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    success, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if success:
        enc.tofile(str(p))
    else:
        raise IOError(f"Could not encode image for {p}")


class RunRecorder:
    def __init__(
        self,
        run_dir: str | Path,
        video_name: str,
        fps: float,
        frame_size: tuple[int, int],
        save_video: bool = False,
    ) -> None:
        p = Path(run_dir)
        if p.name.startswith("run_"):
            self.run_dir = p
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = p / f"run_{ts}"

        self.snapshots_dir = self.run_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.video_name = video_name
        self.fps = fps if fps > 0 else 25.0
        self.frame_size = frame_size
        self.save_video = save_video

        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.alert_no = 0
        self.alerts: list[SavedAlert] = []
        self.alerts_by_class: dict[str, int] = {}
        self.first_alert_video_time_s: float | None = None

        # Prepare CSV file and write header immediately
        self.csv_path = self.run_dir / "events.csv"
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_HEADER)
        self._csv_file.flush()

        # Optional video writer
        self._video_writer: cv2.VideoWriter | None = None
        self.video_out_path = self.run_dir / "annotated.mp4"
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(self.video_out_path), fourcc, self.fps, self.frame_size
            )

    def save_alert(
        self, event: AlertEvent, annotated_frame: np.ndarray, raw_frame: np.ndarray
    ) -> SavedAlert:
        """Saves annotated snapshot with banner, un-annotated cropped intruder, and logs CSV row."""
        self.alert_no += 1
        wall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.first_alert_video_time_s is None:
            self.first_alert_video_time_s = event.video_time_s

        self.alerts_by_class[event.cls_name] = (
            self.alerts_by_class.get(event.cls_name, 0) + 1
        )

        # Sanitise zone name
        safe_zone = re.sub(r"[^A-Za-z0-9_-]+", "_", event.zone_name).strip("_") or "zone"
        base_filename = (
            f"alert_{self.alert_no:04d}_{event.cls_name}_id{event.track_id}_{safe_zone}"
        )
        snapshot_filename = f"{base_filename}.jpg"
        crop_filename = f"{base_filename}_crop.jpg"

        snapshot_path = self.snapshots_dir / snapshot_filename
        crop_path = self.snapshots_dir / crop_filename

        # Add red banner to annotated snapshot
        sim_display = event.sim_time if event.sim_time else "N/A"
        banner_text = (
            f"UNAUTHORISED {event.cls_name} #{event.track_id} in '{event.zone_name}' | "
            f"video t={event.video_time_s:.1f}s | sim {sim_display} | wall {wall_time}"
        )
        snap_frame = annotate_alert(annotated_frame, banner_text)
        _write_image_safe(snapshot_path, snap_frame)

        # Crop from raw frame (pad 10%, clamped to frame bounds)
        rh, rw = raw_frame.shape[:2]
        x1, y1, x2, y2 = event.box
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * 0.1
        pad_y = bh * 0.1
        cx1 = max(0, min(rw - 1, int(round(x1 - pad_x))))
        cy1 = max(0, min(rh - 1, int(round(y1 - pad_y))))
        cx2 = max(cx1 + 1, min(rw, int(round(x2 + pad_x))))
        cy2 = max(cy1 + 1, min(rh, int(round(y2 + pad_y))))

        crop_img = raw_frame[cy1:cy2, cx1:cx2]
        if crop_img.size == 0:
            crop_img = raw_frame
        _write_image_safe(crop_path, crop_img)

        # Append row to CSV
        video_file_val = "annotated.mp4" if self.save_video else ""
        row = [
            self.alert_no,
            wall_time,
            f"{event.video_time_s:.2f}",
            event.sim_time or "",
            event.track_id,
            event.cls_name,
            event.zone_name,
            f"{event.dwell_s:.2f}",
            f"{event.conf:.2f}",
            event.frame_idx,
            f"snapshots/{snapshot_filename}",
            f"snapshots/{crop_filename}",
            video_file_val,
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        saved = SavedAlert(
            event=event,
            alert_no=self.alert_no,
            wall_time=wall_time,
            snapshot_path=str(snapshot_path),
            crop_path=str(crop_path),
        )
        self.alerts.append(saved)
        return saved

    def write_frame(self, annotated_frame: np.ndarray) -> None:
        """Appends frame to annotated video if save_video is active."""
        if self._video_writer is not None:
            self._video_writer.write(annotated_frame)

    def finish(self, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        """Closes all writers and writes summary.json."""
        if stats is None:
            stats = {}

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()

        ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary = {
            "video": self.video_name,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "video_duration_s": stats.get("video_duration_s", 0.0),
            "frames_total": stats.get("frames_total", 0),
            "frames_processed": stats.get("frames_processed", 0),
            "frames_skipped": stats.get("frames_skipped", 0),
            "avg_processing_fps": stats.get("avg_processing_fps", 0.0),
            "settings": stats.get("settings", {}),
            "zones": stats.get("zones", {}),
            "alerts_total": self.alert_no,
            "alerts_by_class": dict(self.alerts_by_class),
            "unique_tracks_by_class": stats.get("unique_tracks_by_class", {}),
            "first_alert_video_time_s": self.first_alert_video_time_s,
        }

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

