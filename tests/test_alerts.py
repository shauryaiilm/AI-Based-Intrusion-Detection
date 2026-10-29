from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from guard.alerts import CSV_HEADER, RunRecorder
from guard.schema import AlertEvent


def test_run_recorder_creates_folder_and_header(tmp_path: Path):
    run_dir = tmp_path / "run_test"
    recorder = RunRecorder(
        run_dir=run_dir,
        video_name="test.mp4",
        fps=25.0,
        frame_size=(640, 480),
        save_video=False,
    )
    assert run_dir.exists()
    assert (run_dir / "snapshots").exists()
    assert (run_dir / "events.csv").exists()

    # Verify CSV header immediately written
    with open(run_dir / "events.csv", "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 1
    assert reader[0] == CSV_HEADER

    # Finish and verify summary.json
    summary = recorder.finish({"frames_total": 100})
    assert (run_dir / "summary.json").exists()
    assert summary["video"] == "test.mp4"
    assert summary["alerts_total"] == 0
    assert summary["first_alert_video_time_s"] is None


def test_save_alert_writes_jpegs_and_csv_row(tmp_path: Path):
    run_dir = tmp_path / "run_alert_test"
    recorder = RunRecorder(
        run_dir=run_dir,
        video_name="test.mp4",
        fps=30.0,
        frame_size=(320, 240),
        save_video=False,
    )

    raw_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    raw_frame[50:150, 50:150] = (200, 200, 200)
    annotated_frame = raw_frame.copy()

    event = AlertEvent(
        track_id=3,
        cls_name="person",
        zone_name="Gate / North?",
        video_time_s=12.45,
        sim_time="22:00:12",
        dwell_s=2.5,
        conf=0.88,
        frame_idx=310,
        box=(50.0, 50.0, 150.0, 150.0),
    )

    saved = recorder.save_alert(event, annotated_frame, raw_frame)

    assert saved.alert_no == 1
    assert Path(saved.snapshot_path).exists()
    assert Path(saved.crop_path).exists()
    # Check sanitized zone name in filename
    assert "Gate_North" in Path(saved.snapshot_path).name

    # Check images can be decoded back with OpenCV
    snap_bytes = np.fromfile(saved.snapshot_path, dtype=np.uint8)
    snap_dec = cv2.imdecode(snap_bytes, cv2.IMREAD_COLOR)
    assert snap_dec is not None
    assert snap_dec.shape == (240, 320, 3)

    crop_bytes = np.fromfile(saved.crop_path, dtype=np.uint8)
    crop_dec = cv2.imdecode(crop_bytes, cv2.IMREAD_COLOR)
    assert crop_dec is not None
    assert crop_dec.shape[0] > 0 and crop_dec.shape[1] > 0

    # Verify CSV row
    with open(run_dir / "events.csv", "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 2  # header + 1 row
    row = reader[1]
    assert row[0] == "1"  # alert_no
    assert row[2] == "12.45"  # video_time_s
    assert row[3] == "22:00:12"  # sim_time
    assert row[4] == "3"  # track_id
    assert row[5] == "person"
    assert row[6] == "Gate / North?"
    assert row[7] == "2.50"
    assert row[8] == "0.88"
    assert row[9] == "310"

    # Finish and verify summary.json keys
    stats = {
        "video_duration_s": 30.0,
        "frames_total": 900,
        "frames_processed": 900,
        "frames_skipped": 0,
        "avg_processing_fps": 30.0,
        "settings": {"dwell_seconds": 2.0},
        "zones": {"Gate / North?": ["car"]},
        "unique_tracks_by_class": {"person": 1},
    }
    summary = recorder.finish(stats)
    assert summary["alerts_total"] == 1
    assert summary["alerts_by_class"] == {"person": 1}
    assert summary["first_alert_video_time_s"] == 12.45

    # Check summary.json on disk
    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        disk_sum = json.load(f)
    assert disk_sum["alerts_total"] == 1
    assert disk_sum["video"] == "test.mp4"

