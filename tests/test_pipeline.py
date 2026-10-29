from __future__ import annotations

import csv
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from guard.alerts import RunRecorder
from guard.pipeline import FramePipeline, VideoWorker
from guard.rules import RuleEngine
from guard.schema import AlertMsg, Detection, DoneMsg, Settings, Zone
from tests.conftest import FakeDetector, create_synthetic_video


def test_frame_pipeline_synthetic_video(tmp_path: Path):
    video_path = create_synthetic_video(tmp_path / "synth.mp4", duration_s=3.0, fps=10.0)
    run_dir = tmp_path / "run_pipeline"

    zone = Zone(
        name="SecurityZone",
        points=[(0.4, 0.0), (0.8, 0.0), (0.8, 1.0), (0.4, 1.0)],
        allowed=[],
    )
    zones_px = [("SecurityZone", [(128.0, 0.0), (256.0, 0.0), (256.0, 240.0), (128.0, 240.0)])]

    recorder = RunRecorder(run_dir, "synth.mp4", fps=10.0, frame_size=(320, 240))
    engine = RuleEngine([zone], frame_size=(320, 240), dwell_seconds=1.0)

    # Scripted detector:
    # frames 0..9 (t=0.0..0.9s): outside (anchor at x=50, y=100)
    # frames 10..29 (t=1.0..2.9s): inside (anchor at x=180, y=100)
    script: dict[int, list[Detection]] = {}
    for i in range(10):
        script[i] = [Detection(track_id=1, cls_name="person", conf=0.9, box=(40.0, 80.0, 60.0, 100.0))]
    for i in range(10, 30):
        script[i] = [Detection(track_id=1, cls_name="person", conf=0.9, box=(170.0, 80.0, 190.0, 100.0))]

    detector = FakeDetector(script=script)
    pipeline = FramePipeline(
        detector=detector,
        engine=engine,
        recorder=recorder,
        zones_px=zones_px,
        class_names=["person", "car", "motorcycle"],
    )

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        video_t = frame_idx / 10.0
        pipeline.process(frame, frame_idx, video_t)
        frame_idx += 1
    cap.release()

    summary = recorder.finish({"frames_total": frame_idx})

    # Exactly one alert fired
    assert summary["alerts_total"] == 1
    assert len(recorder.alerts) == 1

    # Snapshot exists
    snap_path = Path(recorder.alerts[0].snapshot_path)
    assert snap_path.exists()
    crop_path = Path(recorder.alerts[0].crop_path)
    assert crop_path.exists()

    # CSV has exactly one data row
    with open(recorder.csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # Header + 1 row


def test_video_worker_full_run(tmp_path: Path):
    video_path = create_synthetic_video(tmp_path / "worker_test.mp4", duration_s=2.5, fps=10.0)
    run_dir = tmp_path / "run_worker"

    zone = Zone(
        name="ZoneA",
        points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        allowed=[],
    )

    # All frames have an intruder in ZoneA (anchor at 160, 120)
    default_det = [Detection(track_id=1, cls_name="person", conf=0.9, box=(140.0, 100.0, 180.0, 120.0))]
    detector = FakeDetector(default_detections=default_det)

    recorder = RunRecorder(run_dir, "worker_test.mp4", fps=10.0, frame_size=(320, 240))

    settings = Settings(
        dwell_seconds=1.0,
        realtime=False,
        output_dir=str(run_dir),
    )

    frame_q = queue.Queue(maxsize=2)
    event_q = queue.Queue()
    stop_ev = threading.Event()
    pause_ev = threading.Event()

    worker = VideoWorker(
        video_path=str(video_path),
        zones=[zone],
        settings=settings,
        class_map={0: "person", 2: "car", 3: "motorcycle"},
        frame_queue=frame_q,
        event_queue=event_q,
        stop_event=stop_ev,
        pause_event=pause_ev,
        detector=detector,
        recorder=recorder,
    )

    worker.start()

    alerts_received: list[AlertMsg] = []
    done_received: DoneMsg | None = None
    start_time = time.time()

    while time.time() - start_time < 20.0:
        try:
            msg = event_q.get(timeout=0.2)
            if isinstance(msg, AlertMsg):
                alerts_received.append(msg)
            elif isinstance(msg, DoneMsg):
                done_received = msg
                break
        except queue.Empty:
            continue

    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert done_received is not None
    assert len(alerts_received) == 1
    assert done_received.summary["alerts_total"] == 1
    # Check that at least one frame packet was produced
    assert not frame_q.empty()


def test_video_worker_stop_mid_run(tmp_path: Path):
    video_path = create_synthetic_video(tmp_path / "long_video.mp4", duration_s=10.0, fps=10.0)
    run_dir = tmp_path / "run_stop"

    zone = Zone(name="Area", points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], allowed=[])
    detector = FakeDetector()
    recorder = RunRecorder(run_dir, "long_video.mp4", fps=10.0, frame_size=(320, 240))

    settings = Settings(dwell_seconds=1.0, realtime=False)
    frame_q = queue.Queue(maxsize=2)
    event_q = queue.Queue()
    stop_ev = threading.Event()
    pause_ev = threading.Event()

    worker = VideoWorker(
        video_path=str(video_path),
        zones=[zone],
        settings=settings,
        class_map={0: "person", 2: "car", 3: "motorcycle"},
        frame_queue=frame_q,
        event_queue=event_q,
        stop_event=stop_ev,
        pause_event=pause_ev,
        detector=detector,
        recorder=recorder,
    )

    worker.start()
    time.sleep(0.3)

    t0 = time.perf_counter()
    stop_ev.set()
    worker.join(timeout=3.0)
    elapsed = time.perf_counter() - t0

    assert not worker.is_alive()
    assert elapsed < 3.0

