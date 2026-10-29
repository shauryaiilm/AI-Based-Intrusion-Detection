from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import pytest

from guard.schema import Detection


class FakeDetector:
    """Scripted detector returning pre-defined detections by frame index.
    
    Used for unit tests so they run offline without downloading YOLO weights.
    """

    def __init__(
        self,
        script: Mapping[int, list[Detection]] | None = None,
        default_detections: list[Detection] | None = None,
    ) -> None:
        self.script = dict(script) if script else {}
        self.default_detections = list(default_detections) if default_detections else []
        self.conf: float = 0.35
        self.call_count: int = 0

    def set_confidence(self, conf: float) -> None:
        self.conf = conf

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        dets = self.script.get(self.call_count, self.default_detections)
        self.call_count += 1
        return [d for d in dets if d.conf >= self.conf]


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()


def create_synthetic_video(
    path: str | Path,
    width: int = 320,
    height: int = 240,
    fps: float = 10.0,
    duration_s: float = 4.0,
) -> Path:
    """Creates a small synthetic video with a moving rectangle for testing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(p), fourcc, fps, (width, height))

    total_frames = int(fps * duration_s)
    rect_w, rect_h = 40, 40

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Move rectangle horizontally
        x = int((width - rect_w) * (i / max(1, total_frames - 1)))
        y = (height - rect_h) // 2
        cv2.rectangle(frame, (x, y), (x + rect_w, y + rect_h), (0, 255, 0), -1)
        writer.write(frame)

    writer.release()
    return p


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    return create_synthetic_video(tmp_path / "test_synth.mp4")

