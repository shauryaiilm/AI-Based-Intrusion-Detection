from __future__ import annotations

import logging
from typing import Mapping, Protocol

import numpy as np
from ultralytics import YOLO

from guard.schema import Detection

logger = logging.getLogger("guard")


class DetectorError(Exception):
    """Raised when object detection or tracking fails."""
    pass


class DetectorProtocol(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        ...

    def set_confidence(self, conf: float) -> None:
        ...


class YoloDetector:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        class_map: Mapping[int, str] | None = None,
        conf: float = 0.35,
        imgsz: int = 480,
    ) -> None:
        self._class_map = dict(class_map) if class_map else {0: "person", 2: "car", 3: "motorcycle"}
        self._conf = max(0.05, min(0.95, float(conf)))
        self._imgsz = imgsz
        self.model_name = model_name

        try:
            logger.info(f"Loading YOLO model: {model_name}")
            self._model = YOLO(model_name)
        except Exception as exc:
            fallback = "yolov8n.pt"
            logger.warning(f"Failed to load {model_name} ({exc}). Falling back to {fallback}")
            try:
                self._model = YOLO(fallback)
                self.model_name = fallback
            except Exception as fb_exc:
                raise DetectorError(
                    f"Could not load YOLO model {model_name} or {fallback}: {fb_exc}. "
                    f"Check internet connection or place weights file in workspace."
                ) from fb_exc

        logger.info(f"YOLO detector initialized with {self.model_name} (device=cpu)")

    def set_confidence(self, conf: float) -> None:
        """Dynamically updates confidence threshold."""
        self._conf = max(0.05, min(0.95, float(conf)))

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Runs tracking on frame and returns tracked detections."""
        try:
            results = self._model.track(
                frame_bgr,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(self._class_map.keys()),
                conf=self._conf,
                imgsz=self._imgsz,
                device="cpu",
                verbose=False,
            )
            r = results[0]
            if r.boxes is None or r.boxes.id is None:
                return []

            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            detections: list[Detection] = []
            for box, tid, cls_id, conf in zip(boxes_xyxy, track_ids, classes, confs):
                cls_name = self._class_map.get(int(cls_id))
                if cls_name is None:
                    continue  # ignore unmapped classes
                detections.append(
                    Detection(
                        track_id=int(tid),
                        cls_name=cls_name,
                        conf=float(conf),
                        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    )
                )
            return detections
        except Exception as exc:
            logger.error(f"Detector error during track(): {exc}")
            raise DetectorError(f"Detection failed: {exc}") from exc

