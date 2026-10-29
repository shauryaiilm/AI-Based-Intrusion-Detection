"""Check detector script: downloads a test image, runs YoloDetector on CPU, and prints detections."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import urllib.request
import cv2

from guard.detector import YoloDetector


def main() -> None:
    sample_url = "https://ultralytics.com/images/bus.jpg"
    cache_path = Path("bus.jpg")

    if not cache_path.exists():
        print(f"Downloading sample image from {sample_url}...")
        try:
            req = urllib.request.Request(
                sample_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp, open(cache_path, "wb") as out:
                out.write(resp.read())
            print("Sample image downloaded successfully.")
        except Exception as exc:
            print(f"Could not download sample image (offline or network error: {exc}).")
            print("Skipping detector check. Connect to the internet once or place bus.jpg locally.")
            return

    img = cv2.imread(str(cache_path))
    if img is None:
        print(f"Failed to read image from {cache_path}")
        sys.exit(1)

    print("Initializing YOLO detector (device=cpu, classes: person, car, motorcycle)...")
    detector = YoloDetector(
        model_name="yolo11n.pt",
        class_map={0: "person", 2: "car", 3: "motorcycle"},
        conf=0.25,
        imgsz=480,
    )

    print("Running detector on sample image...")
    detections = detector.detect(img)
    print(f"Detections found: {len(detections)}")
    for d in detections:
        print(f" - [{d.cls_name}] ID:{d.track_id} Conf:{d.conf:.2f} Box:{d.box}")

    print("\nNote: Bus is excluded from detection as per project configuration.")
    print("Check detector completed successfully.")


if __name__ == "__main__":
    main()
