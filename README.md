# Intrusion Guard

Intrusion Guard is a lightweight, modular desktop surveillance application (Tkinter + OpenCV + YOLO) built as the Phase-II working prototype for the major project **"AI-Based Intelligent Border Surveillance and Intrusion Detection System"** (IILM University). The system monitors recorded video footage, detects people, cars, and motorcycles in real time on CPU, tracks them across frames with stable IDs using ByteTrack, and enforces virtual boundary rules against user-defined polygon zones with dwell-time and time-of-day schedule restrictions. When an unauthorized intrusion persists beyond a dwell threshold, the system immediately flags the intruder in red, captures visual evidence (annotated snapshot and intruder crop), sounds an alert, and records structured event telemetry to CSV.

---

## Screenshots

> TODO: add screenshots after the first run (place images in `docs/images/` and reference them here).

---

## Features

- **CPU-Optimized Deep Learning Object Detection:** Pretrained YOLOv11 nano model (`yolo11n.pt`) running exclusively on CPU with zero GPU/CUDA requirements.
- **Specific Object Filtering:** Tracks strictly `person`, `car`, and `motorcycle` classes (buses, trucks, and bicycles are excluded by configuration).
- **Interactive Virtual Polygon Zones:** Draw custom zones directly on the video feed. Each zone supports custom allowed classes and optional active hours (`HH:MM`).
- **Temporal Dwell Time & Grace Period:** Filters out transient noise and brief edge crossings. Triggers intrusion alerts only after an object remains inside a restricted zone longer than the dwell duration (default: 2.0s), with a grace period (default: 1.0s) for detection flicker.
- **Simulated Schedule Clock:** Enforces time-of-day access control on pre-recorded footage using simulated clock progression (including midnight-crossing schedules such as 19:00 to 07:00).
- **Instant Visual Evidence Capture:** Automatically generates full annotated alert frames with timestamp banners and tightly cropped images of intruders saved to `output/run_*/snapshots/`.
- **Structured Audit Logging:** Every intrusion appends an entry to `events.csv` with event ID, wall clock time, video time, simulated time, track ID, object class, zone name, and file links.
- **Non-Blocking Pop-Up Toast:** Transient, reusable alert pop-up window updates in place without stalling or pausing the video playback loop.
- **Batch Headless Evaluation:** CLI runner `headless.py` for batch processing test clips and generating Markdown summary reports.

---

## How It Works

Intrusion Guard executes a 6-step processing pipeline for each video frame:

1. **Read & Downscale:** Frames are read from the video file and downscaled to working dimensions (up to `max_work_width`, default 1280 px) maintaining original aspect ratio.
2. **Detect Objects:** The working frame is fed into YOLO on CPU with confidence filtering.
3. **Multi-Object Tracking:** ByteTrack assigns persistent track IDs (`track_id`) across successive frames.
4. **Anchor & Zone Intersection:** Each track's anchor point—its bottom-center coordinate `((x1 + x2)/2, y2)`, approximating where feet or tires contact the ground—is evaluated against all active polygon zones using pure-Python ray-casting.
5. **Evaluate Rules:** The rule engine checks if the object's class is allowed and whether the zone is currently active under the simulated schedule. If unauthorized, a dwell timer increments.
6. **Alert & Record:** When dwell time reaches the threshold, the track status transitions to `UNAUTHORISED`. An alert fires: a snapshot and cropped intruder image are saved, an entry is appended to `events.csv`, and visual indicators appear.

### Visual Color Legend
- 🟩 **Green (Authorised / Outside):** The object is either outside all zones or belongs to an allowed class for that zone.
- 🟧 **Orange (Pending):** The object is inside a restricted zone; the dwell timer is actively running (`dwell < dwell_seconds`).
- 🟥 **Red (Unauthorised):** The object stayed in the restricted zone past the dwell duration. The bounding box thickens, the zone outline turns red, and evidence is recorded.
- ⬜ **Grey ([inactive]):** The zone is currently outside its scheduled active hours and does not trigger alerts.

---

## Requirements and Installation on Windows

### Prerequisites
- Windows 10 or 11
- Python 3.10, 3.11, or 3.12 (standard installer from [python.org](https://www.python.org) with "Add Python to PATH" and "tcl/tk" checked).

### Installation Steps

1. Open PowerShell or Command Prompt in the project directory:
   ```powershell
   cd "AI Based Intrusion Detection"
   ```

2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install required runtime and development dependencies:
   ```powershell
   pip install -r requirements.txt -r requirements-dev.txt
   ```

4. Pre-download model weights and verify the detector:
   ```powershell
   python tools/check_detector.py
   ```
   *(This downloads the lightweight `yolo11n.pt` model weights (~5.4 MB) and runs a verification test on a sample image).*

---

## Quick Start

Launch the desktop application:
```powershell
python app.py
```

### Step-by-Step Usage:
1. Click **Open Video** (`Ctrl+O`) and choose a video file (`.mp4`, `.avi`, `.mov`, `.mkv`).
2. Click **Add Zone**:
   - The first frame is displayed.
   - **Left-Click** to add polygon vertices.
   - **Backspace** to undo the last point.
   - **Right-Click** or press **Enter** (with at least 3 points) to finish the polygon.
   - **Esc** to cancel.
3. In the **Zone Dialog**:
   - Give the zone a name (e.g. `Perimeter Gate`).
   - Check any allowed object classes (leave all unchecked to restrict everyone).
   - Optionally enter active hours (e.g., `19:00` to `07:00`).
   - Click **Save Zone**. The zone is saved directly to `config.yaml`.
4. Click **Start** to begin monitoring.

### Settings Reference
- **Dwell time (s):** Number of seconds an intruder must remain inside a zone before an alert is triggered (live-editable).
- **Confidence:** Detection confidence threshold between 0.10 and 0.90 (live-editable).
- **Image size:** YOLO inference resolution (`320`, `480`, `640`).
- **Real-time (skip frames):** Automatically skips frames if PC hardware cannot keep pace with the video's native framerate.
- **Use schedule / Sim start:** Toggles time-of-day rule enforcement starting at the specified clock time (e.g. `12:00`).
- **Save annotated video:** When enabled, writes `annotated.mp4` into the run folder (processes all frames).
- **Pop-up / Beep:** Toggles the non-blocking toast window and audio bell.

---

## Zone Rules Explained

| Zone Type | Allowed Types | Active Hours | Result for Person | Result for Car |
|---|---|---|---|---|
| **Pedestrian Walkway** | `person` | None (24/7) | 🟩 Green (Authorised) | 🟥 Red (Intrusion alert) |
| **Vehicle Gate** | `car`, `motorcycle` | None (24/7) | 🟥 Red (Intrusion alert) | 🟩 Green (Authorised) |
| **Restricted Perimeter** | None (empty) | None (24/7) | 🟥 Red (Intrusion alert) | 🟥 Red (Intrusion alert) |
| **After-Hours Warehouse** | None (empty) | `19:00` - `07:00` | 🟥 Red if night, ⬜ Grey if day | 🟥 Red if night, ⬜ Grey if day |

---

## Batch Testing with `headless.py`

Run automated batch evaluation on multiple clips without opening the GUI:
```powershell
python headless.py clips/test1.mp4 clips/test2.mp4 --dwell 2.0 --conf 0.35
```

Outputs for each run are saved to `output/run_YYYYmmdd_HHMMSS/`:
- `events.csv`: Table of all intrusion alerts with timestamps and coordinates.
- `snapshots/`: Full annotated frames with alert banners (`alert_0001_person_id1_Zone.jpg`) and cropped intruder images (`..._crop.jpg`).
- `summary.json`: Detailed run telemetry (total frames, skipped frames, FPS, unique tracks, alert counts).
- `annotated.mp4`: Exported video (if `--save-video` is specified).

---

## Project Structure

```
intrusion-guard/
├── app.py                    # GUI entry point: python app.py
├── headless.py               # Headless batch CLI runner
├── config.yaml               # Application configuration and saved zones
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development & testing dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git ignore rules
├── guard/
│   ├── __init__.py           # Package initialization
│   ├── schema.py             # Data classes and enums (pure Python)
│   ├── config.py             # Config file loading, saving, and validation
│   ├── detector.py           # YOLO + ByteTrack wrapper (Ultralytics)
│   ├── rules.py              # Pure Python rule engine, ray-casting, dwell/grace
│   ├── draw.py               # OpenCV frame rendering (zones, tracks, banners)
│   ├── alerts.py             # Evidence recorder (CSV, snapshots, crops, summary)
│   ├── pipeline.py           # FramePipeline and VideoWorker thread
│   └── gui/
│       ├── __init__.py       # GUI package
│       ├── main_window.py    # Main Tkinter desktop interface
│       ├── video_canvas.py   # Letterboxed, responsive video canvas
│       ├── zone_dialog.py    # Modal dialog for zone configuration
│       └── alert_popup.py    # Reusable, non-blocking alert toast window
├── tests/
│   ├── conftest.py           # FakeDetector and synthetic video fixtures
│   ├── test_rules.py         # 12 comprehensive rule engine unit tests
│   ├── test_config.py        # Configuration persistence and repair tests
│   ├── test_alerts.py        # Snapshot and CSV logging tests
│   ├── test_pipeline.py      # Pipeline and worker threading tests
│   ├── test_architecture.py  # Architectural purity & boundary tests
│   └── test_gui_smoke.py     # GUI smoke test
├── tools/
│   └── check_detector.py     # Script to verify detector on sample image
├── clips/
│   └── README.md             # Guide for placing test video files
├── output/                   # Directory where run artifacts are saved
└── docs/
    ├── architecture.md       # Architecture specification and Mermaid diagrams
    ├── test_results.md       # Empirical benchmark evaluation template
    └── images/               # Directory for user demo screenshots
```

---

## Configuration Reference (`config.yaml`)

```yaml
version: 1
model: yolo11n.pt             # Pretrained YOLO model file
classes:                      # COCO ID to class name mapping
  0: person
  2: car
  3: motorcycle
settings:
  dwell_seconds: 2.0          # Seconds in zone before alert fires
  grace_seconds: 1.0          # Seconds track can flicker before timer resets
  confidence: 0.35            # Minimum object detection confidence
  imgsz: 480                  # YOLO inference image size (320, 480, 640)
  realtime: true              # Drop frames when processing lags
  save_video: false           # Export annotated MP4 video
  schedule_enabled: false     # Enforce active hours schedule
  sim_start: "12:00"          # Simulated clock at frame 0 (HH:MM)
  popup_on_alert: true        # Show toast pop-up on intrusion
  sound_on_alert: true        # Audio bell on intrusion
  max_work_width: 1280        # Maximum working frame width
  output_dir: output          # Base folder for run outputs
zones_by_video:               # Saved zones keyed by video basename
  example_clip.mp4:
    - name: Gate
      allowed: [car, motorcycle]
      active_from: null
      active_to: null
      points: [[0.12, 0.55], [0.48, 0.52], [0.52, 0.93], [0.10, 0.95]]
```

---

## Testing

Run the full automated test suite offline without internet or hardware display:
```powershell
pytest -v
```

The test suite validates:
- **`test_rules.py`**: Ray casting, concave polygons, edge/vertex hits, dwell timers, grace period continuation, re-arming, overlapping zones, schedule midnight crossing, and live parameter changes.
- **`test_config.py`**: Default generation, round-trip serialization, corrupt YAML backup (`.bak`), and geometry repairs.
- **`test_alerts.py`**: CSV formatting, immediate header creation, JPEG encoding, and summary generation.
- **`test_pipeline.py`**: Synthetic video pipeline execution, thread synchronization, and clean mid-run stopping.
- **`test_architecture.py`**: Enforces architectural boundaries (e.g. `rules.py` does not import OpenCV, Tkinter, Torch, or `time`).
- **`test_gui_smoke.py`**: Tests Tkinter window loading, canvas binding, and zone configuration.

---

## Decisions

The following implementation choices were resolved in accordance with Part B of the project specification:
1. **Virtual Environment Isolation:** Installed all dependencies inside `.venv` to avoid modifying global system packages.
2. **Default Zone Security Posture:** Newly drawn zones default to all checkboxes unchecked (`allowed: []`), ensuring an unmarked zone functions immediately as fully restricted.
3. **Corrupt Configuration Recovery:** If `config.yaml` is corrupted, invalid, or empty, the file is safely renamed to `config.yaml.bak` and a clean default configuration is created in its place.
4. **Normalized Coordinates:** Zone vertices are normalized (0.0 to 1.0) so user-drawn zones remain perfectly aligned regardless of display scaling or downscaled working dimensions.
5. **Anchor Point Positioning:** Evaluates zone containment using the bottom-center point of bounding boxes `((x1 + x2)/2, y2)` to accurately match where feet or tires touch the ground plane.
6. **Thread-Safe Architecture:** Worker thread communicates exclusively via thread-safe queues (`frame_queue` and `event_queue`). Tkinter widgets are strictly updated from the main GUI thread.
7. **Cross-Platform Path Encoding:** Snapshots and crop images use `cv2.imencode().tofile()` to prevent Windows Unicode path errors with non-ASCII directory names.

---

## Limitations

- **Classification vs. Identification:** The system classifies object categories (`person`, `car`, `motorcycle`) and checks zone permissions; it does not perform facial recognition or license plate identification.
- **Tracker ID Switches:** Occlusion or visual crossover may occasionally result in a tracker ID switch, which can trigger a new dwell cycle for the same individual.
- **Fixed Camera Requirement:** Zones are defined in 2D image coordinates; the system requires a stationary, non-panning surveillance camera.
- **Environmental Factors:** Extreme low light, glare, rain, or heavy blur will degrade detection accuracy and confidence.
- **Single Camera Stream:** Designed for single-camera offline evaluation; multi-camera network coordination is out of scope.

---

## Future Work

- Integration of live RTSP/IP camera streams and USB webcams.
- Facial recognition for whitelisted personnel and Optical Character Recognition (OCR) for license plates.
- Multi-camera spatial tracking and zone handover.
- Push notification integrations (Telegram bot, email alerts, Webhooks).
- Model fine-tuning with low-light / thermal surveillance datasets.
- Web dashboard and historical event search archive.

---

## Credits

- **Ultralytics YOLO:** Object detection model (Licensed under AGPL-3.0; see [ultralytics.com](https://ultralytics.com)).
- **ByteTrack:** Multi-object tracking algorithm.
- **OpenCV & Pillow:** Image and video processing.

