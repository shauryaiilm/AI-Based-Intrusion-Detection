# Intrusion Guard: Build Guide and Coding-Agent Prompt

A minimal, modular version of the Phase-I project **"AI-Based Intelligent Border Surveillance and Intrusion Detection System"** (IILM University, session 2026-27): a desktop app that watches recorded camera footage, finds people and vehicles, and flags the ones that are not allowed in a zone.

This file has three parts:

| Part | For whom | What it is |
|---|---|---|
| **A** | You (the human) | How to use this guide, step by step |
| **B** | The coding agent | The full build prompt. Give this part to the agent |
| **C** | You | Footage tips, demo script, how this feeds the college form, follow-up prompts if something goes wrong |

---

# PART A: How to use this guide

1. **Install Python 3.11** (3.10 to 3.12 all work) from python.org. On the first installer screen, tick **"Add Python to PATH"**. The installer includes tkinter, which the app's window needs.
   - Check: open a terminal and run `python --version`, then `python -m tkinter`. A tiny test window should appear.
2. **Make an empty folder** called `intrusion-guard` and open it in your coding agent (Claude Code, Cursor, Codex, Copilot agent mode, etc.).
3. **Put this `guide.md` in that folder** and tell the agent:
   > Read `guide.md`. Follow **Part B** exactly and build the whole project. Do not ask me questions. Where something is unclear, pick the simplest option that fits Part B and write that choice in the README under "Decisions".

   If your agent cannot read files, paste everything between the lines `===== BEGIN AGENT PROMPT =====` and `===== END AGENT PROMPT =====`.
4. **Internet is needed once**, to install the packages and to download the small YOLO model file (`yolo11n.pt`, about 6 MB).
5. When the agent finishes, run:
   ```
   pip install -r requirements.txt -r requirements-dev.txt
   pytest
   python app.py
   ```
6. Get 6 to 8 test clips (see Part C), draw zones, run them, and fill in `docs/test_results.md` with **your real observations**.
7. Push to GitHub, record a 1 to 2 minute demo video, and fill in the college form (Part C maps each form field to a file).

---

===== BEGIN AGENT PROMPT =====

# PART B: THE PROMPT

You are an expert Python engineer. Build the complete project described below, end to end, in the current empty folder. Read this entire prompt before writing code. Do not ask the user questions. If something is unclear, choose the simplest option that is consistent with this document and record the choice in `README.md` under a "Decisions" heading. The user is a student who will demo this app and submit it for a college project, so the result must **run on the first try on Windows 10/11** and be easy to read and explain.

## 1. Mission (one paragraph)

Build **Intrusion Guard**, a Python desktop app (tkinter GUI) that plays a **recorded video file** and, frame by frame, detects **people, cars and motorcycles**, tracks them with stable IDs, and checks them against **zones the user draws by hand on the video**. Each zone has a list of allowed object types and optional active hours. An object that stays in a zone that does not allow it (while the zone is active) for longer than a **dwell time** is an **unauthorised intrusion**: its box turns red, an alert appears in the window, a snapshot is saved as evidence, and a row is written to a CSV log. The app shows the video live with colour-coded boxes: **green = authorised/ignored, orange = suspicious (dwell timer running), red = unauthorised**.

## 2. Background (so you understand the intent)

This is the Phase-II prototype of a college major project whose Phase-I report states these objectives. The software must visibly demonstrate them:

1. Real-time AI surveillance for the perimeter and restricted areas of an office building.
2. A deep-learning object detector recognising people and vehicles.
3. **Virtual restricted regions and time-based rules** (e.g. a rule that is active only after office hours) so that entering a protected area counts as a possible intrusion.
4. Fewer false alarms, by separating ordinary activity from events needing attention (this is what the dwell time and grace period are for).
5. Alerts with **event time, location and captured visual evidence**.
6. **A modular design** so later versions can add facial recognition, mobile notifications and multiple cameras.
7. Evaluation under different lighting and conditions (the app must therefore make batch testing and result logging easy).

Scope limits from the report: office-building perimeters only, no biometrics, no control of physical access devices, limited training data and compute. This prototype uses a **pretrained model, no training**.

## 3. Hard constraints (do / do not)

**Do**
- Python **3.10 or newer** (must work on 3.10, 3.11, 3.12). Use `from __future__ import annotations` in every module.
- Run on **CPU only**. The target PC is a Ryzen 5 with integrated graphics, no NVIDIA GPU. Never require CUDA. Always pass `device="cpu"`.
- Work on **recorded video files** (mp4, avi, mov, mkv). Live camera support is *not* required.
- Use **tkinter** (+ `ttk`) for the GUI and **Pillow** to display frames. The GUI must be responsive at all times (see section 9).
- Detect **only** `person`, `car`, `motorcycle`. **Do not** detect bus, truck or bicycle. The class list must live in `config.yaml` as a mapping from COCO id to name so more classes can be added later by editing one line.
- Keep the code **simple and readable**: plain functions and small dataclasses, no frameworks, no metaprogramming, no async. A student must be able to explain every file.
- Make it **modular**: `rules.py` must not import `cv2`, `tkinter`, `torch` or `ultralytics`. `pipeline.py` and `rules.py` must not import `tkinter`. The detector must be replaceable through a tiny interface (section 8.3).
- Write **real tests** and run them (section 14).

**Do not**
- Do not implement number-plate recognition, face recognition, OCR, custom model training, databases, web servers, Docker, cloud calls, Telegram/email alerts, multi-camera, or live RTSP. These are explicitly out of scope (section 17).
- Do not add features not listed here.
- Do not name any module `types.py`, `queue.py`, `config.py` at the top level or anything else that shadows the standard library. (The package module `guard/config.py` is fine because it is inside the `guard` package.)
- **Never fabricate** test results, accuracy numbers, FPS numbers or screenshots in any document. Leave clearly marked placeholders for the user to fill in after real runs.

## 4. Environment and dependencies

`requirements.txt`:
```
ultralytics>=8.3
opencv-python>=4.8
numpy>=1.24
Pillow>=10.0
PyYAML>=6.0
lapx>=0.5
```
`requirements-dev.txt`:
```
pytest>=7.4
```
Notes:
- `ultralytics` brings in PyTorch. On Windows the default PyPI wheel is the CPU build. Do not add CUDA instructions.
- `lapx` is needed by the ByteTrack tracker inside Ultralytics. Listing it avoids a surprise auto-install in the middle of the demo.
- tkinter ships with the python.org installer. Do not list it in requirements.
- On first run Ultralytics downloads `yolo11n.pt` into the current working directory. Pre-download is a README step: `python tools/check_detector.py` triggers it.
- If `yolo11n.pt` cannot be loaded (older Ultralytics), `detector.py` falls back to `yolov8n.pt` and logs which one it used.

## 5. Project structure (create exactly this)

```
intrusion-guard/
├── app.py                    # GUI entry point:  python app.py
├── headless.py               # batch runner without a window:  python headless.py clips/a.mp4
├── config.yaml               # created with defaults on first run if missing
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .gitignore
├── guard/
│   ├── __init__.py
│   ├── schema.py             # dataclasses and enums shared everywhere (pure Python)
│   ├── config.py             # load/save config.yaml, validate, normalise zones
│   ├── detector.py           # YOLO + ByteTrack wrapper (the only file that imports ultralytics)
│   ├── rules.py              # zones, dwell, grace, schedule, status decisions (pure Python)
│   ├── draw.py               # drawing boxes, zones, labels, banners on frames (OpenCV)
│   ├── alerts.py             # snapshots, events.csv, summary.json, optional annotated video
│   ├── pipeline.py           # FramePipeline (one frame in, one result out) and VideoWorker (thread)
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py    # the main window and all event handling
│       ├── video_canvas.py   # canvas that shows frames, handles letterboxing and click mapping
│       ├── zone_dialog.py    # dialog for zone name, allowed types, active hours
│       └── alert_popup.py    # small non-blocking alert window
├── tests/
│   ├── conftest.py           # FakeDetector, synthetic video helper
│   ├── test_rules.py
│   ├── test_config.py
│   ├── test_alerts.py
│   ├── test_pipeline.py
│   ├── test_architecture.py  # import-boundary checks
│   └── test_gui_smoke.py     # skipped automatically when no display
├── tools/
│   └── check_detector.py     # downloads weights, runs detector on a sample image, prints detections
├── clips/
│   └── README.md             # "put your test videos here" (videos themselves are gitignored)
├── output/                   # created at runtime, gitignored
└── docs/
    ├── architecture.md       # diagram + explanation
    ├── test_results.md       # template for the user's real results
    └── images/               # empty; user adds screenshots (add .gitkeep)
```

`.gitignore` must ignore: `output/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.pt`, `clips/*` except `clips/README.md`, `.venv/`, `venv/`, `.idea/`, `.vscode/`.

## 6. Core concepts (use these exact words in code and docs)

- **Frame**: one picture from the video.
- **Working frame**: the frame after optional downscaling. If the video is wider than `max_work_width` (default 1280), resize it once on read, keeping the aspect ratio. Everything (detection, drawing, snapshots, saved video) uses the working frame.
- **Detection**: one object found in a frame: class name, confidence, box.
- **Track**: a detection with a stable `track_id` assigned by ByteTrack, so "person #3" stays #3 while moving.
- **Anchor point**: the bottom-centre of a box, `((x1 + x2) / 2, y2)`. It approximates where feet or wheels touch the ground, so it is used for all "is it inside the zone?" tests.
- **Zone**: a user-drawn polygon with a name, a list of allowed class names, and optional active hours.
- **Allowed list**: class names permitted inside that zone. Empty list means nothing is allowed (a fully restricted zone).
- **Active hours**: optional `HH:MM` start and end. The zone is enforced only inside this window. Windows that cross midnight (e.g. 19:00 to 07:00) must work.
- **Dwell time**: how long (in **video time**) a track must stay inside a zone that does not allow it before it counts as unauthorised. Default 2.0 seconds.
- **Grace period**: how long a track may be missing from the zone (detection flicker) without resetting its dwell timer. Default 1.0 seconds.
- **Video time**: `frame_index / fps`. **All timing logic uses video time, never wall-clock time**, so results are identical no matter how fast or slow the PC processes frames.
- **Simulated clock**: because the footage is recorded, the schedule uses a simulated time of day: `sim_time = (sim_start + video_time) mod 24h`, where `sim_start` is a user setting (default `12:00`) meaning "the clock time shown at the first frame". Only used when "Use schedule" is on.
- **Status** of a track (enum `Status`): `OUTSIDE`, `AUTHORISED`, `PENDING`, `UNAUTHORISED`.

## 7. Configuration file: `config.yaml`

Create it with these defaults when missing. Load with `yaml.safe_load`, save with `yaml.safe_dump(..., sort_keys=False)`, and write atomically (write to a temp file in the same folder, then `os.replace`). If the file is corrupt, rename it to `config.yaml.bak`, log a warning and use defaults.

```yaml
version: 1
model: yolo11n.pt
classes:            # COCO class id -> name. Edit here to add classes later.
  0: person
  2: car
  3: motorcycle
settings:
  dwell_seconds: 2.0
  grace_seconds: 1.0
  confidence: 0.35
  imgsz: 480
  realtime: true          # skip frames when the PC is slower than the video
  save_video: false       # write output/<run>/annotated.mp4
  schedule_enabled: false
  sim_start: "12:00"
  popup_on_alert: true
  sound_on_alert: true
  max_work_width: 1280
  output_dir: output
zones_by_video:           # zones are remembered per video file name
  example_clip.mp4:
    - name: Gate
      allowed: [car, motorcycle]
      active_from: null   # "HH:MM" or null
      active_to: null
      points:             # NORMALISED coordinates, 0..1 of width/height
        - [0.12, 0.55]
        - [0.48, 0.52]
        - [0.52, 0.93]
        - [0.10, 0.95]
```
Why normalised: zones then survive any resolution or downscaling. Convert to pixels only at the moment a run starts, using the working-frame size.

Validation on load (log a warning and fix, never crash): drop zones with fewer than 3 points; clamp points to [0,1]; drop unknown class names from `allowed`; make zone names unique within a video (append " 2", " 3"); validate `HH:MM` strings (both or neither of `active_from`/`active_to`); clamp numeric settings to sane ranges (`dwell_seconds` 0.1 to 60, `confidence` 0.05 to 0.95, `imgsz` one of 320/480/640).

The zones key is the video's **file name** (basename). A video with no entry has no zones.

## 8. Module specifications

### 8.1 `guard/schema.py` (pure Python, no third-party imports)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class Status(str, Enum):
    OUTSIDE = "outside"
    AUTHORISED = "authorised"
    PENDING = "pending"
    UNAUTHORISED = "unauthorised"

@dataclass(frozen=True)
class Detection:
    track_id: int
    cls_name: str
    conf: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in working-frame pixels

    @property
    def anchor(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

@dataclass
class Zone:
    name: str
    points: list[tuple[float, float]]        # normalised 0..1
    allowed: list[str] = field(default_factory=list)
    active_from: str | None = None           # "HH:MM"
    active_to: str | None = None

@dataclass
class TrackState:
    detection: Detection
    status: Status
    zone_name: str | None        # the zone that decided the status, if any
    dwell_s: float               # seconds inside the violating zone (0 if none)
    reason: str                  # short human text, e.g. "car allowed in Gate"

@dataclass
class AlertEvent:                # produced by rules.py, no file paths yet
    track_id: int
    cls_name: str
    zone_name: str
    video_time_s: float
    sim_time: str | None         # "HH:MM:SS" if schedule enabled, else None
    dwell_s: float
    conf: float
    frame_idx: int
    box: tuple[float, float, float, float]

@dataclass
class SavedAlert:                # produced by alerts.py after files are written
    event: AlertEvent
    alert_no: int
    wall_time: str               # "YYYY-MM-DD HH:MM:SS"
    snapshot_path: str
    crop_path: str

@dataclass
class Settings:                  # mirrors config.yaml "settings"; defaults as in section 7
    dwell_seconds: float = 2.0
    grace_seconds: float = 1.0
    confidence: float = 0.35
    imgsz: int = 480
    realtime: bool = True
    save_video: bool = False
    schedule_enabled: bool = False
    sim_start: str = "12:00"
    popup_on_alert: bool = True
    sound_on_alert: bool = True
    max_work_width: int = 1280
    output_dir: str = "output"
```
Add any further small dataclasses you need here (`FrameResult`, `RunSummary`, worker message types).

### 8.2 `guard/config.py`

Functions: `load_config(path) -> AppConfig`, `save_config(cfg, path)`, `get_zones(cfg, video_name) -> list[Zone]`, `set_zones(cfg, video_name, zones)`, plus the validation from section 7. `AppConfig` holds `model`, `classes: dict[int, str]`, `settings: Settings`, `zones_by_video`. Save zones **immediately** whenever the user adds, edits or deletes one, and save settings when the window closes.

### 8.3 `guard/detector.py`: the only file that imports `ultralytics`

Define a tiny interface so tests and future models can replace it:

```python
class DetectorProtocol(Protocol):
    def detect(self, frame_bgr: "np.ndarray") -> list[Detection]: ...
    def set_confidence(self, conf: float) -> None: ...
```

`YoloDetector(model_name, class_map, conf, imgsz)`:
- Loads `YOLO(model_name)`; on failure tries `yolov8n.pt`; logs the model used. Loading may download weights, so it is called from the worker thread, never the GUI thread.
- `detect()` calls exactly:
  ```python
  results = self._model.track(
      frame_bgr, persist=True, tracker="bytetrack.yaml",
      classes=list(self._class_map.keys()), conf=self._conf, imgsz=self._imgsz,
      device="cpu", verbose=False,
  )
  r = results[0]
  ```
- If `r.boxes is None` or `r.boxes.id is None`, return `[]`. Otherwise read `r.boxes.xyxy`, `r.boxes.id`, `r.boxes.cls`, `r.boxes.conf` (`.cpu().numpy()`, ids and classes as ints) and build `Detection` objects with names from **`class_map`**, not from the model.
- Boxes without an id are ignored.
- **A new `YoloDetector` instance is created for every run** (every press of Start) so tracker state never leaks between videos or replays. This is the simplest correct approach. Loading takes about a second.
- Wrap the call in try/except: on an unexpected error, log it and re-raise as `DetectorError` so the worker can report it to the GUI.

### 8.4 `guard/rules.py`: the brain (pure Python, heavily unit-tested)

```python
class RuleEngine:
    def __init__(self, zones: list[Zone], frame_size: tuple[int, int],
                 dwell_seconds: float, grace_seconds: float = 1.0,
                 schedule_enabled: bool = False, sim_start: str = "12:00"): ...
    def set_dwell_seconds(self, v: float) -> None: ...      # live-editable
    def update(self, detections: list[Detection], frame_idx: int,
               video_time_s: float) -> tuple[list[TrackState], list[AlertEvent]]: ...
    def zone_states(self, video_time_s: float) -> dict[str, "ZoneInfo"]: ...  # for drawing
```
Pure helper functions (module level, individually tested):
- `point_in_polygon(x, y, polygon) -> bool`: **ray casting implemented by hand, no cv2**. Points exactly on an edge or vertex count as inside.
- `parse_hhmm(s) -> int` seconds since midnight (accept `HH:MM` and `HH:MM:SS`; raise `ValueError` on bad input).
- `is_zone_active(zone, sim_seconds: int | None) -> bool`:
  - schedule disabled (`sim_seconds is None`) or zone has no hours: **True**.
  - `from < to`: active if `from <= s < to`.
  - `from > to` (crosses midnight): active if `s >= from or s < to`.
  - `from == to`: treat as always active.
- `format_hhmmss(seconds) -> str`.

**`update()` algorithm (implement exactly):**

```
sim_s = (parse(sim_start) + video_time_s) % 86400  if schedule_enabled else None
for each detection d:
    ax, ay = d.anchor
    containing = zones whose pixel polygon contains (ax, ay)          # in zone order
    if none:
        status = OUTSIDE, zone_name = None, reason = "outside all zones"
    else:
        active = [z for z in containing if is_zone_active(z, sim_s)]
        if none:
            status = AUTHORISED, reason = "zone not active now"
        else:
            violating = [z for z in active if d.cls_name not in z.allowed]
            if none:
                status = AUTHORISED, zone_name = active[0].name,
                reason = f"{d.cls_name} allowed in {zone_name}"
            else:
                z = violating[0]
                key = (d.track_id, z.name)
                rec = records.get(key)
                if rec is None or video_time_s - rec.last_seen > grace_seconds:
                    rec = Record(first_seen=video_time_s, last_seen=video_time_s, alerted=False)
                    records[key] = rec            # brand-new entry -> dwell restarts, alert re-armed
                rec.last_seen = video_time_s
                dwell = video_time_s - rec.first_seen
                if dwell >= dwell_seconds:
                    status = UNAUTHORISED
                    if not rec.alerted:
                        rec.alerted = True
                        emit AlertEvent(...)
                else:
                    status = PENDING
                zone_name = z.name, reason = f"{d.cls_name} not allowed in {z.name}"
prune records whose last_seen < video_time_s - 30.0
```
Consequences that tests must lock in:
- One alert per (track, zone) entry. A long stay produces **one** alert, not one per frame.
- Leaving and coming back **within** the grace period continues the same dwell timer and does not alert twice.
- Leaving for **longer** than the grace period, then returning, starts a new timer and may alert again.
- A track whose class is allowed in the zone never alerts.
- A track inside an **inactive** zone (schedule) never alerts and is green.
- If several zones overlap, the first violating active zone decides.
- Detections are processed independently; the order of the input list does not change the outcome.

Pixel conversion: the constructor converts each zone's normalised points to pixel tuples using `frame_size = (width, height)` once.

`zone_states()` returns, per zone name, whether the zone is currently active (for the grey "[inactive]" label) and whether any track is currently UNAUTHORISED inside it (for the red outline). Keep what `update()` last computed; do not recompute.

### 8.5 `guard/draw.py`: all OpenCV drawing in one place

Functions (all return nothing, drawing in place on a BGR frame, or return a new frame where stated):
- `draw_zones(frame, zones_px, active_map, violating_map)`: semi-transparent fill (alpha about 0.18) plus a 2 px outline plus the zone name near the first point on a small filled label.
  - Normal zone: sky blue, BGR `(255, 180, 0)`.
  - Zone with a currently UNAUTHORISED track: red `(0, 0, 255)`.
  - Zone not active (schedule): grey `(160, 160, 160)` and the label gets the suffix ` [inactive]`.
- `draw_tracks(frame, states)`: for each `TrackState` draw the box and a label with a filled background.
  - Colours (BGR): AUTHORISED and OUTSIDE green `(0, 200, 0)`; PENDING orange `(0, 165, 255)`; UNAUTHORISED red `(0, 0, 255)`.
  - Box thickness 2 (3 for UNAUTHORISED). Label: `"{cls} #{id}"` plus `" · {dwell:.1f}s"` for PENDING and `" · UNAUTHORISED"` for UNAUTHORISED. A small filled dot at the anchor point.
  - Keep text legible: `cv2.FONT_HERSHEY_SIMPLEX`, scale about 0.5 to 0.6, with a filled rectangle behind the text; clamp the label inside the frame.
- `draw_info_bar(frame, text)`: a dark strip across the top with white text. Used for snapshots and the saved video: `"t=12.4s | frame 310 | sim 22:00:12"`.
- `annotate_alert(frame, saved_text)`: adds a red banner on the top of the snapshot with `UNAUTHORISED {cls} #{id} in '{zone}' | video t=.. | sim .. | wall time ..`.
- Drawing code never converts colour spaces for display; `video_canvas.py` does the BGR to RGB conversion.

### 8.6 `guard/alerts.py`

`RunRecorder(run_dir, video_name, fps, frame_size, save_video: bool)`:
- Creates `output/run_YYYYmmdd_HHMMSS/` with a `snapshots/` subfolder, and `events.csv` **with its header written immediately** (even if no alert ever happens).
- `save_alert(event, annotated_frame, raw_frame) -> SavedAlert`:
  - Increments `alert_no` (starting at 1).
  - Saves the **full annotated frame** (zones and boxes drawn, plus the red alert banner) as `snapshots/alert_{no:04d}_{cls}_id{track}_{zone}.jpg`, and a **crop of the intruder from the raw, un-annotated frame** (box padded by 10 percent, clamped to the frame) as `..._crop.jpg`.
  - Zone name in file names is sanitised with `re.sub(r"[^A-Za-z0-9_-]+", "_", name)`.
  - Appends a row to `events.csv` and **flushes** immediately. Columns, in this order:
    `alert_no, wall_time, video_time_s, sim_time, track_id, class, zone, dwell_s, confidence, frame_idx, snapshot_file, crop_file, video_file`
  - Image writing must be safe for non-ASCII Windows paths: use `cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(path)` instead of `cv2.imwrite`.
- `write_frame(annotated_frame)`: if `save_video`, append to `annotated.mp4` using `cv2.VideoWriter(..., cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)`.
- `finish(stats) -> dict`: closes files and writes `summary.json` with: `video`, `started_at`, `ended_at`, `video_duration_s`, `frames_total`, `frames_processed`, `frames_skipped`, `avg_processing_fps`, `settings` (as used), `zones` (names and allowed lists), `alerts_total`, `alerts_by_class`, `unique_tracks_by_class`, `first_alert_video_time_s` (or null).

### 8.7 `guard/pipeline.py`

**`FramePipeline`** handles one frame at a time, with **no threads and no GUI**:

```python
class FramePipeline:
    def __init__(self, detector: DetectorProtocol, engine: RuleEngine,
                 recorder: RunRecorder, zones_px, class_names): ...
    def process(self, frame_bgr, frame_idx: int, video_time_s: float) -> FrameResult: ...
```
Order inside `process()`:
1. `detections = detector.detect(frame)`
2. `states, events = engine.update(detections, frame_idx, video_time_s)`
3. If `events` is non-empty, first take `raw = frame.copy()` (an untouched copy, used for the intruder crop; skip the copy otherwise to save time). Then draw zones, then tracks, then the small info bar, in place on the frame; that drawn frame is the annotated frame.
4. For each event: `saved = recorder.save_alert(event, annotated, raw)`. (The alert banner is drawn on a **copy** used for the snapshot, so the live view stays clean.)
5. `recorder.write_frame(annotated)` if saving video.
6. Count unique track ids per class for the summary, and return `FrameResult(annotated_frame, states, saved_alerts, counts_now, frame_idx, video_time_s)`.

**`VideoWorker(threading.Thread)`**: see the threading and timing rules in section 9.

### 8.8 `guard/gui/*`: see section 10.

### 8.9 `headless.py`

Command line (use `argparse`):
```
python headless.py clips/a.mp4 [clips/b.mp4 ...] [--config config.yaml] [--dwell 2.0] [--conf 0.35]
                   [--imgsz 480] [--schedule --sim-start 22:00] [--save-video] [--output output]
```
- For each video: load zones for that file name from the config. If it has none, print `"No zones saved for <name>. Open it in the GUI (python app.py) and draw zones first."` and continue with the next file.
- Process **every** frame (no skipping), using `FramePipeline` directly with the real `YoloDetector`, writing the same outputs as the GUI (snapshots, `events.csv`, `summary.json`).
- Print one line per video: `name | duration | frames | avg FPS | alerts | first alert time`, and at the end a Markdown table of those lines that the user can paste into `docs/test_results.md`.

### 8.10 `tools/check_detector.py`

Downloads `https://ultralytics.com/images/bus.jpg` (skip politely with a message if offline), runs `YoloDetector` once on it, and prints the detections. This confirms weights download and detection work. Expect to see several `person` detections and **no** bus (bus is intentionally excluded).

## 9. Threading, timing and performance (the part most likely to go wrong, so follow it exactly)

**Rule: tkinter widgets are only touched from the main thread.** The worker never calls any tkinter function.

**Architecture**
- Main thread: tkinter event loop. A repeating `root.after(15, self._poll)` pulls results from queues.
- `VideoWorker` thread: opens the video, loads the model, reads frames, runs `FramePipeline.process`, pushes results.
- Frame queue: `queue.Queue(maxsize=2)` of `FramePacket(frame_rgb_or_bgr, frame_idx, total_frames, video_time_s, stats)`. If it is full, **drop the oldest** (`get_nowait`) then `put`, so the GUI always shows the newest frame and the worker never blocks on the GUI.
- Event queue (unbounded): messages `AlertMsg(saved_alert)`, `StatusMsg(text)`, `ErrorMsg(text)`, `DoneMsg(summary_dict)`.
- Controls: `stop_event` and `pause_event` (`threading.Event`). Pause is implemented by waiting on the event in the loop (`wait(0.05)` in a loop that also checks `stop_event`).
- `_poll()` drains the event queue fully each tick, takes **only the newest** frame packet, renders it, and reschedules itself. It must finish in a few milliseconds.
- Live settings: the worker reads `dwell_seconds` and `confidence` from a shared `Settings` object each frame (calling `engine.set_dwell_seconds()` and `detector.set_confidence()` when they change), so the sliders work during playback. Other settings (image size, realtime, save video, schedule, sim start) apply on the next Start.
- Closing the window while running: set `stop_event`, `join(timeout=3)`, then destroy. Use `daemon=True` for the thread as a safety net.
- Any exception in the worker is caught at the top of `run()` and sent as `ErrorMsg` (with a readable message), followed by `DoneMsg`. The GUI shows a `messagebox.showerror` and returns to the idle state. Never let the thread die silently.

**Reading and pacing loop (implement this logic):**
```
cap = cv2.VideoCapture(path)             # fail with a clear ErrorMsg if not cap.isOpened()
fps = cap.get(cv2.CAP_PROP_FPS); if fps is NaN or fps < 1: fps = 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # may be 0/-1 for odd files; handle it
realtime = settings.realtime and not settings.save_video   # saving video forces "process every frame"
t0 = time.perf_counter(); frame_idx = 0
while not stop_event.is_set():
    handle pause (while paused, wait; when resuming, shift t0 forward by the paused duration)
    if not cap.grab(): break                             # end of video
    video_t = frame_idx / fps
    if realtime:
        due = t0 + video_t
        now = time.perf_counter()
        if now < due: sleep (in small slices, checking stop/pause) until due
        elif now - due > 1.0 / fps: frame_idx += 1; skipped += 1; continue   # behind: skip cheaply
    ok, frame = cap.retrieve(); if not ok: break
    frame = downscale_if_needed(frame, max_work_width)
    result = pipeline.process(frame, frame_idx, video_t)
    push packet; frame_idx += 1; processed += 1
```
So: on a fast PC the video plays at its natural speed; on a slow PC frames are skipped automatically and playback still keeps real-time pace. With "Real-time" off, every frame is processed (slower but exhaustive). The dwell logic is unaffected because it uses video time.

**Speed settings for CPU**
- Model `yolo11n.pt`, `imgsz=480` by default (selectable 320/480/640).
- Do not run any extra per-frame work (no per-frame disk writes, no per-frame copies except when an alert fires, no per-frame logging above DEBUG).
- Target: at least 8 processed frames per second on a 6-core Ryzen 5 at `imgsz=480`. This is a target, not a hard test. If it is lower the app must still be fully responsive.

## 10. The GUI (tkinter + ttk)

Window title `Intrusion Guard`, initial size 1280x760, minimum size 1100x650, resizable. Use `ttk` widgets. Font: default system font, 10 pt. Use a dark video area (`#111111`) and the default light theme elsewhere. Keep it tidy and aligned, no overlapping widgets at any window size.

**Layout**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ [Open Video] [Start] [Pause] [Stop]                 status banner (colour)  │
├──────────────────────────────────────────────┬──────────────────────────────┤
│                                              │ Zones                        │
│                                              │  [list]  [Add][Edit][Delete] │
│            VIDEO CANVAS                      │          [Clear all]         │
│   (frame fitted, letterboxed, boxes/zones)   │ Settings                     │
│                                              │  Dwell time (s)  [spinbox]   │
│                                              │  Confidence      [slider]    │
│                                              │  Image size      [combobox]  │
│                                              │  [x] Real-time (skip frames) │
│                                              │  [x] Use schedule  Sim start │
│                                              │  [ ] Save annotated video    │
│                                              │  [x] Pop-up on alert         │
│                                              │  [x] Beep on alert           │
│                                              │ Live counters                │
│                                              │ Alerts (table) + preview     │
├──────────────────────────────────────────────┴──────────────────────────────┤
│ file name | frame 310/900 | processing FPS 11.2 | video time 12.4s | sim ... │
└─────────────────────────────────────────────────────────────────────────────┘
```
The video area gets all extra space (`grid` weights); the right panel is a fixed width of about 340 px.

**Buttons and states** (enable/disable correctly at all times)
| State | Open Video | Start | Pause | Stop | Add/Edit/Delete zone |
|---|---|---|---|---|---|
| No video | on | off | off | off | off |
| Video loaded, idle | on | on | off | off | on |
| Running | off | off | on ("Pause") | on | off |
| Paused | off | off | on ("Resume") | on | off |
| Finished | on | on (replays from start) | off | off | on |

**Open Video:** `filedialog.askopenfilename` (mp4, avi, mov, mkv). Read the first frame (downscaled to the working size) and show it still, with the zones saved for that file name drawn on it. Show the file name in the status bar. Show an error dialog if the file cannot be opened.

**Drawing a zone** (Add Zone button, only when idle/finished):
1. Re-show the first frame as a still. Enter drawing mode: the status banner says `Drawing zone: left-click to add points, right-click or Enter to finish, Backspace to undo, Esc to cancel`.
2. Left-click adds a point (small circle) and a line from the previous point; a faint line follows the mouse to the first point. Right-click, double-click or Enter finishes if there are at least 3 points; otherwise show a hint. Backspace removes the last point. Esc cancels.
3. On finish, open `ZoneDialog` (modal): name (default "Zone N", must be unique and non-empty), checkboxes for each class in `classes` (allowed types; **all unticked by default**), and optional active hours `From` and `To` entries (`HH:MM`, both or neither, validate and show an inline error).
4. Save the zone to `config.yaml` immediately, redraw the still with all zones, and list it in the zones list as `Gate  (allowed: car, motorcycle | 19:00-07:00)` or `(allowed: none)`.
- **Edit** opens the same dialog for the selected zone (name, allowed types, hours; points are not editable). **Delete** removes the selected zone after a confirmation. **Clear all** asks for confirmation.
- Points are stored as normalised coordinates. `VideoCanvas` must provide `canvas_to_norm(cx, cy)` and `norm_to_canvas(nx, ny)` that account for the letterbox offset and scale, clamp to the image area, and ignore clicks in the black bars.

**Video canvas** (`video_canvas.py`)
- One `tk.Canvas`; one image item updated with `itemconfig` (do not recreate it every frame). Keep a reference to the `ImageTk.PhotoImage` so it is not garbage collected.
- Scale = `min(canvas_w / frame_w, canvas_h / frame_h)`; centre the image (letterbox). Use `cv2.resize(..., interpolation=cv2.INTER_AREA)` when shrinking and `INTER_LINEAR` when enlarging. Re-render the last frame on `<Configure>` (throttled), so resizing the window works while paused or stopped.
- Convert BGR to RGB with `cv2.cvtColor` before `Image.fromarray`.

**Right panel behaviour**
- **Dwell time** spinbox (0.5 to 30, step 0.5, default from config) and **Confidence** slider (0.10 to 0.90, step 0.05): both apply live while running.
- **Image size** combobox (320, 480, 640). **Real-time** checkbox. **Save annotated video** checkbox (when ticked, show a small note: "processes every frame, may run slower than real time").
- **Use schedule** checkbox and **Sim start** entry (`HH:MM`, validated; tooltip or label: "clock time at the first frame"). When off, all zones are always active.
- **Live counters:** `Tracking now: 2 person, 1 car, 0 motorcycle` (generated from the class list) and `Alerts: 3`.
- **Alerts table** (`ttk.Treeview`, columns `#`, `Video time`, `Type`, `ID`, `Zone`, with a vertical scrollbar). New alerts are appended at the top. Selecting a row shows the **crop thumbnail** under the table (about 160 px wide). Double-clicking a row opens the full snapshot in a separate `Toplevel` window that fits the image to the screen (title shows the file name; Esc closes it).

**Status banner** (top right, a coloured label): `READY - open a video`, `MONITORING` (green), `PAUSED` (grey), `INTRUSION: person #3 in 'Gate'` (red background, white bold text) for 4 seconds after an alert and then back to MONITORING, `FINISHED - 2 alerts` (blue/grey), `ERROR: ...` (red).

**On every alert (handled in the main thread when the `AlertMsg` is polled)**
1. Append to the alerts table and update counters.
2. Flash the banner red for 4 seconds.
3. If "Pop-up on alert" is on, show `AlertPopup`: a **single reusable, non-modal** `Toplevel` (about 360x250) with the crop thumbnail and text (`UNAUTHORISED person #3`, zone, time). It sits at the bottom-right of the main window, is `-topmost`, auto-hides after 5 seconds, and when a new alert arrives it updates in place and restarts the timer (never stacks). It must **never** pause the video, never use `grab_set`, and after it appears give focus back to the main window (`root.focus_force()`) so keyboard shortcuts keep working.
4. If "Beep on alert" is on, call `root.bell()`.

**Keyboard shortcuts:** `Space` pause/resume, `Ctrl+O` open video, `Esc` cancel zone drawing or close the snapshot window, `Enter` finish the polygon, `Backspace` undo last point.

**Finish / Stop:** when the video ends or the user presses Stop: finalise the run (`summary.json`, close the video writer), set the state to Finished (or idle), show the final banner and a small info dialog or status text with the path of the output folder (`output/run_.../`) and a button or menu item **"Open output folder"** (`os.startfile` on Windows, `xdg-open`/`open` elsewhere, wrapped in try/except).

**Menu bar** (small): `File` → Open Video, Open Output Folder, Exit. `Help` → About (project title and one line description).

**Persistence:** window size is not persisted. Settings are saved to `config.yaml` on exit and when Start is pressed. The last-used video folder may be remembered (optional).

## 11. Outputs per run (all under `output/run_YYYYmmdd_HHMMSS/`)

- `events.csv`: one row per alert (columns in section 8.6).
- `snapshots/alert_0001_person_id3_Gate.jpg` and `..._crop.jpg`.
- `annotated.mp4`: only if "Save annotated video" is on.
- `summary.json`: run statistics (section 8.6).
These are the evidence files the user will upload to Google Drive for the college submission, so keep the names and columns stable and human-readable.

## 12. Error handling and edge cases (handle all, none may crash the app)

- Video cannot be opened or has zero frames: clear dialog, state returns to "no video".
- FPS metadata missing or 0: assume 25.
- Frame count unknown: progress shows `frame N` without a total.
- A frame fails to decode mid-video: stop cleanly as if the video ended and log it.
- No zones for the video: still run and draw boxes (all green), and show a one-line hint in the status text: `No zones drawn: nothing can be flagged. Use Add Zone.`
- Zero detections in a frame: just show the frame.
- Very small boxes or boxes partly outside the frame: clamp when cropping and drawing.
- Starting a second run after the first finished: new run folder, new detector instance, counters reset, alert table cleared (ask nothing).
- Model weights missing and no internet: show an error dialog explaining `Could not load the YOLO model. Connect to the internet once so it can download yolo11n.pt, or place the file in the project folder.`
- Non-ASCII paths: all image writes use `imencode(...).tofile`; for reading videos, if `isOpened()` is false and the path has non-ASCII characters, mention that in the error text.
- Zones file validation errors never crash; they log and repair (section 7).
- Closing the window at any moment (including during model load) exits cleanly with no stack trace and no zombie thread.

## 13. Code quality

- Type hints on all public functions, short docstrings, comments that explain **why**, not what.
- Use the `logging` module (logger name `guard`), one `logging.basicConfig` in `app.py` and `headless.py`. No `print` in library code (only `headless.py` prints its table).
- No global mutable state. Constants (colours, defaults) live at the top of the module that uses them or in `schema.py`.
- Each module under about 400 lines; split if larger.
- No unused imports or dead code. If `ruff` or `flake8` is available, run it and fix findings.
- Names: `snake_case` functions, `PascalCase` classes.

## 14. Tests (write them, run them, make them pass)

Use `pytest`. Tests must run **without a display, without internet, and without downloading any model**. Use a `FakeDetector` (in `conftest.py`) that returns scripted `Detection` lists by frame index. Also provide a helper that writes a small synthetic video (e.g. 320x240, 10 fps, 4 seconds, plain background with a moving rectangle) using `cv2.VideoWriter`.

`tests/test_rules.py` (the most important file; at least these cases):
1. `point_in_polygon`: inside, outside, on an edge, on a vertex, concave polygon.
2. A person walks from outside into a restricted zone and stays: statuses go `OUTSIDE` → `PENDING` → `UNAUTHORISED`; exactly **one** `AlertEvent`.
3. A person touches the zone for less than the dwell time and leaves: no alert.
4. A car in a zone that allows `car`: always `AUTHORISED`, no alert. A person in the same zone: alert.
5. A zone with empty `allowed`: any class alerts.
6. Grace period: the object leaves for 0.5 s (less than grace) and comes back: the dwell timer continues and there is **no second alert**.
7. Re-arm: the object leaves for longer than grace, returns, stays past dwell: a **second** alert.
8. Schedule: with `schedule_enabled` and hours 19:00-07:00, `sim_start=12:00`: no alert at sim 12:00; alert when sim time is 20:00; the midnight-crossing window works (e.g. 23:30 and 03:00 are active, 12:00 is not); a zone without hours is always active.
9. Overlapping zones: the first violating active zone decides the status.
10. Two tracks at once are handled independently; shuffling the detection list changes nothing.
11. Uses **video time**: feeding the same scripted sequence with the same video times gives the same result regardless of real elapsed time (no `time.time()` inside `rules.py`; assert this by reading the source file in the architecture test).
12. `dwell_seconds` changed live via `set_dwell_seconds` takes effect on the next update.

`tests/test_config.py`: defaults created when the file is missing; save then load round trip keeps zones and settings; a corrupt file is renamed to `.bak` and defaults are used; invalid zones (2 points, out-of-range points, unknown class, duplicate names, bad `HH:MM`) are repaired or dropped; numeric clamps.

`tests/test_alerts.py`: `RunRecorder` creates the folder, writes the CSV header immediately, `save_alert` writes both JPEGs (readable back with `cv2.imdecode`), the CSV row has the right values and column order, file names are sanitised (zone name like `Gate / North?`), and `finish()` writes a valid `summary.json` with the keys from section 8.6.

`tests/test_pipeline.py`: (a) run `FramePipeline` over a synthetic video with a `FakeDetector` that moves a person into a zone: exactly one saved alert, snapshot exists, CSV has exactly one data row. (b) Run a real `VideoWorker` with `FakeDetector` on the synthetic video with `realtime=False`: collect queue messages until `DoneMsg` (timeout 20 s); assert one `AlertMsg`, at least one frame packet, `DoneMsg` summary present, and that the thread has ended. (c) Stop mid-run: `stop_event` ends the worker quickly (under 3 s) with no exception.

`tests/test_architecture.py`: read source text and assert: `guard/rules.py` does not import `cv2`, `tkinter`, `torch`, `ultralytics`, or `time`; `guard/pipeline.py` and `guard/alerts.py` do not import `tkinter`; only `guard/detector.py` imports `ultralytics`; no file in the repo is named `types.py` or `queue.py`.

`tests/test_gui_smoke.py`: try to create `tk.Tk()`; **skip** the test on `TclError` (no display). Otherwise build the main window, call `update_idletasks()`, load a synthetic video, assert the Start button becomes enabled, simulate adding a zone through the model/controller method (not by real clicks), and destroy the window.

Also run these checks yourself before you finish: `python -m compileall guard app.py headless.py`, `pytest -q`, and, if internet is available, `python tools/check_detector.py`. If you can run the GUI (for example under Xvfb on Linux), start it, open a synthetic video, draw a zone programmatically, run it with a fake detector or the real one, and confirm the red status and alert row appear. If you cannot run the GUI in your environment, say so plainly in your final report instead of claiming it works.

## 15. Documentation deliverables

**`README.md`** (clear and student-friendly), sections in this order:
1. Title and one-paragraph description. Mention that it is the prototype for "AI-Based Intelligent Border Surveillance and Intrusion Detection System".
2. Screenshots placeholder: `docs/images/` with a note `TODO: add screenshots after the first run` (do not invent images).
3. Features (bullets, matching section 1).
4. How it works: the 6-step flow (read frame → detect → track → zone check → rules → alert) in plain words, plus the colour legend.
5. Requirements and installation on Windows (Python 3.10-3.12, `python -m venv .venv`, activate, `pip install -r requirements.txt`), including the one-time model download and `python tools/check_detector.py`.
6. Quick start: `python app.py` → Open Video → Add Zone (with the drawing instructions) → Start. Explain what each setting does in one line.
7. Zone rules explained with a small table of example zones (Walkway: person; Vehicle gate: car, motorcycle; Restricted area: nothing) and what turns red.
8. Batch testing with `headless.py` and what the output files are.
9. Project structure (the tree from section 5 with one-line descriptions).
10. Configuration reference (`config.yaml` fields).
11. Testing: `pytest`.
12. **Decisions**: every ambiguity you resolved yourself.
13. Limitations (be honest): detects any person/vehicle of a disallowed class and cannot tell who is authorised; tracker ID switches can cause a duplicate alert; reduced accuracy at night, in blur or heavy occlusion; pretrained model only; fixed camera required because zones are drawn in screen space; single camera and recorded video only.
14. Future work: live RTSP/webcam source, multiple cameras, authorised-vehicle list or number-plate recognition, face recognition for authorised persons, Telegram/email/mobile alerts, web dashboard, fine-tuning the model on night footage, video clip evidence around alerts.
15. Credits: Ultralytics YOLO (note it is licensed AGPL-3.0, see their site), OpenCV, ByteTrack. Do **not** create a LICENSE file; leave that choice to the author.

**`docs/architecture.md`:** a Mermaid flowchart of the pipeline (`Video file → Reader → Detector (YOLO) → Tracker (ByteTrack) → Rule engine (zones, dwell, grace, schedule) → Drawing → GUI / Alert recorder (snapshot, CSV, summary)`), the thread diagram (worker thread, queues, main thread), a table of modules with their single responsibility, and a short "how to extend" section (add a class, swap the detector, add an alert channel).

**`docs/test_results.md`:** a **template** only. Include: the test environment block (CPU, RAM, OS, Python version, model, imgsz, dwell, grace, conf, all with `TODO` values), a results table with columns `Clip | Scenario | Lighting | Real intrusions | Alerts fired | Correct | False alerts | Missed | First alert at (s) | Avg FPS | Notes`, a short definition of each column, 6 to 8 empty example rows with scenario suggestions (person enters zone day; person walks outside zone; car in allowed zone; car in person-only zone; person at night; brief touch below dwell time; person leaves and returns; crowded scene), and a "Findings and failures" section with `TODO`. Add this line at the top: `All numbers in this file must come from real runs. Do not estimate.`

## 16. Quality bar

Before declaring done, re-read this prompt top to bottom and tick every item in section 19. The user will give this app to evaluators; a window that freezes, boxes that do not line up with the drawn zone, or a crash on Start is a failure. Prefer a smaller feature that works over a larger one that does not.

## 17. Explicitly out of scope (do not build, do not stub, only mention in "Future work")

Number-plate recognition and OCR; face or person identification; model training or fine-tuning; live camera, webcam or RTSP input; multiple cameras; databases; web or mobile apps; Docker; email, SMS, Telegram or push alerts; cloud services; installer or `.exe` packaging; seeking or scrubbing the video timeline; editing zone points after creation (delete and redraw instead); bus, truck, bicycle or any class other than person, car, motorcycle (the config makes adding them a one-line change).

## 18. Work order (follow it, and run the code after each step)

1. Scaffold folders, `requirements*.txt`, `.gitignore`, empty modules, `schema.py`, `config.py` + `test_config.py`. Run the tests.
2. `rules.py` + `test_rules.py`. Get all rule tests green **before** moving on.
3. `draw.py`, `alerts.py` + `test_alerts.py`.
4. `detector.py` + `tools/check_detector.py`. Run it if internet is available.
5. `pipeline.py` (`FramePipeline` first, then `VideoWorker`) + `test_pipeline.py`.
6. `headless.py`. Try it on a synthetic video with a hand-made config to confirm outputs are produced.
7. GUI: `video_canvas.py`, then `main_window.py` (open video, show frame, zone drawing, start/pause/stop, live view), then `zone_dialog.py`, then `alert_popup.py`, then right-panel widgets. Run the GUI smoke test.
8. `app.py`, README, docs, `.gitignore` check.
9. Full pass: `compileall`, `pytest -q`, lint if available, re-read this prompt against the code.

## 19. Definition of done (all must be true)

- [ ] `pip install -r requirements.txt -r requirements-dev.txt` then `pytest -q` passes with no network and no display (GUI smoke test may be skipped).
- [ ] `python app.py` opens the window on Windows 10/11 with Python 3.10 to 3.12.
- [ ] Open Video shows the first frame; Add Zone lets me click a polygon, finish it, name it and choose allowed types; zones persist in `config.yaml` and reload when the same video is opened again.
- [ ] Start plays the video smoothly with boxes: green when outside zones or allowed, orange while the dwell timer runs, red when unauthorised; the zone outline turns red while an intruder is inside it.
- [ ] The window never freezes while playing; Pause, Resume and Stop work; closing the window mid-run exits cleanly.
- [ ] On an intrusion: banner turns red for 4 s, a row is added to the alerts table, the pop-up appears without blocking the video, a snapshot and crop are saved, and a row is appended to `events.csv`.
- [ ] Only person, car, motorcycle are detected. No bus or truck boxes ever appear.
- [ ] Dwell time and confidence changes take effect during playback. The schedule with simulated start time works, including windows that cross midnight.
- [ ] `python headless.py clips/<video>` produces the same outputs and prints a summary table.
- [ ] `summary.json`, `events.csv`, snapshots exist after each run; the optional annotated video works when ticked.
- [ ] `rules.py` has no `cv2`, `tkinter`, `torch`, `ultralytics`, or `time` imports (architecture test passes).
- [ ] README, `docs/architecture.md` and `docs/test_results.md` exist; no fabricated numbers anywhere.

## 20. Final message to the user

When finished, reply with a short report: what was built; the exact commands to install, test and run; what you verified and **what you could not verify** (for example "GUI not run because this environment has no display"); the decisions you made; known limitations. Keep it plain and brief.

===== END AGENT PROMPT =====

---

# PART C: Everything else you need

## C1. Getting test footage (do this while the agent builds)

You need **fixed-camera** clips (the camera must not move, because zones are drawn on the screen).

- **Free stock sites:** Pexels, Pixabay, Mixkit, Videvo. Search: `security camera view`, `CCTV`, `parking lot`, `gate entrance`, `driveway`, `fence`, `pedestrians`, `street camera`. Pick one continuous shot, no panning or zoom, 10 to 60 seconds, objects clearly visible but not huge.
- **Research datasets (real surveillance footage):** VIRAT (closest match), UA-DETRAC (cars, fixed traffic cameras), MOT17/MOT20 and PETS2009 (pedestrians). Search the name plus "download".
- **Record your own (often best):** phone on a tripod or resting on a window or balcony railing, pointed at a gate, lane or open area. 30 to 60 seconds with you or a friend walking into the area, plus a bike or car passing. Make sure everyone filmed is fine with it.
- **A good starter set of 6 to 8 clips:**
  1. Person walks into the zone, daytime (should alert)
  2. Person walks past outside the zone (no alert)
  3. Car drives through a zone that allows cars (no alert)
  4. Car or person in a zone that does not allow it (alert)
  5. Person briefly touches the edge, leaves (no alert, shows the dwell time working)
  6. Evening or dark clip (shows the limits honestly)
  7. Person leaves and returns (tests the grace period)
- Trim long clips to 20 to 60 seconds. Note each clip's source and licence for your report.

## C2. Demo script (1 to 2 minutes, record the screen)

1. Open the app, open clip 1, draw a zone and tick no allowed types. Name it "Restricted".
2. Press Start. Person outside: green. Steps in: orange with a counting timer, then red, pop-up and alert row appear.
3. Double-click the alert row to show the snapshot. Open `output/run_.../events.csv`.
4. Open clip 3 or 4, show a zone that allows cars: car green; then a person in the same zone: red.
5. Turn on **Use schedule**, set the zone hours and the simulated start time, and show that the same footage is ignored during allowed hours and flagged at night.

## C3. How this feeds the college form

| Form field | Where the content comes from |
|---|---|
| Implementation Completed | README "Features" (completed) plus "Future work" (in progress / not done) |
| Code / Hardware / System Integration Status | `docs/architecture.md` (modules, config, YOLO model, CSV and snapshot storage; no hardware) |
| Testing / Experimental Results So Far | `docs/test_results.md`, filled with your **real** numbers from `headless.py` runs |
| Supporting Evidence links | GitHub repo, a Drive folder (screenshots, `output/` runs, demo video, test clips list), architecture doc |

Set every link to "Anyone with the link can view" and open it in a private window to check.

## C4. Quick checks after the agent finishes

- `pytest -q` is green.
- `python app.py` opens, and a test clip plays smoothly.
- Boxes line up with people and vehicles, and the drawn zone sits exactly where you clicked.
- The pop-up never freezes the video.
- `output/run_.../` contains `events.csv`, `snapshots/`, `summary.json`.
- Open `docs/test_results.md`: it should contain placeholders, not made-up numbers.

## C5. Follow-up prompts if something goes wrong

- **Window freezes while playing:** "The GUI freezes during playback. Check that tkinter is only touched from the main thread, that the frame queue has maxsize 2 and drops the oldest, and that `_poll` takes only the newest frame. Fix it and add a test."
- **Boxes do not match the zone:** "Zones and detections do not line up. Check `video_canvas.py` letterbox offset and scale in `canvas_to_norm` and `norm_to_canvas`, and that zones are converted to working-frame pixels using the same frame size used for detection."
- **Too slow:** "Playback is slower than the video. Profile one frame, confirm imgsz 480, realtime skipping with `cap.grab()`, and no per-frame copies or disk writes. Report milliseconds per stage."
- **Same person alerts twice:** "A single person produced two alerts. Check whether the ByteTrack ID changed (log track IDs per frame) or the grace period is too short. If it is an ID switch, document it as a limitation rather than adding complexity."
- **Nothing detected at night:** "Detection fails on dark clips. Lower confidence to 0.25, test imgsz 640, and report the effect. Do not add training."
- **Model will not load:** "Loading `yolo11n.pt` fails. Print the Ultralytics version and the full error, try the `yolov8n.pt` fallback, and show a clear message to the user."
- **tkinter missing:** reinstall Python from python.org (not the Microsoft Store version) and tick "tcl/tk and IDLE" in the installer's optional features.
- **A feature is missing:** "Compare the code against section 19 of guide.md line by line and list anything not implemented or not verified, then fix it."

## C6. Honest limitations to state in your report

- The system flags any object of a disallowed type in a zone. It cannot tell *who* a person is, so authorisation is decided by **zone, object type and time**, not identity. Face recognition and number-plate checks are listed as future work (the Phase-I report already lists facial recognition as a later extension).
- A fixed camera is required, because zones are drawn in screen coordinates.
- Night, blur, rain and heavy occlusion reduce accuracy (the Phase-I literature survey notes the same problems).
- Tracker ID switches can cause a duplicate alert for one person.
- Pretrained model only; recorded video only; one camera.
