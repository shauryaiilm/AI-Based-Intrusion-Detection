from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in working-frame pixels

    @property
    def anchor(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)


@dataclass
class Zone:
    name: str
    points: list[tuple[float, float]]  # normalised 0..1
    allowed: list[str] = field(default_factory=list)
    active_from: str | None = None  # "HH:MM"
    active_to: str | None = None


@dataclass
class ZoneInfo:
    is_active: bool
    has_unauthorised: bool


@dataclass
class TrackState:
    detection: Detection
    status: Status
    zone_name: str | None  # the zone that decided the status, if any
    dwell_s: float  # seconds inside the violating zone (0 if none)
    reason: str  # short human text, e.g. "car allowed in Gate"


@dataclass
class AlertEvent:  # produced by rules.py, no file paths yet
    track_id: int
    cls_name: str
    zone_name: str
    video_time_s: float
    sim_time: str | None  # "HH:MM:SS" if schedule enabled, else None
    dwell_s: float
    conf: float
    frame_idx: int
    box: tuple[float, float, float, float]


@dataclass
class SavedAlert:  # produced by alerts.py after files are written
    event: AlertEvent
    alert_no: int
    wall_time: str  # "YYYY-MM-DD HH:MM:SS"
    snapshot_path: str
    crop_path: str


@dataclass
class Settings:  # mirrors config.yaml "settings"; defaults as in section 7
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


@dataclass
class FrameResult:
    annotated_frame: Any
    states: list[TrackState]
    saved_alerts: list[SavedAlert]
    counts_now: dict[str, int]
    frame_idx: int
    video_time_s: float


@dataclass
class FramePacket:
    frame: Any
    frame_idx: int
    total_frames: int
    video_time_s: float
    stats: dict[str, Any]


@dataclass
class AlertMsg:
    saved_alert: SavedAlert


@dataclass
class StatusMsg:
    text: str


@dataclass
class ErrorMsg:
    text: str


@dataclass
class DoneMsg:
    summary: dict[str, Any]

