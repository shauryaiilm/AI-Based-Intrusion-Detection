from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from guard.schema import AlertEvent, Detection, Status, TrackState, Zone, ZoneInfo


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray casting algorithm to test if point (x, y) is inside polygon.
    
    Points on edges or vertices count as inside.
    Pure Python implementation, no cv2 or other dependencies.
    """
    n = len(polygon)
    if n < 3:
        return False

    # First check if the point lies exactly on an edge or vertex
    for i in range(n):
        p1x, p1y = polygon[i]
        p2x, p2y = polygon[(i + 1) % n]

        # Check bounding box of segment with float tolerance
        min_x, max_x = (p1x, p2x) if p1x <= p2x else (p2x, p1x)
        min_y, max_y = (p1y, p2y) if p1y <= p2y else (p2y, p1y)

        if (min_x - 1e-7 <= x <= max_x + 1e-7) and (min_y - 1e-7 <= y <= max_y + 1e-7):
            # Collinearity check via cross product
            cross = (x - p1x) * (p2y - p1y) - (y - p1y) * (p2x - p1x)
            if abs(cross) <= 1e-6:
                return True

    # Standard ray-casting (horizontal ray towards +x)
    inside = False
    for i in range(n):
        p1x, p1y = polygon[i]
        p2x, p2y = polygon[(i + 1) % n]

        if (p1y > y) != (p2y > y):
            x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if x < x_inters:
                inside = not inside

    return inside


def parse_hhmm(s: str) -> int:
    """Parses HH:MM or HH:MM:SS to total seconds since midnight.
    
    Raises ValueError on invalid formats or out-of-range times.
    """
    if not isinstance(s, str):
        raise ValueError(f"Expected string for time, got {type(s)}")
    s = s.strip()
    m = re.fullmatch(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        raise ValueError(f"Invalid time format: '{s}' (expected HH:MM or HH:MM:SS)")

    hours = int(m.group(1))
    minutes = int(m.group(2))
    seconds = int(m.group(3)) if m.group(3) is not None else 0

    if not (0 <= hours < 24):
        raise ValueError(f"Hour out of range [0, 23]: {hours}")
    if not (0 <= minutes < 60):
        raise ValueError(f"Minute out of range [0, 59]: {minutes}")
    if not (0 <= seconds < 60):
        raise ValueError(f"Second out of range [0, 59]: {seconds}")

    return hours * 3600 + minutes * 60 + seconds


def format_hhmmss(seconds: int | float) -> str:
    """Formats seconds since midnight into HH:MM:SS string."""
    s = int(seconds) % 86400
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def is_zone_active(zone: Zone, sim_seconds: int | None) -> bool:
    """Determines whether a zone is active at the given simulated second.
    
    - If schedule is disabled (sim_seconds is None) or zone has no hours: True
    - from < to: active if from <= s < to
    - from > to: active if s >= from or s < to (crosses midnight)
    - from == to: always active
    """
    if sim_seconds is None or not zone.active_from or not zone.active_to:
        return True

    try:
        f = parse_hhmm(zone.active_from)
        t = parse_hhmm(zone.active_to)
    except ValueError:
        return True

    if f == t:
        return True
    if f < t:
        return f <= sim_seconds < t
    # Crosses midnight
    return sim_seconds >= f or sim_seconds < t


@dataclass
class _TrackRecord:
    first_seen: float
    last_seen: float
    alerted: bool


class RuleEngine:
    def __init__(
        self,
        zones: list[Zone],
        frame_size: tuple[int, int],
        dwell_seconds: float,
        grace_seconds: float = 1.0,
        schedule_enabled: bool = False,
        sim_start: str = "12:00",
    ) -> None:
        self._zones = list(zones)
        self._frame_size = frame_size  # (width, height)
        self._dwell_seconds = max(0.1, float(dwell_seconds))
        self._grace_seconds = max(0.0, float(grace_seconds))
        self._schedule_enabled = bool(schedule_enabled)
        self._sim_start = sim_start
        try:
            self._sim_start_sec = parse_hhmm(sim_start)
        except ValueError:
            self._sim_start_sec = 12 * 3600

        # Convert normalised points to pixel coordinates once
        w, h = frame_size
        self._zones_px: list[tuple[Zone, list[tuple[float, float]]]] = []
        for z in self._zones:
            px_pts = [(pt[0] * w, pt[1] * h) for pt in z.points]
            self._zones_px.append((z, px_pts))

        self._records: dict[tuple[int, str], _TrackRecord] = {}
        self._last_zone_states: dict[str, ZoneInfo] = {
            z.name: ZoneInfo(is_active=True, has_unauthorised=False) for z in self._zones
        }

    def set_dwell_seconds(self, v: float) -> None:
        """Live-editable dwell seconds."""
        self._dwell_seconds = max(0.1, float(v))

    def update(
        self, detections: list[Detection], frame_idx: int, video_time_s: float
    ) -> tuple[list[TrackState], list[AlertEvent]]:
        """Evaluates detections against zones and dwell timers."""
        if self._schedule_enabled:
            sim_s = int(self._sim_start_sec + video_time_s) % 86400
            sim_time_str = format_hhmmss(sim_s)
        else:
            sim_s = None
            sim_time_str = None

        states: list[TrackState] = []
        events: list[AlertEvent] = []
        unauthorised_zones: set[str] = set()

        for d in detections:
            ax, ay = d.anchor
            # Find containing zones in zone order
            containing: list[tuple[Zone, list[tuple[float, float]]]] = [
                (z, pts) for z, pts in self._zones_px if point_in_polygon(ax, ay, pts)
            ]

            if not containing:
                states.append(
                    TrackState(
                        detection=d,
                        status=Status.OUTSIDE,
                        zone_name=None,
                        dwell_s=0.0,
                        reason="outside all zones",
                    )
                )
                continue

            active = [z for z, _ in containing if is_zone_active(z, sim_s)]
            if not active:
                states.append(
                    TrackState(
                        detection=d,
                        status=Status.AUTHORISED,
                        zone_name=containing[0][0].name,
                        dwell_s=0.0,
                        reason="zone not active now",
                    )
                )
                continue

            violating = [z for z in active if d.cls_name not in z.allowed]
            if not violating:
                z_auth = active[0]
                states.append(
                    TrackState(
                        detection=d,
                        status=Status.AUTHORISED,
                        zone_name=z_auth.name,
                        dwell_s=0.0,
                        reason=f"{d.cls_name} allowed in {z_auth.name}",
                    )
                )
                continue

            # First violating active zone decides
            z = violating[0]
            key = (d.track_id, z.name)
            rec = self._records.get(key)
            if rec is None or (video_time_s - rec.last_seen > self._grace_seconds):
                # Brand-new entry -> dwell restarts, alert re-armed
                rec = _TrackRecord(
                    first_seen=video_time_s,
                    last_seen=video_time_s,
                    alerted=False,
                )
                self._records[key] = rec

            rec.last_seen = video_time_s
            dwell = video_time_s - rec.first_seen

            if dwell >= self._dwell_seconds:
                status = Status.UNAUTHORISED
                unauthorised_zones.add(z.name)
                if not rec.alerted:
                    rec.alerted = True
                    events.append(
                        AlertEvent(
                            track_id=d.track_id,
                            cls_name=d.cls_name,
                            zone_name=z.name,
                            video_time_s=video_time_s,
                            sim_time=sim_time_str,
                            dwell_s=dwell,
                            conf=d.conf,
                            frame_idx=frame_idx,
                            box=d.box,
                        )
                    )
            else:
                status = Status.PENDING

            states.append(
                TrackState(
                    detection=d,
                    status=status,
                    zone_name=z.name,
                    dwell_s=dwell,
                    reason=f"{d.cls_name} not allowed in {z.name}",
                )
            )

        # Prune old records
        cutoff = video_time_s - 30.0
        prune_keys = [k for k, r in self._records.items() if r.last_seen < cutoff]
        for k in prune_keys:
            del self._records[k]

        # Update cached zone states
        for z in self._zones:
            self._last_zone_states[z.name] = ZoneInfo(
                is_active=is_zone_active(z, sim_s),
                has_unauthorised=(z.name in unauthorised_zones),
            )

        return states, events

    def zone_states(self, video_time_s: float) -> dict[str, ZoneInfo]:
        """Returns cached active & unauthorised status per zone from the latest update."""
        return self._last_zone_states

