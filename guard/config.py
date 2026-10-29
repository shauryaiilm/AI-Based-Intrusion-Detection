from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from guard.schema import Settings, Zone

logger = logging.getLogger("guard")

DEFAULT_CLASSES: dict[int, str] = {
    0: "person",
    2: "car",
    3: "motorcycle",
}

DEFAULT_EXAMPLE_ZONES: dict[str, list[Zone]] = {
    "example_clip.mp4": [
        Zone(
            name="Gate",
            allowed=["car", "motorcycle"],
            active_from=None,
            active_to=None,
            points=[
                (0.12, 0.55),
                (0.48, 0.52),
                (0.52, 0.93),
                (0.10, 0.95),
            ],
        )
    ]
}


@dataclass
class AppConfig:
    version: int = 1
    model: str = "yolo11n.pt"
    classes: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_CLASSES))
    settings: Settings = field(default_factory=Settings)
    zones_by_video: dict[str, list[Zone]] = field(default_factory=lambda: dict(DEFAULT_EXAMPLE_ZONES))


def _validate_hhmm(val: Any) -> str | None:
    if val is None or val == "":
        return None
    s = str(val).strip()
    if re.fullmatch(r"^(?:[01]\d|2[0-3]):[0-5]\d$", s):
        return s
    return None


def _validate_settings(raw: dict[str, Any] | None) -> Settings:
    st = Settings()
    if not isinstance(raw, dict):
        return st

    try:
        if "dwell_seconds" in raw:
            st.dwell_seconds = max(0.1, min(60.0, float(raw["dwell_seconds"])))
    except (ValueError, TypeError):
        pass

    try:
        if "grace_seconds" in raw:
            st.grace_seconds = max(0.0, min(60.0, float(raw["grace_seconds"])))
    except (ValueError, TypeError):
        pass

    try:
        if "confidence" in raw:
            st.confidence = max(0.05, min(0.95, float(raw["confidence"])))
    except (ValueError, TypeError):
        pass

    try:
        if "imgsz" in raw:
            val = int(raw["imgsz"])
            if val in (320, 480, 640):
                st.imgsz = val
            else:
                st.imgsz = 480
    except (ValueError, TypeError):
        st.imgsz = 480

    for bool_field in ("realtime", "save_video", "schedule_enabled", "popup_on_alert", "sound_on_alert"):
        if bool_field in raw:
            setattr(st, bool_field, bool(raw[bool_field]))

    if "sim_start" in raw:
        validated = _validate_hhmm(raw["sim_start"])
        if validated:
            st.sim_start = validated

    try:
        if "max_work_width" in raw:
            st.max_work_width = max(320, int(raw["max_work_width"]))
    except (ValueError, TypeError):
        pass

    if "output_dir" in raw and isinstance(raw["output_dir"], str) and raw["output_dir"].strip():
        st.output_dir = raw["output_dir"].strip()

    return st


def _validate_zone(
    raw_z: dict[str, Any], known_classes: set[str], existing_names: set[str]
) -> Zone | None:
    if not isinstance(raw_z, dict):
        return None

    name = str(raw_z.get("name", "Zone")).strip() or "Zone"
    # Ensure unique name within video
    base_name = name
    counter = 2
    while name in existing_names:
        name = f"{base_name} {counter}"
        counter += 1
    existing_names.add(name)

    raw_pts = raw_z.get("points", [])
    if not isinstance(raw_pts, (list, tuple)):
        return None

    valid_pts: list[tuple[float, float]] = []
    for pt in raw_pts:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                x = max(0.0, min(1.0, float(pt[0])))
                y = max(0.0, min(1.0, float(pt[1])))
                valid_pts.append((x, y))
            except (ValueError, TypeError):
                continue

    if len(valid_pts) < 3:
        logger.warning(f"Dropping zone '{name}' because it has fewer than 3 valid points.")
        return None

    raw_allowed = raw_z.get("allowed", [])
    if not isinstance(raw_allowed, (list, tuple)):
        raw_allowed = []
    allowed = [str(cls_name).strip() for cls_name in raw_allowed if str(cls_name).strip() in known_classes]

    active_from = _validate_hhmm(raw_z.get("active_from"))
    active_to = _validate_hhmm(raw_z.get("active_to"))
    if not (active_from and active_to):
        active_from = None
        active_to = None

    return Zone(
        name=name,
        points=valid_pts,
        allowed=allowed,
        active_from=active_from,
        active_to=active_to,
    )


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        cfg = AppConfig()
        save_config(cfg, p)
        return cfg

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Configuration root is not a mapping")
        if "version" not in data and "classes" not in data and "settings" not in data:
            raise ValueError("Configuration missing required root keys")
    except Exception as exc:
        bak_path = p.with_name(p.name + ".bak")
        logger.warning(f"Corrupt config file '{p}': {exc}. Renaming to '{bak_path}' and using defaults.")
        try:
            if bak_path.exists():
                bak_path.unlink()
            p.rename(bak_path)
        except Exception as ren_exc:
            logger.error(f"Failed to backup corrupt config: {ren_exc}")
        cfg = AppConfig()
        save_config(cfg, p)
        return cfg

    version = int(data.get("version", 1))
    model = str(data.get("model", "yolo11n.pt"))

    classes_raw = data.get("classes", DEFAULT_CLASSES)
    classes: dict[int, str] = {}
    if isinstance(classes_raw, dict):
        for k, v in classes_raw.items():
            try:
                classes[int(k)] = str(v).strip()
            except (ValueError, TypeError):
                continue
    if not classes:
        classes = dict(DEFAULT_CLASSES)

    known_classes = set(classes.values())
    settings = _validate_settings(data.get("settings"))

    zones_by_video: dict[str, list[Zone]] = {}
    raw_zbv = data.get("zones_by_video", {})
    if isinstance(raw_zbv, dict):
        for video_key, zone_list in raw_zbv.items():
            v_name = os.path.basename(str(video_key).strip())
            if not v_name or not isinstance(zone_list, (list, tuple)):
                continue
            validated_zones: list[Zone] = []
            seen_names: set[str] = set()
            for z_raw in zone_list:
                z = _validate_zone(z_raw, known_classes, seen_names)
                if z is not None:
                    validated_zones.append(z)
            zones_by_video[v_name] = validated_zones

    return AppConfig(
        version=version,
        model=model,
        classes=classes,
        settings=settings,
        zones_by_video=zones_by_video,
    )


def save_config(cfg: AppConfig, path: str | Path = "config.yaml") -> None:
    p = Path(path)
    parent = p.parent
    parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "version": cfg.version,
        "model": cfg.model,
        "classes": cfg.classes,
        "settings": asdict(cfg.settings),
        "zones_by_video": {},
    }

    for v_name, zones in cfg.zones_by_video.items():
        base_name = os.path.basename(v_name)
        data["zones_by_video"][base_name] = [
            {
                "name": z.name,
                "allowed": z.allowed,
                "active_from": z.active_from,
                "active_to": z.active_to,
                "points": [[round(pt[0], 4), round(pt[1], 4)] for pt in z.points],
            }
            for z in zones
        ]

    # Write atomically
    temp_file = tempfile.NamedTemporaryFile("w", dir=parent, delete=False, encoding="utf-8")
    try:
        yaml.safe_dump(data, temp_file, sort_keys=False, default_flow_style=False)
        temp_file.flush()
        temp_file.close()
        os.replace(temp_file.name, p)
    except Exception:
        if os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
        raise


def get_zones(cfg: AppConfig, video_name: str) -> list[Zone]:
    base_name = os.path.basename(video_name)
    return list(cfg.zones_by_video.get(base_name, []))


def set_zones(cfg: AppConfig, video_name: str, zones: list[Zone]) -> None:
    base_name = os.path.basename(video_name)
    cfg.zones_by_video[base_name] = list(zones)
