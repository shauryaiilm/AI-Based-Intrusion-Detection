from __future__ import annotations

from pathlib import Path

from guard.config import AppConfig, get_zones, load_config, save_config, set_zones
from guard.schema import Settings, Zone


def test_defaults_created_when_missing(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    assert not cfg_file.exists()
    cfg = load_config(cfg_file)
    assert cfg_file.exists()
    assert cfg.model == "yolo11n.pt"
    assert "Gate" in [z.name for z in cfg.zones_by_video.get("example_clip.mp4", [])]
    assert cfg.settings.dwell_seconds == 2.0


def test_round_trip(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg = AppConfig(
        version=1,
        model="yolo11n.pt",
        classes={0: "person", 2: "car", 3: "motorcycle"},
        settings=Settings(dwell_seconds=3.5, confidence=0.45, imgsz=640),
        zones_by_video={
            "video1.mp4": [
                Zone(
                    name="Perimeter",
                    points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
                    allowed=["car"],
                    active_from="08:00",
                    active_to="17:00",
                )
            ]
        },
    )
    save_config(cfg, cfg_file)

    loaded = load_config(cfg_file)
    assert loaded.settings.dwell_seconds == 3.5
    assert loaded.settings.confidence == 0.45
    assert loaded.settings.imgsz == 640
    zones = get_zones(loaded, "video1.mp4")
    assert len(zones) == 1
    assert zones[0].name == "Perimeter"
    assert zones[0].allowed == ["car"]
    assert zones[0].active_from == "08:00"
    assert zones[0].active_to == "17:00"
    assert len(zones[0].points) == 4


def test_corrupt_file_handling(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("::: bad yaml {[[", encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.model == "yolo11n.pt"
    bak_file = tmp_path / "config.yaml.bak"
    assert bak_file.exists()
    assert bak_file.read_text(encoding="utf-8") == "::: bad yaml {[["


def test_validation_and_repairs(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    content = """
version: 1
model: yolo11n.pt
classes:
  0: person
  2: car
settings:
  dwell_seconds: 999.0
  confidence: -0.5
  imgsz: 500
  sim_start: "invalid_time"
zones_by_video:
  test.mp4:
    - name: BadZone
      points: [[0.1, 0.2], [0.3, 0.4]]
      allowed: [person]
    - name: Duplicate
      points: [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]
      allowed: [person, alien_spaceship]
      active_from: "25:00"
      active_to: "12:00"
    - name: Duplicate
      points: [[-0.5, 1.5], [0.8, 0.0], [0.8, 0.8]]
      allowed: [car]
      active_from: "09:00"
      active_to: "18:00"
"""
    cfg_file.write_text(content, encoding="utf-8")
    cfg = load_config(cfg_file)

    # Numeric clamps
    assert cfg.settings.dwell_seconds == 60.0  # clamped to 60.0 max
    assert cfg.settings.confidence == 0.05  # clamped to 0.05 min
    assert cfg.settings.imgsz == 480  # fallback to 480 since 500 not in (320, 480, 640)
    assert cfg.settings.sim_start == "12:00"  # fallback default because invalid_time

    zones = get_zones(cfg, "test.mp4")
    # BadZone had 2 points -> dropped
    # Duplicate (1) had bad active_from -> active_from and active_to both set to None, alien_spaceship dropped
    # Duplicate (2) points clamped to [0, 1], name deduplicated to "Duplicate 2"
    assert len(zones) == 2
    z1 = zones[0]
    assert z1.name == "Duplicate"
    assert z1.allowed == ["person"]
    assert z1.active_from is None
    assert z1.active_to is None

    z2 = zones[1]
    assert z2.name == "Duplicate 2"
    assert z2.allowed == ["car"]
    assert z2.points[0] == (0.0, 1.0)  # clamped from (-0.5, 1.5)
    assert z2.active_from == "09:00"
    assert z2.active_to == "18:00"


def test_set_zones(tmp_path: Path):
    cfg = AppConfig()
    new_zone = Zone(
        name="Zone A",
        points=[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)],
        allowed=["person"],
    )
    set_zones(cfg, "my_video.mp4", [new_zone])
    assert get_zones(cfg, "my_video.mp4") == [new_zone]

