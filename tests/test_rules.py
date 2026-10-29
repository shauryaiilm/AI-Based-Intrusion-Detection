from __future__ import annotations

import random
from guard.rules import (
    RuleEngine,
    format_hhmmss,
    is_zone_active,
    parse_hhmm,
    point_in_polygon,
)
from guard.schema import Detection, Status, Zone


def test_point_in_polygon():
    # Square: (10, 10) to (50, 50)
    poly = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]

    # Inside
    assert point_in_polygon(30.0, 30.0, poly) is True
    # Outside
    assert point_in_polygon(5.0, 30.0, poly) is False
    assert point_in_polygon(60.0, 30.0, poly) is False
    # On an edge
    assert point_in_polygon(30.0, 10.0, poly) is True
    assert point_in_polygon(50.0, 30.0, poly) is True
    # On a vertex
    assert point_in_polygon(10.0, 10.0, poly) is True
    assert point_in_polygon(50.0, 50.0, poly) is True

    # Concave polygon (V-shape)
    # (0,0), (10,0), (10,10), (5,5), (0,10)
    concave = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 5.0), (0.0, 10.0)]
    assert point_in_polygon(5.0, 2.0, concave) is True  # In bottom part
    assert point_in_polygon(5.0, 7.0, concave) is False  # In notch


def test_parse_hhmm_and_format():
    assert parse_hhmm("12:00") == 12 * 3600
    assert parse_hhmm("00:00") == 0
    assert parse_hhmm("23:59:59") == 23 * 3600 + 59 * 60 + 59
    assert format_hhmmss(12 * 3600) == "12:00:00"
    assert format_hhmmss(86399) == "23:59:59"

    import pytest

    with pytest.raises(ValueError):
        parse_hhmm("24:00")
    with pytest.raises(ValueError):
        parse_hhmm("12:60")
    with pytest.raises(ValueError):
        parse_hhmm("not-a-time")


def test_is_zone_active():
    z_standard = Zone(name="Day", points=[(0, 0)], active_from="09:00", active_to="17:00")
    z_midnight = Zone(name="Night", points=[(0, 0)], active_from="19:00", active_to="07:00")
    z_all_day = Zone(name="AllDay", points=[(0, 0)], active_from="12:00", active_to="12:00")
    z_no_hours = Zone(name="NoHours", points=[(0, 0)])

    # No schedule enabled (sim_seconds is None)
    assert is_zone_active(z_standard, None) is True
    assert is_zone_active(z_no_hours, 1000) is True
    assert is_zone_active(z_all_day, 5000) is True

    # Standard (09:00 to 17:00)
    t_12 = parse_hhmm("12:00")
    t_08 = parse_hhmm("08:00")
    t_18 = parse_hhmm("18:00")
    assert is_zone_active(z_standard, t_12) is True
    assert is_zone_active(z_standard, t_08) is False
    assert is_zone_active(z_standard, t_18) is False

    # Midnight crossing (19:00 to 07:00)
    t_2330 = parse_hhmm("23:30")
    t_0300 = parse_hhmm("03:00")
    assert is_zone_active(z_midnight, t_2330) is True
    assert is_zone_active(z_midnight, t_0300) is True
    assert is_zone_active(z_midnight, t_12) is False


def test_person_enters_restricted_zone_and_stays():
    zone = Zone(
        name="Restricted",
        points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
        allowed=[],
    )
    # Frame size 100x100 -> Zone pixels: (20,20) to (80,80)
    engine = RuleEngine(zones=[zone], frame_size=(100, 100), dwell_seconds=2.0)

    # Outside at t=0.0: box anchor at ((0+10)/2, 10) = (5, 10)
    d_out = Detection(track_id=1, cls_name="person", conf=0.9, box=(0.0, 0.0, 10.0, 10.0))
    states, events = engine.update([d_out], frame_idx=0, video_time_s=0.0)
    assert states[0].status == Status.OUTSIDE
    assert len(events) == 0

    # Inside at t=1.0: box anchor at ((40+60)/2, 60) = (50, 60)
    d_in = Detection(track_id=1, cls_name="person", conf=0.9, box=(40.0, 40.0, 60.0, 60.0))
    states, events = engine.update([d_in], frame_idx=10, video_time_s=1.0)
    assert states[0].status == Status.PENDING
    assert len(events) == 0

    # Intermediate frame at t=1.8 (gap 0.8s <= grace of 1.0s)
    states, events = engine.update([d_in], frame_idx=18, video_time_s=1.8)
    assert states[0].status == Status.PENDING
    assert len(events) == 0

    # Still inside at t=2.5 (dwell 1.5s < 2.0s, gap 0.7s <= grace) -> still PENDING
    states, events = engine.update([d_in], frame_idx=25, video_time_s=2.5)
    assert states[0].status == Status.PENDING
    assert len(events) == 0

    # Still inside at t=3.0 (dwell 2.0s == dwell_seconds, gap 0.5s <= grace) -> UNAUTHORISED, 1 AlertEvent
    states, events = engine.update([d_in], frame_idx=30, video_time_s=3.0)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].zone_name == "Restricted"
    assert events[0].dwell_s == 2.0

    # Continued stay at t=4.0 -> UNAUTHORISED, but NO second alert!
    states, events = engine.update([d_in], frame_idx=40, video_time_s=4.0)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 0


def test_brief_touch_leaves_no_alert():
    zone = Zone(name="Restricted", points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], allowed=[])
    engine = RuleEngine(zones=[zone], frame_size=(100, 100), dwell_seconds=2.0)

    d_in = Detection(track_id=1, cls_name="person", conf=0.9, box=(40.0, 40.0, 60.0, 60.0))
    d_out = Detection(track_id=1, cls_name="person", conf=0.9, box=(0.0, 0.0, 10.0, 10.0))

    # In zone at t=1.0, 1.5
    engine.update([d_in], 10, 1.0)
    engine.update([d_in], 15, 1.5)

    # Leaves at t=2.0
    states, events = engine.update([d_out], 20, 2.0)
    assert states[0].status == Status.OUTSIDE
    assert len(events) == 0


def test_allowed_class_vs_disallowed():
    zone = Zone(name="Gate", points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], allowed=["car"])
    engine = RuleEngine(zones=[zone], frame_size=(100, 100), dwell_seconds=1.0)

    d_car = Detection(track_id=1, cls_name="car", conf=0.9, box=(40.0, 40.0, 60.0, 60.0))
    d_person = Detection(track_id=2, cls_name="person", conf=0.9, box=(40.0, 40.0, 60.0, 60.0))

    # Car stays for 3 seconds -> always AUTHORISED
    for t in [0.0, 1.0, 2.0, 3.0]:
        states, events = engine.update([d_car], int(t * 10), t)
        assert states[0].status == Status.AUTHORISED
        assert len(events) == 0

    # Person enters at t=4.0 and stays until t=5.5 -> alerts at t=5.0
    engine.update([d_person], 40, 4.0)
    states, events = engine.update([d_person], 50, 5.0)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 1
    assert events[0].cls_name == "person"


def test_empty_allowed_flags_all():
    zone = Zone(name="Vault", points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], allowed=[])
    engine = RuleEngine(zones=[zone], frame_size=(100, 100), dwell_seconds=1.0)

    d_car = Detection(track_id=1, cls_name="car", conf=0.9, box=(10.0, 10.0, 30.0, 30.0))
    engine.update([d_car], 0, 0.0)
    states, events = engine.update([d_car], 10, 1.0)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 1


def test_grace_period_continues_timer_without_second_alert():
    zone = Zone(name="Restricted", points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], allowed=[])
    engine = RuleEngine(zones=[zone], frame_size=(100, 100), dwell_seconds=2.0, grace_seconds=1.0)

    d_in = Detection(track_id=1, cls_name="person", conf=0.9, box=(40.0, 40.0, 60.0, 60.0))
    d_out = Detection(track_id=1, cls_name="person", conf=0.9, box=(0.0, 0.0, 10.0, 10.0))

    # Enter at t=1.0
    engine.update([d_in], 10, 1.0)
    # Outside at t=1.5 (missing for 0.5s < grace of 1.0s)
    engine.update([d_out], 15, 1.5)
    # Back inside at t=2.0 (last_seen was 1.0, difference is 1.0 <= grace_seconds)
    states, events = engine.update([d_in], 20, 2.0)
    assert states[0].status == Status.PENDING

    # Inside at t=3.0 (first_seen was 1.0 -> dwell = 2.0s) -> UNAUTHORISED, alert emitted
    states, events = engine.update([d_in], 30, 3.0)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 1

    # Leaves again for 0.4s and returns: no second alert
    engine.update([d_out], 34, 3.4)
    states, events = engine.update([d_in], 38, 3.8)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 0


def test_rearm_after_grace_period_expires():
    zone = Zone(name="Restricted", points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], allowed=[])
    engine = RuleEngine(zones=[zone], frame_size=(100, 100), dwell_seconds=1.0, grace_seconds=1.0)

    d_in = Detection(track_id=1, cls_name="person", conf=0.9, box=(40.0, 40.0, 60.0, 60.0))
    d_out = Detection(track_id=1, cls_name="person", conf=0.9, box=(0.0, 0.0, 10.0, 10.0))

    # First intrusion and alert
    engine.update([d_in], 0, 0.0)
    states, events = engine.update([d_in], 10, 1.0)
    assert len(events) == 1

    # Leaves at t=1.5 and stays out until t=4.0 (out for 2.5s > grace 1.0s)
    engine.update([d_out], 15, 1.5)
    engine.update([d_out], 30, 3.0)

    # Returns at t=4.0 -> re-armed, status PENDING
    states, events = engine.update([d_in], 40, 4.0)
    assert states[0].status == Status.PENDING
    assert len(events) == 0

    # Stays until t=5.0 (dwell 1.0s) -> Second alert!
    states, events = engine.update([d_in], 50, 5.0)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 1


def test_schedule_and_simulated_time():
    zone = Zone(
        name="NightOnly",
        points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        allowed=[],
        active_from="19:00",
        active_to="07:00",
    )
    # Start at 12:00
    engine = RuleEngine(
        zones=[zone],
        frame_size=(100, 100),
        dwell_seconds=1.0,
        schedule_enabled=True,
        sim_start="12:00",
    )
    d = Detection(track_id=1, cls_name="person", conf=0.9, box=(20.0, 20.0, 40.0, 40.0))

    # At video_time = 0.0 (sim 12:00), zone is inactive -> AUTHORISED
    states, events = engine.update([d], 0, 0.0)
    assert states[0].status == Status.AUTHORISED
    assert states[0].reason == "zone not active now"
    assert len(events) == 0

    # Fast forward to 8 hours later (video_time = 8 * 3600 = 28800s -> sim 20:00)
    t = 28800.0
    states, events = engine.update([d], 100, t)
    assert states[0].status == Status.PENDING

    states, events = engine.update([d], 110, t + 1.0)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 1
    assert events[0].sim_time == "20:00:01"


def test_overlapping_zones_first_violating_decides():
    # Zone 1 allows car, Zone 2 is restricted to all
    z1 = Zone(name="Zone1", points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], allowed=["car"])
    z2 = Zone(name="Zone2", points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], allowed=[])

    engine = RuleEngine(zones=[z1, z2], frame_size=(100, 100), dwell_seconds=1.0)

    # Car: allowed in z1 -> z1 is active & non-violating. Since z1 contains it and is active and allowed,
    # let's verify violating check: z1 is NOT violating, but z2 IS violating.
    # From spec:
    # active = [z for z in containing if is_zone_active(z, sim_s)]
    # violating = [z for z in active if d.cls_name not in z.allowed]
    # if none: authorised
    # else: z = violating[0]
    # Here violating is [z2]. So z2 decides!
    d_car = Detection(track_id=1, cls_name="car", conf=0.9, box=(20.0, 20.0, 40.0, 40.0))
    engine.update([d_car], 0, 0.0)
    states, events = engine.update([d_car], 10, 1.0)
    assert states[0].status == Status.UNAUTHORISED
    assert states[0].zone_name == "Zone2"


def test_independent_tracks_and_shuffle_invariance():
    zone = Zone(name="Area", points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], allowed=[])
    engine = RuleEngine(zones=[zone], frame_size=(100, 100), dwell_seconds=1.0)

    d1 = Detection(track_id=1, cls_name="person", conf=0.9, box=(10.0, 10.0, 20.0, 20.0))
    d2 = Detection(track_id=2, cls_name="car", conf=0.9, box=(30.0, 30.0, 40.0, 40.0))

    engine.update([d1, d2], 0, 0.0)

    # Shuffled order
    dets = [d2, d1]
    states, events = engine.update(dets, 10, 1.0)
    # Both should be unauthorised
    assert len(events) == 2
    state_map = {s.detection.track_id: s.status for s in states}
    assert state_map[1] == Status.UNAUTHORISED
    assert state_map[2] == Status.UNAUTHORISED


def test_live_set_dwell_seconds():
    zone = Zone(name="Area", points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], allowed=[])
    engine = RuleEngine(zones=[zone], frame_size=(100, 100), dwell_seconds=5.0)

    d = Detection(track_id=1, cls_name="person", conf=0.9, box=(10.0, 10.0, 20.0, 20.0))
    engine.update([d], 0, 0.0)
    engine.update([d], 10, 1.0)

    # At t=2.0, dwell is 2.0s < 5.0s -> PENDING
    states, events = engine.update([d], 20, 2.0)
    assert states[0].status == Status.PENDING

    # Dynamically lower dwell to 1.5s
    engine.set_dwell_seconds(1.5)

    # At t=2.1, dwell is 2.1s >= 1.5s -> UNAUTHORISED!
    states, events = engine.update([d], 21, 2.1)
    assert states[0].status == Status.UNAUTHORISED
    assert len(events) == 1
