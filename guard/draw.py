from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from guard.schema import Status, TrackState

# BGR Colours
COLOR_SKY_BLUE = (255, 180, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREY = (160, 160, 160)
COLOR_GREEN = (0, 200, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_DARK_BAR = (30, 30, 30)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def draw_zones(
    frame: np.ndarray,
    zones_px: Sequence[tuple[str, Sequence[tuple[float, float]]]]
    | Mapping[str, Sequence[tuple[float, float]]],
    active_map: Mapping[str, bool],
    violating_map: Mapping[str, bool],
) -> None:
    """Draws semi-transparent polygon fills, outlines and name labels for all zones."""
    if isinstance(zones_px, Mapping):
        items = list(zones_px.items())
    else:
        items = list(zones_px)

    if not items:
        return

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Pass 1: Semi-transparent fills
    for name, points in items:
        if len(points) < 3:
            continue
        is_active = active_map.get(name, True)
        is_violating = violating_map.get(name, False)

        if not is_active:
            color = COLOR_GREY
        elif is_violating:
            color = COLOR_RED
        else:
            color = COLOR_SKY_BLUE

        pts_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_np], color)

    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # Pass 2: Outlines and labels
    for name, points in items:
        if len(points) < 3:
            continue
        is_active = active_map.get(name, True)
        is_violating = violating_map.get(name, False)

        if not is_active:
            color = COLOR_GREY
            label_text = f"{name} [inactive]"
        elif is_violating:
            color = COLOR_RED
            label_text = name
        else:
            color = COLOR_SKY_BLUE
            label_text = name

        pts_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_np], isClosed=True, color=color, thickness=2)

        # Place label near the first point
        x0, y0 = int(round(points[0][0])), int(round(points[0][1]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        lx = max(4, min(w - tw - 8, x0))
        ly = max(th + 6, min(h - 6, y0 - 6))

        # Filled background rectangle for text
        cv2.rectangle(
            frame,
            (lx - 2, ly - th - 4),
            (lx + tw + 4, ly + baseline),
            color,
            -1,
        )
        text_color = COLOR_WHITE if (color == COLOR_RED or color == COLOR_GREY) else COLOR_BLACK
        cv2.putText(
            frame,
            label_text,
            (lx + 1, ly - 2),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def draw_tracks(frame: np.ndarray, states: Sequence[TrackState]) -> None:
    """Draws bounding boxes, labels with background, and anchor points for all tracks."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1

    for st in states:
        d = st.detection
        status = st.status

        if status == Status.UNAUTHORISED:
            color = COLOR_RED
            box_thick = 3
            label = f"{d.cls_name} #{d.track_id} \u00b7 UNAUTHORISED"
        elif status == Status.PENDING:
            color = COLOR_ORANGE
            box_thick = 2
            label = f"{d.cls_name} #{d.track_id} \u00b7 {st.dwell_s:.1f}s"
        else:  # AUTHORISED or OUTSIDE
            color = COLOR_GREEN
            box_thick = 2
            label = f"{d.cls_name} #{d.track_id}"

        x1 = int(round(d.box[0]))
        y1 = int(round(d.box[1]))
        x2 = int(round(d.box[2]))
        y2 = int(round(d.box[3]))

        # Clamp box coordinates
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)

        # Anchor point (bottom-centre)
        ax = int(round((x1 + x2) / 2.0))
        ay = y2
        cv2.circle(frame, (ax, ay), 3, color, -1)

        # Label with filled background
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        lx = max(2, min(w - tw - 4, x1))
        # Place above box if room, else inside top of box
        if y1 - th - 8 >= 0:
            ly = y1 - 4
        else:
            ly = y1 + th + 4

        cv2.rectangle(
            frame,
            (lx - 2, ly - th - 3),
            (lx + tw + 2, ly + baseline),
            color,
            -1,
        )
        text_color = COLOR_WHITE if (status == Status.UNAUTHORISED or status == Status.PENDING) else COLOR_BLACK
        cv2.putText(
            frame,
            label,
            (lx, ly),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def draw_info_bar(frame: np.ndarray, text: str) -> None:
    """Draws a dark strip across the top with white text."""
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 24), COLOR_DARK_BAR, -1)
    cv2.putText(
        frame,
        text,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )


def annotate_alert(frame: np.ndarray, alert_text: str) -> np.ndarray:
    """Creates a copy with a red banner across the top for snapshot evidence."""
    annotated = frame.copy()
    w = annotated.shape[1]
    cv2.rectangle(annotated, (0, 0), (w, 28), COLOR_RED, -1)
    cv2.putText(
        annotated,
        alert_text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    return annotated

