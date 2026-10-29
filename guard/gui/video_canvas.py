from __future__ import annotations

import tkinter as tk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk


class VideoCanvas(tk.Frame):
    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, bg="#111111", **kwargs)

        self.canvas = tk.Canvas(self, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._image_item: int | None = None
        self._current_frame: np.ndarray | None = None
        self._photo_ref: ImageTk.PhotoImage | None = None

        # Letterbox layout cache: (offset_x, offset_y, display_w, display_h)
        self._layout = (0, 0, 0, 0)
        self._resize_timer: str | None = None

        self.canvas.bind("<Configure>", self._on_configure)

    def _on_configure(self, event: tk.Event) -> None:
        if self._resize_timer is not None:
            self.after_cancel(self._resize_timer)
        self._resize_timer = self.after(50, self._rerender_current)

    def _rerender_current(self) -> None:
        self._resize_timer = None
        if self._current_frame is not None:
            self._render(self._current_frame)

    def set_frame(self, frame_bgr: np.ndarray | None) -> None:
        """Sets and displays a BGR frame, scaled and letterboxed in the canvas."""
        self._current_frame = frame_bgr
        if frame_bgr is None:
            if self._image_item is not None:
                self.canvas.delete(self._image_item)
                self._image_item = None
            self._photo_ref = None
            self._layout = (0, 0, 0, 0)
            return

        self._render(frame_bgr)

    def _render(self, frame_bgr: np.ndarray) -> None:
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        fh, fw = frame_bgr.shape[:2]
        if fw == 0 or fh == 0:
            return

        scale = min(canvas_w / float(fw), canvas_h / float(fh))
        disp_w = max(1, int(round(fw * scale)))
        disp_h = max(1, int(round(fh * scale)))

        ox = (canvas_w - disp_w) // 2
        oy = (canvas_h - disp_h) // 2
        self._layout = (ox, oy, disp_w, disp_h)

        if scale < 1.0:
            resized = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        elif scale > 1.0:
            resized = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame_bgr

        # Convert BGR to RGB for PIL / Tkinter
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo_ref = ImageTk.PhotoImage(image=img)

        if self._image_item is None:
            self._image_item = self.canvas.create_image(
                ox, oy, anchor=tk.NW, image=self._photo_ref
            )
        else:
            self.canvas.coords(self._image_item, ox, oy)
            self.canvas.itemconfig(self._image_item, image=self._photo_ref)

    def canvas_to_norm(self, cx: float, cy: float) -> tuple[float, float] | None:
        """Converts canvas click coordinates to normalized coordinates (0..1).
        
        Returns None if click falls within the black bars.
        """
        ox, oy, dw, dh = self._layout
        if dw <= 0 or dh <= 0:
            return None

        # Check if inside active video region
        if not (ox <= cx <= ox + dw and oy <= cy <= oy + dh):
            return None

        nx = max(0.0, min(1.0, (cx - ox) / float(dw)))
        ny = max(0.0, min(1.0, (cy - oy) / float(dh)))
        return nx, ny

    def norm_to_canvas(self, nx: float, ny: float) -> tuple[float, float]:
        """Converts normalized (0..1) coordinates back to canvas pixel coordinates."""
        ox, oy, dw, dh = self._layout
        cx = ox + nx * dw
        cy = oy + ny * dh
        return cx, cy

    def get_layout(self) -> tuple[int, int, int, int]:
        """Returns the letterbox layout: (offset_x, offset_y, display_w, display_h)."""
        return self._layout

