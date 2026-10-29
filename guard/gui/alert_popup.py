from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from guard.schema import SavedAlert


class AlertPopup(tk.Toplevel):
    """Single reusable, non-modal, auto-dismissing pop-up window for intrusion alerts."""

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.parent = parent
        self.title("Intrusion Alert")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self._hide_timer: str | None = None
        self._thumb_ref: ImageTk.PhotoImage | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self.withdraw()  # Start hidden

    def _build_ui(self) -> None:
        # Header banner
        header = tk.Frame(self, bg="#d9534f", height=32)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        ttk.Label(
            header,
            text="\u26a0 INTRUSION DETECTED",
            font=("TkDefaultFont", 11, "bold"),
            foreground="white",
            background="#d9534f",
        ).pack(side=tk.LEFT, padx=10, pady=4)

        # Content area
        content = ttk.Frame(self, padding="10 10 10 10")
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Thumbnail image
        self.thumb_label = ttk.Label(content, background="#222222")
        self.thumb_label.grid(row=0, column=0, rowspan=4, padx=(0, 10), sticky=tk.NW)

        # Right: Info details
        self.title_label = ttk.Label(
            content,
            text="",
            font=("TkDefaultFont", 10, "bold"),
            foreground="#c9302c",
        )
        self.title_label.grid(row=0, column=1, sticky=tk.W, pady=(0, 2))

        self.zone_label = ttk.Label(content, text="", font=("TkDefaultFont", 9))
        self.zone_label.grid(row=1, column=1, sticky=tk.W, pady=(0, 2))

        self.time_label = ttk.Label(content, text="", font=("TkDefaultFont", 9))
        self.time_label.grid(row=2, column=1, sticky=tk.W, pady=(0, 2))

        self.dwell_label = ttk.Label(content, text="", font=("TkDefaultFont", 9))
        self.dwell_label.grid(row=3, column=1, sticky=tk.W, pady=(0, 2))

        # Close button
        btn_frame = ttk.Frame(content)
        btn_frame.grid(row=4, column=0, columnspan=2, sticky=tk.E, pady=(8, 0))
        ttk.Button(btn_frame, text="Dismiss", command=self.hide, width=10).pack(side=tk.RIGHT)

    def show_alert(self, saved_alert: SavedAlert) -> None:
        """Displays or updates the pop-up in place and restarts the 5-second timer."""
        ev = saved_alert.event

        self.title_label.config(text=f"UNAUTHORISED {ev.cls_name} #{ev.track_id}")
        self.zone_label.config(text=f"Zone: {ev.zone_name}")

        sim_text = f" (sim {ev.sim_time})" if ev.sim_time else ""
        self.time_label.config(text=f"Video: {ev.video_time_s:.1f}s{sim_text}")
        self.dwell_label.config(text=f"Dwell: {ev.dwell_s:.1f}s")

        # Load thumbnail from crop
        crop_path = saved_alert.crop_path
        if os.path.exists(crop_path):
            try:
                crop_bytes = np.fromfile(crop_path, dtype=np.uint8)
                crop_bgr = cv2.imdecode(crop_bytes, cv2.IMREAD_COLOR)
                if crop_bgr is not None:
                    ch, cw = crop_bgr.shape[:2]
                    target_w = 120
                    target_h = max(1, int(round(ch * (target_w / float(cw)))))
                    if target_h > 120:
                        target_h = 120
                        target_w = max(1, int(round(cw * (target_h / float(ch)))))
                    resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    self._thumb_ref = ImageTk.PhotoImage(img)
                    self.thumb_label.config(image=self._thumb_ref)
            except Exception:
                self.thumb_label.config(image="")
        else:
            self.thumb_label.config(image="")

        # Position at bottom right of main window
        try:
            rx = self.parent.winfo_rootx()
            ry = self.parent.winfo_rooty()
            rw = self.parent.winfo_width()
            rh = self.parent.winfo_height()
            px = max(0, rx + rw - 380)
            py = max(0, ry + rh - 260)
            self.geometry(f"360x220+{px}+{py}")
        except Exception:
            self.geometry("360x220")

        self.deiconify()
        self.lift()

        # Keep main window focused so keyboard shortcuts still work
        self.parent.focus_force()

        # Restart auto-hide timer (5 seconds)
        if self._hide_timer is not None:
            self.after_cancel(self._hide_timer)
        self._hide_timer = self.after(5000, self.hide)

    def hide(self) -> None:
        if self._hide_timer is not None:
            self.after_cancel(self._hide_timer)
            self._hide_timer = None
        self.withdraw()

