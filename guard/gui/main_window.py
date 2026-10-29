from __future__ import annotations

import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from guard.alerts import RunRecorder
from guard.config import AppConfig, get_zones, load_config, save_config, set_zones
from guard.draw import draw_zones
from guard.gui.alert_popup import AlertPopup
from guard.gui.video_canvas import VideoCanvas
from guard.gui.zone_dialog import ZoneDialog
from guard.pipeline import VideoWorker, compute_working_size, downscale_if_needed
from guard.schema import (
    AlertMsg,
    DoneMsg,
    ErrorMsg,
    FramePacket,
    SavedAlert,
    Settings,
    StatusMsg,
    Zone,
)

logger = logging.getLogger("guard")


class MainWindow:
    def __init__(self, root: tk.Tk, config_path: str = "config.yaml") -> None:
        self.root = root
        self.config_path = config_path
        self.config: AppConfig = load_config(config_path)
        self.settings: Settings = self.config.settings

        self.root.title("Intrusion Guard")
        self.root.geometry("1280x760")
        self.root.minsize(1100, 650)

        # State variables
        self.current_video_path: str | None = None
        self.first_frame_bgr: np.ndarray | None = None
        self.working_size: tuple[int, int] = (640, 480)
        self.current_zones: list[Zone] = []
        self.latest_run_dir: str | None = None

        self.worker: VideoWorker | None = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.event_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # Zone drawing state
        self.is_drawing_zone = False
        self.drawing_points: list[tuple[float, float]] = []  # normalized (0..1)
        self._temp_canvas_items: list[int] = []

        # Alerts and UI timers
        self.alert_popup = AlertPopup(self.root)
        self.alerts_list: list[SavedAlert] = []
        self._banner_reset_timer: str | None = None
        self._thumb_ref: ImageTk.PhotoImage | None = None
        self._snapshot_window: tk.Toplevel | None = None

        self._build_menu()
        self._build_ui()
        self._bind_keys()

        self._update_button_states("NO_VIDEO")
        self._poll()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video...", command=self._open_video, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Output Folder", command=self._open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _build_ui(self) -> None:
        # 1. Top toolbar
        toolbar = ttk.Frame(self.root, padding="6 6 6 6")
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttk.Button(toolbar, text="Open Video", command=self._open_video)
        self.btn_open.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_start = ttk.Button(toolbar, text="Start", command=self._start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_pause = ttk.Button(toolbar, text="Pause", command=self._toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_stop = ttk.Button(toolbar, text="Stop", command=self._stop)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 4))

        # Status banner on top right
        self.status_banner = tk.Label(
            toolbar,
            text="READY - open a video",
            font=("TkDefaultFont", 10, "bold"),
            bg="#e9ecef",
            fg="#495057",
            padx=12,
            pady=4,
            relief=tk.FLAT,
        )
        self.status_banner.pack(side=tk.RIGHT)

        # 2. Main content area (split into canvas and right panel)
        content_frame = ttk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=0)
        content_frame.rowconfigure(0, weight=1)

        # Left: Video canvas
        self.canvas_panel = VideoCanvas(content_frame)
        self.canvas_panel.grid(row=0, column=0, sticky=tk.NSEW)

        # Canvas drawing bindings
        self.canvas_panel.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas_panel.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas_panel.canvas.bind("<Button-3>", self._on_canvas_finish_click)
        self.canvas_panel.canvas.bind("<Double-Button-1>", self._on_canvas_finish_click)

        # Right: Fixed control & information panel
        right_panel = ttk.Frame(content_frame, width=340, padding="6 6 6 6")
        right_panel.grid(row=0, column=1, sticky=tk.NSEW)
        right_panel.grid_propagate(False)

        self._build_right_panel(right_panel)

        # 3. Bottom status bar
        self.status_bar = ttk.Label(
            self.root,
            text="No video loaded",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding="4 2 4 2",
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        # Zones section
        zones_group = ttk.LabelFrame(parent, text="Zones", padding="6 6 6 6")
        zones_group.pack(fill=tk.X, pady=(0, 6))

        self.zones_listbox = tk.Listbox(zones_group, height=4, font=("TkDefaultFont", 9))
        self.zones_listbox.pack(fill=tk.X, pady=(0, 4))

        z_btn_row1 = ttk.Frame(zones_group)
        z_btn_row1.pack(fill=tk.X)
        self.btn_add_zone = ttk.Button(z_btn_row1, text="Add Zone", command=self._start_draw_zone)
        self.btn_add_zone.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self.btn_edit_zone = ttk.Button(z_btn_row1, text="Edit", command=self._edit_zone)
        self.btn_edit_zone.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 2))
        self.btn_del_zone = ttk.Button(z_btn_row1, text="Delete", command=self._delete_zone)
        self.btn_del_zone.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        self.btn_clear_zones = ttk.Button(zones_group, text="Clear All", command=self._clear_all_zones)
        self.btn_clear_zones.pack(fill=tk.X, pady=(4, 0))

        # Settings section
        settings_group = ttk.LabelFrame(parent, text="Settings", padding="6 6 6 6")
        settings_group.pack(fill=tk.X, pady=(0, 6))

        # Dwell time
        dwell_row = ttk.Frame(settings_group)
        dwell_row.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(dwell_row, text="Dwell time (s):").pack(side=tk.LEFT)
        self.dwell_var = tk.DoubleVar(value=self.settings.dwell_seconds)
        self.dwell_spin = ttk.Spinbox(
            dwell_row,
            from_=0.5,
            to=30.0,
            increment=0.5,
            textvariable=self.dwell_var,
            width=6,
            command=self._on_dwell_changed,
        )
        self.dwell_spin.pack(side=tk.RIGHT)
        self.dwell_spin.bind("<FocusOut>", lambda e: self._on_dwell_changed())
        self.dwell_spin.bind("<Return>", lambda e: self._on_dwell_changed())

        # Confidence
        conf_row = ttk.Frame(settings_group)
        conf_row.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(conf_row, text="Confidence:").pack(side=tk.LEFT)
        self.conf_val_lbl = ttk.Label(conf_row, text=f"{self.settings.confidence:.2f}")
        self.conf_val_lbl.pack(side=tk.RIGHT)
        self.conf_var = tk.DoubleVar(value=self.settings.confidence)
        self.conf_slider = ttk.Scale(
            settings_group,
            from_=0.10,
            to=0.90,
            variable=self.conf_var,
            command=self._on_conf_changed,
        )
        self.conf_slider.pack(fill=tk.X, pady=(0, 4))

        # Imgsz
        imgsz_row = ttk.Frame(settings_group)
        imgsz_row.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(imgsz_row, text="Image size:").pack(side=tk.LEFT)
        self.imgsz_var = tk.IntVar(value=self.settings.imgsz)
        self.imgsz_combo = ttk.Combobox(
            imgsz_row,
            values=[320, 480, 640],
            textvariable=self.imgsz_var,
            state="readonly",
            width=6,
        )
        self.imgsz_combo.pack(side=tk.RIGHT)

        # Real-time
        self.realtime_var = tk.BooleanVar(value=self.settings.realtime)
        ttk.Checkbutton(
            settings_group, text="Real-time (skip frames)", variable=self.realtime_var
        ).pack(anchor=tk.W)

        # Schedule
        sched_row = ttk.Frame(settings_group)
        sched_row.pack(fill=tk.X, pady=(2, 0))
        self.sched_var = tk.BooleanVar(value=self.settings.schedule_enabled)
        ttk.Checkbutton(sched_row, text="Use schedule", variable=self.sched_var).pack(side=tk.LEFT)
        ttk.Label(sched_row, text="Sim start:").pack(side=tk.LEFT, padx=(8, 2))
        self.sim_start_var = tk.StringVar(value=self.settings.sim_start)
        self.sim_start_entry = ttk.Entry(sched_row, textvariable=self.sim_start_var, width=6)
        self.sim_start_entry.pack(side=tk.RIGHT)

        # Save video
        self.save_video_var = tk.BooleanVar(value=self.settings.save_video)
        ttk.Checkbutton(
            settings_group, text="Save annotated video", variable=self.save_video_var
        ).pack(anchor=tk.W, pady=(2, 0))

        # Popup & Sound
        alert_opt_row = ttk.Frame(settings_group)
        alert_opt_row.pack(fill=tk.X, pady=(2, 0))
        self.popup_var = tk.BooleanVar(value=self.settings.popup_on_alert)
        ttk.Checkbutton(alert_opt_row, text="Pop-up", variable=self.popup_var).pack(side=tk.LEFT)
        self.sound_var = tk.BooleanVar(value=self.settings.sound_on_alert)
        ttk.Checkbutton(alert_opt_row, text="Beep", variable=self.sound_var).pack(side=tk.LEFT, padx=(8, 0))

        # Live counters
        counters_group = ttk.LabelFrame(parent, text="Live Status", padding="6 4 6 4")
        counters_group.pack(fill=tk.X, pady=(0, 6))

        self.lbl_tracking = ttk.Label(
            counters_group, text="Tracking: 0 person, 0 car, 0 motorcycle"
        )
        self.lbl_tracking.pack(anchor=tk.W)
        self.lbl_alerts_cnt = ttk.Label(
            counters_group, text="Alerts: 0", font=("TkDefaultFont", 9, "bold")
        )
        self.lbl_alerts_cnt.pack(anchor=tk.W)

        # Alerts table
        alerts_group = ttk.LabelFrame(parent, text="Alerts", padding="6 6 6 6")
        alerts_group.pack(fill=tk.BOTH, expand=True)

        tree_frame = ttk.Frame(alerts_group)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.alerts_tree = ttk.Treeview(
            tree_frame,
            columns=("#", "Time", "Type", "ID", "Zone"),
            show="headings",
            height=5,
        )
        self.alerts_tree.heading("#", text="#")
        self.alerts_tree.column("#", width=30, stretch=False)
        self.alerts_tree.heading("Time", text="Time")
        self.alerts_tree.column("Time", width=55, stretch=False)
        self.alerts_tree.heading("Type", text="Type")
        self.alerts_tree.column("Type", width=65, stretch=False)
        self.alerts_tree.heading("ID", text="ID")
        self.alerts_tree.column("ID", width=35, stretch=False)
        self.alerts_tree.heading("Zone", text="Zone")
        self.alerts_tree.column("Zone", width=85, stretch=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.alerts_tree.yview)
        self.alerts_tree.configure(yscrollcommand=scrollbar.set)
        self.alerts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.alerts_tree.bind("<<TreeviewSelect>>", self._on_alert_selected)
        self.alerts_tree.bind("<Double-1>", self._on_alert_double_clicked)

        # Thumbnail preview under table
        thumb_frame = ttk.Frame(alerts_group, height=110)
        thumb_frame.pack(fill=tk.X, pady=(6, 0))
        self.lbl_crop_preview = ttk.Label(
            thumb_frame, text="Select an alert to preview crop", anchor=tk.CENTER
        )
        self.lbl_crop_preview.pack(fill=tk.BOTH, expand=True)

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda e: self._toggle_pause())
        self.root.bind("<Control-o>", lambda e: self._open_video())
        self.root.bind("<Control-O>", lambda e: self._open_video())
        self.root.bind("<Escape>", self._on_esc_key)
        self.root.bind("<Return>", self._on_enter_key)
        self.root.bind("<BackSpace>", self._on_backspace_key)

    def _on_dwell_changed(self) -> None:
        try:
            val = float(self.dwell_var.get())
            self.settings.dwell_seconds = max(0.1, min(60.0, val))
        except (ValueError, tk.TclError):
            pass

    def _on_conf_changed(self, event: Any = None) -> None:
        val = round(self.conf_var.get(), 2)
        self.settings.confidence = val
        self.conf_val_lbl.config(text=f"{val:.2f}")

    def _update_button_states(self, state: str) -> None:
        if state == "NO_VIDEO":
            self.btn_open.config(state=tk.NORMAL)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.DISABLED, text="Pause")
            self.btn_stop.config(state=tk.DISABLED)
            self._set_zone_btns(tk.DISABLED)
        elif state == "IDLE":
            self.btn_open.config(state=tk.NORMAL)
            self.btn_start.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED, text="Pause")
            self.btn_stop.config(state=tk.DISABLED)
            self._set_zone_btns(tk.NORMAL)
        elif state == "RUNNING":
            self.btn_open.config(state=tk.DISABLED)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL, text="Pause")
            self.btn_stop.config(state=tk.NORMAL)
            self._set_zone_btns(tk.DISABLED)
        elif state == "PAUSED":
            self.btn_open.config(state=tk.DISABLED)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL, text="Resume")
            self.btn_stop.config(state=tk.NORMAL)
            self._set_zone_btns(tk.DISABLED)
        elif state == "FINISHED":
            self.btn_open.config(state=tk.NORMAL)
            self.btn_start.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED, text="Pause")
            self.btn_stop.config(state=tk.DISABLED)
            self._set_zone_btns(tk.NORMAL)

    def _set_zone_btns(self, state: str) -> None:
        self.btn_add_zone.config(state=state)
        self.btn_edit_zone.config(state=state)
        self.btn_del_zone.config(state=state)
        self.btn_clear_zones.config(state=state)

    def _set_banner(self, text: str, bg: str = "#e9ecef", fg: str = "#495057") -> None:
        self.status_banner.config(text=text, bg=bg, fg=fg)

    def _open_video(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return

        filetypes = [
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*"),
        ]
        chosen = filedialog.askopenfilename(title="Open Video File", filetypes=filetypes)
        if not chosen:
            return

        cap = cv2.VideoCapture(chosen)
        if not cap.isOpened():
            messagebox.showerror(
                "Error Opening Video",
                f"Could not open video file:\n{chosen}\n"
                "Please verify the format and ensure codecs are installed.",
            )
            return

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            messagebox.showerror(
                "Empty Video",
                f"Video contains no readable frames:\n{chosen}",
            )
            return

        self.current_video_path = chosen
        max_w = self.settings.max_work_width
        self.first_frame_bgr = downscale_if_needed(frame, max_w)
        fh, fw = self.first_frame_bgr.shape[:2]
        self.working_size = (fw, fh)

        # Load saved zones for this video
        v_name = os.path.basename(chosen)
        self.current_zones = get_zones(self.config, v_name)

        # Refresh UI
        self._refresh_zones_list()
        self._show_first_frame_with_zones()

        self._update_button_states("IDLE")
        self._set_banner("READY - press Start", bg="#28a745", fg="white")

        hint = "" if self.current_zones else " | Note: No zones drawn. Use 'Add Zone' to define restricted areas."
        self.status_bar.config(
            text=f"Loaded: {v_name} ({fw}x{fh}){hint}"
        )

    def _show_first_frame_with_zones(self) -> None:
        if self.first_frame_bgr is None:
            return

        frame = self.first_frame_bgr.copy()
        fw, fh = self.working_size
        zones_px = [
            (z.name, [(pt[0] * fw, pt[1] * fh) for pt in z.points])
            for z in self.current_zones
        ]
        active_map = {z.name: True for z in self.current_zones}
        violating_map = {z.name: False for z in self.current_zones}
        draw_zones(frame, zones_px, active_map, violating_map)
        self.canvas_panel.set_frame(frame)

    def _refresh_zones_list(self) -> None:
        self.zones_listbox.delete(0, tk.END)
        for z in self.current_zones:
            allowed_str = ", ".join(z.allowed) if z.allowed else "none"
            hours_str = f" | {z.active_from}-{z.active_to}" if z.active_from else ""
            desc = f"{z.name} (allowed: {allowed_str}{hours_str})"
            self.zones_listbox.insert(tk.END, desc)

    def _start_draw_zone(self) -> None:
        if self.first_frame_bgr is None:
            return

        self.is_drawing_zone = True
        self.drawing_points.clear()
        self._clear_temp_canvas_items()
        self._show_first_frame_with_zones()

        self._set_banner(
            "DRAWING ZONE: Click to add points, Right-click/Enter to finish, Backspace undo, Esc cancel",
            bg="#ffc107",
            fg="#212529",
        )
        self.status_bar.config(text="Drawing zone mode active. Click on video to place polygon vertices.")

    def _clear_temp_canvas_items(self) -> None:
        for item in self._temp_canvas_items:
            self.canvas_panel.canvas.delete(item)
        self._temp_canvas_items.clear()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self.is_drawing_zone:
            return

        norm = self.canvas_panel.canvas_to_norm(event.x, event.y)
        if norm is None:
            return  # Click in black bars

        self.drawing_points.append(norm)
        self._redraw_drawing_preview()

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if not self.is_drawing_zone or not self.drawing_points:
            return

        self._redraw_drawing_preview(mouse_pos=(event.x, event.y))

    def _on_canvas_finish_click(self, event: tk.Event) -> None:
        if not self.is_drawing_zone:
            return
        self._finish_drawing_zone()

    def _on_enter_key(self, event: tk.Event) -> None:
        if self.is_drawing_zone:
            self._finish_drawing_zone()

    def _on_backspace_key(self, event: tk.Event) -> None:
        if self.is_drawing_zone and self.drawing_points:
            self.drawing_points.pop()
            self._redraw_drawing_preview()

    def _on_esc_key(self, event: tk.Event) -> None:
        if self.is_drawing_zone:
            self.is_drawing_zone = False
            self.drawing_points.clear()
            self._clear_temp_canvas_items()
            self._show_first_frame_with_zones()
            self._set_banner("READY - press Start", bg="#28a745", fg="white")
            self.status_bar.config(text="Zone drawing cancelled.")
        elif self._snapshot_window is not None:
            self._snapshot_window.destroy()
            self._snapshot_window = None

    def _redraw_drawing_preview(self, mouse_pos: tuple[int, int] | None = None) -> None:
        self._clear_temp_canvas_items()
        c = self.canvas_panel.canvas

        pixel_pts: list[tuple[float, float]] = []
        for nx, ny in self.drawing_points:
            cx, cy = self.canvas_panel.norm_to_canvas(nx, ny)
            pixel_pts.append((cx, cy))
            # Draw vertex dot
            r = 4
            dot = c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#00ff00", outline="#ffffff", width=1)
            self._temp_canvas_items.append(dot)

        # Draw connecting lines
        for i in range(len(pixel_pts) - 1):
            line = c.create_line(pixel_pts[i], pixel_pts[i + 1], fill="#00ff00", width=2)
            self._temp_canvas_items.append(line)

        # Draw faint guideline to mouse cursor
        if mouse_pos and pixel_pts:
            guide1 = c.create_line(pixel_pts[-1], mouse_pos, fill="#88ff88", width=1, dash=(3, 3))
            self._temp_canvas_items.append(guide1)
            if len(pixel_pts) >= 2:
                guide2 = c.create_line(mouse_pos, pixel_pts[0], fill="#88ff88", width=1, dash=(3, 3))
                self._temp_canvas_items.append(guide2)

    def _finish_drawing_zone(self) -> None:
        if len(self.drawing_points) < 3:
            messagebox.showinfo("Zone Drawing", "A zone needs at least 3 points. Click more points or Esc to cancel.")
            return

        self.is_drawing_zone = False
        self._clear_temp_canvas_items()

        # Suggest default unique name
        existing_names = {z.name for z in self.current_zones}
        counter = len(self.current_zones) + 1
        default_name = f"Zone {counter}"
        while default_name in existing_names:
            counter += 1
            default_name = f"Zone {counter}"

        dlg = ZoneDialog(
            parent=self.root,
            title="Configure Zone",
            available_classes=list(self.config.classes.values()),
            existing_names=existing_names,
            initial_name=default_name,
        )
        self.root.wait_window(dlg)

        if dlg.result:
            new_zone = Zone(
                name=dlg.result["name"],
                points=list(self.drawing_points),
                allowed=dlg.result["allowed"],
                active_from=dlg.result["active_from"],
                active_to=dlg.result["active_to"],
            )
            self.current_zones.append(new_zone)
            if self.current_video_path:
                v_name = os.path.basename(self.current_video_path)
                set_zones(self.config, v_name, self.current_zones)
                save_config(self.config, self.config_path)

            self._refresh_zones_list()
            self._show_first_frame_with_zones()
            self._set_banner("READY - press Start", bg="#28a745", fg="white")
            self.status_bar.config(text=f"Zone '{new_zone.name}' saved successfully.")
        else:
            self._show_first_frame_with_zones()
            self._set_banner("READY - press Start", bg="#28a745", fg="white")

    def _edit_zone(self) -> None:
        sel = self.zones_listbox.curselection()
        if not sel:
            messagebox.showinfo("Edit Zone", "Select a zone from the list to edit.")
            return

        idx = sel[0]
        z = self.current_zones[idx]
        existing_names = {other.name for i, other in enumerate(self.current_zones) if i != idx}

        dlg = ZoneDialog(
            parent=self.root,
            title=f"Edit Zone: {z.name}",
            available_classes=list(self.config.classes.values()),
            existing_names=existing_names,
            initial_name=z.name,
            initial_allowed=z.allowed,
            initial_from=z.active_from,
            initial_to=z.active_to,
        )
        self.root.wait_window(dlg)

        if dlg.result:
            z.name = dlg.result["name"]
            z.allowed = dlg.result["allowed"]
            z.active_from = dlg.result["active_from"]
            z.active_to = dlg.result["active_to"]

            if self.current_video_path:
                v_name = os.path.basename(self.current_video_path)
                set_zones(self.config, v_name, self.current_zones)
                save_config(self.config, self.config_path)

            self._refresh_zones_list()
            self._show_first_frame_with_zones()
            self.status_bar.config(text=f"Zone '{z.name}' updated.")

    def _delete_zone(self) -> None:
        sel = self.zones_listbox.curselection()
        if not sel:
            messagebox.showinfo("Delete Zone", "Select a zone from the list to delete.")
            return

        idx = sel[0]
        z = self.current_zones[idx]
        if not messagebox.askyesno("Delete Zone", f"Are you sure you want to delete zone '{z.name}'?"):
            return

        del self.current_zones[idx]
        if self.current_video_path:
            v_name = os.path.basename(self.current_video_path)
            set_zones(self.config, v_name, self.current_zones)
            save_config(self.config, self.config_path)

        self._refresh_zones_list()
        self._show_first_frame_with_zones()
        self.status_bar.config(text=f"Zone '{z.name}' deleted.")

    def _clear_all_zones(self) -> None:
        if not self.current_zones:
            return
        if not messagebox.askyesno("Clear Zones", "Delete ALL zones for this video?"):
            return

        self.current_zones.clear()
        if self.current_video_path:
            v_name = os.path.basename(self.current_video_path)
            set_zones(self.config, v_name, [])
            save_config(self.config, self.config_path)

        self._refresh_zones_list()
        self._show_first_frame_with_zones()
        self.status_bar.config(text="All zones cleared.")

    def _start(self) -> None:
        if not self.current_video_path:
            return

        # Synchronize UI settings to Settings dataclass
        self._on_dwell_changed()
        self._on_conf_changed()
        try:
            self.settings.imgsz = int(self.imgsz_var.get())
        except ValueError:
            self.settings.imgsz = 480
        self.settings.realtime = bool(self.realtime_var.get())
        self.settings.schedule_enabled = bool(self.sched_var.get())
        self.settings.sim_start = self.sim_start_var.get().strip() or "12:00"
        self.settings.save_video = bool(self.save_video_var.get())
        self.settings.popup_on_alert = bool(self.popup_var.get())
        self.settings.sound_on_alert = bool(self.sound_var.get())

        # Save settings on Start
        save_config(self.config, self.config_path)

        # Clear alerts UI
        self.alerts_list.clear()
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        self.lbl_alerts_cnt.config(text="Alerts: 0")
        self.lbl_crop_preview.config(image="", text="No alert selected")

        # Drain queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

        self.stop_event.clear()
        self.pause_event.clear()

        # Start VideoWorker
        self.worker = VideoWorker(
            video_path=self.current_video_path,
            zones=self.current_zones,
            settings=self.settings,
            class_map=self.config.classes,
            frame_queue=self.frame_queue,
            event_queue=self.event_queue,
            stop_event=self.stop_event,
            pause_event=self.pause_event,
        )
        self.worker.start()

        self._update_button_states("RUNNING")
        self._set_banner("MONITORING", bg="#28a745", fg="white")

    def _toggle_pause(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return

        if self.pause_event.is_set():
            # Resume
            self.pause_event.clear()
            self._update_button_states("RUNNING")
            self._set_banner("MONITORING", bg="#28a745", fg="white")
        else:
            # Pause
            self.pause_event.set()
            self._update_button_states("PAUSED")
            self._set_banner("PAUSED", bg="#6c757d", fg="white")

    def _stop(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self.pause_event.clear()

    def _poll(self) -> None:
        # 1. Drain event queue
        while True:
            try:
                msg = self.event_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(msg, AlertMsg):
                self._handle_alert_msg(msg.saved_alert)
            elif isinstance(msg, StatusMsg):
                self.status_bar.config(text=msg.text)
            elif isinstance(msg, ErrorMsg):
                messagebox.showerror("Error", msg.text)
                self._set_banner(f"ERROR: {msg.text[:30]}", bg="#dc3545", fg="white")
                self._update_button_states("IDLE")
            elif isinstance(msg, DoneMsg):
                self._handle_done_msg(msg.summary)

        # 2. Get newest frame packet
        newest_packet: FramePacket | None = None
        while True:
            try:
                newest_packet = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if newest_packet is not None:
            self.canvas_panel.set_frame(newest_packet.frame)

            # Update status bar
            v_name = (
                os.path.basename(self.current_video_path)
                if self.current_video_path
                else ""
            )
            total_str = f"/{newest_packet.total_frames}" if newest_packet.total_frames > 0 else ""
            stats = newest_packet.stats
            fps_val = stats.get("fps", 0.0)
            self.status_bar.config(
                text=(
                    f"{v_name} | frame {newest_packet.frame_idx}{total_str} | "
                    f"FPS: {fps_val:.1f} | time: {newest_packet.video_time_s:.1f}s"
                )
            )

            # Update live track counts
            counts_now = stats.get("counts_now", {})
            parts = [f"{cnt} {cls_name}" for cls_name, cnt in counts_now.items()]
            self.lbl_tracking.config(text=f"Tracking: {', '.join(parts)}")

        # Reschedule poll
        self.root.after(15, self._poll)

    def _handle_alert_msg(self, saved_alert: SavedAlert) -> None:
        self.alerts_list.insert(0, saved_alert)
        ev = saved_alert.event

        # Insert at top of tree
        self.alerts_tree.insert(
            "",
            0,
            values=(
                saved_alert.alert_no,
                f"{ev.video_time_s:.1f}s",
                ev.cls_name,
                ev.track_id,
                ev.zone_name,
            ),
        )

        self.lbl_alerts_cnt.config(text=f"Alerts: {len(self.alerts_list)}")

        # Flash red banner for 4 seconds
        banner_text = f"INTRUSION: {ev.cls_name} #{ev.track_id} in '{ev.zone_name}'"
        self._set_banner(banner_text, bg="#dc3545", fg="white")

        if self._banner_reset_timer is not None:
            self.root.after_cancel(self._banner_reset_timer)
        self._banner_reset_timer = self.root.after(
            4000,
            lambda: self._set_banner("MONITORING", bg="#28a745", fg="white")
            if (self.worker and self.worker.is_alive() and not self.pause_event.is_set())
            else None,
        )

        # Non-blocking popup
        if self.settings.popup_on_alert:
            self.alert_popup.show_alert(saved_alert)

        # Beep
        if self.settings.sound_on_alert:
            self.root.bell()

    def _handle_done_msg(self, summary: dict[str, Any]) -> None:
        self._update_button_states("FINISHED")
        n_alerts = summary.get("alerts_total", len(self.alerts_list))
        self._set_banner(f"FINISHED - {n_alerts} alerts", bg="#17a2b8", fg="white")

        # Remember latest run output folder
        if self.worker and hasattr(self.worker, "_custom_recorder") and self.worker._custom_recorder:
            self.latest_run_dir = str(self.worker._custom_recorder.run_dir)
        elif self.worker:
            # Look in output directory for newest run
            out_base = Path(self.settings.output_dir)
            if out_base.exists():
                runs = sorted(out_base.glob("run_*"))
                if runs:
                    self.latest_run_dir = str(runs[-1])

        msg = f"Video processing finished. Total alerts: {n_alerts}."
        if self.latest_run_dir:
            msg += f"\nOutputs saved to:\n{self.latest_run_dir}"
        self.status_bar.config(text=msg.replace("\n", " "))

    def _on_alert_selected(self, event: tk.Event) -> None:
        sel = self.alerts_tree.selection()
        if not sel:
            return

        item = self.alerts_tree.item(sel[0])
        alert_no = int(item["values"][0])
        matched = next((a for a in self.alerts_list if a.alert_no == alert_no), None)
        if not matched or not os.path.exists(matched.crop_path):
            self.lbl_crop_preview.config(image="", text="Crop file missing")
            return

        try:
            crop_bytes = np.fromfile(matched.crop_path, dtype=np.uint8)
            crop_bgr = cv2.imdecode(crop_bytes, cv2.IMREAD_COLOR)
            if crop_bgr is not None:
                ch, cw = crop_bgr.shape[:2]
                target_w = 160
                target_h = max(1, int(round(ch * (target_w / float(cw)))))
                if target_h > 100:
                    target_h = 100
                    target_w = max(1, int(round(cw * (target_h / float(ch)))))
                resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self._thumb_ref = ImageTk.PhotoImage(img)
                self.lbl_crop_preview.config(image=self._thumb_ref, text="")
        except Exception:
            self.lbl_crop_preview.config(image="", text="Failed to load crop")

    def _on_alert_double_clicked(self, event: tk.Event) -> None:
        sel = self.alerts_tree.selection()
        if not sel:
            return

        item = self.alerts_tree.item(sel[0])
        alert_no = int(item["values"][0])
        matched = next((a for a in self.alerts_list if a.alert_no == alert_no), None)
        if not matched or not os.path.exists(matched.snapshot_path):
            return

        # Open snapshot in separate window
        snap_path = matched.snapshot_path
        snap_bytes = np.fromfile(snap_path, dtype=np.uint8)
        snap_bgr = cv2.imdecode(snap_bytes, cv2.IMREAD_COLOR)
        if snap_bgr is None:
            return

        win = tk.Toplevel(self.root)
        self._snapshot_window = win
        win.title(f"Snapshot: {os.path.basename(snap_path)}")

        # Fit image to screen
        screen_w = self.root.winfo_screenwidth() - 100
        screen_h = self.root.winfo_screenheight() - 120
        sh, sw = snap_bgr.shape[:2]
        scale = min(screen_w / float(sw), screen_h / float(sh), 1.0)
        disp_w = int(round(sw * scale))
        disp_h = int(round(sh * scale))

        resized = cv2.resize(snap_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        lbl = ttk.Label(win, image=photo)
        lbl.image = photo  # keep reference
        lbl.pack(fill=tk.BOTH, expand=True)

        win.bind("<Escape>", lambda e: win.destroy())

    def _open_output_folder(self) -> None:
        target_dir = self.latest_run_dir if self.latest_run_dir else self.settings.output_dir
        p = Path(target_dir)
        p.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))
            elif sys.platform.startswith("darwin"):
                subprocess.run(["open", str(p)], check=False)
            else:
                subprocess.run(["xdg-open", str(p)], check=False)
        except Exception as exc:
            messagebox.showinfo("Output Folder", f"Path: {p.resolve()}\n(Could not open automatically: {exc})")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About Intrusion Guard",
            "Intrusion Guard\n"
            "AI-Based Intelligent Border Surveillance and Intrusion Detection System\n"
            "Version 1.0.0\n\n"
            "Detects people and vehicles, tracks them across virtual zones, "
            "and captures intrusion evidence with time-based rules.",
        )

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self.pause_event.clear()
            self.worker.join(timeout=3.0)

        # Save settings to config.yaml on exit
        try:
            self._on_dwell_changed()
            self._on_conf_changed()
            save_config(self.config, self.config_path)
        except Exception:
            pass

        self.root.destroy()

