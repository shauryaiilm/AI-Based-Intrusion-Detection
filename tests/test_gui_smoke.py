from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk

import pytest

from guard.gui.main_window import MainWindow
from guard.schema import Zone
from tests.conftest import create_synthetic_video


def test_gui_smoke(tmp_path: Path):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No display available for GUI test: {exc}")

    try:
        cfg_file = tmp_path / "config.yaml"
        app = MainWindow(root, config_path=str(cfg_file))
        root.update_idletasks()

        # Load synthetic video
        synth_vid = create_synthetic_video(tmp_path / "smoke_synth.mp4", duration_s=1.0)
        app.current_video_path = str(synth_vid)

        import cv2
        cap = cv2.VideoCapture(str(synth_vid))
        ok, frame = cap.read()
        cap.release()
        assert ok
        app.first_frame_bgr = frame
        app.working_size = (frame.shape[1], frame.shape[0])
        app._update_button_states("IDLE")
        root.update_idletasks()

        # Check start button becomes enabled
        assert str(app.btn_start["state"]) == tk.NORMAL

        # Simulate adding a zone programmatically
        new_zone = Zone(
            name="TestZone",
            points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
            allowed=["car"],
        )
        app.current_zones.append(new_zone)
        app._refresh_zones_list()
        app._show_first_frame_with_zones()
        root.update_idletasks()

        assert len(app.current_zones) == 1
        assert app.zones_listbox.size() == 1
    finally:
        root.destroy()

