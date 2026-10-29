from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk
from typing import Any, Sequence


def _is_valid_hhmm(s: str) -> bool:
    return bool(re.fullmatch(r"^(?:[01]\d|2[0-3]):[0-5]\d$", s.strip()))


class ZoneDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        available_classes: Sequence[str],
        existing_names: set[str],
        initial_name: str = "Zone 1",
        initial_allowed: Sequence[str] | None = None,
        initial_from: str | None = None,
        initial_to: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)

        self.available_classes = list(available_classes)
        self.existing_names = set(existing_names)
        self.initial_name = initial_name

        self.result: dict[str, Any] | None = None

        self._build_ui(
            initial_name,
            initial_allowed or [],
            initial_from or "",
            initial_to or "",
        )

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda e: self._on_cancel())
        self.bind("<Return>", lambda e: self._on_save())

        self.geometry("+%d+%d" % (parent.winfo_rootx() + 100, parent.winfo_rooty() + 100))
        self.name_entry.focus_set()
        self.name_entry.select_range(0, tk.END)

    def _build_ui(
        self,
        initial_name: str,
        initial_allowed: Sequence[str],
        initial_from: str,
        initial_to: str,
    ) -> None:
        frame = ttk.Frame(self, padding="16 16 16 16")
        frame.pack(fill=tk.BOTH, expand=True)

        # Name row
        ttk.Label(frame, text="Zone Name:").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        self.name_var = tk.StringVar(value=initial_name)
        self.name_entry = ttk.Entry(frame, textvariable=self.name_var, width=28)
        self.name_entry.grid(row=0, column=1, sticky=tk.EW, pady=(0, 4))

        # Allowed types
        ttk.Label(frame, text="Allowed Types:").grid(row=1, column=0, sticky=tk.NW, pady=(8, 4))
        types_frame = ttk.Frame(frame)
        types_frame.grid(row=1, column=1, sticky=tk.W, pady=(8, 4))

        self.allowed_vars: dict[str, tk.BooleanVar] = {}
        for cls_name in self.available_classes:
            var = tk.BooleanVar(value=(cls_name in initial_allowed))
            self.allowed_vars[cls_name] = var
            cb = ttk.Checkbutton(types_frame, text=cls_name, variable=var)
            cb.pack(anchor=tk.W, pady=1)

        ttk.Label(
            types_frame,
            text="(Unticked = not allowed \u2192 triggers alert)",
            font=("TkDefaultFont", 8, "italic"),
        ).pack(anchor=tk.W, pady=(2, 0))

        # Active hours
        ttk.Label(frame, text="Active Hours:").grid(row=2, column=0, sticky=tk.NW, pady=(8, 4))
        hours_frame = ttk.Frame(frame)
        hours_frame.grid(row=2, column=1, sticky=tk.W, pady=(8, 4))

        ttk.Label(hours_frame, text="From:").grid(row=0, column=0, sticky=tk.W)
        self.from_var = tk.StringVar(value=initial_from)
        self.from_entry = ttk.Entry(hours_frame, textvariable=self.from_var, width=8)
        self.from_entry.grid(row=0, column=1, padx=(4, 12))

        ttk.Label(hours_frame, text="To:").grid(row=0, column=2, sticky=tk.W)
        self.to_var = tk.StringVar(value=initial_to)
        self.to_entry = ttk.Entry(hours_frame, textvariable=self.to_var, width=8)
        self.to_entry.grid(row=0, column=3, padx=(4, 0))

        ttk.Label(
            hours_frame,
            text="(Format HH:MM, e.g. 19:00 - 07:00, or leave both empty for 24/7)",
            font=("TkDefaultFont", 8, "italic"),
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))

        # Error label
        self.error_label = ttk.Label(frame, text="", foreground="red")
        self.error_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=2, sticky=tk.E, pady=(12, 0))

        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=(6, 0))

        save_btn = ttk.Button(btn_frame, text="Save Zone", command=self._on_save)
        save_btn.pack(side=tk.RIGHT)

    def _on_save(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            self.error_label.config(text="Zone name cannot be empty.")
            return

        # Check unique if changed
        if name != self.initial_name and name in self.existing_names:
            self.error_label.config(text=f"A zone named '{name}' already exists.")
            return

        from_str = self.from_var.get().strip()
        to_str = self.to_var.get().strip()

        if (from_str and not to_str) or (to_str and not from_str):
            self.error_label.config(text="Provide both 'From' and 'To', or leave both empty.")
            return

        if from_str and to_str:
            if not _is_valid_hhmm(from_str):
                self.error_label.config(text=f"Invalid 'From' time: '{from_str}'. Expected HH:MM.")
                return
            if not _is_valid_hhmm(to_str):
                self.error_label.config(text=f"Invalid 'To' time: '{to_str}'. Expected HH:MM.")
                return
            active_from: str | None = from_str
            active_to: str | None = to_str
        else:
            active_from = None
            active_to = None

        allowed = [cls_name for cls_name, var in self.allowed_vars.items() if var.get()]

        self.result = {
            "name": name,
            "allowed": allowed,
            "active_from": active_from,
            "active_to": active_to,
        }
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()

