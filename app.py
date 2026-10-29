"""Intrusion Guard - Desktop Application Entry Point."""
from __future__ import annotations

import logging
import sys
import tkinter as tk

from guard.gui.main_window import MainWindow


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("guard")
    logger.info("Starting Intrusion Guard GUI application...")

    try:
        root = tk.Tk()
    except Exception as exc:
        logger.error(f"Failed to initialize Tkinter display: {exc}")
        print(f"Error: Could not open window display: {exc}", file=sys.stderr)
        sys.exit(1)

    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()

