from __future__ import annotations

from pathlib import Path


def _get_imports(file_path: Path) -> set[str]:
    """Extracts top-level imported module names from Python source."""
    imports = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("import "):
            parts = line.split()
            if len(parts) > 1:
                module = parts[1].split(".")[0].split(",")[0]
                imports.add(module)
        elif line.startswith("from "):
            parts = line.split()
            if len(parts) > 1:
                module = parts[1].split(".")[0]
                imports.add(module)
    return imports


def test_rules_purity():
    rules_file = Path("guard/rules.py")
    assert rules_file.exists()
    imports = _get_imports(rules_file)

    forbidden = {"cv2", "tkinter", "torch", "ultralytics", "time"}
    found_forbidden = imports.intersection(forbidden)
    assert not found_forbidden, f"guard/rules.py imports forbidden modules: {found_forbidden}"


def test_pipeline_and_alerts_no_tkinter():
    for f in [Path("guard/pipeline.py"), Path("guard/alerts.py")]:
        assert f.exists()
        imports = _get_imports(f)
        assert "tkinter" not in imports, f"{f} imports tkinter!"


def test_only_detector_imports_ultralytics():
    guard_dir = Path("guard")
    for py_file in guard_dir.rglob("*.py"):
        if py_file.name == "detector.py":
            continue
        imports = _get_imports(py_file)
        assert "ultralytics" not in imports, f"{py_file} imports ultralytics directly!"


def test_no_shadowing_module_names():
    root = Path(".")
    shadowing = {"types.py", "queue.py", "config.py"}
    # Check top-level root
    for f in root.glob("*.py"):
        assert f.name not in shadowing, f"Top-level file {f.name} shadows standard library!"

