#!/usr/bin/env python3
"""Remove generated caches and binary artifacts from a source checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".pyre", ".hypothesis", "htmlcov",
}
BINARY_SUFFIXES = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib",
    ".h5", ".hdf5", ".bin", ".dat",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--apply", action="store_true", help="Delete files. Without this flag, only list them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    targets: list[Path] = []

    for path in root.rglob("*"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_dir() and path.name in CACHE_DIRS:
            targets.append(path)
        elif path.is_file() and path.suffix.lower() in BINARY_SUFFIXES:
            targets.append(path)

    # Remove nested files/directories only through their highest selected parent.
    selected: list[Path] = []
    for target in sorted(targets, key=lambda item: (len(item.parts), str(item))):
        if not any(parent == target or parent in target.parents for parent in selected):
            selected.append(target)

    action = "REMOVE" if args.apply else "WOULD REMOVE"
    for target in selected:
        print(f"{action}: {target.relative_to(root)}")
        if args.apply:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)

    print(f"Artifacts {'removed' if args.apply else 'found'}: {len(selected)}")


if __name__ == "__main__":
    main()
