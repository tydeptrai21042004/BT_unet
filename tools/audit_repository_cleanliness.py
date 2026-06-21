#!/usr/bin/env python3
"""Reject bytecode, caches, archives, checkpoints, and other binary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".pyre",
    ".hypothesis",
    "htmlcov",
}

FORBIDDEN_SUFFIXES = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib",
    ".h5", ".hdf5", ".bin", ".dat",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".pdf",
}

SKIP_TOP_LEVEL = {".git"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def contains_nul(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            while chunk := file.read(65536):
                if b"\x00" in chunk:
                    return True
    except OSError:
        return True
    return False


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    violations: list[dict[str, str]] = []

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in SKIP_TOP_LEVEL:
            continue

        if path.is_dir() and path.name in FORBIDDEN_DIR_NAMES:
            violations.append({"path": str(relative), "reason": "forbidden cache directory"})
            continue

        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            violations.append({"path": str(relative), "reason": f"forbidden binary suffix {suffix}"})
            continue

        if contains_nul(path):
            violations.append({"path": str(relative), "reason": "contains NUL bytes"})

    report = {
        "root": str(root),
        "clean": not violations,
        "violation_count": len(violations),
        "violations": violations,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Repository root: {root}")
        print(f"Clean: {report['clean']}")
        print(f"Violations: {len(violations)}")
        for item in violations:
            print(f"  - {item['path']}: {item['reason']}")

    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
