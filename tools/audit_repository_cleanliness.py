#!/usr/bin/env python3
"""Reject source-code artifacts while ignoring runtime data/output folders.

This audit is intended to keep the committed repository clean. It should not fail
only because a Kaggle run downloaded datasets into ``data/`` or wrote metrics to
``outputs_*``. Runtime folders are skipped by default and can be overridden from
CLI if a stricter check is needed.
"""

from __future__ import annotations

import argparse
import fnmatch
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

DEFAULT_SKIP_PATTERNS = (
    ".git",
    ".venv",
    "venv",
    "data",
    "outputs",
    "outputs_*",
    "wandb",
    "mlruns",
    "runs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict-runtime-folders",
        action="store_true",
        help="Also scan runtime folders such as data/ and outputs_*/.",
    )
    parser.add_argument(
        "--skip-pattern",
        action="append",
        default=[],
        help=(
            "Additional top-level path pattern to skip. Can be repeated. "
            "Examples: --skip-pattern cache --skip-pattern checkpoints_*"
        ),
    )
    return parser.parse_args()


def _matches_top_level(relative: Path, patterns: tuple[str, ...]) -> bool:
    if not relative.parts:
        return False
    top = relative.parts[0]
    return any(fnmatch.fnmatch(top, pattern) for pattern in patterns)


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
    skip_patterns = tuple(args.skip_pattern)
    if not args.strict_runtime_folders:
        skip_patterns = DEFAULT_SKIP_PATTERNS + skip_patterns

    violations: list[dict[str, str]] = []

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _matches_top_level(relative, skip_patterns):
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
        "skipped_top_level_patterns": list(skip_patterns),
        "violations": violations,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Repository root: {root}")
        print(f"Clean: {report['clean']}")
        print(f"Violations: {len(violations)}")
        if skip_patterns:
            print("Skipped top-level patterns: " + ", ".join(skip_patterns))
        for item in violations:
            print(f"  - {item['path']}: {item['reason']}")

    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
