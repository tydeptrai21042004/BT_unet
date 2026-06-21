#!/usr/bin/env python3
"""Run the canonical official-faithful benchmark."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .model_sets import BASELINE_MODELS
except ImportError:  # Direct script execution
    from model_sets import BASELINE_MODELS

MODELS = BASELINE_MODELS

def main() -> None:
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "benchmark_all.py"),
        "--config-dir", "configs/official_faithful",
        "--models", ",".join(MODELS),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
