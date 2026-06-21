#!/usr/bin/env python3
"""Download public ImageNet backbone checkpoints used by official-style adapters."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.common.official_backbones import DEFAULT_CHECKPOINT_URLS

URLS = dict(DEFAULT_CHECKPOINT_URLS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public official/pretrained backbone checkpoints used by the benchmark adapters.")
    parser.add_argument("--output-dir", type=Path, default=Path("weights/official_backbones"))
    parser.add_argument("--models", nargs="*", default=list(URLS.keys()))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        if name not in URLS:
            raise SystemExit(f"Unknown model key: {name}. Available: {', '.join(sorted(URLS))}")
        suffix = Path(URLS[name]).name
        dest = args.output_dir / suffix
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[SKIP] {name}: already exists at {dest}")
            continue
        print(f"[DOWNLOAD] {name} -> {dest}")
        urllib.request.urlretrieve(URLS[name], dest)
    print("[DONE]")


if __name__ == "__main__":
    main()
