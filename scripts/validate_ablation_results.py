#!/usr/bin/env python3
"""Strictly validate aggregate and per-seed ablation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

METRICS = ("dice", "iou", "precision", "recall", "mae", "loss")


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--training-summary-path", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models", required=True, type=parse_csv)
    parser.add_argument("--seeds", required=True, type=parse_csv)
    parser.add_argument("--dataset", default="etis")
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def finite(value: object, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"ERROR: non-finite {label}")
    return number


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    output_root = Path(args.output_root)
    expected_models = set(args.models)
    expected_seeds = set(args.seeds)

    if not summary_path.is_file():
        raise SystemExit(f"ERROR: missing aggregate summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        headers = reader.fieldnames or []

    rows_by_model = {str(row.get("model", "")): row for row in rows}
    if set(rows_by_model) != expected_models:
        raise SystemExit(
            f"ERROR: aggregate models={sorted(rows_by_model)}, "
            f"expected={sorted(expected_models)}"
        )
    if len(rows) != len(expected_models):
        raise SystemExit("ERROR: duplicate aggregate model rows detected")

    for model, row in rows_by_model.items():
        if str(row.get("dataset", "")) != args.dataset:
            raise SystemExit(f"ERROR: wrong aggregate dataset for {model}")
        if str(row.get("split", "")) != args.split:
            raise SystemExit(f"ERROR: wrong aggregate split for {model}")
        if int(row.get("num_seeds", 0)) != len(expected_seeds):
            raise SystemExit(f"ERROR: wrong seed count for {model}")
        found_seeds = set(re.findall(r"-?\d+", str(row.get("seeds", ""))))
        if found_seeds != expected_seeds:
            raise SystemExit(f"ERROR: wrong seed list for {model}: {found_seeds}")
        for metric in METRICS:
            for suffix in ("mean", "std", "mean_pm_std"):
                column = f"{metric}_{suffix}"
                if column not in headers or not str(row.get(column, "")).strip():
                    raise SystemExit(f"ERROR: missing {column} for {model}")
            mean = finite(row[f"{metric}_mean"], label=f"{model}/{metric}_mean")
            std = finite(row[f"{metric}_std"], label=f"{model}/{metric}_std")
            if std < 0:
                raise SystemExit(f"ERROR: negative standard deviation for {model}/{metric}")
            if metric in {"dice", "iou", "precision", "recall", "mae"} and not 0 <= mean <= 1:
                raise SystemExit(f"ERROR: {model}/{metric} outside [0,1]")
            if metric == "loss" and mean < 0:
                raise SystemExit(f"ERROR: negative loss for {model}")

    records: dict[tuple[str, str], Path] = {}
    for seed in args.seeds:
        seed_root = output_root / f"seed_{seed}"
        candidates = sorted((seed_root / "results" / "tables").rglob("*_metrics.json"))
        if not candidates:
            candidates = sorted(seed_root.rglob("metrics_*.json"))
        for path in candidates:
            payload = json.loads(path.read_text(encoding="utf-8"))
            model = str(payload.get("model", ""))
            if model not in expected_models:
                continue
            if str(payload.get("dataset", "")) != args.dataset:
                raise SystemExit(f"ERROR: wrong dataset in {path}")
            if str(payload.get("split", args.split)) != args.split:
                raise SystemExit(f"ERROR: wrong split in {path}")
            if str(payload.get("seed", seed)) != seed:
                raise SystemExit(f"ERROR: wrong seed in {path}")
            key = (seed, model)
            if key in records:
                raise SystemExit(f"ERROR: duplicate test record {key}")
            values = payload.get("metrics", {})
            for metric in METRICS:
                if metric not in values:
                    raise SystemExit(f"ERROR: {path} lacks {metric}")
                finite(values[metric], label=f"{path}/{metric}")
            records[key] = path

    expected_records = {
        (seed, model)
        for seed in args.seeds
        for model in args.models
    }
    if set(records) != expected_records:
        missing = expected_records - set(records)
        extra = set(records) - expected_records
        raise SystemExit(f"ERROR: missing={sorted(missing)}, extra={sorted(extra)}")

    if args.training_summary_path:
        training_path = Path(args.training_summary_path)
        if not training_path.is_file():
            raise SystemExit(f"ERROR: missing training summary: {training_path}")
        with training_path.open("r", encoding="utf-8-sig", newline="") as file:
            training_rows = list(csv.DictReader(file))
        training_models = {str(row.get("model", "")) for row in training_rows}
        if training_models != expected_models or len(training_rows) != len(expected_models):
            raise SystemExit("ERROR: training summary model scope mismatch")
        for row in training_rows:
            if str(row.get("dataset", "")) != args.dataset:
                raise SystemExit("ERROR: training summary dataset mismatch")
            if int(row.get("num_seeds", 0)) != len(expected_seeds):
                raise SystemExit("ERROR: training summary seed count mismatch")

    print("Ablation result validation passed.")
    print(f"Aggregate rows: {len(rows)}")
    print(f"Individual test records: {len(records)}")


if __name__ == "__main__":
    main()
