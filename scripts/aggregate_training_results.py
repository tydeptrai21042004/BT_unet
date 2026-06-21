#!/usr/bin/env python3
"""Aggregate per-seed training/validation summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

TRAINING_METRICS = (
    "train_loss",
    "val_loss",
    "val_dice",
    "val_iou",
    "val_precision",
    "val_recall",
    "val_mae",
    "best_epoch",
)


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--models", required=True, type=parse_csv)
    parser.add_argument("--seeds", required=True, type=parse_csv)
    parser.add_argument("--dataset", default="etis")
    parser.add_argument("--expected-batch-size", type=int, default=None)
    parser.add_argument(
        "--require-no-auxiliary-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-no-boundary-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def finite_float(raw: object, *, name: str, path: Path) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"ERROR: invalid {name}={raw!r} in {path}") from error
    if not math.isfinite(value):
        raise SystemExit(f"ERROR: non-finite {name} in {path}")
    return value


def load_record(
    path: Path,
    *,
    model: str,
    seed: str,
    dataset: str,
    expected_batch_size: int | None,
    require_no_auxiliary_loss: bool,
    require_no_boundary_loss: bool,
) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"ERROR: missing training summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("model", "")).strip() != model:
        raise SystemExit(f"ERROR: wrong model metadata in {path}")
    if str(payload.get("dataset", "")).strip() != dataset:
        raise SystemExit(f"ERROR: wrong dataset metadata in {path}")

    if expected_batch_size is not None:
        physical = int(payload.get("physical_batch_size", 0))
        effective = int(payload.get("effective_batch_size", 0))
        if physical != expected_batch_size:
            raise SystemExit(f"ERROR: physical batch size mismatch in {path}")
        if effective != expected_batch_size:
            raise SystemExit(f"ERROR: effective batch size mismatch in {path}")
    if require_no_auxiliary_loss and bool(payload.get("use_aux_outputs_loss", True)):
        raise SystemExit(f"ERROR: auxiliary-output loss enabled in {path}")
    if require_no_boundary_loss and bool(payload.get("use_boundary_loss", True)):
        raise SystemExit(f"ERROR: boundary loss enabled in {path}")

    best = payload.get("best_record", {})
    mapping = {
        "train_loss": best.get("train/loss"),
        "val_loss": best.get("val/loss"),
        "val_dice": best.get("val/dice"),
        "val_iou": best.get("val/iou"),
        "val_precision": best.get("val/precision"),
        "val_recall": best.get("val/recall"),
        "val_mae": best.get("val/mae"),
        "best_epoch": payload.get("best_epoch"),
    }
    record: dict[str, object] = {"model": model, "seed": seed}
    for name, raw in mapping.items():
        record[name] = finite_float(raw, name=name, path=path)
    return record


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else output_root / "results" / "tables" / "training_summary.csv"
    )

    records = []
    for seed in args.seeds:
        for model in args.models:
            path = output_root / f"seed_{seed}" / model / "results" / "summary.json"
            records.append(
                load_record(
                    path,
                    model=model,
                    seed=seed,
                    dataset=args.dataset,
                    expected_batch_size=args.expected_batch_size,
                    require_no_auxiliary_loss=args.require_no_auxiliary_loss,
                    require_no_boundary_loss=args.require_no_boundary_loss,
                )
            )

    rows = []
    for model in args.models:
        items = [item for item in records if item["model"] == model]
        row: dict[str, object] = {
            "model": model,
            "dataset": args.dataset,
            "num_seeds": len(items),
            "seeds": ",".join(sorted((str(item["seed"]) for item in items), key=int)),
        }
        for metric in TRAINING_METRICS:
            values = [float(item[metric]) for item in items]
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_pm_std"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Training records: {len(records)}")
    print(f"Training summary: {output_path}")


if __name__ == "__main__":
    main()
