#!/usr/bin/env python3
"""Export the DAPR U-Net hyperparameter-sensitivity table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .dapr_hyperparameter_spec import (
        DAPR_HYPERPARAMETER_MODELS,
        DISPLAY_NAMES,
        SENSITIVITY_COMPARISONS,
    )
except ImportError:
    from dapr_hyperparameter_spec import (
        DAPR_HYPERPARAMETER_MODELS,
        DISPLAY_NAMES,
        SENSITIVITY_COMPARISONS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--latex-path", required=True)
    parser.add_argument("--delta-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    latex_path = Path(args.latex_path)
    delta_path = Path(args.delta_path) if args.delta_path else latex_path.with_suffix(".deltas.csv")

    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = {row["model"]: row for row in csv.DictReader(file)}
    missing = set(DAPR_HYPERPARAMETER_MODELS) - set(rows)
    if missing:
        raise SystemExit(f"ERROR: summary lacks models: {sorted(missing)}")

    dataset = next(iter(rows.values())).get("dataset", "ETIS-LaribPolypDB")
    seed_count = next(iter(rows.values())).get("num_seeds", "three")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Hyperparameter sensitivity of DAPR U-Net on "
            f"{dataset} over {seed_count} random seeds.}}"
        ),
        r"\label{tab:dapr-hyperparameter-sensitivity}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Setting & Dice $\uparrow$ & IoU $\uparrow$ & Precision $\uparrow$ & Recall $\uparrow$ & MAE $\downarrow$ & Loss $\downarrow$ \\",
        r"\midrule",
    ]
    for model in DAPR_HYPERPARAMETER_MODELS:
        row = rows[model]
        lines.append(
            f"{DISPLAY_NAMES[model]} & {row['dice_mean_pm_std']} & "
            f"{row['iou_mean_pm_std']} & {row['precision_mean_pm_std']} & "
            f"{row['recall_mean_pm_std']} & {row['mae_mean_pm_std']} & "
            f"{row['loss_mean_pm_std']} \\\\"  # noqa: W605
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    deltas = []
    for variant, default in SENSITIVITY_COMPARISONS:
        deltas.append({
            "default_model": default,
            "variant_model": variant,
            "dice_change_vs_default": float(rows[variant]["dice_mean"]) - float(rows[default]["dice_mean"]),
            "iou_change_vs_default": float(rows[variant]["iou_mean"]) - float(rows[default]["iou_mean"]),
            "mae_change_vs_default": float(rows[variant]["mae_mean"]) - float(rows[default]["mae_mean"]),
        })
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    with delta_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    print(f"LaTeX table: {latex_path}")
    print(f"Sensitivity deltas: {delta_path}")


if __name__ == "__main__":
    main()
