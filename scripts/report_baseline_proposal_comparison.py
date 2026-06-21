#!/usr/bin/env python3
"""Print and export the 11-baseline versus DAPR proposal comparison table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .baseline_proposal_spec import (
        BASELINE_MODELS,
        BASELINE_PROPOSAL_MODELS,
        DISPLAY_NAMES,
        PROPOSAL_MODELS,
    )
except ImportError:  # Direct script execution
    from baseline_proposal_spec import (
        BASELINE_MODELS,
        BASELINE_PROPOSAL_MODELS,
        DISPLAY_NAMES,
        PROPOSAL_MODELS,
    )


METRICS = ("dice", "iou", "precision", "recall", "mae", "loss")
HIGHER_IS_BETTER = {"dice", "iou", "precision", "recall"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--latex-path", required=True)
    parser.add_argument("--delta-path", default=None)
    return parser.parse_args()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    latex_path = Path(args.latex_path)
    delta_path = Path(args.delta_path) if args.delta_path else latex_path.with_suffix(".deltas.csv")

    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = {row["model"]: row for row in csv.DictReader(file)}

    missing = set(BASELINE_PROPOSAL_MODELS) - set(rows)
    unexpected = set(rows) - set(BASELINE_PROPOSAL_MODELS)
    if missing or unexpected:
        raise SystemExit(
            f"ERROR: summary model mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    columns = (
        "method",
        "dice_mean_pm_std",
        "iou_mean_pm_std",
        "precision_mean_pm_std",
        "recall_mean_pm_std",
        "mae_mean_pm_std",
        "loss_mean_pm_std",
    )
    pretty: list[dict[str, str]] = []
    for model in BASELINE_PROPOSAL_MODELS:
        row = dict(rows[model])
        row["method"] = DISPLAY_NAMES[model]
        pretty.append(row)

    widths = {
        column: max(len(column), *(len(str(row[column])) for row in pretty))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in pretty:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))

    ranking = sorted(
        BASELINE_PROPOSAL_MODELS,
        key=lambda model: (
            -float(rows[model]["dice_mean"]),
            -float(rows[model]["iou_mean"]),
            float(rows[model]["mae_mean"]),
        ),
    )
    print("\nOverall ranking by Dice, IoU, then MAE:")
    for index, model in enumerate(ranking, start=1):
        print(
            f"{index:2d}. {DISPLAY_NAMES[model]}: "
            f"Dice={rows[model]['dice_mean_pm_std']}, "
            f"IoU={rows[model]['iou_mean_pm_std']}, "
            f"MAE={rows[model]['mae_mean_pm_std']}"
        )

    best_baseline_by_metric: dict[str, str] = {}
    for metric in METRICS:
        if metric in HIGHER_IS_BETTER:
            best = max(BASELINE_MODELS, key=lambda model: float(rows[model][f"{metric}_mean"]))
        else:
            best = min(BASELINE_MODELS, key=lambda model: float(rows[model][f"{metric}_mean"]))
        best_baseline_by_metric[metric] = best

    deltas: list[dict[str, object]] = []
    print("\nProposal changes relative to the strongest baseline for each metric:")
    for proposal in PROPOSAL_MODELS:
        for metric in METRICS:
            baseline = best_baseline_by_metric[metric]
            baseline_value = float(rows[baseline][f"{metric}_mean"])
            proposal_value = float(rows[proposal][f"{metric}_mean"])
            improvement = (
                proposal_value - baseline_value
                if metric in HIGHER_IS_BETTER
                else baseline_value - proposal_value
            )
            item = {
                "proposal": proposal,
                "metric": metric,
                "best_baseline": baseline,
                "best_baseline_value": baseline_value,
                "proposal_value": proposal_value,
                "improvement": improvement,
            }
            deltas.append(item)
            print(
                f"{DISPLAY_NAMES[proposal]:38s} | {metric:9s} | "
                f"best baseline={DISPLAY_NAMES[baseline]:18s} | "
                f"improvement={improvement:+.6f}"
            )

    dataset = next(iter(rows.values())).get("dataset", "dataset")
    seed_count = next(iter(rows.values())).get("num_seeds", "")
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Fair comparison of eleven published baselines and the proposed "
            f"DAPR U-Net method on {latex_escape(str(dataset))} over {seed_count} random seeds."
            r" All methods use the same data split, augmentation, optimization, loss, "
            r"epoch count, threshold, and evaluation protocol.}"
        ),
        r"\label{tab:baseline-proposal-comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Dice $\uparrow$ & IoU $\uparrow$ & Precision $\uparrow$ & Recall $\uparrow$ & MAE $\downarrow$ & Loss $\downarrow$ \\",
        r"\midrule",
    ]
    for model in BASELINE_PROPOSAL_MODELS:
        if model == PROPOSAL_MODELS[0]:
            latex_lines.append(r"\midrule")
        row = rows[model]
        latex_lines.append(
            f"{latex_escape(DISPLAY_NAMES[model])} & {row['dice_mean_pm_std']} & "
            f"{row['iou_mean_pm_std']} & {row['precision_mean_pm_std']} & "
            f"{row['recall_mean_pm_std']} & {row['mae_mean_pm_std']} & "
            f"{row['loss_mean_pm_std']} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    delta_path.parent.mkdir(parents=True, exist_ok=True)
    with delta_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)

    print(f"\nLaTeX table: {latex_path}")
    print(f"Proposal deltas: {delta_path}")


if __name__ == "__main__":
    main()
