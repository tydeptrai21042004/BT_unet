#!/usr/bin/env python3
"""Run the permanent DAPR U-Net component ablation end to end."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from .dapr_ablation_spec import DEFAULT_MODELS_CSV, DEFAULT_SEEDS
except ImportError:
    from dapr_ablation_spec import DEFAULT_MODELS_CSV, DEFAULT_SEEDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="etis")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--zip-path", default=None)
    parser.add_argument("--download-url", default=None)
    parser.add_argument("--download-dst", default=None)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--models", default=DEFAULT_MODELS_CSV)
    parser.add_argument("--config-dir", default="configs/dapr_ablation")
    parser.add_argument("--output-root", default="outputs_dapr_ablation/etis")
    parser.add_argument("--allow-insecure-download", action="store_true")
    parser.add_argument("--delete-checkpoints-after-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    tables = output_root / "results" / "tables"
    test_summary = tables / "multi_seed_summary.csv"
    training_summary = tables / "full_training_summary.csv"
    latex_table = tables / "dapr_ablation.tex"
    delta_table = tables / "dapr_ablation_deltas.csv"

    run([py, str(PROJECT_ROOT / "tools" / "audit_repository_cleanliness.py")])
    run([py, str(PROJECT_ROOT / "tools" / "audit_dapr_ablation.py"), "--config-dir", args.config_dir])

    if args.run_tests:
        run([
            py, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "tests/test_dapr_unet.py",
            "tests/test_baseline_proposal_comparison.py",
            "tests/test_repository_scope_and_fairness.py",
            "tests/test_pipeline_contracts.py",
        ])

    benchmark = [
        py, str(PROJECT_ROOT / "scripts" / "benchmark_multi_seed.py"),
        "--models", args.models,
        "--config-dir", args.config_dir,
        "--dataset", args.dataset,
        "--data-root", args.data_root,
        "--image-size", str(args.image_size),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--device", args.device,
        "--num-workers", str(args.num_workers),
        "--seeds", args.seeds,
        "--output-root", str(output_root),
    ]
    for flag, value in {
        "--source-dir": args.source_dir,
        "--zip-path": args.zip_path,
        "--download-url": args.download_url,
        "--download-dst": args.download_dst,
    }.items():
        if value:
            benchmark.extend([flag, value])
    if args.allow_insecure_download:
        benchmark.append("--allow-insecure-download")
    if args.delete_checkpoints_after_eval:
        benchmark.append("--delete-checkpoints-after-eval")
    if args.save_predictions:
        benchmark.append("--save-predictions")
    if args.save_visualizations:
        benchmark.append("--save-visualizations")
    run(benchmark)

    run([
        py, str(PROJECT_ROOT / "scripts" / "aggregate_training_results.py"),
        "--output-root", str(output_root),
        "--output-path", str(training_summary),
        "--models", args.models,
        "--seeds", args.seeds,
        "--dataset", args.dataset,
        "--expected-batch-size", str(args.batch_size),
    ])
    run([
        py, str(PROJECT_ROOT / "scripts" / "validate_ablation_results.py"),
        "--summary-path", str(test_summary),
        "--training-summary-path", str(training_summary),
        "--output-root", str(output_root),
        "--models", args.models,
        "--seeds", args.seeds,
        "--dataset", args.dataset,
    ])
    run([
        py, str(PROJECT_ROOT / "scripts" / "report_dapr_ablation.py"),
        "--summary-path", str(test_summary),
        "--latex-path", str(latex_table),
        "--delta-path", str(delta_table),
    ])

    if args.delete_checkpoints_after_eval:
        remaining = [path for path in output_root.rglob("checkpoints") if path.is_dir()]
        if remaining:
            raise SystemExit("ERROR: checkpoint cleanup failed:\n" + "\n".join(str(path) for path in remaining))

    print("\nDAPR U-Net component ablation completed successfully.")
    print(f"Test summary:     {test_summary}")
    print(f"Training summary: {training_summary}")
    print(f"LaTeX table:      {latex_table}")
    print(f"Delta table:      {delta_table}")


if __name__ == "__main__":
    main()
