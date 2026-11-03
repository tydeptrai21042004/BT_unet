# Running the new reviewer experiment package

Run all commands from the repository root. First prepare genuine medical
segmentation datasets and dataset-specific `splits/<dataset>/train.txt`,
`val.txt`, and `test.txt` manifests. Check images and masks, patient/video
grouping, and ensure that training has no access to target test sets.

```bash
pip install -r requirements.txt -r requirements/revision.txt
python -m pytest -q tests/test_final_revision_package.py tests/test_revision_extensions.py
```

## 1. Inspect the proposed run matrix (default: does NOT train)

```bash
python scripts/revision/run_full_revision.py \
  --groups factorial,residual,convergence \
  --datasets kvasir_seg --seeds 42,1337,2024 \
  --epochs 100 --image-size 352 --data-root data \
  --output-root outputs_revision/final
```

The default plan is **57 jobs** (8 factorial + 7 residual + 4
convergence, times 3 seeds). This is computationally substantial. Start with
one seed and one epoch, then verify actual data and GPU availability.

## 2. Run the experiments

```bash
python scripts/revision/run_full_revision.py \
  --groups factorial,residual,convergence \
  --datasets kvasir_seg --seeds 42,1337,2024 \
  --epochs 100 --image-size 352 --data-root data \
  --output-root outputs_revision/final --device auto \
  --num-workers 2 --save-predictions --run --resume
```

`--resume` skips training only when its best checkpoint is present and skips
evaluation only when its CSV and summary are present. Keep saved checkpoints
for downstream frequency and robustness analyses.

## 3. Compile results into LaTeX directly from real model outputs

```bash
python scripts/revision/build_manuscript_tables.py \
  --plan outputs_revision/final/revision_plan.json \
  --output-dir outputs_revision/final/tables
```

Incomplete runs raise an error; `--allow-incomplete` exports explicitly
partial aggregates. The three-seed sample SD is not a statistical guarantee.

## 4. Plot multi-seed convergence

```bash
python scripts/revision/plot_convergence_multiseed.py \
  --runs \
  U-Net=PATH/seed_42/convergence_unet/results/history.csv \
  U-Net=PATH/seed_1337/convergence_unet/results/history.csv \
  BT=PATH/seed_42/convergence_dapr_unet/results/history.csv \
  BT=PATH/seed_1337/convergence_dapr_unet/results/history.csv \
  --metric val/dice --output outputs_revision/final/convergence.png
```

Use actual job paths from `revision_plan.json` rather than guessing names.

## 5. Analyze actual bottleneck activations

```bash
python scripts/revision/analyze_spectrum.py \
  --checkpoint PATH/TO/best.pt --output-dir outputs_revision/spectral_weights

python scripts/revision/analyze_feature_spectrum.py \
  --checkpoint PATH/TO/best.pt --dataset kvasir_seg --data-root data \
  --max-images 100 --output-dir outputs_revision/spectral_activations
```

## 6. Object-size and reproducible case selection

```bash
python scripts/revision/analyze_object_size.py \
  --runs U-Net=PATH/UNET/per_image.csv BT=PATH/BT/per_image.csv \
  --output outputs_revision/size_metrics.csv
python scripts/revision/select_qualitative_cases.py \
  --reference PATH/UNET/per_image.csv --max-cases 8 \
  --output outputs_revision/case_selection.json
```

Use the emitted IDs for `plot_qualitative.py`; curate glare, flat lesions,
and blur manually using a recorded annotation protocol where metadata is not
available. Do not claim automatic medical annotation of image artifacts.

## 7. Original GFUNet / AFDSeg sources (STRICT, incomplete until verified)

In a networked environment obtain an **actual upstream commit SHA**, then:

```bash
python scripts/revision/install_official_source.py \
  --name gfunet --commit EXACT_UPSTREAM_SHA --destination third_party/GFUNet
python scripts/revision/install_official_source.py \
  --name afdseg --commit EXACT_UPSTREAM_SHA --destination third_party/AFDSeg
```

Inspect each pinned checkout and fill a separate audit JSON from
`configs/controlled_spectral/AUDIT_TEMPLATE.json`, with the specific paper
components mapped to existing original source files (`path/to/module.py::Symbol`).
Do not set `status=verified` until the architecture and dependencies have been
validated and the official training protocol documented.

Only after that, edit and rename the `*_OFFICIAL_TEMPLATE.yaml` files with the
actual entrypoint, constructor kwargs, output contract, and faithful training
settings. If using a controlled loss/augmentation/pretraining regime instead of
the official protocol, disclose these deviations; never call it a faithful
reproduction of the paper's training.

Then `--groups official --external-configs path/to/gfunet.yaml,path/to/afdseg.yaml`
can plan and run strictly verified wrappers. A source Git clone without this
audit is deliberately NOT enough.

## Important

No real datasets, official third-party weights, or medical benchmark results
are included in this ZIP. Download and verify them separately.
