# ISBI 2012 dataset support

The retained challenge dataset is:

- `isbi2012`: neuronal-structure electron-microscopy segmentation benchmark.

ISBI 2012 is distributed as multipage TIFF stacks. The preparation script
expands the labeled training stacks into paired 2D PNG slices:

```text
train-volume.tif  -> slice_000.png ... slice_029.png
train-labels.tif  -> slice_000.png ... slice_029.png
```

The public challenge test stack has no public ground-truth labels. Therefore,
the supervised repository benchmark uses only the 30 labeled training slices
and creates a reproducible internal train/validation/test split.

## Prepare automatically

```bash
python scripts/prepare_dataset.py \
  --dataset isbi2012 \
  --data-root data \
  --image-size 352
```

## Prepare from an extracted directory

The directory must contain `train-volume.tif` and `train-labels.tif`.

```bash
python scripts/prepare_dataset.py \
  --dataset isbi2012 \
  --data-root data \
  --source-dir /path/to/ISBI-2012-challenge \
  --image-size 352
```

## Prepare from a ZIP

```bash
python scripts/prepare_dataset.py \
  --dataset isbi2012 \
  --data-root data \
  --zip-path /path/to/ISBI-2012-challenge.zip \
  --image-size 352
```

## Recommended contiguous split

Adjacent EM slices are highly correlated. A random slice split can leak nearly
identical neighboring structures between training and test sets. Use the
contiguous strategy:

```bash
python scripts/make_splits.py \
  --dataset isbi2012 \
  --data-root data \
  --image-size 352 \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --strategy contiguous \
  --seed 42
```

This produces 18 training, 6 validation, and 6 test slices.

## Run the two proposed methods

```bash
python scripts/benchmark_multi_seed.py \
  --models dapr_unet \
  --config-dir configs/ablation \
  --dataset isbi2012 \
  --data-root data \
  --image-size 352 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_dapr_ablation/isbi2012 \
  --skip-prepare \
  --skip-splits
```

The `--skip-prepare --skip-splits` flags preserve the manually created
contiguous split.

## Interpretation limitation

These results are an internal 2D slice-level benchmark and are not directly
comparable with the original challenge leaderboard, which used a hidden test
volume and topology-aware metrics.
