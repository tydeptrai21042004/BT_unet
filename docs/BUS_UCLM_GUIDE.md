# BUS-UCLM dataset support

This repository supports **BUS-UCLM: Breast ultrasound lesion segmentation dataset** as a small DOI-backed medical segmentation dataset for efficiency experiments.

## Official source

- Dataset page: <https://data.mendeley.com/datasets/7fvgj4jsp7/3>
- DOI: `10.17632/7fvgj4jsp7.3`
- Official direct archive URL used by the repository:
  `https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7fvgj4jsp7-1.zip`

BUS-UCLM contains 683 ultrasound images from 38 patients with RGB lesion masks. Green pixels denote benign lesions, red pixels denote malignant lesions, and black pixels denote background or normal tissue. The repository converts BUS-UCLM masks to binary segmentation masks using the rule:

```text
any non-black RGB mask pixel = foreground
black RGB mask pixel = background
```

This is important because a red malignant mask has low grayscale luminance and can be lost by a simple `>127` grayscale threshold.

## Automatic preparation

```bash
python scripts/prepare_dataset.py \
  --dataset bus_uclm \
  --data-root data \
  --image-size 352 \
  --force
```

The prepared layout is:

```text
data/processed/bus_uclm/images_352/
data/processed/bus_uclm/masks_352/
data/processed/bus_uclm/metadata.csv
```

## Manual preparation from an already downloaded folder

```bash
python scripts/prepare_dataset.py \
  --dataset bus_uclm \
  --data-root data \
  --source-dir /path/to/BUS-UCLM \
  --image-size 352 \
  --force
```

The source folder may be the dataset root itself or a parent folder containing a BUS-UCLM folder. The resolver expects an `images/` directory and a `masks/` directory.

## Splits

```bash
python scripts/make_splits.py \
  --dataset bus_uclm \
  --data-root data \
  --image-size 352 \
  --seed 42
```

## Efficiency-table benchmark

For the manuscript efficiency table, run the same input size and batch size for all models:

```bash
python scripts/benchmark_multi_seed.py \
  --models unet,unetpp,resunetpp,dapr_unet \
  --config-dir configs/fair \
  --dataset bus_uclm \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_efficiency/bus_uclm
```

In the manuscript table, report `dapr_unet` as **BT U-Net** if that is the paper-facing method name.
