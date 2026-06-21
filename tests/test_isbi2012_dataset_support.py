from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

from src.datasets import DATASET_SPECS, get_dataset_spec, normalize_dataset_name
from src.datasets.factory import SUPPORTED_BINARY_SEG_DATASETS, build_dataset


def _write_stack(path: Path, values: list[int]) -> None:
    frames = [Image.new("L", (16, 16), color=value) for value in values]
    frames[0].save(path, save_all=True, append_images=frames[1:])


def test_only_isbi2012_is_retained_from_the_two_challenge_entries() -> None:
    assert "isbi2012" in DATASET_SPECS
    assert "isbi2012" in SUPPORTED_BINARY_SEG_DATASETS
    assert "isic2018" not in DATASET_SPECS
    assert "isic2018" not in SUPPORTED_BINARY_SEG_DATASETS
    assert normalize_dataset_name("ISBI-2012-challenge") == "isbi2012"
    assert get_dataset_spec("isbi2012").canonical_dir == "ISBI2012"
    try:
        normalize_dataset_name("isic2018")
    except ValueError:
        pass
    else:
        raise AssertionError("ISIC 2018 must no longer be registered")


def test_prepare_isbi2012_multipage_tiff_stacks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_stack(source / "train-volume.tif", [10, 20, 30, 40])
    _write_stack(source / "train-labels.tif", [0, 255, 0, 255])

    data_root = tmp_path / "data"
    command = [
        sys.executable,
        "scripts/prepare_dataset.py",
        "--dataset",
        "isbi2012",
        "--data-root",
        str(data_root),
        "--source-dir",
        str(source),
        "--image-size",
        "32",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Prepared 4 samples" in completed.stdout

    image_dir = data_root / "processed" / "isbi2012" / "images_32"
    mask_dir = data_root / "processed" / "isbi2012" / "masks_32"
    assert sorted(path.name for path in image_dir.glob("*.png")) == [
        "slice_000.png",
        "slice_001.png",
        "slice_002.png",
        "slice_003.png",
    ]
    assert sorted(path.name for path in mask_dir.glob("*.png")) == [
        "slice_000.png",
        "slice_001.png",
        "slice_002.png",
        "slice_003.png",
    ]

    mask_values = set(Image.open(mask_dir / "slice_001.png").get_flattened_data())
    assert mask_values <= {0, 255}

    dataset = build_dataset(
        "isbi2012",
        data_root,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=32,
    )
    assert len(dataset) == 4


def test_isbi2012_rejects_mismatched_stack_lengths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_stack(source / "train-volume.tif", [10, 20, 30])
    _write_stack(source / "train-labels.tif", [0, 255])

    command = [
        sys.executable,
        "scripts/prepare_dataset.py",
        "--dataset",
        "isbi2012",
        "--data-root",
        str(tmp_path / "data"),
        "--source-dir",
        str(source),
        "--image-size",
        "32",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "stack length mismatch" in (completed.stdout + completed.stderr)


def test_contiguous_split_preserves_slice_blocks() -> None:
    from scripts.make_splits import _split_ids

    ids = [f"slice_{index:03d}" for index in range(30)]
    train_ids, val_ids, test_ids = _split_ids(
        ids,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=42,
        strategy="contiguous",
    )
    assert train_ids == ids[:18]
    assert val_ids == ids[18:24]
    assert test_ids == ids[24:]
