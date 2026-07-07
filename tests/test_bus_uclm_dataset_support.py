from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import prepare_kvasir_seg
from src.datasets import build_dataset, get_dataset_spec, infer_dataset_paths, load_binary_mask, normalize_dataset_name


def _write_bus_uclm_pair(root: Path, stem: str, mask_color: tuple[int, int, int]) -> None:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(25, 50, 75)).save(image_dir / f"{stem}.png")
    mask = Image.new("RGB", (8, 8), color=(0, 0, 0))
    for x in range(2, 6):
        for y in range(2, 6):
            mask.putpixel((x, y), mask_color)
    mask.save(mask_dir / f"{stem}.png")


def test_bus_uclm_registry_has_official_doi_source():
    assert normalize_dataset_name("BUS-UCLM") == "bus_uclm"
    spec = get_dataset_spec("breast-ultrasound-uclm")
    assert spec.name == "bus_uclm"
    assert spec.official_source_url == "https://data.mendeley.com/datasets/7fvgj4jsp7/3"
    assert spec.official_download_urls
    assert "10.17632/7fvgj4jsp7.3" in spec.description


def test_bus_uclm_image_mask_dirs_are_inferred_and_loaded(tmp_path: Path):
    root = tmp_path / "BUS-UCLM Breast ultrasound lesion segmentation dataset"
    _write_bus_uclm_pair(root, "case_red", (255, 0, 0))
    _write_bus_uclm_pair(root, "case_green", (0, 255, 0))

    paths = infer_dataset_paths(tmp_path, dataset_name="bus_uclm")
    assert paths.image_dir.name == "images"
    assert paths.mask_dir.name == "masks"

    dataset = build_dataset(name="bus-uclm", root=tmp_path, image_size=None, return_paths=True)
    assert len(dataset) == 2
    assert dataset.get_ids() == ["case_green", "case_red"]
    sample = dataset[0]
    assert sample["image"].size == (8, 8)
    assert set(np.asarray(sample["mask"]).reshape(-1).tolist()) == {0, 255}


def test_bus_uclm_rgb_masks_preserve_red_and_green_foreground(tmp_path: Path):
    red_mask = tmp_path / "red.png"
    green_mask = tmp_path / "green.png"
    black_mask = tmp_path / "black.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(red_mask)
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(green_mask)
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(black_mask)

    assert set(np.asarray(load_binary_mask(red_mask)).reshape(-1).tolist()) == {255}
    assert set(np.asarray(load_binary_mask(green_mask)).reshape(-1).tolist()) == {255}
    assert set(np.asarray(load_binary_mask(black_mask)).reshape(-1).tolist()) == {0}


def test_prepare_bus_uclm_from_source_dir_binarizes_color_masks(tmp_path: Path, monkeypatch):
    source = tmp_path / "source" / "BUS-UCLM"
    _write_bus_uclm_pair(source, "case_red", (255, 0, 0))
    _write_bus_uclm_pair(source, "case_green", (0, 255, 0))
    _write_bus_uclm_pair(source, "case_normal", (0, 0, 0))
    data_root = tmp_path / "data"

    monkeypatch.setattr(
        prepare_kvasir_seg,
        "parse_args",
        lambda: Namespace(
            dataset="bus_uclm",
            data_root=str(data_root),
            source_dir=str(source),
            zip_path=None,
            download_url=None,
            download_dst=None,
            image_size=16,
            skip_raw_copy=False,
            force=False,
            allow_insecure_download=False,
        ),
    )
    prepare_kvasir_seg.main()

    mask_dir = data_root / "processed" / "bus_uclm" / "masks_16"
    assert sorted(p.name for p in mask_dir.glob("*.png")) == [
        "case_green.png",
        "case_normal.png",
        "case_red.png",
    ]
    assert set(np.asarray(Image.open(mask_dir / "case_red.png")).reshape(-1).tolist()) == {0, 255}
    assert set(np.asarray(Image.open(mask_dir / "case_green.png")).reshape(-1).tolist()) == {0, 255}
    assert set(np.asarray(Image.open(mask_dir / "case_normal.png")).reshape(-1).tolist()) == {0}


def test_prepare_main_uses_bus_uclm_official_download(monkeypatch, tmp_path: Path):
    extracted = tmp_path / "official_extract" / "BUS-UCLM"
    _write_bus_uclm_pair(extracted, "case_red", (255, 0, 0))
    data_root = tmp_path / "data"

    called = {}

    def fake_download_official_archives(urls, downloads_dir, extract_dir, dataset_name, *, verify=True):
        called["urls"] = tuple(urls)
        called["dataset_name"] = dataset_name
        called["verify"] = verify
        return extracted

    monkeypatch.setattr(prepare_kvasir_seg, "_download_official_archives", fake_download_official_archives)
    monkeypatch.setattr(
        prepare_kvasir_seg,
        "parse_args",
        lambda: Namespace(
            dataset="bus_uclm",
            data_root=str(data_root),
            source_dir=None,
            zip_path=None,
            download_url=None,
            download_dst=None,
            image_size=16,
            skip_raw_copy=True,
            force=False,
            allow_insecure_download=False,
        ),
    )
    prepare_kvasir_seg.main()

    assert called["dataset_name"] == "bus_uclm"
    assert called["urls"] == get_dataset_spec("bus_uclm").official_download_urls
    assert (data_root / "processed" / "bus_uclm" / "images_16" / "case_red.png").is_file()
