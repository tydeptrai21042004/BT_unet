from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import subprocess
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import prepare_kvasir_seg
from src.datasets import DATASET_SPECS, get_dataset_spec


def test_removed_datasets_are_not_registered():
    assert "busi" not in DATASET_SPECS
    assert "drive" not in DATASET_SPECS
    assert "custom" not in DATASET_SPECS


def test_new_registry_uses_public_institutional_sources():
    instrument = get_dataset_spec("kvasir_instrument")
    hyper = get_dataset_spec("hyper_kvasir_seg")
    assert instrument.default_download_url == "https://datasets.simula.no/downloads/kvasir-instrument.zip"
    assert hyper.default_download_url.endswith("hyper-kvasir-segmented-images.zip")
    assert "58.6 GB" in hyper.description


def test_prepare_main_auto_downloads_kvasir_instrument(monkeypatch, tmp_path: Path):
    extracted = tmp_path / "Kvasir-Instrument"
    images = extracted / "images"
    masks = extracted / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(images / "tool.jpg")
    Image.new("L", (8, 8), color=255).save(masks / "tool.png")

    monkeypatch.setattr(prepare_kvasir_seg, "_maybe_download", lambda *a, **k: tmp_path / "instrument.zip")
    monkeypatch.setattr(prepare_kvasir_seg, "_extract_zip", lambda *a, **k: extracted)
    data_root = tmp_path / "data"
    monkeypatch.setattr(prepare_kvasir_seg, "parse_args", lambda: Namespace(
        dataset="kvasir_instrument", data_root=str(data_root), source_dir=None,
        zip_path=None, download_url=None, download_dst=None, image_size=32,
        skip_raw_copy=False, force=False, allow_insecure_download=False,
    ))
    prepare_kvasir_seg.main()
    assert (data_root / "processed/kvasir_instrument/images_32/tool.png").is_file()
    assert (data_root / "processed/kvasir_instrument/masks_32/tool.png").is_file()


