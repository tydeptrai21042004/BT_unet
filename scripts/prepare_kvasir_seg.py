#!/usr/bin/env python3
"""Prepare supported binary-segmentation datasets for the Fourier U-Net benchmark.

Supported input modes:
1) Existing extracted dataset folder with images/ and masks/
2) Zip archive containing a supported dataset
3) Optional direct download URL (including Google Drive links via gdown if installed)
4) Automatic default download when a direct URL is configured in the dataset registry
5) Automatic download from official dataset archives configured in the registry

Outputs a benchmark-friendly layout:
    data/
      raw/<dataset>/images
      raw/<dataset>/masks
      processed/images_<size>
      processed/masks_<size>
      processed/metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import zipfile
from html.parser import HTMLParser
from urllib.parse import urljoin
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

from PIL import Image, ImageChops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import get_dataset_spec, normalize_dataset_name
from src.datasets.kvasir_seg_dataset import (
    KvasirPaths,
    _dir_name_variants,
    _resolve_image_mask_dirs,
    canonical_sample_id,
    looks_like_mask_stem,
)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def _has_image_mask_dirs(path: Path) -> bool:
    return _resolve_image_mask_dirs(path) is not None



def _find_dataset_root(root: Path, dataset_name: str) -> Optional[Path]:
    normalized = normalize_dataset_name(dataset_name)
    if _has_image_mask_dirs(root):
        return root

    for token in _dir_name_variants(normalized):
        for prefix in (
            root,
            root / "raw",
            root / "dataset",
            root / "datasets",
            root / "TrainDataset",
            root / "TestDataset",
            root / "train",
            root / "test",
        ):
            candidate = prefix / token
            if _has_image_mask_dirs(candidate):
                return candidate

    keywords = {token.lower() for token in _dir_name_variants(normalized)}
    for cand in root.rglob("*"):
        if not cand.is_dir() or not _has_image_mask_dirs(cand):
            continue
        path_text = cand.as_posix().lower()
        if any(keyword in path_text for keyword in keywords):
            return cand
    return None



def _is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS



def _iter_images(directory: Path) -> List[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and _is_image(p))


def _collect_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[str, Path, Path]]:
    image_map = {}
    mask_map = {}
    for image_path in _iter_images(image_dir):
        key = canonical_sample_id(image_path.stem)
        # When image_dir == mask_dir, skip mask-like files from the image map.
        if image_dir == mask_dir and looks_like_mask_stem(image_path.stem):
            continue
        image_map.setdefault(key, image_path)
    for mask_path in _iter_images(mask_dir):
        key = canonical_sample_id(mask_path.stem)
        # In separate mask folders, exact-stem masks are valid; in shared BUSI-like
        # folders, require a mask-like suffix so original images are not treated as masks.
        if image_dir == mask_dir and not looks_like_mask_stem(mask_path.stem):
            continue
        mask_map.setdefault(key, mask_path)

    pairs: List[Tuple[str, Path, Path]] = []
    missing: List[str] = []
    for sample_id, image_path in sorted(image_map.items()):
        mask_path = mask_map.get(sample_id)
        if mask_path is None:
            missing.append(sample_id)
            continue
        pairs.append((sample_id, image_path, mask_path))
    if not pairs:
        raise RuntimeError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
    if missing:
        print(f"[WARN] Missing masks for {len(missing)} images. They will be skipped.", file=sys.stderr)
    return pairs



def _copy_raw_pairs(pairs: Sequence[Tuple[str, Path, Path]], raw_images: Path, raw_masks: Path) -> None:
    raw_images.mkdir(parents=True, exist_ok=True)
    raw_masks.mkdir(parents=True, exist_ok=True)
    for sample_id, image_path, mask_path in pairs:
        shutil.copy2(image_path, raw_images / f"{sample_id}{image_path.suffix.lower()}")
        shutil.copy2(mask_path, raw_masks / f"{sample_id}{mask_path.suffix.lower()}")



def _resize_pair(image_path: Path, mask_path: Path, out_image: Path, out_mask: Path, size: int) -> Tuple[int, int]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    orig_h, orig_w = image.height, image.width

    image_resized = image.resize((size, size), resample=Image.BILINEAR)
    mask_resized = mask.resize((size, size), resample=Image.NEAREST)
    mask_binary = mask_resized.point(lambda x: 255 if x > 127 else 0)

    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    image_resized.save(out_image)
    mask_binary.save(out_mask)
    return orig_h, orig_w



def _write_metadata(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def _find_isbi2012_stacks(root: Path) -> Optional[Tuple[Path, Path]]:
    """Locate the labeled ISBI 2012 training image and mask stacks."""
    image_names = {"train-volume.tif", "train-volume.tiff", "train_volume.tif", "train_volume.tiff"}
    mask_names = {"train-labels.tif", "train-labels.tiff", "train_labels.tif", "train_labels.tiff"}
    image_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in image_names and image_path is None:
            image_path = path
        elif name in mask_names and mask_path is None:
            mask_path = path
    if image_path is not None and mask_path is not None:
        return image_path, mask_path
    return None


def _expand_isbi2012_stacks(
    image_stack_path: Path,
    mask_stack_path: Path,
    output_root: Path,
) -> KvasirPaths:
    """Expand paired multipage TIFF stacks into deterministic PNG slice pairs."""
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    if output_root.exists():
        shutil.rmtree(output_root)
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_stack_path) as image_stack, Image.open(mask_stack_path) as mask_stack:
        image_frames = int(getattr(image_stack, "n_frames", 1))
        mask_frames = int(getattr(mask_stack, "n_frames", 1))
        if image_frames != mask_frames:
            raise RuntimeError(
                "ISBI 2012 image/mask stack length mismatch: "
                f"images={image_frames}, masks={mask_frames}"
            )
        if image_frames <= 0:
            raise RuntimeError("ISBI 2012 stacks contain no slices.")

        for index in range(image_frames):
            image_stack.seek(index)
            mask_stack.seek(index)
            image = image_stack.convert("L")
            mask = mask_stack.convert("L").point(lambda value: 255 if value > 127 else 0)
            sample_id = f"slice_{index:03d}"
            image.save(image_dir / f"{sample_id}.png")
            mask.save(mask_dir / f"{sample_id}.png")

    return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)


def _extract_zip(zip_path: Path, extract_dir: Path, dataset_name: str) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    dataset_root = _find_dataset_root(extract_dir, dataset_name)
    if dataset_root is None and normalize_dataset_name(dataset_name) == "isbi2012":
        if _find_isbi2012_stacks(extract_dir) is not None:
            return extract_dir
    if dataset_root is None:
        spec = get_dataset_spec(dataset_name)
        raise FileNotFoundError(
            f"Could not find extracted dataset={spec.name} under {extract_dir}. Expected images/ and masks/ folders."
        )
    return dataset_root



def _extract_archive_contents(zip_path: Path, extract_dir: Path) -> None:
    """Extract one ZIP archive into a shared directory."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)


def _download_filename(url: str, index: int) -> str:
    name = Path(unquote(urlparse(url).path)).name
    return name or f"official_archive_{index:02d}.zip"


def _download_official_archives(
    urls: Sequence[str],
    downloads_dir: Path,
    extract_dir: Path,
    dataset_name: str,
    *,
    verify: bool = True,
) -> Path:
    """Download and merge all configured official archives for a dataset.

    Every archive is extracted into one shared directory before the generic
    image/mask layout resolver is applied.
    """
    if not urls:
        raise ValueError(f"No official archive URLs configured for dataset={dataset_name}")

    downloads_dir.mkdir(parents=True, exist_ok=True)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(urls):
        archive_path = downloads_dir / _download_filename(url, index)
        if archive_path.exists() and zipfile.is_zipfile(archive_path):
            print(f"Using cached official archive: {archive_path}")
        else:
            archive_path.unlink(missing_ok=True)
            print(f"Downloading official archive {index + 1}/{len(urls)}: {url}")
            _maybe_download(url, archive_path, verify=verify)
        _extract_archive_contents(archive_path, extract_dir)

    dataset_root = _find_dataset_root(extract_dir, dataset_name)
    if dataset_root is None:
        raise FileNotFoundError(
            f"Official archives were downloaded and extracted under {extract_dir}, "
            f"but no compatible image/mask layout was found for dataset={dataset_name}."
        )
    return dataset_root


def _maybe_download(url: str, dst: Path, *, verify: bool = True) -> Path:
    if "drive.google.com" in url:
        try:
            import gdown
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Google Drive download requested but gdown is not installed. Install gdown or pass --zip-path/--source-dir."
            ) from exc
        dst.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(url=url, output=str(dst), quiet=False, fuzzy=True)
        if not result:
            raise RuntimeError(f"Failed to download Google Drive URL: {url}")
        return dst

    try:
        import requests
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("requests is required for download mode. Install it or pass --zip-path/--source-dir.") from exc

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_suffix(dst.suffix + ".part")
    temporary.unlink(missing_ok=True)
    headers = {"User-Agent": "DT-unet-dataset-downloader/1.0"}
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(30, 300),
                verify=verify,
                allow_redirects=True,
                headers=headers,
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            temporary.replace(dst)
            return dst
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                break
            print(f"[WARN] Download attempt {attempt}/3 failed for {url}: {exc}", file=sys.stderr)
    raise RuntimeError(f"Failed to download after 3 attempts: {url}") from last_error




def _warn_hyper_kvasir_overlap(data_root: Path, hyper_pairs: Sequence[Tuple[str, Path, Path]]) -> None:
    """Warn about filename overlap; never merge Kvasir collections implicitly."""
    candidates = (
        data_root / "raw" / get_dataset_spec("kvasir_seg").canonical_dir,
        data_root / "processed" / "kvasir_seg",
    )
    kvasir_root = next((path for path in candidates if path.is_dir() and _has_image_mask_dirs(path)), None)
    if kvasir_root is None:
        return
    resolved = _resolve_image_mask_dirs(kvasir_root)
    if resolved is None:
        return
    existing_ids = {canonical_sample_id(path.stem) for path in _iter_images(resolved.image_dir)}
    overlap = sorted(existing_ids.intersection(sample_id for sample_id, _, _ in hyper_pairs))
    if overlap:
        report = data_root / "processed" / "hyper_kvasir_seg" / "overlap_with_kvasir_seg.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(overlap) + "\n", encoding="utf-8")
        print(f"[WARN] HyperKvasir has {len(overlap)} filename IDs overlapping Kvasir-SEG. They remain separate; report: {report}", file=sys.stderr)

def prepared_dataset_exists(data_root: Path, image_size: int, dataset_name: str = "kvasir_seg") -> bool:
    dataset_name = normalize_dataset_name(dataset_name)
    legacy = (data_root / "processed" / f"images_{image_size}").is_dir() and (data_root / "processed" / f"masks_{image_size}").is_dir()
    dataset_specific = (
        (data_root / "processed" / dataset_name / f"images_{image_size}").is_dir()
        and (data_root / "processed" / dataset_name / f"masks_{image_size}").is_dir()
    )
    return bool(dataset_specific or legacy)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a supported binary segmentation dataset for benchmark training.")
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help="Registered dataset key. Public datasets are downloaded from their configured institutional sources when needed.")
    parser.add_argument("--data-root", type=str, default="data", help="Benchmark data root.")
    parser.add_argument("--source-dir", type=str, default=None, help="Path to an extracted dataset folder or its parent.")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to a dataset zip archive.")
    parser.add_argument("--download-url", type=str, default=None, help="Optional URL to download a zip archive.")
    parser.add_argument("--download-dst", type=str, default=None, help="Optional destination path for the downloaded zip.")
    parser.add_argument("--image-size", type=int, default=352, help="Output square size for processed images/masks.")
    parser.add_argument("--skip-raw-copy", action="store_true", help="Do not copy files into data/raw/<dataset>.")
    parser.add_argument("--force", action="store_true", help="Rebuild processed outputs even if they already exist.")
    parser.add_argument("--allow-insecure-download", action="store_true", help="Disable TLS certificate verification for dataset download.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    spec = get_dataset_spec(dataset_name)
    data_root = Path(args.data_root)
    raw_root = data_root / "raw" / spec.canonical_dir
    processed_root = data_root / "processed" / dataset_name
    processed_images = processed_root / f"images_{args.image_size}"
    processed_masks = processed_root / f"masks_{args.image_size}"

    if prepared_dataset_exists(data_root, args.image_size, dataset_name=dataset_name) and not args.force:
        print(f"Dataset already prepared for image_size={args.image_size} at: {processed_root}")
        print(f"Processed images: {processed_images}")
        print(f"Processed masks : {processed_masks}")
        return

    dataset_root: Optional[Path] = None

    if args.source_dir:
        source_root = Path(args.source_dir)
        dataset_root = _find_dataset_root(source_root, dataset_name)
        if dataset_root is None and dataset_name == "isbi2012":
            if _find_isbi2012_stacks(source_root) is not None:
                dataset_root = source_root
        if dataset_root is None:
            raise FileNotFoundError(f"Could not locate compatible dataset data under source-dir: {args.source_dir}")
    elif args.zip_path:
        dataset_root = _extract_zip(Path(args.zip_path), data_root / "_tmp_extract", dataset_name)
    else:
        explicit_download_url = args.download_url
        existing_root = _find_dataset_root(data_root, dataset_name)
        if existing_root is None and dataset_name == "isbi2012":
            if _find_isbi2012_stacks(data_root) is not None:
                existing_root = data_root
        allow_insecure = bool(
            args.allow_insecure_download
            or os.environ.get("ALLOW_INSECURE_DOWNLOAD", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
        )

        if existing_root is not None:
            dataset_root = existing_root
        elif explicit_download_url:
            download_dst = Path(args.download_dst) if args.download_dst else data_root / "downloads" / f"{dataset_name}.zip"
            print(f"Downloading {dataset_name} from explicit URL: {explicit_download_url}")
            zip_path = _maybe_download(explicit_download_url, download_dst, verify=not allow_insecure)
            dataset_root = _extract_zip(zip_path, data_root / "_tmp_extract" / dataset_name, dataset_name)
        elif spec.official_download_urls:
            print(f"Downloading {dataset_name} from its official dataset source.")
            if spec.official_source_url:
                print(f"Official source page: {spec.official_source_url}")
            dataset_root = _download_official_archives(
                spec.official_download_urls,
                data_root / "downloads" / dataset_name,
                data_root / "_tmp_official_extract" / dataset_name,
                dataset_name,
                verify=not allow_insecure,
            )
        elif spec.default_download_url:
            download_dst = Path(args.download_dst) if args.download_dst else data_root / "downloads" / f"{dataset_name}.zip"
            print(f"Downloading {dataset_name} from registry URL: {spec.default_download_url}")
            zip_path = _maybe_download(spec.default_download_url, download_dst, verify=not allow_insecure)
            dataset_root = _extract_zip(zip_path, data_root / "_tmp_extract" / dataset_name, dataset_name)
        else:
            raise ValueError(
                f"No automatic official download source is configured for dataset={dataset_name}. "
                "Use --source-dir, --zip-path, or --download-url."
            )

    if dataset_name == "isbi2012":
        stacks = _find_isbi2012_stacks(dataset_root)
        if stacks is not None:
            resolved_dirs = _expand_isbi2012_stacks(
                stacks[0],
                stacks[1],
                data_root / "_tmp_isbi2012_slices",
            )
        else:
            resolved_dirs = _resolve_image_mask_dirs(dataset_root)
    else:
        resolved_dirs = _resolve_image_mask_dirs(dataset_root)

    if resolved_dirs is None:
        raise FileNotFoundError(f"Could not resolve compatible image/mask data inside dataset root: {dataset_root}")
    image_dir = resolved_dirs.image_dir
    mask_dir = resolved_dirs.mask_dir
    pairs = _collect_pairs(image_dir, mask_dir)
    if dataset_name == "hyper_kvasir_seg":
        _warn_hyper_kvasir_overlap(data_root, pairs)

    if not args.skip_raw_copy:
        _copy_raw_pairs(pairs, raw_root / "images", raw_root / "masks")

    metadata_rows: List[dict] = []
    for sample_id, image_path, mask_path in pairs:
        out_image = processed_images / f"{sample_id}.png"
        out_mask = processed_masks / f"{sample_id}.png"
        orig_h, orig_w = _resize_pair(image_path, mask_path, out_image, out_mask, args.image_size)
        metadata_rows.append(
            {
                "id": sample_id,
                "dataset": dataset_name,
                "image_path": str(out_image.as_posix()),
                "mask_path": str(out_mask.as_posix()),
                "orig_height": orig_h,
                "orig_width": orig_w,
                "proc_height": args.image_size,
                "proc_width": args.image_size,
            }
        )

    _write_metadata(metadata_rows, processed_root / "metadata.csv")
    print(f"Prepared {len(metadata_rows)} samples for dataset={dataset_name} at: {processed_root}")
    print(f"Processed images: {processed_images}")
    print(f"Processed masks : {processed_masks}")


if __name__ == "__main__":
    main()
