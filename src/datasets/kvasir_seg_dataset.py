"""Binary segmentation dataset helpers for Kvasir-SEG and compatible datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .registry import get_dataset_spec, normalize_dataset_name
from .transforms import build_eval_transforms, build_train_transforms

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")
Sample = MutableMapping[str, object]


@dataclass(frozen=True)
class KvasirPaths:
    image_dir: Path
    mask_dir: Path


COMMON_DATASET_DIR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "kvasir_seg": ("Kvasir-SEG", "kvasir-seg", "kvasir_seg", "Kvasir", "kvasir"),
    "cvc_clinicdb": ("CVC-ClinicDB", "CVC612", "cvc-clinicdb", "clinicdb", "ClinicDB", "cvc612"),
    "etis": ("ETIS-LaribPolypDB", "ETIS", "etis-larib", "etis_larib", "etis-laribpolypdb"),
    "cvc_colondb": ("CVC-ColonDB", "cvc-colondb", "ColonDB", "colondb"),
    "cvc_300": ("CVC-300", "cvc-300", "CVC300", "cvc300", "EndoScene", "CVC-T"),
    "isbi2012": ("ISBI2012", "ISBI-2012", "ISBI_2012", "ISBI-2012-challenge"),
    "kvasir_instrument": ("Kvasir-Instrument", "kvasir-instrument", "kvasir_instrument"),
    "hyper_kvasir_seg": ("HyperKvasir-SEG", "hyper-kvasir-segmented-images", "segmented-images", "segmented_images"),
}


IMAGE_DIR_NAMES = {
    "image",
    "images",
    "imgs",
    "img",
    "original",
    "originals",
    "frame",
    "frames",
    "jpegimages",
    "bbdd",
    "training images",
    "test images",
    "training",
    "test",
    "cxr_png",
}
MASK_DIR_NAMES = {
    "mask",
    "masks",
    "gt",
    "groundtruth",
    "groundtruths",
    "ground_truth",
    "ground_truths",
    "ground truth",
    "ground truths",
    "annotation",
    "annotations",
    "label",
    "labels",
    "segmentationclass",
    "segmentationclasses",
    "manual",
    "manual1",
    "1st_manual",
    "training masks",
    "test masks",
    "manualmask",
    "leftmask",
    "rightmask",
}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def _canonical_dir_key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_sample_id(value: str | Path) -> str:
    """Return a robust pairing key for common binary-segmentation datasets.

    Kvasir/CVC masks usually share the exact image stem. Cross-domain datasets
    may add suffixes such as ``_segmentation``. Removing common annotation suffixes allows one generic image/mask loader to support those layouts.
    """
    stem = Path(value).stem.lower().strip()
    stem = stem.replace(" ", "_")
    stem = re.sub(r"\([0-9]+\)", lambda m: f"_{m.group(0)[1:-1]}", stem)
    suffix_patterns = [
        r"_segmentation$",
        r"_seg$",
        r"_mask(_\d+)?$",
        r"_manual\d*$",
        r"_manual$",
        r"_gt$",
        r"_groundtruth$",
        r"_ground_truth$",
        r"_truth$",
        r"_label$",
        r"_labels$",
        r"_annotation$",
        r"_annotations$",
        r"_1st_manual$",
        r"_training$",
        r"_test$",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in suffix_patterns:
            new_stem = re.sub(pattern, "", stem)
            if new_stem != stem:
                stem = new_stem
                changed = True
    return stem




def looks_like_mask_stem(value: str | Path) -> bool:
    stem = Path(value).stem.lower().replace(" ", "_")
    return bool(re.search(r"(_segmentation$|_seg$|_mask(_\d+)?$|_manual\d*$|_gt$|_groundtruth$|_ground_truth$|_label$|_annotation$|_1st_manual$)", stem))


def _resolve_existing_dir(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def _resolve_processed_pair(root: Path, dataset_name: str = "kvasir_seg", image_size: Optional[int] = None) -> Optional[KvasirPaths]:
    processed_root = root / "processed"
    if not processed_root.is_dir():
        return None

    normalized = normalize_dataset_name(dataset_name)
    candidate_roots = [processed_root / normalized, processed_root / get_dataset_spec(normalized).canonical_dir, processed_root]

    for base in candidate_roots:
        if not base.is_dir():
            continue
        if image_size is not None:
            image_dir = base / f"images_{image_size}"
            mask_dir = base / f"masks_{image_size}"
            if image_dir.is_dir() and mask_dir.is_dir():
                return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)
        image_dir = base / "images"
        mask_dir = base / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)

        suffixes: list[str] = []
        for path in base.iterdir():
            if path.is_dir() and path.name.startswith("images_"):
                suffixes.append(path.name[len("images_"):])
        for suffix in sorted(set(suffixes)):
            image_dir = base / f"images_{suffix}"
            mask_dir = base / f"masks_{suffix}"
            if image_dir.is_dir() and mask_dir.is_dir():
                return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)
    return None


def _dir_name_variants(dataset_name: str) -> Tuple[str, ...]:
    normalized = normalize_dataset_name(dataset_name)
    spec = get_dataset_spec(normalized)
    values: List[str] = []
    for token in (spec.canonical_dir, spec.name, *spec.aliases, *COMMON_DATASET_DIR_ALIASES.get(normalized, tuple())):
        token = str(token).strip()
        if not token:
            continue
        values.extend(
            [
                token,
                token.replace("_", "-"),
                token.replace("-", "_"),
                token.lower(),
                token.upper(),
                token.title(),
            ]
        )
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return tuple(deduped)


def _resolve_image_mask_dirs(path: Path) -> Optional[KvasirPaths]:
    """Return compatible image/mask directories under ``path``.

    Public segmentation archives are inconsistent. This resolver accepts common
    Kvasir/PraNet, ISIC, Kvasir-Instrument, HyperKvasir, and Montgomery directory names without changing the
    dataset API.
    """
    if not path.is_dir():
        return None

    children = [child for child in path.iterdir() if child.is_dir()]
    by_key = {_canonical_dir_key(child.name): child for child in children}

    image_dir: Optional[Path] = None
    mask_dir: Optional[Path] = None
    for key in IMAGE_DIR_NAMES:
        image_dir = by_key.get(_canonical_dir_key(key))
        if image_dir is not None:
            break
    for key in MASK_DIR_NAMES:
        mask_dir = by_key.get(_canonical_dir_key(key))
        if mask_dir is not None:
            break

    if image_dir is not None and mask_dir is not None:
        return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)

    # Some archives place class subfolders directly below the root, e.g.
    # BUSI/{benign,malignant,normal}. Treat the same directory as both image and
    # mask root and let canonical_sample_id pair image/mask stems recursively.
    recursive_files = [p for p in path.rglob("*") if p.is_file() and _is_image_file(p)]
    if recursive_files:
        has_mask_like = any(looks_like_mask_stem(p.stem) for p in recursive_files)
        has_image_like = any(not looks_like_mask_stem(p.stem) for p in recursive_files)
        if has_mask_like and has_image_like:
            return KvasirPaths(image_dir=path, mask_dir=path)
    return None


def _has_image_mask_dirs(path: Path) -> bool:
    return _resolve_image_mask_dirs(path) is not None


def _iter_dataset_root_candidates(root: Path, dataset_name: str) -> Iterable[Path]:
    if _has_image_mask_dirs(root):
        yield root

    variants = _dir_name_variants(dataset_name)
    for name in variants:
        for prefix in (
            root,
            root / "raw",
            root / "datasets",
            root / "dataset",
            root / "TestDataset",
            root / "TrainDataset",
            root / "test",
            root / "train",
        ):
            yield prefix / name

    for candidate in (
        root / "images",
        root / "masks",
    ):
        if candidate.is_dir() and _has_image_mask_dirs(root):
            yield root
            break


def _find_dataset_root(root: Path, dataset_name: str) -> Optional[Path]:
    normalized = normalize_dataset_name(dataset_name)
    for candidate in _iter_dataset_root_candidates(root, normalized):
        if _has_image_mask_dirs(candidate):
            return candidate

    keywords = {token.lower() for token in _dir_name_variants(normalized)}
    for candidate in root.rglob("*"):
        if not candidate.is_dir() or not _has_image_mask_dirs(candidate):
            continue
        path_text = candidate.as_posix().lower()
        if any(keyword in path_text for keyword in keywords):
            return candidate
    return None


def infer_dataset_paths(root: str | Path, dataset_name: str = "kvasir_seg", image_size: Optional[int] = None) -> KvasirPaths:
    """Infer image and mask directories from a benchmark-style root.

    Supported layouts include dataset-specific processed folders, legacy flat
    processed folders, direct dataset roots, and common public archive layouts.
    """
    normalized = normalize_dataset_name(dataset_name)
    root = Path(root)

    processed_pair = _resolve_processed_pair(root, dataset_name=normalized, image_size=image_size)
    if processed_pair is not None:
        return processed_pair

    dataset_root = _find_dataset_root(root, normalized)
    if dataset_root is None:
        spec = get_dataset_spec(normalized)
        expected = (
            "processed/<dataset>/images_<size> + processed/<dataset>/masks_<size>, "
            "legacy processed/images_<size> + processed/masks_<size>, "
            f"raw/{spec.canonical_dir}/images + raw/{spec.canonical_dir}/masks, "
            f"or a dataset folder named like {spec.canonical_dir} containing images/ and masks/."
        )
        raise FileNotFoundError(
            f"Could not infer image/mask directories for dataset={normalized!r} from root={root}. Expected {expected}"
        )

    resolved = _resolve_image_mask_dirs(dataset_root)
    if resolved is None:  # defensive: _find_dataset_root only returns compatible roots
        raise FileNotFoundError(f"Could not resolve image/mask directories inside {dataset_root}")
    return resolved


def infer_kvasir_paths(root: str | Path, image_size: Optional[int] = None) -> KvasirPaths:
    return infer_dataset_paths(root=root, dataset_name="kvasir_seg", image_size=image_size)


class KvasirSegDataset(Dataset):
    """Binary segmentation dataset for Kvasir-SEG or compatible image/mask layouts."""

    def __init__(
        self,
        root: str | Path,
        split: Optional[str] = None,
        split_file: Optional[str | Path] = None,
        image_dir: Optional[str | Path] = None,
        mask_dir: Optional[str | Path] = None,
        image_size: Optional[int] = None,
        transform: Optional[Callable[[Sample], Sample]] = None,
        return_paths: bool = False,
        strict_pairing: bool = True,
        dataset_name: str = "kvasir_seg",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.return_paths = return_paths
        self.strict_pairing = strict_pairing
        self.transform = transform
        self.image_size = image_size
        self.dataset_name = normalize_dataset_name(dataset_name)

        if image_dir is None or mask_dir is None:
            inferred = infer_dataset_paths(self.root, dataset_name=self.dataset_name, image_size=image_size)
            self.image_dir = inferred.image_dir if image_dir is None else Path(image_dir)
            self.mask_dir = inferred.mask_dir if mask_dir is None else Path(mask_dir)
        else:
            self.image_dir = Path(image_dir)
            self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        if split_file is None and split is not None:
            candidates = [
                self.root / "splits" / self.dataset_name / f"{split}.txt",
                self.root / "splits" / f"{self.dataset_name}_{split}.txt",
                self.root / "splits" / f"{split}.txt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    split_file = candidate
                    break

        self.samples = self._build_samples(split_file=split_file)
        if not self.samples:
            raise RuntimeError(
                "No valid image-mask pairs found for dataset. "
                f"dataset={self.dataset_name}, image_dir={self.image_dir}, mask_dir={self.mask_dir}, split_file={split_file}"
            )

    def _load_split_ids(self, split_file: Optional[str | Path]) -> Optional[List[str]]:
        if split_file is None:
            return None

        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        ids: List[str] = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                ids.append(canonical_sample_id(item))
        return ids

    def _iter_files(self, directory: Path) -> List[Path]:
        return sorted(p for p in directory.rglob("*") if p.is_file() and _is_image_file(p))

    def _build_file_map(self, directory: Path, *, want_masks: bool) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        for path in self._iter_files(directory):
            key = canonical_sample_id(path.stem)
            is_mask_like = looks_like_mask_stem(path.stem)
            if want_masks and not is_mask_like and directory == self.image_dir:
                continue
            if not want_masks and is_mask_like and directory == self.mask_dir:
                continue
            # Prefer the first deterministic match. This avoids BUSI duplicate
            # _mask_1 variants overriding the canonical _mask file.
            result.setdefault(key, path)
        return result

    def _find_image_by_stem(self, directory: Path, stem: str, *, want_masks: bool = False) -> Optional[Path]:
        for ext in VALID_IMAGE_EXTENSIONS:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        key = canonical_sample_id(stem)
        return self._build_file_map(directory, want_masks=want_masks).get(key)

    def _build_samples(self, split_file: Optional[str | Path]) -> List[Tuple[str, Path, Path]]:
        split_ids = self._load_split_ids(split_file)
        samples: List[Tuple[str, Path, Path]] = []
        image_map = self._build_file_map(self.image_dir, want_masks=False)
        mask_map = self._build_file_map(self.mask_dir, want_masks=True)

        if split_ids is not None:
            for sample_id in split_ids:
                key = canonical_sample_id(sample_id)
                image_path = image_map.get(key) or self._find_image_by_stem(self.image_dir, sample_id, want_masks=False)
                mask_path = mask_map.get(key) or self._find_image_by_stem(self.mask_dir, sample_id, want_masks=True)

                if image_path is None or mask_path is None:
                    if self.strict_pairing:
                        raise FileNotFoundError(
                            f"Missing image or mask for sample '{sample_id}'. image_dir={self.image_dir}, mask_dir={self.mask_dir}"
                        )
                    continue
                samples.append((key, image_path, mask_path))
            return samples

        for key, image_path in sorted(image_map.items()):
            mask_path = mask_map.get(key)
            if mask_path is None:
                if self.strict_pairing:
                    raise FileNotFoundError(
                        f"No corresponding mask found for image '{image_path.name}' in {self.mask_dir}"
                    )
                continue
            samples.append((key, image_path, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id, image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        sample: Dict[str, object] = {
            "id": sample_id,
            "image": image,
            "mask": mask,
            "orig_size": (image.height, image.width),
        }

        if self.return_paths:
            sample["image_path"] = str(image_path)
            sample["mask_path"] = str(mask_path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_ids(self) -> List[str]:
        return [sample_id for sample_id, _, _ in self.samples]


def build_kvasir_datasets(
    root: str | Path,
    image_size: int | Sequence[int] = 352,
    normalize: bool = True,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    return_paths: bool = False,
) -> Dict[str, KvasirSegDataset]:
    """Convenience factory for train/val/test datasets."""
    train_transform = build_train_transforms(image_size=image_size, normalize=normalize)
    eval_transform = build_eval_transforms(image_size=image_size, normalize=normalize)
    resolved_size = image_size if isinstance(image_size, int) else None

    datasets = {
        "train": KvasirSegDataset(
            root=root,
            split=train_split,
            image_size=resolved_size,
            transform=train_transform,
            return_paths=return_paths,
            dataset_name="kvasir_seg",
        ),
        "val": KvasirSegDataset(
            root=root,
            split=val_split,
            image_size=resolved_size,
            transform=eval_transform,
            return_paths=return_paths,
            dataset_name="kvasir_seg",
        ),
        "test": KvasirSegDataset(
            root=root,
            split=test_split,
            image_size=resolved_size,
            transform=eval_transform,
            return_paths=return_paths,
            dataset_name="kvasir_seg",
        ),
    }
    return datasets


__all__ = [
    "KvasirPaths",
    "KvasirSegDataset",
    "canonical_sample_id",
    "looks_like_mask_stem",
    "infer_dataset_paths",
    "infer_kvasir_paths",
    "build_kvasir_datasets",
]
