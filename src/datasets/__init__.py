"""Dataset package for HF-U-Net benchmark."""

from .factory import SUPPORTED_BINARY_SEG_DATASETS, build_dataset
from .kvasir_seg_dataset import KvasirSegDataset, build_kvasir_datasets, canonical_sample_id, looks_like_mask_stem, infer_dataset_paths, infer_kvasir_paths
from .registry import DATASET_SPECS, DatasetSpec, SUPPORTED_DATASETS, get_dataset_spec, normalize_dataset_name
from .transforms import (
    SegCompose,
    SegResize,
    SegRandomHorizontalFlip,
    SegRandomVerticalFlip,
    SegRandomRotate90,
    SegRandomBrightnessContrast,
    SegToTensor,
    SegNormalize,
    build_train_transforms,
    build_eval_transforms,
)

__all__ = [
    "KvasirSegDataset",
    "build_kvasir_datasets",
    "build_dataset",
    "infer_dataset_paths",
    "infer_kvasir_paths",
    "canonical_sample_id",
    "looks_like_mask_stem",
    "SUPPORTED_BINARY_SEG_DATASETS",
    "DATASET_SPECS",
    "SUPPORTED_DATASETS",
    "DatasetSpec",
    "get_dataset_spec",
    "normalize_dataset_name",
    "SegCompose",
    "SegResize",
    "SegRandomHorizontalFlip",
    "SegRandomVerticalFlip",
    "SegRandomRotate90",
    "SegRandomBrightnessContrast",
    "SegToTensor",
    "SegNormalize",
    "build_train_transforms",
    "build_eval_transforms",
]
