from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    canonical_dir: str
    default_download_url: Optional[str] = None
    official_download_urls: Tuple[str, ...] = ()
    official_source_url: Optional[str] = None
    description: str = ""


DATASET_SPECS = {
    "kvasir_seg": DatasetSpec(
        name="kvasir_seg",
        aliases=("kvasir_seg", "kvasir-seg", "kvasir"),
        canonical_dir="Kvasir-SEG",
        default_download_url="https://datasets.simula.no/downloads/kvasir-seg.zip",
        description="Kvasir-SEG polyp segmentation dataset.",
    ),
    "cvc_clinicdb": DatasetSpec(
        name="cvc_clinicdb",
        aliases=("cvc_clinicdb", "cvc-clinicdb", "clinicdb", "cvc612", "cvc-612"),
        canonical_dir="CVC-ClinicDB",
        # PraNet public training bundle: TrainDataset.zip contains Kvasir-SEG and CVC-ClinicDB.
        # The preparation script searches inside the archive and extracts only the requested dataset.
        default_download_url="https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing",
        description="CVC-ClinicDB polyp segmentation dataset.",
    ),
    "etis": DatasetSpec(
        name="etis",
        aliases=("etis", "etis-larib", "etis_larib", "etis-laribpolypdb", "etis_laribpolypdb"),
        canonical_dir="ETIS-LaribPolypDB",
        # PraNet public testing bundle: TestDataset.zip contains ETIS-LaribPolypDB,
        # CVC-ColonDB, CVC-300, CVC-ClinicDB, and Kvasir test subsets.
        default_download_url="https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing",
        description="ETIS-LaribPolypDB polyp segmentation dataset.",
    ),
    "cvc_colondb": DatasetSpec(
        name="cvc_colondb",
        aliases=("cvc_colondb", "cvc-colondb", "colondb", "cvc-colon"),
        canonical_dir="CVC-ColonDB",
        default_download_url="https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing",
        description="CVC-ColonDB polyp segmentation dataset.",
    ),
    "cvc_300": DatasetSpec(
        name="cvc_300",
        aliases=("cvc_300", "cvc-300", "cvc300"),
        canonical_dir="CVC-300",
        default_download_url="https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing",
        description="CVC-300 polyp segmentation dataset.",
    ),

    "isbi2012": DatasetSpec(
        name="isbi2012",
        aliases=("isbi2012", "isbi-2012", "isbi_2012", "isbi2012_challenge", "isbi-2012-challenge"),
        canonical_dir="ISBI2012",
        default_download_url="https://downloads.imagej.net/ISBI-2012-challenge.zip",
        official_source_url="https://imagej.net/events/isbi-2012-segmentation-challenge",
        description=(
            "ISBI 2012 neuronal-structure EM segmentation dataset. The labeled "
            "training volume contains 30 paired 512x512 slices; the public challenge "
            "test volume has no public ground-truth labels and is not used by the "
            "standard supervised benchmark pipeline."
        ),
    ),

    "kvasir_instrument": DatasetSpec(
        name="kvasir_instrument",
        aliases=("kvasir_instrument", "kvasir-instrument", "instrument", "kvasir_tool"),
        canonical_dir="Kvasir-Instrument",
        default_download_url="https://datasets.simula.no/downloads/kvasir-instrument.zip",
        official_source_url="https://datasets.simula.no/kvasir-instrument/",
        description="Kvasir-Instrument gastrointestinal endoscopic tool binary segmentation dataset.",
    ),
    "hyper_kvasir_seg": DatasetSpec(
        name="hyper_kvasir_seg",
        aliases=("hyper_kvasir_seg", "hyper-kvasir-seg", "hyper_kvasir_segmentation", "hyper-kvasir-segmentation"),
        canonical_dir="HyperKvasir-SEG",
        default_download_url="https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip",
        official_source_url="https://datasets.simula.no/hyper-kvasir/",
        description="The segmentation-only HyperKvasir subset; the full 58.6 GB archive is intentionally not downloaded.",
    ),

}

_ALIAS_TO_NAME: Dict[str, str] = {}
for _name, _spec in DATASET_SPECS.items():
    for _alias in _spec.aliases:
        _ALIAS_TO_NAME[_alias.lower()] = _name

SUPPORTED_DATASETS = tuple(sorted(DATASET_SPECS))


def normalize_dataset_name(name: Optional[str]) -> str:
    if name is None:
        return "kvasir_seg"
    value = str(name).strip().lower().replace(" ", "_")
    if not value:
        return "kvasir_seg"
    try:
        return _ALIAS_TO_NAME[value]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {supported}") from exc



def get_dataset_spec(name: Optional[str] = None) -> DatasetSpec:
    return DATASET_SPECS[normalize_dataset_name(name)]


__all__ = [
    "DatasetSpec",
    "DATASET_SPECS",
    "SUPPORTED_DATASETS",
    "normalize_dataset_name",
    "get_dataset_spec",
]
