"""
Centralized path definitions for the face verification project.

Data layout (after moving nested data/ to project root):
  data/raw/{lfw,calfw,cplfw}/     -> raw datasets
  data/processed/{lfw,calfw,cplfw}/{S,M,L}/ -> preprocessed crops
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

LFW_RAW = DATA_RAW / "lfw"
CALFW_RAW = DATA_RAW / "calfw" / "calfw"
CPLFW_RAW = DATA_RAW / "cplfw" / "cplfw"

LFW_PAIRS_CSV = LFW_RAW / "pairs.csv"
CALFW_PAIRS_CSV = CALFW_RAW / "pairs.csv"
CPLFW_PAIRS_CSV = CPLFW_RAW / "pairs.csv"

LFW_PROCESSED = DATA_PROCESSED / "lfw"
CALFW_PROCESSED = DATA_PROCESSED / "calfw"
CPLFW_PROCESSED = DATA_PROCESSED / "cplfw"

CHECKPOINTS_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
ERROR_CASES_DIR = PROJECT_ROOT / "outputs" / "error_cases"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

VARIANT_INPUT_SIZES = {"s": 300, "m": 384, "l": 480}


def variant_folder(variant: str) -> str:
    """Map config variant 's'/'m'/'l' to processed folder name 'S'/'M'/'L'."""
    return variant.upper()


def processed_variant_dir(dataset: str, variant: str) -> Path:
    """Return processed image root for a dataset and model variant."""
    mapping = {
        "lfw": LFW_PROCESSED,
        "calfw": CALFW_PROCESSED,
        "cplfw": CPLFW_PROCESSED,
    }
    return mapping[dataset] / variant_folder(variant)


def ensure_output_dirs():
    for directory in (CHECKPOINTS_DIR, METRICS_DIR, ERROR_CASES_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_lfw_pairs_file() -> Path:
    return LFW_PAIRS_CSV
