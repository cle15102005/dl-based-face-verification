"""
Centralized path definitions for the face verification project.

Data protocol
-------------
  Train : CASIA-WebFace  -> data/processed/casia-webface/{S,M,L}/{identity_id}/*.jpg
  Val   : CALFW + CPLFW  -> data/processed/{calfw,cplfw}/{S,M,L}/...
  Test  : LFW only       -> data/processed/lfw/{S,M,L}/{name}/*.jpg

Raw / archive layout
--------------------
  archive/casia-webface/     MXNet RecordIO pack (train.rec, train.idx, train.lst)
  data/raw/lfw/              LFW deepfunneled images + pairs.csv
  data/raw/calfw/calfw/      CALFW images + pairs.csv
  data/raw/cplfw/cplfw/      CPLFW images + pairs.csv
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Raw evaluation datasets (MTCNN input) ---
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

LFW_RAW = DATA_RAW / "lfw"
CALFW_RAW = DATA_RAW / "calfw" / "calfw"
CPLFW_RAW = DATA_RAW / "cplfw" / "cplfw"

LFW_PAIRS_CSV = LFW_RAW / "pairs.csv"
CALFW_PAIRS_CSV = CALFW_RAW / "pairs.csv"
CPLFW_PAIRS_CSV = CPLFW_RAW / "pairs.csv"

# --- CASIA-WebFace (training) ---
CASIA_WEBFACE_ARCHIVE = PROJECT_ROOT / "archive" / "casia-webface"
CASIA_WEBFACE_PROCESSED = DATA_PROCESSED / "casia-webface"

# --- Processed evaluation datasets ---
LFW_PROCESSED = DATA_PROCESSED / "lfw"
CALFW_PROCESSED = DATA_PROCESSED / "calfw"
CPLFW_PROCESSED = DATA_PROCESSED / "cplfw"

# --- Outputs ---
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
    """
    Return processed image root for a dataset and model variant.

    Step 2 of preprocessing writes crops under these folders (see mtcnn_preprocess.py
    and extract_casia_webface.py).
    """
    mapping = {
        "lfw": LFW_PROCESSED,
        "calfw": CALFW_PROCESSED,
        "cplfw": CPLFW_PROCESSED,
        "casia-webface": CASIA_WEBFACE_PROCESSED,
        "webface": CASIA_WEBFACE_PROCESSED,
    }
    if dataset not in mapping:
        raise KeyError(f"Unknown dataset '{dataset}'. Expected one of {list(mapping)}.")
    return mapping[dataset] / variant_folder(variant)


def ensure_output_dirs():
    """Create checkpoint / metrics / error-case output folders if missing."""
    for directory in (CHECKPOINTS_DIR, METRICS_DIR, ERROR_CASES_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_lfw_pairs_file() -> Path:
    """Return path to the LFW 6,000-pair test protocol."""
    return LFW_PAIRS_CSV
