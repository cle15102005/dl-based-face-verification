"""
CLI entry point: MTCNN preprocessing for LFW / CALFW / CPLFW.

Step 2 — run after placing raw eval datasets under data/raw/:
    python scripts/mtcnn_preprocess.py
    python scripts/mtcnn_preprocess.py --datasets lfw calfw cplfw
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mtcnn_preprocess import preprocess_and_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MTCNN preprocess LFW, CALFW, and CPLFW into data/processed/."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["lfw", "calfw", "cplfw"],
        choices=["lfw", "calfw", "cplfw"],
        help="Which evaluation datasets to preprocess.",
    )
    args = parser.parse_args()
    preprocess_and_split(datasets=args.datasets)
