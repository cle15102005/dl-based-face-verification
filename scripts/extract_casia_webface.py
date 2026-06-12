"""
CLI entry point: extract CASIA-WebFace from archive RecordIO to processed JPGs.

Pure-Python reader — no mxnet needed (works with NumPy 2.x / Python 3.12).

Step 1 — run once before training:
    python scripts/extract_casia_webface.py

Optional smoke test (first 1000 images):
    python scripts/extract_casia_webface.py --max-images 1000
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.casia_extract import extract_casia_webface
from src.utils.paths import CASIA_WEBFACE_ARCHIVE, CASIA_WEBFACE_PROCESSED


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CASIA-WebFace .rec files to data/processed/casia-webface/."
    )
    parser.add_argument(
        "--archive-dir",
        default=str(CASIA_WEBFACE_ARCHIVE),
        help="Path to folder with train.rec, train.idx, train.lst.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CASIA_WEBFACE_PROCESSED),
        help="Output root for resized JPG crops.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    args = parser.parse_args()
    extract_casia_webface(
        archive_dir=args.archive_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )
