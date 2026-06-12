"""
CLI entry point: extract CASIA-WebFace from archive RecordIO to processed JPGs.

Pure-Python reader — no mxnet needed (works with NumPy 2.x / Python 3.12).

Incremental: already-extracted images are skipped, so re-running only adds
what is missing (e.g. when increasing max-identities or max-images-per-identity).

Step 1 — run once before training:
    python scripts/extract_casia_webface.py --max-identities 5000 --max-images-per-identity 30

Increase limits later without re-extracting everything:
    python scripts/extract_casia_webface.py --max-identities 10000 --max-images-per-identity 45

Quick smoke test (first 1000 images):
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
        "--max-identities",
        type=int,
        default=5000,
        help="Number of identities to extract (default: 5000).",
    )
    parser.add_argument(
        "--max-images-per-identity",
        type=int,
        default=30,
        help="Max images per identity (default: 30).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Hard cap on total images, for quick smoke tests.",
    )
    args = parser.parse_args()
    extract_casia_webface(
        archive_dir=args.archive_dir,
        output_dir=args.output_dir,
        max_identities=args.max_identities,
        max_images_per_identity=args.max_images_per_identity,
        max_images=args.max_images,
    )
