"""
CLI: trim CASIA-WebFace processed data to a smaller train subset.

Default: 2000 identities, 20 images each. Extra files are deleted from disk.

    python scripts/subset_casia_webface.py
    python scripts/subset_casia_webface.py --max-identities 2000 --max-images 20
    python scripts/subset_casia_webface.py --dry-run
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.casia_subset import subset_casia_webface
from src.utils.paths import CASIA_WEBFACE_PROCESSED


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset CASIA-WebFace processed JPG folders.")
    parser.add_argument("--processed-dir", default=str(CASIA_WEBFACE_PROCESSED))
    parser.add_argument("--max-identities", type=int, default=2000)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    subset_casia_webface(
        processed_root=args.processed_dir,
        max_identities=args.max_identities,
        max_images_per_identity=args.max_images,
        seed=args.seed,
        dry_run=args.dry_run,
    )
