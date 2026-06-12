"""
Trim CASIA-WebFace processed folders to a fixed train subset.

Keeps N identities (sorted, seeded) and M images per identity, deletes everything else
from data/processed/casia-webface/{S,M,L}/.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from src.utils.paths import CASIA_WEBFACE_PROCESSED, VARIANT_INPUT_SIZES


def subset_casia_webface(
    processed_root: Path | str | None = None,
    max_identities: int = 2000,
    max_images_per_identity: int = 20,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Delete extra identities/images from all S/M/L variant folders.

    Uses variant S as the canonical identity/image list; the same keep/delete
    decisions are applied to M and L so all variants stay aligned.
    """
    processed_root = Path(processed_root or CASIA_WEBFACE_PROCESSED)
    canonical = processed_root / "S"
    if not canonical.exists():
        raise FileNotFoundError(
            f"Missing {canonical}. Run scripts/extract_casia_webface.py first."
        )

    identity_dirs = sorted(
        d for d in canonical.iterdir() if d.is_dir()
    )
    rng = random.Random(seed)
    rng.shuffle(identity_dirs)
    keep_identities = sorted(d.name for d in identity_dirs[:max_identities])
    keep_set = set(keep_identities)

    stats = {
        "identities_before": len(identity_dirs),
        "identities_after": len(keep_identities),
        "images_deleted": 0,
        "identity_dirs_deleted": 0,
    }

    variants = [v.upper() for v in VARIANT_INPUT_SIZES]

    for variant in variants:
        variant_root = processed_root / variant
        if not variant_root.exists():
            continue

        for identity_dir in sorted(variant_root.iterdir()):
            if not identity_dir.is_dir():
                continue

            identity_name = identity_dir.name
            if identity_name not in keep_set:
                stats["identity_dirs_deleted"] += 1
                if not dry_run:
                    shutil.rmtree(identity_dir)
                continue

            images = sorted(
                p for p in identity_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for extra in images[max_images_per_identity:]:
                stats["images_deleted"] += 1
                if not dry_run:
                    extra.unlink()

    print(
        f"[INFO] CASIA subset: {stats['identities_before']} -> "
        f"{stats['identities_after']} identities, "
        f"max {max_images_per_identity} images each"
    )
    print(
        f"[INFO] Deleted {stats['images_deleted']} images and "
        f"{stats['identity_dirs_deleted']} identity folders"
        f"{' (dry run)' if dry_run else ''}."
    )
    return stats
