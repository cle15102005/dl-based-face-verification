"""
Resize all processed face crops from S/ (300px) to I/ (160px).

Reads  : data/processed/{dataset}/S/**/*.jpg
Writes : data/processed/{dataset}/I/**/*.jpg

No face detection needed — crops are already aligned.
Much faster than re-running MTCNN.

Usage:
    python scripts/resize_to_160.py
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.utils.paths import DATA_PROCESSED

TARGET_SIZE = 160
DATASETS = ["lfw", "calfw", "cplfw", "casia-webface"]


def resize_image(src: Path, dst: Path):
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    img.save(dst, "JPEG", quality=95)
    return True


def resize_dataset(dataset: str):
    src_root = DATA_PROCESSED / dataset / "S"
    dst_root = DATA_PROCESSED / dataset / "I"

    if not src_root.exists():
        print(f"[SKIP] {dataset}/S not found, skipping.")
        return

    src_files = list(src_root.rglob("*.jpg"))
    if not src_files:
        print(f"[SKIP] {dataset}/S has no .jpg files.")
        return

    print(f"[INFO] {dataset}: {len(src_files)} images -> {dst_root}")

    tasks = []
    for src in src_files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        tasks.append((src, dst))

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(resize_image, s, d): (s, d) for s, d in tasks}
        for fut in as_completed(futures):
            if fut.result():
                done += 1

    print(f"[INFO] {dataset}: wrote {done} new files ({len(tasks) - done} already existed).")


if __name__ == "__main__":
    for ds in DATASETS:
        resize_dataset(ds)
    print("[INFO] Done.")
