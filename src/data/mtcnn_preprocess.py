"""
MTCNN face detection, alignment, and multi-resolution preprocessing.

Step 2 of the pipeline (evaluation datasets only)
-------------------------------------------------
  Input  : data/raw/{lfw,calfw,cplfw}/
  Output : data/processed/{lfw,calfw,cplfw}/{S,M,L}/...

CASIA-WebFace is already aligned inside the .rec pack — use extract_casia_webface.py
instead of MTCNN for that dataset.

For each raw image:
  1. Detect and align the face with MTCNN (facenet-pytorch).
  2. Crop and resize to EfficientNetV2 sizes: S=300, M=384, L=480.
  3. Log failed detections to outputs/results/mtcnn_failed_detections.log.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

from src.utils.paths import (
    CALFW_RAW,
    CPLFW_RAW,
    DATA_PROCESSED,
    LFW_RAW,
    RESULTS_DIR,
    VARIANT_INPUT_SIZES,
)


logging.basicConfig(
    filename=os.path.join(RESULTS_DIR, "mtcnn_failed_detections.log"),
    level=logging.WARNING,
    format="%(asctime)s - FAILED DETECTION: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Raw image roots and where processed crops are written.
DATASET_LAYOUT = {
    "lfw": {
        "raw_root": LFW_RAW / "lfw-deepfunneled",
        "processed_root": DATA_PROCESSED / "lfw",
        "layout": "identity_folders",
    },
    "calfw": {
        "raw_root": CALFW_RAW,
        "processed_root": DATA_PROCESSED / "calfw",
        "layout": "flat_or_nested",
    },
    "cplfw": {
        "raw_root": CPLFW_RAW,
        "processed_root": DATA_PROCESSED / "cplfw",
        "layout": "flat_or_nested",
    },
}


def _iter_raw_images(raw_root: Path, layout: str):
    """
    Yield (relative_subdir, image_path) tuples for supported raw layouts.

    LFW uses one folder per identity. CALFW/CPLFW ship flat or nested JPG folders.
    """
    if not raw_root.exists():
        return

    if layout == "identity_folders":
        for identity_dir in sorted(raw_root.iterdir()):
            if not identity_dir.is_dir():
                continue
            for image_path in sorted(identity_dir.iterdir()):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    yield identity_dir.name, image_path
        return

    image_dirs = ["aligned images", "images", ""]
    for sub in image_dirs:
        search_root = raw_root / sub if sub else raw_root
        if not search_root.exists():
            continue
        for image_path in sorted(search_root.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield "", image_path
        break


def preprocess_dataset(dataset_name: str, mtcnn: MTCNN, variant_sizes: dict[str, int]) -> None:
    """Run MTCNN on one evaluation dataset and save S/M/L crops."""
    if dataset_name not in DATASET_LAYOUT:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from {list(DATASET_LAYOUT)}")

    spec = DATASET_LAYOUT[dataset_name]
    raw_root = Path(spec["raw_root"])
    processed_root = Path(spec["processed_root"])
    layout = spec["layout"]

    images = list(_iter_raw_images(raw_root, layout))
    if not images:
        print(f"[WARN] No raw images found for '{dataset_name}' under {raw_root}")
        return

    print(f"[INFO] Preprocessing {dataset_name}: {len(images)} images from {raw_root}")

    for subdir, image_path in tqdm(images, desc=f"MTCNN {dataset_name}", unit="img"):
        try:
            img = Image.open(image_path).convert("RGB")
            face = mtcnn(img)
            if face is None:
                logging.warning(str(image_path))
                continue

            if isinstance(face, torch.Tensor):
                face_np = face.permute(1, 2, 0).numpy()
                face_pil = Image.fromarray(face_np.astype("uint8"))
            else:
                face_pil = face

            if layout == "identity_folders":
                save_subdir = subdir
                save_name = image_path.name
            else:
                save_subdir = "aligned images"
                save_name = image_path.name

            for variant, size in variant_sizes.items():
                out_dir = processed_root / variant / save_subdir
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / save_name
                resized = face_pil.resize((size, size), Image.Resampling.LANCZOS)
                resized.save(out_path)
        except Exception as exc:
            logging.warning(f"{image_path} | ERROR: {exc}")


def preprocess_and_split(datasets: list[str] | None = None):
    """
    Preprocess LFW / CALFW / CPLFW with MTCNN.

    Args:
        datasets: Subset to process, e.g. ["lfw", "calfw", "cplfw"]. Default: all three.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    datasets = datasets or list(DATASET_LAYOUT.keys())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing MTCNN on device: {device}")

    mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=device)
    variant_sizes = {variant.upper(): size for variant, size in VARIANT_INPUT_SIZES.items()}

    for dataset_name in datasets:
        preprocess_dataset(dataset_name, mtcnn, variant_sizes)

    print(f"\n[INFO] MTCNN preprocessing complete for: {', '.join(datasets)}")
    print(f"[INFO] Failed detections logged to: {os.path.join(RESULTS_DIR, 'mtcnn_failed_detections.log')}")


if __name__ == "__main__":
    preprocess_and_split()
