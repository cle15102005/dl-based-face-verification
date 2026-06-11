"""
MTCNN face detection, alignment, and multi-resolution preprocessing.

Pipeline:
  1. Load raw LFW images from data/raw/lfw/.
  2. Detect and align faces using MTCNN (facenet-pytorch).
  3. Crop faces and resize to EfficientNetV2 input sizes: S=300, M=384, L=480.
  4. Save to data/processed/{s,m,l}/{identity}/.
  5. Log failed detections to outputs/results/mtcnn_failed_detections.log.

Imports canonical paths from src/utils/paths.py.
"""

import os
import torch
import logging
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

from src.utils.paths import LFW_RAW, LFW_PROCESSED, RESULTS_DIR


logging.basicConfig(
    filename=os.path.join(RESULTS_DIR, "mtcnn_failed_detections.log"),
    level=logging.WARNING,
    format="%(asctime)s - FAILED DETECTION: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def preprocess_and_split(raw_dir=None, processed_dir=None):
    """
    Detect faces using MTCNN and save multi-resolution crops for all EfficientNetV2 variants.

    Steps:
      1. Initialize MTCNN on GPU (or CPU fallback).
      2. Scan raw LFW identity folders.
      3. For each image: detect face, crop, resize to S/M/L, save to processed dirs.
      4. Log any images where face detection failed.
    """
    raw_dir = raw_dir or str(LFW_RAW / "lfw-deepfunneled")
    processed_dir = processed_dir or str(LFW_PROCESSED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing MTCNN on device: {device}")

    # margin=20 preserves jawline/forehead for better embedding quality
    mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=device)
    variant_sizes = {"S": 300, "M": 384, "L": 480}

    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print(f"[ERROR] Raw data directory '{raw_dir}' is empty or missing.")
        print(f"[INFO] Download LFW to: {raw_dir}")
        return

    identities = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    total_images = sum(len(os.listdir(os.path.join(raw_dir, d))) for d in identities)
    print(f"[INFO] Found {len(identities)} identities and {total_images} total images.")

    with tqdm(total=total_images, desc="Processing & Splitting Faces") as pbar:
        for identity_name in identities:
            identity_path = os.path.join(raw_dir, identity_name)

            for variant in variant_sizes:
                os.makedirs(os.path.join(processed_dir, variant, identity_name), exist_ok=True)

            for image_name in os.listdir(identity_path):
                img_path = os.path.join(identity_path, image_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    face = mtcnn(img)

                    if face is None:
                        logging.warning(img_path)
                        pbar.update(1)
                        continue

                    if isinstance(face, torch.Tensor):
                        face_np = face.permute(1, 2, 0).numpy()
                        face_pil = Image.fromarray(face_np.astype("uint8"))
                    else:
                        face_pil = face

                    for variant, size in variant_sizes.items():
                        out_path = os.path.join(processed_dir, variant, identity_name, image_name)
                        resized_face = face_pil.resize((size, size), Image.Resampling.LANCZOS)
                        resized_face.save(out_path)

                except Exception as exc:
                    logging.warning(f"{img_path} | ERROR: {exc}")

                pbar.update(1)

    print(f"\n[INFO] Preprocessing complete! Faces saved to: {processed_dir}")
    print(f"[INFO] Failed detections logged to: {os.path.join(RESULTS_DIR, 'mtcnn_failed_detections.log')}")


if __name__ == "__main__":
    preprocess_and_split()
