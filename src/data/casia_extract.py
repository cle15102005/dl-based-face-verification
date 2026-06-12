"""
Extract CASIA-WebFace images from MXNet RecordIO into processed JPG folders.

Step 1 of the training pipeline
--------------------------------
  Input  : archive/casia-webface/{train.rec, train.idx, train.lst}
  Output : data/processed/casia-webface/{S,M,L}/{identity_id}/*.jpg

Uses a pure-Python RecordIO reader (no mxnet required).
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.data.recordio_reader import MXIndexedRecordReader
from src.utils.paths import (
    CASIA_WEBFACE_ARCHIVE,
    CASIA_WEBFACE_PROCESSED,
    RESULTS_DIR,
    VARIANT_INPUT_SIZES,
)


def _parse_train_lst(lst_path: Path) -> list[tuple[int, str, str]]:
    """
    Parse train.lst lines.

    CASIA-WebFace lst format:
      field0  original_path  identity_label  bbox/landmarks...

    field0 is NOT the RecordIO key. train.idx keys start at 1 (key 1 -> first image).
    Lst line order matches idx keys, so record_key = line_number + 1.
    """
    entries: list[tuple[int, str, str]] = []
    with open(lst_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            original_path = parts[1]
            identity_folder = Path(original_path).parent.name
            image_name = Path(original_path).name
            entries.append((line_no + 1, identity_folder, image_name))
    return entries


def extract_casia_webface(
    archive_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    max_images: int | None = None,
) -> None:
    """
    Decode CASIA-WebFace RecordIO and save resized crops for S/M/L variants.

    Args:
        archive_dir: Folder containing train.rec / train.idx / train.lst.
        output_dir:  Root for processed images (default: data/processed/casia-webface).
        max_images:  Optional cap for quick smoke tests.
    """
    archive_dir = Path(archive_dir or CASIA_WEBFACE_ARCHIVE)
    output_dir = Path(output_dir or CASIA_WEBFACE_PROCESSED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    idx_path = archive_dir / "train.idx"
    rec_path = archive_dir / "train.rec"
    lst_path = archive_dir / "train.lst"

    for required in (idx_path, rec_path, lst_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing CASIA-WebFace file: {required}. "
                "Place the InsightFace RecordIO pack in archive/casia-webface/."
            )

    entries = _parse_train_lst(lst_path)
    if max_images is not None:
        entries = entries[:max_images]

    variant_sizes = {variant.upper(): size for variant, size in VARIANT_INPUT_SIZES.items()}
    for variant in variant_sizes:
        (output_dir / variant).mkdir(parents=True, exist_ok=True)

    failed_log = RESULTS_DIR / "casia_extract_failed.log"
    failed = 0

    with MXIndexedRecordReader(idx_path, rec_path) as reader, open(
        failed_log, "w", encoding="utf-8"
    ) as log_handle:
        for record_key, identity_folder, image_name in tqdm(
            entries,
            desc="Extracting CASIA-WebFace",
            unit="img",
        ):
            try:
                rgb = reader.read_image(record_key)

                for variant, size in variant_sizes.items():
                    out_dir = output_dir / variant / identity_folder
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / image_name
                    if not out_path.exists():
                        resized = rgb.resize((size, size), Image.Resampling.LANCZOS)
                        resized.save(out_path, quality=95)
            except Exception as exc:
                failed += 1
                log_handle.write(f"{record_key}\t{identity_folder}\t{exc}\n")

    print(f"[INFO] CASIA-WebFace extraction complete -> {output_dir}")
    if failed:
        print(f"[WARN] {failed} records failed. See {failed_log}")
