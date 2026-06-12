"""
PyTorch datasets for face verification.

Data protocol
-------------
  Train : CASIA-WebFace identities / randomly sampled pairs (no augmentation)
  Val   : merged CALFW + CPLFW official eval pairs (no augmentation)
  Test  : LFW official eval pairs only (horizontal-flip TTA at eval time)

Preprocessing prerequisites
---------------------------
  Step 1: python scripts/extract_casia_webface.py
  Step 2: python scripts/mtcnn_preprocess.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from src.data.pair_parsers import (
    identity_from_filename,
    parse_calfw_pairs,
    parse_cplfw_pairs,
    parse_lfw_pairs_csv,
)
from src.utils.paths import (
    CALFW_PAIRS_CSV,
    CPLFW_PAIRS_CSV,
    DATA_PROCESSED,
    LFW_PAIRS_CSV,
    processed_variant_dir,
)


def _default_transform():
    """Standard ImageNet normalization — used for train, validation, and val/test pair loading."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def resolve_pair_image_paths(entry: dict, variant: str, processed_root: Path | None = None) -> tuple[str, str]:
    """
    Resolve absolute image paths for a unified pair entry.

    Supports:
      - LFW      : identity folder + numbered JPG
      - CALFW/CPLFW : flat 'aligned images/' layout
    """
    processed_root = processed_root or DATA_PROCESSED
    dataset = entry["dataset"]

    if entry.get("kind") == "lfw_id":
        img_root = processed_variant_dir("lfw", variant)
        if not img_root.exists():
            img_root = Path(processed_root) / "lfw" / variant.upper()
        name1, id1 = entry["name1"], entry["id1"]
        name2, id2 = entry["name2"], entry["id2"]
        img1 = img_root / name1 / f"{name1}_{id1:04d}.jpg"
        img2 = img_root / name2 / f"{name2}_{id2:04d}.jpg"
        return str(img1), str(img2)

    img_root = processed_variant_dir(dataset, variant)
    if not img_root.exists():
        img_root = Path(processed_root) / dataset / variant.upper()

    candidates = ["aligned images", "images", ""]
    img1_name, img2_name = entry["img1"], entry["img2"]

    def _find(filename: str) -> str:
        for sub in candidates:
            path = img_root / sub / filename if sub else img_root / filename
            if path.exists():
                return str(path)
        return str(img_root / "aligned images" / filename)

    return _find(img1_name), _find(img2_name)


def load_validation_pairs() -> list[dict]:
    """
    Load the full CALFW + CPLFW validation set (12,000 pairs total).

    Both benchmarks are used together for model selection during training.
    """
    calfw_pairs = parse_calfw_pairs(CALFW_PAIRS_CSV)
    cplfw_pairs = parse_cplfw_pairs(CPLFW_PAIRS_CSV)
    val_pairs = calfw_pairs + cplfw_pairs
    print(
        f"[INFO] Validation pairs: CALFW={len(calfw_pairs)} + "
        f"CPLFW={len(cplfw_pairs)} -> {len(val_pairs)} total"
    )
    return val_pairs


def load_lfw_test_pairs() -> list[dict]:
    """Load LFW evaluation pairs (6,000 pairs) for final held-out testing."""
    pairs = parse_lfw_pairs_csv(LFW_PAIRS_CSV)
    print(f"[INFO] LFW test pairs: {len(pairs)}")
    return pairs


def _scan_identity_folders(root: Path) -> dict[str, list[str]]:
    """Map identity folder name -> list of absolute image paths."""
    identity_to_paths: dict[str, list[str]] = {}
    if not root.exists():
        return identity_to_paths

    for identity_dir in sorted(root.iterdir()):
        if not identity_dir.is_dir():
            continue
        paths = [
            str(path)
            for path in sorted(identity_dir.iterdir())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if paths:
            identity_to_paths[identity_dir.name] = paths
    return identity_to_paths


class VerificationPairsDataset(Dataset):
    """
    Pair-based dataset for contrastive training, validation, or LFW testing.

    Each item returns (img1, img2, label) where label 1 = same person, 0 = different.
    """

    def __init__(
        self,
        pair_entries: list[dict],
        variant: str = "s",
        transform=None,
        processed_root: Path | str | None = None,
    ):
        self.variant = variant
        self.transform = transform if transform else _default_transform()
        self.processed_root = Path(processed_root) if processed_root else DATA_PROCESSED
        self.pair_entries = pair_entries
        self.pairs = self._resolve_all_paths()

    def _resolve_all_paths(self):
        pairs = []
        missing = 0
        for entry in self.pair_entries:
            img1_path, img2_path = resolve_pair_image_paths(
                entry, self.variant, self.processed_root
            )
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                pairs.append((img1_path, img2_path, entry["label"]))
            else:
                missing += 1
        if missing:
            print(f"[WARN] Skipped {missing} pairs with missing image files.")
        print(f"[INFO] VerificationPairsDataset: {len(pairs)} usable pairs.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(float(label), dtype=torch.float32)


class WebFacePairsDataset(Dataset):
    """
    Randomly sampled same/different pairs from CASIA-WebFace for baseline training.

    Each __getitem__ call draws a fresh pair, so every epoch sees different samples.
    """

    def __init__(
        self,
        variant: str = "s",
        transform=None,
        processed_root: Path | str | None = None,
        num_pairs: int = 50000,
        pos_fraction: float = 0.5,
        seed: int = 42,
        min_images_per_identity: int = 2,
    ):
        self.variant = variant
        self.transform = transform if transform else _default_transform()
        self.processed_root = Path(processed_root) if processed_root else DATA_PROCESSED
        self.num_pairs = num_pairs
        self.pos_fraction = pos_fraction
        self.seed = seed

        img_root = processed_variant_dir("casia-webface", variant)
        if not img_root.exists():
            img_root = self.processed_root / "casia-webface" / variant.upper()

        self.identity_to_paths = {
            identity: paths
            for identity, paths in _scan_identity_folders(img_root).items()
            if len(paths) >= min_images_per_identity
        }
        self.identities = sorted(self.identity_to_paths.keys())

        if len(self.identities) < 2:
            raise RuntimeError(
                f"Need at least 2 CASIA-WebFace identities with >= {min_images_per_identity} "
                f"images under {img_root}. Run scripts/extract_casia_webface.py first."
            )

        print(
            f"[INFO] WebFacePairsDataset: {len(self.identities)} identities, "
            f"{self.num_pairs} random pairs per epoch."
        )

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx)
        if rng.random() < self.pos_fraction:
            identity = rng.choice(self.identities)
            path_a, path_b = rng.sample(self.identity_to_paths[identity], 2)
            label = 1.0
        else:
            identity_a, identity_b = rng.sample(self.identities, 2)
            path_a = rng.choice(self.identity_to_paths[identity_a])
            path_b = rng.choice(self.identity_to_paths[identity_b])
            label = 0.0

        img1 = Image.open(path_a).convert("RGB")
        img2 = Image.open(path_b).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)


class CasiaWebFaceIdentityDataset(Dataset):
    """
    Identity-labeled CASIA-WebFace images for ArcFace + hard pair mining training.

    Reads every JPG under data/processed/casia-webface/{S,M,L}/{identity_id}/.
    """

    def __init__(
        self,
        variant: str = "s",
        transform=None,
        processed_root: Path | str | None = None,
        min_images: int = 2,
    ):
        self.variant = variant
        self.transform = transform if transform else _default_transform()
        self.processed_root = Path(processed_root) if processed_root else DATA_PROCESSED

        img_root = processed_variant_dir("casia-webface", variant)
        if not img_root.exists():
            img_root = self.processed_root / "casia-webface" / variant.upper()

        identity_to_paths = _scan_identity_folders(img_root)

        self.samples: list[str] = []
        self.labels: list[int] = []
        self.label_to_indices: dict[int, list[int]] = {}
        label_id = 0

        for _identity, paths in sorted(identity_to_paths.items()):
            if len(paths) < min_images:
                continue
            self.label_to_indices[label_id] = []
            for path in paths:
                self.samples.append(path)
                self.labels.append(label_id)
                self.label_to_indices[label_id].append(len(self.samples) - 1)
            label_id += 1

        if label_id < 2:
            raise RuntimeError(
                f"Not enough CASIA-WebFace identities under {img_root}. "
                "Run scripts/extract_casia_webface.py first."
            )

        print(
            f"[INFO] CasiaWebFaceIdentityDataset: {label_id} identities, "
            f"{len(self.samples)} images."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


class PKBatchSampler(Sampler):
    """P identities x K images per batch for ArcFace + hard pair mining."""

    def __init__(self, dataset: CasiaWebFaceIdentityDataset, p: int = 16, k: int = 4):
        self.dataset = dataset
        self.p = min(p, len(dataset.label_to_indices))
        self.k = k
        self.batch_size = self.p * self.k
        self.label_to_indices = dataset.label_to_indices
        self.available_labels = list(self.label_to_indices.keys())
        self.num_batches = max(1, len(self.dataset) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            selected = random.sample(self.available_labels, self.p)
            for cls in selected:
                indices = self.label_to_indices[cls]
                replace = len(indices) < self.k
                chosen = np.random.choice(indices, self.k, replace=replace)
                batch_indices.extend(chosen.tolist())
            yield batch_indices

    def __len__(self):
        return self.num_batches


class LFWPairsDataset(VerificationPairsDataset):
    """Backward-compatible wrapper for LFW test evaluation."""

    def __init__(self, pairs_file_path=None, processed_data_dir=None, img_size="s", transform=None, fold=None):
        del pairs_file_path, processed_data_dir, fold
        pairs = load_lfw_test_pairs()
        super().__init__(pair_entries=pairs, variant=img_size, transform=transform)


def load_calfw_cplfw_train_val(*_args, **_kwargs):
    """Deprecated: training now uses CASIA-WebFace. Use load_validation_pairs() instead."""
    raise NotImplementedError(
        "Training data is CASIA-WebFace. Use WebFacePairsDataset / CasiaWebFaceIdentityDataset "
        "for training and load_validation_pairs() for validation."
    )
