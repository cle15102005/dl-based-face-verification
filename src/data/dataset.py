"""
PyTorch datasets for face verification.

Data protocol:
  - Train / Val: merged CALFW + CPLFW eval pairs (80/20 split)
  - Test:        LFW eval pairs only

All three datasets use their official evaluation pair lists only.
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
from src.data.splits import merge_train_val_split
from src.utils.paths import (
    CALFW_PAIRS_CSV,
    CPLFW_PAIRS_CSV,
    DATA_PROCESSED,
    LFW_PAIRS_CSV,
    processed_variant_dir,
)


def _default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def resolve_pair_image_paths(entry: dict, variant: str, processed_root: Path | None = None) -> tuple[str, str]:
    """
    Resolve absolute image paths for a unified pair entry.

    Supports LFW (identity + image id) and CALFW/CPLFW (flat filename layout).
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


def load_calfw_cplfw_train_val(
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Load and split merged CALFW + CPLFW eval pairs."""
    calfw_pairs = parse_calfw_pairs(CALFW_PAIRS_CSV)
    cplfw_pairs = parse_cplfw_pairs(CPLFW_PAIRS_CSV)
    train_pairs, val_pairs = merge_train_val_split(
        [calfw_pairs, cplfw_pairs],
        train_ratio=train_ratio,
        seed=seed,
    )
    print(
        f"[INFO] CALFW pairs: {len(calfw_pairs)} | CPLFW pairs: {len(cplfw_pairs)} | "
        f"Train: {len(train_pairs)} | Val: {len(val_pairs)}"
    )
    return train_pairs, val_pairs


def load_lfw_test_pairs() -> list[dict]:
    """Load LFW evaluation pairs (test set)."""
    pairs = parse_lfw_pairs_csv(LFW_PAIRS_CSV)
    print(f"[INFO] LFW test pairs: {len(pairs)}")
    return pairs


class VerificationPairsDataset(Dataset):
    """
    Pair-based dataset for contrastive training or evaluation.

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


class IdentityDatasetFromPairs(Dataset):
    """
    Identity-labeled images extracted from evaluation pair lists (for ArcFace training).

    Only includes images that appear in the supplied pair entries.
    """

    def __init__(
        self,
        pair_entries: list[dict],
        variant: str = "s",
        transform=None,
        processed_root: Path | str | None = None,
        min_images: int = 2,
    ):
        self.variant = variant
        self.transform = transform if transform else _default_transform()
        self.processed_root = Path(processed_root) if processed_root else DATA_PROCESSED

        identity_to_paths: dict[str, list[str]] = {}
        seen_paths: set[str] = set()

        for entry in pair_entries:
            if entry.get("kind") == "lfw_id":
                continue
            img1_path, img2_path = resolve_pair_image_paths(
                entry, self.variant, self.processed_root
            )
            for path in (img1_path, img2_path):
                if not os.path.exists(path) or path in seen_paths:
                    continue
                seen_paths.add(path)
                identity = identity_from_filename(os.path.basename(path))
                identity_to_paths.setdefault(identity, []).append(path)

        self.samples: list[str] = []
        self.labels: list[int] = []
        self.label_to_indices: dict[int, list[int]] = {}
        label_id = 0

        for identity, paths in sorted(identity_to_paths.items()):
            if len(paths) < min_images:
                continue
            self.label_to_indices[label_id] = []
            for path in paths:
                self.samples.append(path)
                self.labels.append(label_id)
                self.label_to_indices[label_id].append(len(self.samples) - 1)
            label_id += 1

        print(
            f"[INFO] IdentityDatasetFromPairs: {label_id} identities, "
            f"{len(self.samples)} images (from eval pairs)."
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

    def __init__(self, dataset: IdentityDatasetFromPairs, p: int = 16, k: int = 4):
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


# Backward-compatible alias used by evaluation scripts
class LFWPairsDataset(VerificationPairsDataset):
    """LFW test set wrapper — loads all LFW eval pairs."""

    def __init__(self, pairs_file_path=None, processed_data_dir=None, img_size="s", transform=None, fold=None):
        del pairs_file_path, processed_data_dir, fold
        pairs = load_lfw_test_pairs()
        super().__init__(pair_entries=pairs, variant=img_size, transform=transform)
