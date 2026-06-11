"""
Train/validation splitting for merged CALFW + CPLFW evaluation pairs.
"""

from __future__ import annotations

import random


def merge_train_val_split(
    pair_lists: list[list[dict]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Merge multiple pair lists and split into train / validation sets.

    Args:
        pair_lists:  e.g. [calfw_pairs, cplfw_pairs]
        train_ratio: fraction for training (default 0.8)
        seed:        random seed for reproducible split

    Returns:
        train_pairs, val_pairs
    """
    merged = []
    for pairs in pair_lists:
        merged.extend(pairs)

    rng = random.Random(seed)
    rng.shuffle(merged)

    split_idx = int(len(merged) * train_ratio)
    train_pairs = merged[:split_idx]
    val_pairs = merged[split_idx:]
    return train_pairs, val_pairs
