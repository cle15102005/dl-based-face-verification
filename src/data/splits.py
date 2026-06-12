"""
Train/validation splitting helpers.

The current project protocol no longer splits CALFW/CPLFW for train/val.
Those datasets are used entirely as the validation set while CASIA-WebFace
provides training data. The functions here remain for optional experiments.
"""

from __future__ import annotations

import random

from src.data.pair_parsers import identity_from_filename


def _pair_identities(entry: dict) -> set[str]:
    """Return the identity/identities referenced by a pair entry."""
    if entry.get("kind") == "lfw_id":
        return {entry["name1"]} if entry["label"] == 1 else {entry["name1"], entry["name2"]}

    id1 = identity_from_filename(entry["img1"])
    if entry["label"] == 1:
        return {id1}
    id2 = identity_from_filename(entry["img2"])
    return {id1, id2}


def merge_train_val_split(
    pair_lists: list[list[dict]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Merge multiple pair lists and split into train / validation sets by identity.

    Not used by the default WebFace-train / CALFW+CPLFW-val protocol, but kept
    for ablation experiments.
    """
    merged = []
    for pairs in pair_lists:
        merged.extend(pairs)

    identities = sorted({_id for entry in merged for _id in _pair_identities(entry)})
    rng = random.Random(seed)
    rng.shuffle(identities)

    split_idx = max(1, int(len(identities) * train_ratio))
    if split_idx >= len(identities):
        split_idx = len(identities) - 1

    train_ids = set(identities[:split_idx])
    val_ids = set(identities[split_idx:])

    train_pairs, val_pairs, skipped = [], [], 0
    for entry in merged:
        pair_ids = _pair_identities(entry)
        if pair_ids <= train_ids:
            train_pairs.append(entry)
        elif pair_ids <= val_ids:
            val_pairs.append(entry)
        else:
            skipped += 1

    if skipped:
        print(
            f"[INFO] Split excluded {skipped} cross-split pairs "
            f"(identities mixed between train and val)."
        )
    print(
        f"[INFO] Identity split: {len(train_ids)} train identities, "
        f"{len(val_ids)} val identities."
    )
    return train_pairs, val_pairs


def split_pair_list(
    pairs: list[dict],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Randomly split one pair list into two disjoint subsets."""
    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * train_ratio))
    if split_idx >= len(shuffled):
        split_idx = len(shuffled) - 1
    return shuffled[:split_idx], shuffled[split_idx:]
