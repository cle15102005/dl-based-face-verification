"""
Parse evaluation pair protocols for LFW, CALFW, and CPLFW.

Used for:
  - Validation : CALFW + CPLFW (merged, all pairs)
  - Test       : LFW only

All parsers return a unified list of dicts resolved later by dataset.resolve_pair_image_paths().
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


def parse_lfw_pairs_csv(pairs_file: str | Path) -> list[dict]:
    """
    Parse LFW pairs.csv (6,000 eval pairs).

    Match row:    name, imagenum1, imagenum2          -> label 1
    Mismatch row: name1, id1, name2, id2              -> label 0
    """
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            row = [cell.strip() for cell in row if cell.strip()]
            if len(row) == 3:
                name, id1, id2 = row
                pairs.append({
                    "dataset": "lfw",
                    "kind": "lfw_id",
                    "name1": name,
                    "id1": int(id1),
                    "name2": name,
                    "id2": int(id2),
                    "label": 1,
                })
            elif len(row) == 4:
                name1, id1, name2, id2 = row
                pairs.append({
                    "dataset": "lfw",
                    "kind": "lfw_id",
                    "name1": name1,
                    "id1": int(id1),
                    "name2": name2,
                    "id2": int(id2),
                    "label": 0,
                })
    return pairs


def _parse_filename_pairs(lines: list[tuple[str, int]], positive_labels: set[int]) -> list[dict]:
    """Group consecutive lines with the same label into pairs (2 lines = 1 pair)."""
    pairs = []
    i = 0
    while i + 1 < len(lines):
        fname1, lab1 = lines[i]
        fname2, lab2 = lines[i + 1]
        if lab1 == lab2 and ((lab1 in positive_labels) or lab1 == 0):
            pairs.append({
                "kind": "filename",
                "img1": fname1,
                "img2": fname2,
                "label": 1 if lab1 in positive_labels else 0,
            })
            i += 2
        else:
            i += 1
    return pairs


def parse_calfw_pairs(pairs_file: str | Path) -> list[dict]:
    """
    Parse CALFW pairs.csv (6,000 eval pairs).

    Format: `{filename} {label}` per line.
    Labels 1-10: positive pairs (group consecutive lines with same label).
    Label 0:     negative pairs (group consecutive lines with label 0).
    """
    lines = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            lines.append((parts[0], int(parts[1])))

    positive_labels = set(range(1, 11))
    parsed = _parse_filename_pairs(lines, positive_labels)
    for entry in parsed:
        entry["dataset"] = "calfw"
    return parsed


def parse_cplfw_pairs(pairs_file: str | Path) -> list[dict]:
    """
    Parse CPLFW pairs.csv (6,000 eval pairs).

    Label 1: positive pairs (consecutive lines).
    Label 0: negative pairs (consecutive lines).
    """
    lines = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            lines.append((parts[0], int(parts[1])))

    parsed = _parse_filename_pairs(lines, positive_labels={1})
    for entry in parsed:
        entry["dataset"] = "cplfw"
    return parsed


def identity_from_filename(filename: str) -> str:
    """Extract person name from CALFW/CPLFW image filename."""
    stem = Path(filename).stem
    match = re.match(r"^(.*)_\d+$", stem)
    return match.group(1) if match else stem
