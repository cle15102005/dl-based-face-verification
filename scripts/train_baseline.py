"""
CLI entry point: Baseline Siamese model training (Contrastive Loss).

Runs from project root with --config. Previously train_baseline was invoked directly
from src/training/ with hardcoded paths and hyperparameters.

Usage (from project root):
    python scripts/train_baseline.py --config configs/baseline_s.yaml
    python scripts/train_baseline.py --config configs/baseline_m.yaml
    python scripts/train_baseline.py --config configs/baseline_l.yaml
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_baseline import train_baseline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train baseline Siamese model with contrastive loss.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    train_baseline(args.config)
