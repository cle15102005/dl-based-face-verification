"""
CLI entry point: ArcFace model training (InceptionResnetV1 backbone).

Step 3b — train on CASIA-WebFace identities, validate on CALFW+CPLFW:
    python scripts/train_arcface.py --config configs/arcface.yaml
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_arcface import train_arcface

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ArcFace model with hard pair mining.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    train_arcface(args.config)
