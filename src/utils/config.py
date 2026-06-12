"""
YAML configuration loader with shared hyperparameter defaults.
"""

import os

import yaml
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cpu_count = os.cpu_count() or 4
    defaults = {
        # --- Paths & data ---
        "processed_data_dir": "data/processed",
        "split_seed": 42,
        "max_train_identities": 5000,     # CASIA identities used for training
        "max_images_per_identity": 30,    # Max images per identity
        # --- Training loop ---
        "epochs": 32,
        "early_stopping_patience": 10,
        # --- Backbone ---
        "dropout": 0.1,
        "unfreeze_ratio": 0.5,
        "embedding_dim": 512,
        # --- Optimizer: SGD (ArcFace paper defaults) ---
        "learning_rate": 1.0e-2,         # Fallback; train_arcface uses backbone_lr / arcface_lr
        "backbone_lr": 1.0e-4,           # Pretrained backbone: conservative to avoid overflow
        "arcface_lr": 1.0e-3,            # ArcFace head + embedding head (from scratch, clipped)
        "weight_decay": 5.0e-4,          # Paper value
        "lr_milestones": [20, 28],        # Epochs at which LR is multiplied by lr_decay
        "lr_decay": 0.1,                  # LR multiplier at each milestone (÷10)
        # --- ArcFace loss ---
        "arcface_scale": 64.0,            # Paper optimal s=64
        "arcface_margin": 0.5,            # Paper optimal m=0.5
        # --- PK batch sampler ---
        "p_identities": 16,
        "k_images": 4,
        "min_images": 2,
        # --- Baseline (contrastive / cosine pair) ---
        "contrastive_margin": 1.0,
        "cosine_pos_threshold": 0.5,
        "cosine_neg_threshold": 0.3,
        # --- Evaluation ---
        "lfw_test_augmentation": True,
        # --- DataLoader speed ---
        "num_workers": min(8, max(2, cpu_count - 1)),
        "prefetch_factor": 2,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    return config
