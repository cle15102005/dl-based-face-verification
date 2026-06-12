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
        "processed_data_dir": "data/processed",
        "split_seed": 42,
        "learning_rate": 1.0e-4,
        "weight_decay": 0.4,
        "epochs": 30,
        "dropout": 0.3,
        "unfreeze_ratio": 0.3,
        "early_stopping_patience": 5,
        "embedding_dim": 512,
        "arcface_scale": 32.0,
        "arcface_margin": 0.5,
        "backbone_lr": 5.0e-5,
        "arcface_lr": 1.0e-4,
        "contrastive_margin": 1.0,
        "p_identities": 16,
        "k_images": 4,
        "hard_mining_weight": 0.1,
        "mining_strategy": "hard",
        "min_images": 2,
        "webface_pairs_per_epoch": 10000,
        "verification_weight": 0.4,
        "lfw_test_augmentation": True,
        "num_workers": min(8, max(2, cpu_count - 1)),
        "prefetch_factor": 2,
        "max_train_identities": 2000,
        "max_images_per_identity": 20,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    return config
