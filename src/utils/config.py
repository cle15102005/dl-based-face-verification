"""
YAML configuration loader with shared hyperparameter defaults.
"""

import yaml
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = {
        "processed_data_dir": "data/processed",
        "train_val_split": 0.8,
        "split_seed": 42,
        "learning_rate": 1.0e-4,
        "weight_decay": 0.4,
        "epochs": 30,
        "dropout": 0.3,
        "unfreeze_ratio": 0.3,
        "early_stopping_patience": 5,
        "arcface_scale": 64.0,
        "arcface_margin": 0.5,
        "contrastive_margin": 1.0,
        "p_identities": 16,
        "k_images": 4,
        "hard_mining_weight": 0.1,
        "mining_strategy": "hard",
        "min_images": 2,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    return config
