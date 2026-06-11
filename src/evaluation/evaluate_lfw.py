"""
Evaluate all trained models on the LFW test set (eval pairs only).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import VerificationPairsDataset, load_lfw_test_pairs
from src.evaluation.metrics import (
    compute_eer,
    far_at_frr,
    plot_far_frr_vs_threshold,
    plot_roc_curve,
)
from src.models.backbone import EfficientNetV2Backbone
from src.models.siamese import SiameseNetwork
from src.training.train_utils import (
    extract_backbone_pair_similarities,
    extract_pair_similarities,
    measure_pair_latency,
    resolve_project_path,
)
from src.utils.paths import CHECKPOINTS_DIR, METRICS_DIR, ensure_output_dirs


MODEL_CONFIGS = [
    ("baseline_s", "configs/baseline_s.yaml", "contrastive"),
    ("baseline_m", "configs/baseline_m.yaml", "contrastive"),
    ("baseline_l", "configs/baseline_l.yaml", "contrastive"),
    ("arcface_s", "configs/arcface_s.yaml", "arcface"),
    ("arcface_m", "configs/arcface_m.yaml", "arcface"),
    ("arcface_l", "configs/arcface_l.yaml", "arcface"),
]


def load_yaml_config(relative_path):
    config_path = PROJECT_ROOT / relative_path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config, loss_type, device):
    backbone = EfficientNetV2Backbone(
        variant=config["variant"],
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
    )
    if loss_type == "contrastive":
        return SiameseNetwork(backbone).to(device), True
    return backbone.to(device), False


def evaluate_model(checkpoint_name, config_path, loss_type, device, processed_dir):
    config = load_yaml_config(config_path)
    checkpoint_path = CHECKPOINTS_DIR / f"{checkpoint_name}.pt"

    if not checkpoint_path.exists():
        print(f"[WARN] Checkpoint not found, skipping: {checkpoint_path}")
        return None

    test_pairs = load_lfw_test_pairs()
    eval_dataset = VerificationPairsDataset(
        pair_entries=test_pairs,
        variant=config["variant"],
        processed_root=processed_dir,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)

    model, use_siamese = build_model(config, loss_type, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if use_siamese:
        scores, labels = extract_pair_similarities(model, eval_loader, device)
    else:
        scores, labels = extract_backbone_pair_similarities(model, eval_loader, device)

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    eer, eer_threshold, _, _, _ = compute_eer(scores_arr, labels_arr)
    far_at_1pct_frr, _ = far_at_frr(scores_arr, labels_arr, target_frr=0.01)

    if len(eval_dataset) > 0:
        sample_img1, sample_img2, _ = eval_dataset[0]
        latency_ms = measure_pair_latency(
            model, (sample_img1, sample_img2), device, use_siamese=use_siamese
        )
    else:
        latency_ms = 0.0

    plot_roc_curve(
        scores_arr, labels_arr,
        METRICS_DIR / f"{checkpoint_name}_roc.png",
        title=f"ROC - {checkpoint_name} (LFW test)",
    )
    plot_far_frr_vs_threshold(
        scores_arr, labels_arr,
        METRICS_DIR / f"{checkpoint_name}_far_frr.png",
        title=f"FAR/FRR - {checkpoint_name} (LFW test)",
    )

    return {
        "model": f"EfficientNetV2-{config['variant'].upper()}",
        "loss_type": loss_type,
        "variant": config["variant"],
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_frr_1pct": far_at_1pct_frr,
        "latency_ms": latency_ms,
        "checkpoint": str(checkpoint_path),
    }


def run_evaluation():
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating all models on LFW test set | device: {device}")

    processed_dir = str(PROJECT_ROOT / "data" / "processed")

    results = []
    for checkpoint_name, config_path, loss_type in MODEL_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"[INFO] Evaluating: {checkpoint_name}")
        print(f"{'=' * 60}")
        result = evaluate_model(
            checkpoint_name, config_path, loss_type, device, processed_dir
        )
        if result:
            results.append(result)
            print(
                f"[INFO] EER={result['eer']:.4f} | FAR@FRR1%={result['far_at_frr_1pct']:.4f} | "
                f"Latency={result['latency_ms']:.2f}ms"
            )

    if not results:
        print("[ERROR] No checkpoints found. Train models first.")
        return

    df = pd.DataFrame(results)
    csv_path = METRICS_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Comparison table saved to: {csv_path}")
    print(df[["model", "loss_type", "eer", "far_at_frr_1pct", "latency_ms"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all trained models on LFW test pairs.")
    parser.parse_args()
    run_evaluation()
