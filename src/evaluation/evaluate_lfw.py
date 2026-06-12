"""
Evaluate all trained models on the LFW test set (6,000 eval pairs).

Step 4 of the pipeline — run after training:
    python scripts/evaluate_lfw.py

LFW is never used during training. When lfw_test_augmentation=true in the config,
horizontal-flip test-time augmentation (TTA) is applied at evaluation.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import VerificationPairsDataset, load_lfw_test_pairs
from src.evaluation.metrics import (
    compute_eer,
    far_at_frr,
    plot_far_frr_vs_threshold,
    plot_roc_curve,
)
from src.models.backbone import EfficientNetV2Backbone, InceptionResnetV1Backbone
from src.models.siamese import SiameseNetwork
from src.training.train_utils import (
    extract_backbone_pair_similarities,
    extract_pair_similarities,
    measure_pair_latency,
    resolve_project_path,
)
from src.utils.paths import CHECKPOINTS_DIR, METRICS_DIR, ensure_output_dirs


MODEL_CONFIGS = [
    ("baseline", "configs/baseline.yaml", "contrastive"),
    ("arcface",  "configs/arcface.yaml",  "arcface"),
]


def load_yaml_config(relative_path):
    config_path = PROJECT_ROOT / relative_path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config, loss_type, device):
    if config.get("backbone_type") == "inception":
        backbone = InceptionResnetV1Backbone(
            unfreeze_ratio=config["unfreeze_ratio"],
            dropout=config["dropout"],
            pretrained=None,  # weights loaded from checkpoint
        )
    else:
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
    use_tta = config.get("lfw_test_augmentation", True)

    if not checkpoint_path.exists():
        print(f"[WARN] Checkpoint not found, skipping: {checkpoint_path}")
        return None

    test_pairs = load_lfw_test_pairs()
    eval_dataset = VerificationPairsDataset(
        pair_entries=test_pairs,
        variant=config["variant"],
        processed_root=processed_dir,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model, use_siamese = build_model(config, loss_type, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    if use_siamese:
        scores, labels = extract_pair_similarities(model, eval_loader, device, use_tta=use_tta)
    else:
        scores, labels = extract_backbone_pair_similarities(model, eval_loader, device, use_tta=use_tta)

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

    tta_label = "TTA" if use_tta else "no-TTA"
    plot_roc_curve(
        scores_arr,
        labels_arr,
        METRICS_DIR / f"{checkpoint_name}_roc.png",
        title=f"ROC - {checkpoint_name} (LFW test, {tta_label})",
    )
    plot_far_frr_vs_threshold(
        scores_arr,
        labels_arr,
        METRICS_DIR / f"{checkpoint_name}_far_frr.png",
        title=f"FAR/FRR - {checkpoint_name} (LFW test, {tta_label})",
    )

    backbone_label = "InceptionResnetV1" if config.get("backbone_type") == "inception" else f"EfficientNetV2-{config['variant'].upper()}"
    return {
        "model": backbone_label,
        "loss_type": loss_type,
        "variant": config["variant"],
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_frr_1pct": far_at_1pct_frr,
        "latency_ms": latency_ms,
        "lfw_tta": use_tta,
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
                f"Latency={result['latency_ms']:.2f}ms | TTA={result['lfw_tta']}"
            )

    if not results:
        print("[ERROR] No checkpoints found. Train models first.")
        return

    df = pd.DataFrame(results)
    csv_path = METRICS_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Comparison table saved to: {csv_path}")
    print(df[["model", "loss_type", "eer", "far_at_frr_1pct", "latency_ms", "lfw_tta"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all trained models on LFW test pairs.")
    parser.parse_args()
    run_evaluation()
