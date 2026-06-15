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
    frr_at_far,
    plot_far_frr_vs_threshold,
    plot_roc_curve,
    score_distribution_stats,
)
from src.models.backbone import InceptionResnetV1Backbone
from src.models.siamese import SiameseNetwork
from src.training.train_utils import (
    extract_backbone_pair_similarities,
    extract_pair_similarities,
    measure_pair_latency,
    resolve_project_path,
)
from src.utils.paths import CHECKPOINTS_DIR, METRICS_DIR, ensure_output_dirs


MODEL_CONFIGS = [
    ("baseline",     "configs/baseline.yaml",     "contrastive"),
    ("arcface_pure", "configs/arcface_pure.yaml",  "arcface"),
    ("arcface",      "configs/arcface.yaml",       "arcface"),
]


def load_yaml_config(relative_path):
    config_path = PROJECT_ROOT / relative_path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config, loss_type, device):
    backbone = InceptionResnetV1Backbone(
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
        pretrained=None,  # weights loaded from checkpoint
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

    # FRR@FAR1%  — threshold HIGH (reject 99% impostors), security operating point
    frr_at_1pct_far, thr_at_far1 = frr_at_far(scores_arr, labels_arr, target_far=0.01)

    # Score distributions
    stats = score_distribution_stats(scores_arr, labels_arr)

    genuine  = scores_arr[labels_arr == 1]
    impostor = scores_arr[labels_arr == 0]

    # ── Detailed diagnostic print ──────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Score distribution  ({checkpoint_name})")
    print(f"{'─' * 60}")
    print(f"  Genuine  ({len(genuine):5d} pairs)  "
          f"min={stats['genuine_min']:.4f}  "
          f"p1={stats['genuine_p1']:.4f}  "
          f"p5={stats['genuine_p5']:.4f}  "
          f"mean={stats['genuine_mean']:.4f}  "
          f"std={stats['genuine_std']:.4f}")
    print(f"  Impostor ({len(impostor):5d} pairs)  "
          f"mean={stats['impostor_mean']:.4f}  "
          f"std={stats['impostor_std']:.4f}  "
          f"p95={stats['impostor_p95']:.4f}  "
          f"p99={stats['impostor_p99']:.4f}  "
          f"max={stats['impostor_max']:.4f}")

    overlap = np.sum(impostor > stats["genuine_p5"]) / len(impostor) * 100
    print(f"\n  Overlap: {overlap:.1f}% of impostors score ABOVE genuine 5th-percentile")

    print(f"\n{'─' * 60}")
    print(f"  Operating points  ({checkpoint_name})")
    print(f"{'─' * 60}")
    print(f"  EER              {eer*100:5.2f}%   @ threshold={eer_threshold:.4f}")
    print(f"  FRR@FAR=1%       {frr_at_1pct_far*100:5.2f}%   @ threshold={thr_at_far1:.4f}"
          f"  ← HIGH threshold, rejects 99% impostors")

    # Spot-check around EER ± 0.2 / ± 0.3 (mirrors demo experience)
    print(f"\n  Spot-check  (EER threshold ± offset)")
    from sklearn.metrics import roc_curve as _roc
    _fpr, _tpr, _thr = _roc(labels_arr, scores_arr, pos_label=1)
    _fnr = 1 - _tpr
    for offset in (-0.2, -0.1, 0.0, +0.1, +0.2, +0.3):
        thr = eer_threshold + offset
        idx = np.argmin(np.abs(_thr - thr))
        print(f"    thr={thr:.3f} (+{offset:+.1f})  "
              f"FAR={_fpr[idx]*100:5.2f}%  FRR={_fnr[idx]*100:5.2f}%")
    print(f"{'─' * 60}")
    # ──────────────────────────────────────────────────────────────────────

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
        "model":             backbone_label,
        "loss_type":         loss_type,
        "variant":           config["variant"],
        "eer":               eer,
        "eer_threshold":     eer_threshold,
        "frr_at_far_1pct":   frr_at_1pct_far,
        "latency_ms":        latency_ms,
        "lfw_tta":           use_tta,
        "checkpoint":        str(checkpoint_path),
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
                f"[INFO] EER={result['eer']:.4f} | FRR@FAR1%={result['frr_at_far_1pct']:.4f} | "
                f"Latency={result['latency_ms']:.2f}ms | TTA={result['lfw_tta']}"
            )

    if not results:
        print("[ERROR] No checkpoints found. Train models first.")
        return

    df = pd.DataFrame(results)
    csv_path = METRICS_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Comparison table saved to: {csv_path}")
    print("\n" + "=" * 70)
    print("  Final summary")
    print("=" * 70)
    cols = ["model", "loss_type", "eer", "eer_threshold", "frr_at_far_1pct", "latency_ms"]
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    print("\n  frr_at_far_1pct : threshold HIGH, rejects 99% impostors → security operating point")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all trained models on LFW test pairs.")
    parser.parse_args()
    run_evaluation()
