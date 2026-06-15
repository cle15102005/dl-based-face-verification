"""
Visual error analysis: False Accept and False Reject case visualization.

Identifies and visualizes:
  - False Accepts:  impostor pairs incorrectly accepted (lookalike security risk).
  - False Rejects:  genuine pairs incorrectly rejected (same person blocked).

Outputs side-by-side images with similarity score, threshold, and verdict
to outputs/error_cases/.

Automatically selects the best model (lowest EER) from comparison_table.csv,
or accepts a --checkpoint override.

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --max-cases 20
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.dataset import VerificationPairsDataset, load_lfw_test_pairs
from src.evaluation.metrics import compute_eer
from src.models.backbone import InceptionResnetV1Backbone
from src.models.siamese import SiameseNetwork
from src.training.train_utils import extract_backbone_pair_similarities, extract_pair_similarities
from src.utils.paths import CHECKPOINTS_DIR, ERROR_CASES_DIR, METRICS_DIR, ensure_output_dirs

# Models available in the current pipeline
MODEL_CONFIGS = [
    ("baseline",     "configs/baseline.yaml",     "contrastive"),
    ("arcface_pure", "configs/arcface_pure.yaml",  "arcface"),
    ("arcface",      "configs/arcface.yaml",        "arcface"),
]


def load_best_model_from_metrics():
    """Select the best model from comparison_table.csv (lowest EER)."""
    csv_path = METRICS_DIR / "comparison_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Run evaluate_lfw.py first to generate comparison_table.csv")

    df = pd.read_csv(csv_path)
    best_row = df.loc[df["eer"].idxmin()]
    checkpoint = Path(best_row["checkpoint"])
    return checkpoint, best_row["loss_type"], None


def build_model(loss_type, device, config):
    backbone = InceptionResnetV1Backbone(
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
        pretrained=None,
    )
    if loss_type == "contrastive":
        return SiameseNetwork(backbone).to(device), True
    return backbone.to(device), False


def visualize_error_pair(img1_path, img2_path, score, threshold, label, save_path, error_type):
    """
    Save a side-by-side visualization of an error case.

    Shows: pair images, similarity score, EER threshold, predicted vs true label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(Image.open(img1_path))
    axes[0].set_title("Image A")
    axes[0].axis("off")
    axes[1].imshow(Image.open(img2_path))
    axes[1].set_title("Image B")
    axes[1].axis("off")

    verdict = "MATCH" if score >= threshold else "REJECT"
    true_label = "SAME" if label == 1 else "DIFFERENT"
    fig.suptitle(
        f"{error_type} | Score={score:.3f} | Threshold={threshold:.3f} | "
        f"Predicted={verdict} | True={true_label}",
        fontsize=10,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()


def _infer_loss_type(checkpoint_stem):
    """Infer loss type from checkpoint name."""
    for name, _, loss in MODEL_CONFIGS:
        if name == checkpoint_stem:
            return loss
    return "contrastive" if checkpoint_stem == "baseline" else "arcface"


def run_error_analysis(checkpoint=None, loss_type=None, max_cases=10):
    """
    Run full error analysis on the LFW 6,000-pair evaluation set.

    Steps:
      1. Select model (best from comparison_table.csv or --checkpoint override).
      2. Run inference on all 6,000 pairs and compute EER threshold.
      3. Collect False Accepts and False Rejects.
      4. Visualize up to max_cases of each error type.
    """
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint is None:
        checkpoint, loss_type, _ = load_best_model_from_metrics()  # variant read from config below
    else:
        checkpoint = Path(checkpoint)
        if loss_type is None:
            loss_type = _infer_loss_type(checkpoint.stem)

    config_path = PROJECT_ROOT / "configs" / f"{checkpoint.stem}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    variant = config["variant"]
    test_pairs = load_lfw_test_pairs()
    eval_dataset = VerificationPairsDataset(
        pair_entries=test_pairs,
        variant=variant,
        processed_root=str(PROJECT_ROOT / "data" / "processed"),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    model, use_siamese = build_model(loss_type, device, config)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)

    if use_siamese:
        scores, labels = extract_pair_similarities(model, eval_loader, device)
    else:
        scores, labels = extract_backbone_pair_similarities(model, eval_loader, device)

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    _, threshold, _, _, _ = compute_eer(scores_arr, labels_arr)
    predictions = scores_arr >= threshold

    false_accepts = []
    false_rejects = []

    for idx, (score, label, pred) in enumerate(zip(scores_arr, labels_arr, predictions)):
        img1_path, img2_path, _ = eval_dataset.pairs[idx]
        if label == 0 and pred:
            false_accepts.append((idx, score, label, img1_path, img2_path))
        elif label == 1 and not pred:
            false_rejects.append((idx, score, label, img1_path, img2_path))

    print(f"[INFO] EER threshold: {threshold:.4f}")
    print(f"[INFO] False Accepts: {len(false_accepts)} | False Rejects: {len(false_rejects)}")

    for i, (_, score, label, img1_path, img2_path) in enumerate(false_accepts[:max_cases]):
        visualize_error_pair(
            img1_path, img2_path, score, threshold, label,
            ERROR_CASES_DIR / f"false_accept_{i:03d}.png", "FALSE ACCEPT"
        )

    for i, (_, score, label, img1_path, img2_path) in enumerate(false_rejects[:max_cases]):
        visualize_error_pair(
            img1_path, img2_path, score, threshold, label,
            ERROR_CASES_DIR / f"false_reject_{i:03d}.png", "FALSE REJECT"
        )

    print(f"[INFO] Error case visualizations saved to: {ERROR_CASES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize false accept and false reject cases.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override.")
    parser.add_argument("--max-cases", type=int, default=10, help="Max cases per error type to visualize.")
    args = parser.parse_args()
    run_error_analysis(checkpoint=Path(args.checkpoint) if args.checkpoint else None, max_cases=args.max_cases)

