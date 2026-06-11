"""
Face verification evaluation metrics: FAR, FRR, EER, ROC plots.

  - FAR  (False Accept Rate):  impostor pairs incorrectly accepted.
  - FRR  (False Reject Rate):  genuine pairs incorrectly rejected.
  - EER  (Equal Error Rate):   threshold where FAR == FRR. Target: EER < 5% on LFW.
  - FAR@FRR1%:                 FAR at 1% FRR operating point.
  - ROC and FAR/FRR vs threshold plots for all 6 models.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path


def compute_far_frr(scores: np.ndarray, labels: np.ndarray, threshold: float):
    """
    Compute FAR and FRR at a given similarity threshold.

    Decision rule: score >= threshold -> MATCH, else REJECT.
    labels: 1 = genuine (same person), 0 = impostor (different person).
    """
    predictions = scores >= threshold
    genuine = labels == 1
    impostor = labels == 0

    frr = np.mean(predictions[genuine] == 0) if genuine.any() else 0.0
    far = np.mean(predictions[impostor] == 1) if impostor.any() else 0.0
    return float(far), float(frr)


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """
    Compute Equal Error Rate and the optimal decision threshold.

    EER = point where FAR == FRR on the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])
    return eer, eer_threshold, fpr, fnr, thresholds


def far_at_frr(scores: np.ndarray, labels: np.ndarray, target_frr: float = 0.01):
    """Compute FAR at a fixed FRR operating point (default 1%)."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - target_frr))
    return float(fpr[idx]), float(thresholds[idx])


def plot_roc_curve(scores, labels, save_path: Path, title: str = "ROC Curve"):
    """Plot and save ROC curve. Output: outputs/metrics/{model_name}_roc.png."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("True Accept Rate (1 - FRR)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_far_frr_vs_threshold(scores, labels, save_path: Path, title: str = "FAR/FRR vs Threshold"):
    """Plot FAR and FRR as functions of similarity threshold."""
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    fars, frrs = [], []
    for threshold in thresholds:
        far, frr = compute_far_frr(scores, labels, threshold)
        fars.append(far)
        frrs.append(frr)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fars, label="FAR")
    plt.plot(thresholds, frrs, label="FRR")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
