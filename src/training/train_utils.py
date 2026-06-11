"""
Shared training utilities: evaluation helpers, latency measurement, early stopping.
  - EarlyStopping(patience=5) monitoring validation EER (previously missing).
  - evaluate_val_eer() for per-epoch validation (previously only training loss was logged).
  - Checkpoints saved on best validation EER, not last epoch.
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_eer


def resolve_project_path(project_root, relative_path):
    """Convert a config-relative path to an absolute path."""
    from pathlib import Path
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str(project_root / path)


@torch.no_grad()
def extract_pair_similarities(model, dataloader, device):
    """
    Run SiameseNetwork on all pair batches and collect cosine similarity scores.

    Used for baseline (contrastive) model evaluation.
    """
    model.eval()
    scores, labels = [], []

    for img1, img2, label in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        _, _, cos_sim = model(img1, img2)
        scores.extend(cos_sim.cpu().numpy().tolist())
        labels.extend(label.numpy().tolist())

    return scores, labels


@torch.no_grad()
def extract_backbone_pair_similarities(backbone, dataloader, device):
    """
    Run EfficientNetV2Backbone on pair batches for ArcFace-trained model evaluation.

    ArcFace checkpoints save backbone weights only, so similarity is computed
    directly from backbone embeddings without a Siamese wrapper.
    """
    backbone.eval()
    scores, labels = [], []

    for img1, img2, label in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        emb1 = backbone(img1)
        emb2 = backbone(img2)
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1)
        scores.extend(cos_sim.cpu().numpy().tolist())
        labels.extend(label.numpy().tolist())

    return scores, labels


def evaluate_val_eer(model, val_loader, device, use_siamese=True, progress_desc=None):
    """
    Compute Equal Error Rate (EER) on a validation pair set.

    EER = threshold where FAR == FRR. Lower is better.
    Called each epoch during training for early stopping and checkpoint selection.
    """
    import numpy as np
    from tqdm import tqdm

    loader = val_loader
    if progress_desc:
        loader = tqdm(val_loader, desc=progress_desc, unit="batch", leave=False)

    if use_siamese:
        scores, labels = extract_pair_similarities(model, loader, device)
    else:
        scores, labels = extract_backbone_pair_similarities(model, loader, device)

    eer, _, _, _, _ = compute_eer(np.array(scores), np.array(labels))
    return eer


def measure_pair_latency(model, sample_pair, device, use_siamese=True, repeats=50):
    """
    Measure average inference latency (ms) for one verification pair.

    Includes 5 warmup runs before timing to exclude GPU initialization overhead.
    """
    img1, img2 = sample_pair
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(5):
            if use_siamese:
                model(img1, img2)
            else:
                model(img1)
                model(img2)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            if use_siamese:
                model(img1, img2)
            else:
                model(img1)
                model(img2)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / repeats * 1000


class EarlyStopping:
    """
    Stop training when validation EER does not improve for `patience` consecutive epochs.

    Also signals when a new best checkpoint should be saved.
    Previously missing — training ran all epochs and saved the last one only.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        """
        Args:
            metric: Current epoch validation EER (lower is better).

        Returns:
            True if this is a new best score (caller should save checkpoint).
        """
        if metric < self.best_score:
            self.best_score = metric
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
