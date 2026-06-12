"""
Shared training utilities: evaluation helpers, latency measurement, early stopping.

Validation EER is computed on CALFW + CPLFW pairs during training.
LFW test evaluation (scripts/evaluate_lfw.py) can optionally apply horizontal-flip TTA.
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


def dataloader_kwargs(config: dict, device: torch.device) -> dict:
    """
    Shared DataLoader settings for faster image loading.

    num_workers > 0 enables parallel decode in worker processes (much faster than 0).
    """
    num_workers = int(config.get("num_workers", 4))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return kwargs


def _average_pair_similarity(model, img1, img2, device, use_siamese, use_tta):
    """
    Compute cosine similarity for one pair.

    When use_tta=True (LFW test only), average embeddings from the original image
    and its horizontal flip before computing similarity.
    """
    del device
    if use_siamese:
        emb_a, emb_b, _ = model(img1, img2)
        if use_tta:
            flip_a, flip_b, _ = model(torch.flip(img1, dims=[3]), torch.flip(img2, dims=[3]))
            emb_a = 0.5 * (emb_a + flip_a)
            emb_b = 0.5 * (emb_b + flip_b)
        return F.cosine_similarity(emb_a, emb_b, dim=1)

    emb1 = model(img1)
    emb2 = model(img2)
    if use_tta:
        emb1 = 0.5 * (emb1 + model(torch.flip(img1, dims=[3])))
        emb2 = 0.5 * (emb2 + model(torch.flip(img2, dims=[3])))
    return F.cosine_similarity(emb1, emb2, dim=1)


@torch.no_grad()
def extract_pair_similarities(model, dataloader, device, use_tta=False):
    """Collect cosine similarities from a SiameseNetwork over all pair batches."""
    model.eval()
    scores, labels = [], []

    for img1, img2, label in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        cos_sim = _average_pair_similarity(model, img1, img2, device, use_siamese=True, use_tta=use_tta)
        scores.extend(cos_sim.cpu().numpy().tolist())
        labels.extend(label.numpy().tolist())

    return scores, labels


@torch.no_grad()
def extract_backbone_pair_similarities(backbone, dataloader, device, use_tta=False):
    """Collect cosine similarities from backbone embeddings over all pair batches."""
    backbone.eval()
    scores, labels = [], []

    for img1, img2, label in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        cos_sim = _average_pair_similarity(
            backbone, img1, img2, device, use_siamese=False, use_tta=use_tta
        )
        scores.extend(cos_sim.cpu().numpy().tolist())
        labels.extend(label.numpy().tolist())

    return scores, labels


def evaluate_val_eer(
    model,
    val_loader,
    device,
    use_siamese=True,
    progress_desc=None,
    return_stats=False,
    use_tta=False,
):
    """
    Compute Equal Error Rate (EER) on a pair set.

    Training validation uses CALFW + CPLFW without TTA.
    LFW test evaluation can pass use_tta=True for horizontal-flip augmentation.
    """
    import numpy as np
    from tqdm import tqdm

    loader = val_loader
    if progress_desc:
        loader = tqdm(val_loader, desc=progress_desc, unit="batch", leave=False)

    if use_siamese:
        scores, labels = extract_pair_similarities(model, loader, device, use_tta=use_tta)
    else:
        scores, labels = extract_backbone_pair_similarities(model, loader, device, use_tta=use_tta)

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    eer, _, _, _, _ = compute_eer(scores_arr, labels_arr)

    if not return_stats:
        return eer

    genuine = scores_arr[labels_arr == 1]
    impostor = scores_arr[labels_arr == 0]
    pos_mean = float(genuine.mean()) if genuine.size else 0.0
    neg_mean = float(impostor.mean()) if impostor.size else 0.0
    return eer, {
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "gap": pos_mean - neg_mean,
    }


def measure_pair_latency(model, sample_pair, device, use_siamese=True, repeats=50):
    """Measure average inference latency (ms) for one verification pair."""
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
    """Stop training when validation EER does not improve for `patience` consecutive epochs."""

    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        if metric < self.best_score:
            self.best_score = metric
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
