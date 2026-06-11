"""
Advanced ArcFace + Hard Pair Mining training script.

Train/Val: merged CALFW + CPLFW eval pairs (80/20 split).
  - ArcFace uses images extracted from train eval pairs only.
  - Validation EER computed on val eval pairs.
Test: LFW eval pairs only (via evaluate_lfw.py).

Usage:
    python scripts/train_arcface.py --config configs/arcface_s.yaml
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    IdentityDatasetFromPairs,
    PKBatchSampler,
    VerificationPairsDataset,
    load_calfw_cplfw_train_val,
)
from src.data.hard_pair_mining import HardPairMiner
from src.models.backbone import EfficientNetV2Backbone
from src.models.losses import ArcFaceLoss, HardPairContrastiveLoss
from src.models.siamese import SiameseNetwork
from src.training.train_utils import EarlyStopping, evaluate_val_eer, resolve_project_path
from src.utils.config import load_config
from src.utils.paths import CHECKPOINTS_DIR, ensure_output_dirs


def train_arcface(config_path):
    config = load_config(config_path)
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training ArcFace model on device: {device}")

    processed_dir = resolve_project_path(PROJECT_ROOT, config["processed_data_dir"])
    variant = config["variant"]

    train_entries, val_entries = load_calfw_cplfw_train_val(
        train_ratio=config["train_val_split"],
        seed=config["split_seed"],
    )

    train_dataset = IdentityDatasetFromPairs(
        pair_entries=train_entries,
        variant=variant,
        processed_root=processed_dir,
        min_images=config.get("min_images", 2),
    )
    sampler = PKBatchSampler(
        train_dataset,
        p=config["p_identities"],
        k=config["k_images"],
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)

    val_dataset = VerificationPairsDataset(
        pair_entries=val_entries,
        variant=variant,
        processed_root=processed_dir,
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    backbone = EfficientNetV2Backbone(
        variant=variant,
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
    ).to(device)

    num_classes = len(train_dataset.label_to_indices)
    if num_classes < 2:
        raise RuntimeError("Not enough identities in train eval pairs for ArcFace training.")

    arcface_loss = ArcFaceLoss(
        in_features=backbone.embedding_dim,
        out_features=num_classes,
        s=config["arcface_scale"],
        m=config["arcface_margin"],
    ).to(device)

    hard_miner = HardPairMiner()
    hard_pair_loss = HardPairContrastiveLoss(margin=config["contrastive_margin"])
    mining_weight = config["hard_mining_weight"] if config["mining_strategy"] == "hard" else 0.0

    val_siamese = SiameseNetwork(backbone).to(device)

    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(arcface_loss.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    checkpoint_path = CHECKPOINTS_DIR / f"{config['checkpoint_name']}.pt"
    best_eer = float("inf")

    epochs = config["epochs"]
    for epoch in range(epochs):
        backbone.train()
        arcface_loss.train()
        start_time = time.time()
        total_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            unit="batch",
            leave=False,
        )
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = backbone(images)
            loss = arcface_loss(embeddings, labels)

            if mining_weight > 0:
                hard_pos_idx, hard_neg_idx = hard_miner(embeddings, labels)
                loss = loss + mining_weight * hard_pair_loss(embeddings, hard_pos_idx, hard_neg_idx)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        avg_loss = total_loss / max(len(train_loader), 1)
        val_eer = evaluate_val_eer(
            val_siamese,
            val_loader,
            device,
            use_siamese=True,
            progress_desc=f"Epoch {epoch + 1}/{epochs} [Val]",
        )
        epoch_time = time.time() - start_time
        print(
            f"[INFO] Epoch [{epoch + 1:02d}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | Val EER: {val_eer:.4f} | Time: {epoch_time:.1f}s"
        )

        improved = early_stopping.step(val_eer)
        if improved:
            best_eer = val_eer
            torch.save(backbone.state_dict(), checkpoint_path)
            print(f"[INFO] Saved best checkpoint (EER={best_eer:.4f}) -> {checkpoint_path}")

        if early_stopping.should_stop:
            print(f"[INFO] Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"[INFO] ArcFace training complete. Best Val EER: {best_eer:.4f}")

    del backbone, arcface_loss, optimizer, train_loader, val_loader, val_siamese
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArcFace model with hard pair mining.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    train_arcface(args.config)
