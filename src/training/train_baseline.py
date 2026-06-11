"""
Baseline Siamese Network training with Contrastive Loss.

Train/Val: merged CALFW + CPLFW eval pairs (80/20 split).
Test:     LFW eval pairs only (via evaluate_lfw.py).

Usage:
    python scripts/train_baseline.py --config configs/baseline_s.yaml
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

from src.data.dataset import VerificationPairsDataset, load_calfw_cplfw_train_val
from src.models.backbone import EfficientNetV2Backbone
from src.models.losses import ContrastiveLoss
from src.models.siamese import SiameseNetwork
from src.training.train_utils import EarlyStopping, evaluate_val_eer, resolve_project_path
from src.utils.config import load_config
from src.utils.paths import CHECKPOINTS_DIR, ensure_output_dirs


def train_baseline(config_path):
    config = load_config(config_path)
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training baseline on device: {device}")

    processed_dir = resolve_project_path(PROJECT_ROOT, config["processed_data_dir"])
    variant = config["variant"]
    batch_size = config["batch_size"]

    train_entries, val_entries = load_calfw_cplfw_train_val(
        train_ratio=config["train_val_split"],
        seed=config["split_seed"],
    )

    train_dataset = VerificationPairsDataset(
        pair_entries=train_entries,
        variant=variant,
        processed_root=processed_dir,
    )
    val_dataset = VerificationPairsDataset(
        pair_entries=val_entries,
        variant=variant,
        processed_root=processed_dir,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    backbone = EfficientNetV2Backbone(
        variant=variant,
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
    )
    model = SiameseNetwork(backbone).to(device)
    criterion = ContrastiveLoss(margin=config["contrastive_margin"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    checkpoint_path = CHECKPOINTS_DIR / f"{config['checkpoint_name']}.pt"
    best_eer = float("inf")

    epochs = config["epochs"]
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            unit="batch",
            leave=False,
        )
        for img1, img2, labels in train_pbar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            emb_a, emb_b, _ = model(img1, img2)
            loss = criterion(emb_a, emb_b, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        avg_loss = total_loss / max(len(train_loader), 1)
        val_eer = evaluate_val_eer(
            model,
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
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[INFO] Saved best checkpoint (EER={best_eer:.4f}) -> {checkpoint_path}")

        if early_stopping.should_stop:
            print(f"[INFO] Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"[INFO] Baseline training complete. Best Val EER: {best_eer:.4f}")

    del model, backbone, optimizer, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline Siamese model with contrastive loss.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    train_baseline(args.config)
