"""
Advanced ArcFace + Hard Pair Mining training script.

Data protocol
-------------
  Train : CASIA-WebFace identities (ArcFace + in-batch hard mining)
          + optional WebFace pair verification loss (verification_weight)
  Val   : CALFW + CPLFW merged eval pairs (early stopping on validation EER)
  Test  : LFW only (run scripts/evaluate_lfw.py after training)

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    CasiaWebFaceIdentityDataset,
    PKBatchSampler,
    VerificationPairsDataset,
    WebFacePairsDataset,
    load_validation_pairs,
)
from src.data.hard_pair_mining import HardPairMiner
from src.models.backbone import EfficientNetV2Backbone
from src.models.losses import ArcFaceLoss, CosinePairLoss, HardPairContrastiveLoss
from src.training.train_utils import (
    EarlyStopping,
    dataloader_kwargs,
    evaluate_val_eer,
    resolve_project_path,
)
from src.utils.config import load_config
from src.utils.paths import CHECKPOINTS_DIR, ensure_output_dirs


def train_arcface(config_path):
    config = load_config(config_path)
    ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = dataloader_kwargs(config, device)
    print(f"[INFO] Training ArcFace model on device: {device} | num_workers={loader_kwargs['num_workers']}")

    processed_dir = resolve_project_path(PROJECT_ROOT, config["processed_data_dir"])
    variant = config["variant"]
    batch_size = config["batch_size"]
    freeze_bn = config.get("freeze_batchnorm", True)
    verification_weight = float(config.get("verification_weight", 0.2))

    train_dataset = CasiaWebFaceIdentityDataset(
        variant=variant,
        processed_root=processed_dir,
        min_images=config.get("min_images", 2),
    )
    sampler = PKBatchSampler(
        train_dataset,
        p=config["p_identities"],
        k=config["k_images"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        **loader_kwargs,
    )

    pair_train_loader = None
    pair_train_dataset = None
    if verification_weight > 0:
        pair_train_dataset = WebFacePairsDataset(
            variant=variant,
            processed_root=processed_dir,
            num_pairs=config.get("webface_pairs_per_epoch", 10000),
            seed=config["split_seed"] + 10,
        )
        pair_train_loader = DataLoader(
            pair_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(pair_train_dataset) > batch_size,
            **loader_kwargs,
        )

    val_entries = load_validation_pairs()
    val_dataset = VerificationPairsDataset(
        pair_entries=val_entries,
        variant=variant,
        processed_root=processed_dir,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    backbone = EfficientNetV2Backbone(
        variant=variant,
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
        embedding_dim=config.get("embedding_dim", 512),
    ).to(device)
    backbone.set_train_mode(freeze_batchnorm=freeze_bn)

    num_classes = len(train_dataset.label_to_indices)
    arcface_loss = ArcFaceLoss(
        in_features=backbone.embedding_dim,
        out_features=num_classes,
        s=config["arcface_scale"],
        m=config["arcface_margin"],
    ).to(device)

    contrastive_loss = CosinePairLoss(
        pos_threshold=config.get("cosine_pos_threshold", 0.5),
        neg_threshold=config.get("cosine_neg_threshold", 0.0),
    )
    hard_miner = HardPairMiner()
    hard_pair_loss = HardPairContrastiveLoss(margin=config["contrastive_margin"])
    mining_weight = config["hard_mining_weight"] if config["mining_strategy"] == "hard" else 0.0

    backbone_lr = config.get("backbone_lr", config["learning_rate"] * 0.5)
    arcface_lr = config.get("arcface_lr", config["learning_rate"])
    optimizer = optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": arcface_loss.parameters(), "lr": arcface_lr},
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=backbone_lr * 0.1)

    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    checkpoint_path = CHECKPOINTS_DIR / f"{config['checkpoint_name']}.pt"
    best_eer = float("inf")

    verify_msg = (
        f"verify_weight={verification_weight}, "
        f"pairs/epoch={len(pair_train_dataset)}"
        if pair_train_dataset is not None
        else "verify_weight=0 (disabled)"
    )
    print(
        f"[INFO] ArcFace classes: {num_classes} | "
        f"Val pairs: {len(val_entries)} | "
        f"Batches/epoch: {len(train_loader)} | {verify_msg}"
    )

    epochs = config["epochs"]
    for epoch in range(epochs):
        backbone.set_train_mode(freeze_batchnorm=freeze_bn)
        arcface_loss.train()

        start_time = time.time()
        total_loss = 0.0
        total_arcface = 0.0
        total_verify = 0.0
        num_steps = 0

        pair_iter = iter(pair_train_loader) if pair_train_loader is not None else None
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train/WebFace]",
            unit="batch",
            leave=False,
        )
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = backbone(images)
            arcface_term = arcface_loss(embeddings, labels)
            loss = arcface_term

            if mining_weight > 0:
                hard_pos_idx, hard_neg_idx = hard_miner(embeddings, labels)
                loss = loss + mining_weight * hard_pair_loss(embeddings, hard_pos_idx, hard_neg_idx)

            verify_term = torch.tensor(0.0, device=device)
            if pair_iter is not None:
                try:
                    img1, img2, pair_labels = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(pair_train_loader)
                    img1, img2, pair_labels = next(pair_iter)

                img1 = img1.to(device)
                img2 = img2.to(device)
                pair_labels = pair_labels.to(device)
                emb_a = backbone(img1)
                emb_b = backbone(img2)
                verify_term = contrastive_loss(emb_a, emb_b, pair_labels)
                loss = loss + verification_weight * verify_term

            loss.backward()
            optimizer.step()

            num_steps += 1
            total_loss += loss.item()
            total_arcface += arcface_term.item()
            total_verify += verify_term.item()
            train_pbar.set_postfix(
                arcface=f"{arcface_term.item():.2f}",
                verify=f"{verify_term.item():.4f}",
                refresh=False,
            )

        scheduler.step()

        avg_loss = total_loss / max(num_steps, 1)
        avg_arcface = total_arcface / max(num_steps, 1)
        avg_verify = total_verify / max(num_steps, 1)

        backbone.eval()
        val_eer, val_stats = evaluate_val_eer(
            backbone,
            val_loader,
            device,
            use_siamese=False,
            progress_desc=f"Epoch {epoch + 1}/{epochs} [Val/CALFW+CPLFW]",
            return_stats=True,
        )

        epoch_time = time.time() - start_time
        print(
            f"[INFO] Epoch [{epoch + 1:02d}/{epochs}] | "
            f"Loss: {avg_loss:.4f} (arcface={avg_arcface:.2f}, verify={avg_verify:.4f}) | "
            f"Val EER: {val_eer:.4f} (gap={val_stats['gap']:.4f}, "
            f"pos={val_stats['pos_mean']:.3f}, neg={val_stats['neg_mean']:.3f}) | "
            f"Time: {epoch_time:.1f}s"
        )

        improved = early_stopping.step(val_eer)
        if improved:
            best_eer = val_eer
            torch.save(backbone.state_dict(), checkpoint_path)
            print(f"[INFO] Saved best checkpoint (Val EER={best_eer:.4f}) -> {checkpoint_path}")

        if early_stopping.should_stop:
            print(f"[INFO] Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"[INFO] ArcFace training complete. Best Val EER: {best_eer:.4f}")

    del backbone, arcface_loss, optimizer, train_loader, val_loader, scheduler
    if pair_train_loader is not None:
        del pair_train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArcFace model with hard pair mining.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    train_arcface(args.config)
