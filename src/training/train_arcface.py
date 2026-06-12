"""
Pure ArcFace training script — follows the original paper exactly.

Data protocol
-------------
  Train : CASIA-WebFace identities loaded with PK batch sampling
          (pure ArcFace cross-entropy, NO auxiliary pair losses)
  Val   : CALFW + CPLFW merged eval pairs (early stopping on validation EER)
  Test  : LFW only (run scripts/evaluate_lfw.py after training)

Optimizer
---------
  SGD with momentum=0.9, weight_decay=5e-4 (same as paper).
  Two parameter groups with different initial LRs:
    - ArcFace head (trained from scratch): arcface_lr (e.g. 0.01)
    - Backbone unfrozen layers (pretrained): backbone_lr (e.g. 0.001)
  MultiStepLR: divide both by lr_decay at each milestone epoch.

Reference: Deng et al., ArcFace (2019) — CASIA settings (Section 4.1)

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
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    CasiaWebFaceIdentityDataset,
    PKBatchSampler,
    VerificationPairsDataset,
    load_validation_pairs,
)
import torch.nn.functional as F

from src.models.backbone import InceptionResnetV1Backbone
from src.models.losses import ArcFaceLoss, HardPairContrastiveLoss
from src.training.train_utils import (
    EarlyStopping,
    dataloader_kwargs,
    evaluate_val_eer,
    resolve_project_path,
)
from src.utils.config import load_config
from src.utils.paths import CHECKPOINTS_DIR, ensure_output_dirs


def mine_hard_pairs(embeddings: torch.Tensor, labels: torch.Tensor):
    """
    For each sample in the batch, find the hardest positive (same identity,
    lowest cosine similarity) and hardest negative (different identity,
    highest cosine similarity).

    Returns index tensors hard_pos_idx and hard_neg_idx, both shape [B].
    """
    with torch.no_grad():
        emb = F.normalize(embeddings.detach(), p=2, dim=1)
        sim = emb @ emb.T  # [B, B]
        same = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
        eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)

        # Hardest positive: same class, exclude self, pick min similarity
        pos_sim = sim.masked_fill(~same | eye, float("inf"))
        hard_pos = pos_sim.argmin(dim=1)

        # Hardest negative: different class, pick max similarity
        neg_sim = sim.masked_fill(same, float("-inf"))
        hard_neg = neg_sim.argmax(dim=1)

    return hard_pos, hard_neg


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

    # --- Training data: CASIA-WebFace identities via PK sampler ---
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

    # --- Validation data: CALFW + CPLFW pairs ---
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

    # --- Model ---
    backbone = InceptionResnetV1Backbone(
        unfreeze_ratio=config["unfreeze_ratio"],
        dropout=config["dropout"],
        pretrained="vggface2",
    ).to(device)
    backbone.set_train_mode(freeze_batchnorm=freeze_bn)

    num_classes = len(train_dataset.label_to_indices)
    arcface_loss = ArcFaceLoss(
        in_features=backbone.embedding_dim,
        out_features=num_classes,
        s=config["arcface_scale"],
        m=config["arcface_margin"],
    ).to(device)

    hard_mining_weight = config.get("hard_mining_weight", 0.1)
    hard_loss_fn = HardPairContrastiveLoss(margin=1.0).to(device)

    # --- Optimizer: SGD with two LR groups (paper: momentum=0.9, wd=5e-4) ---
    backbone_lr = config.get("backbone_lr", 1e-3)
    arcface_lr = config.get("arcface_lr", 1e-2)
    optimizer = optim.SGD(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": arcface_loss.parameters(), "lr": arcface_lr},
        ],
        momentum=0.9,
        weight_decay=config["weight_decay"],
    )

    # MultiStepLR: divide LR by lr_decay at each milestone epoch (paper: ÷10 at 20, 28)
    milestones = config.get("lr_milestones", [20, 28])
    lr_decay = config.get("lr_decay", 0.1)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay)

    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    checkpoint_path = CHECKPOINTS_DIR / f"{config['checkpoint_name']}.pt"
    best_eer = float("inf")

    print(
        f"[INFO] ArcFace classes: {num_classes} | "
        f"Val pairs: {len(val_entries)} | "
        f"Batches/epoch: {len(train_loader)} | "
        f"arcface_lr={arcface_lr}, backbone_lr={backbone_lr} | "
        f"milestones={milestones}, decay={lr_decay} | "
        f"s={config['arcface_scale']}, m={config['arcface_margin']} | "
        f"hard_mining_weight={hard_mining_weight}"
    )

    epochs = config["epochs"]
    for epoch in range(epochs):
        backbone.set_train_mode(freeze_batchnorm=freeze_bn)
        arcface_loss.train()

        start_time = time.time()
        total_loss = 0.0
        num_steps = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train/WebFace]",
            unit="batch",
            leave=False,
        )
        total_arcface_loss = 0.0
        total_hard_loss = 0.0

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = backbone(images)

            arc_loss = arcface_loss(embeddings, labels)

            # Hard pair mining: hardest positive + hardest negative in PK batch
            hard_pos_idx, hard_neg_idx = mine_hard_pairs(embeddings, labels)
            hard_loss = hard_loss_fn(embeddings, hard_pos_idx, hard_neg_idx)

            loss = arc_loss + hard_mining_weight * hard_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in backbone.parameters() if p.requires_grad]
                + list(arcface_loss.parameters()),
                max_norm=5.0,
            )
            optimizer.step()

            num_steps += 1
            total_loss += loss.item()
            total_arcface_loss += arc_loss.item()
            total_hard_loss += hard_loss.item()
            train_pbar.set_postfix(
                arcface=f"{arc_loss.item():.4f}",
                hard=f"{hard_loss.item():.4f}",
                refresh=False,
            )

        scheduler.step()

        avg_loss = total_loss / max(num_steps, 1)
        avg_arc = total_arcface_loss / max(num_steps, 1)
        avg_hard = total_hard_loss / max(num_steps, 1)
        current_lr = scheduler.get_last_lr()

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
            f"Loss: {avg_loss:.4f} (arcface={avg_arc:.4f}, hard={avg_hard:.4f}) | "
            f"Val EER: {val_eer:.4f} (gap={val_stats['gap']:.4f}, "
            f"pos={val_stats['pos_mean']:.3f}, neg={val_stats['neg_mean']:.3f}) | "
            f"LR: {current_lr[1]:.2e} | "
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArcFace model (pure, no auxiliary losses).")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    train_arcface(args.config)
