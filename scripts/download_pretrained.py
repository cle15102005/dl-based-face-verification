"""
Initialize model checkpoints from VGGFace2 pretrained weights.

InceptionResnetV1 pretrained on VGGFace2 is used as the starting point for
fine-tuning on CASIA-WebFace — two distinct datasets, meaningful transfer.

Saves state_dicts as:
  outputs/checkpoints/arcface.pt   — for InceptionResnetV1Backbone (ArcFace training)
  outputs/checkpoints/baseline.pt  — for SiameseNetwork(InceptionResnetV1Backbone)

Usage:
    python scripts/download_pretrained.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.backbone import InceptionResnetV1Backbone
from src.models.siamese import SiameseNetwork
from src.utils.paths import CHECKPOINTS_DIR, ensure_output_dirs


def save_checkpoints():
    print("[INFO] Loading VGGFace2 pretrained weights ...")
    backbone = InceptionResnetV1Backbone(
        unfreeze_ratio=1.0,
        pretrained="vggface2",
        dropout=0.1,
    )
    backbone.eval()

    dummy = torch.zeros(1, 3, 160, 160)
    with torch.no_grad():
        out = backbone(dummy)
    assert out.shape == (1, 512), f"Unexpected output shape: {out.shape}"
    assert abs(out.norm(dim=1).item() - 1.0) < 1e-4, "Output not normalized"
    print(f"[INFO] OK — shape={out.shape}, norm={out.norm(dim=1).item():.6f}")

    arcface_path = CHECKPOINTS_DIR / "arcface.pt"
    torch.save(backbone.state_dict(), arcface_path)
    print(f"[INFO] Saved arcface.pt  ({arcface_path.stat().st_size / 1024 / 1024:.1f} MB)")

    siamese = SiameseNetwork(backbone)
    siamese.eval()
    baseline_path = CHECKPOINTS_DIR / "baseline.pt"
    torch.save(siamese.state_dict(), baseline_path)
    print(f"[INFO] Saved baseline.pt ({baseline_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    ensure_output_dirs()
    save_checkpoints()
