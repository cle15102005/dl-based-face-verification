"""
Siamese Network for pairwise face verification.

Processes two face images through a single shared backbone (not two separate models).
Returns both embeddings and their cosine similarity for MATCH/REJECT decisions.

Forward: emb_a = backbone(img_a), emb_b = backbone(img_b), cos_sim = cosine(emb_a, emb_b)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Siamese architecture with weight sharing between two input branches.

    Used by:
      - Baseline models (ContrastiveLoss training + pair evaluation).
      - ArcFace validation wrapper (evaluate_val_eer during ArcFace training).
    """

    def __init__(self, backbone):
        """
        Args:
            backbone: EfficientNetV2Backbone instance (shared between both inputs).
        """
        super().__init__()
        self.backbone = backbone

    def forward(self, img_a, img_b):
        """
        Extract embeddings for a pair and compute cosine similarity.

        Steps:
          1. Pass img_a through shared backbone -> emb_a.
          2. Pass img_b through shared backbone -> emb_b.
          3. Compute cosine similarity between emb_a and emb_b.

        Returns:
            emb_a, emb_b, cos_sim
        """
        emb_a = self.backbone(img_a)
        emb_b = self.backbone(img_b)
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)
        return emb_a, emb_b, cos_sim
