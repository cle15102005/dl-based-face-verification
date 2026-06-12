"""
Loss functions for baseline and advanced face verification training.

  - ContrastiveLoss:         baseline Siamese models (Euclidean, margin=1.0).
  - CosinePairLoss:          verify auxiliary loss for ArcFace — same metric as val EER.
  - ArcFaceLoss:             advanced models (Additive Angular Margin, s=32, m=0.5).
  - HardPairContrastiveLoss: auxiliary hard-mining loss inside ArcFace PK batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Standard contrastive loss for Siamese baseline training.

    Positive pairs (same person):   minimize Euclidean distance squared.
    Negative pairs (different):     push apart up to margin=1.0.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, label):
        """
        Args:
            emb_a:  Embedding of face A [batch, 1280].
            emb_b:  Embedding of face B [batch, 1280].
            label:  1.0 = same person, 0.0 = different person [batch].
        """
        euclidean_distance = F.pairwise_distance(emb_a, emb_b)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2)
            + (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


class CosinePairLoss(nn.Module):
    """
    Cosine-based pair loss — directly aligned with the cosine similarity used at val/test.

    Positive pairs (same person):   penalize when cos < pos_threshold  (want cos > 0.5)
    Negative pairs (different):     penalize when cos > neg_threshold  (want cos < 0.0)

    The gap between pos_threshold (0.5) and neg_threshold (0.0) acts as a margin —
    the model is not penalized for pairs already in the correct region.
    This avoids the previous problem where margin=0 required negatives to be at cos≤0
    while positives needed cos=1.0 — an unrealistically wide gap that collapsed training.
    """

    def __init__(self, pos_threshold: float = 0.5, neg_threshold: float = 0.0):
        super().__init__()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(emb_a, emb_b, dim=1)
        # same-person: penalize if cos < pos_threshold
        pos_loss = torch.clamp(self.pos_threshold - cos, min=0.0) * label
        # different-person: penalize if cos > neg_threshold
        neg_loss = torch.clamp(cos - self.neg_threshold, min=0.0) * (1.0 - label)
        return (pos_loss + neg_loss).mean()


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).

    Applies angular margin m=0.5 to the target class: cos(theta + m).
    Scale s=64 amplifies logits before cross-entropy.
    Tightens intra-class clusters and widens inter-class gaps for lookalike rejection.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        """
        Args:
            in_features:  Embedding dimension (1280 for EfficientNetV2).
            out_features: Number of identity classes in training set.
            s:            Logit scale factor (default 64).
            m:            Angular margin in radians (default 0.5).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Steps:
          1. L2-normalize embeddings and class weight vectors.
          2. Compute cosine similarity (dot product of normalized vectors).
          3. Compute angle theta = arccos(cosine).
          4. Add angular margin m to target class: cos(theta + m).
          5. Scale logits by s and apply cross-entropy.
        """
        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)
        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = cosine * (1.0 - one_hot) + target_logits * one_hot
        logits *= self.s
        return F.cross_entropy(logits, labels)


class HardPairContrastiveLoss(nn.Module):
    """
    Contrastive loss applied to hard-mined pairs within a batch.

    Used as auxiliary loss alongside ArcFaceLoss.
    Emphasizes hard positives (same person, low similarity) and
    hard negatives (different person, high similarity / lookalikes).
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, hard_pos_idx, hard_neg_idx):
        """
        Args:
            embeddings:    Batch embeddings         [batch, 1280].
            hard_pos_idx:  Hardest positive index per sample [batch].
            hard_neg_idx:  Hardest negative index per sample [batch].
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)

        cos_pos = (embeddings * embeddings[hard_pos_idx]).sum(dim=1)
        pos_loss = torch.pow(1.0 - cos_pos, 2).mean()

        cos_neg = (embeddings * embeddings[hard_neg_idx]).sum(dim=1)
        # Euclidean margin on unit vectors: dist >= margin  <=>  cos <= 1 - margin^2 / 2
        cos_threshold = 1.0 - 0.5 * (self.margin ** 2)
        neg_loss = torch.pow(torch.clamp(cos_neg - cos_threshold, min=0.0), 2).mean()

        return pos_loss + neg_loss
