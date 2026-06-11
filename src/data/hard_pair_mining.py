"""
Online Hard Pair Miner (OHPM) for dynamic batch-level pair mining.

Identifies the hardest pairs within each training batch:
  - Hard Positive: same person, lowest cosine similarity (pose/lighting variation).
  - Hard Negative: different person, highest cosine similarity (lookalike impostor).

Used as auxiliary loss in train_arcface.py together with ArcFaceLoss.
Requires PKBatchSampler to ensure multiple identities per batch.
"""

import torch
import torch.nn.functional as F


class HardPairMiner:
    """
    Finds the hardest positive and hardest negative pair for each sample in a batch.

    Operates entirely within a single mini-batch — no pre-computation needed.
    """

    def __init__(self):
        pass

    def __call__(self, embeddings, labels):
        """
        Mine hard pairs from a batch of embeddings.

        Steps:
          1. L2-normalize embeddings (cosine similarity = dot product).
          2. Build full cosine similarity matrix [batch x batch].
          3. Build same-identity and different-identity masks (exclude self-pairs).
          4. Hard Positive: same identity with lowest similarity.
          5. Hard Negative: different identity with highest similarity.

        Args:
            embeddings: Feature vectors [batch_size, 1280].
            labels:     Identity labels       [batch_size].

        Returns:
            hard_pos_idx, hard_neg_idx — index tensors [batch_size].
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.t())

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        device = embeddings.device
        mask_self = torch.eye(labels.size(0), dtype=torch.bool, device=device)
        labels_equal = labels_equal & ~mask_self
        labels_not_equal = ~labels_equal & ~mask_self

        # Hard positives: same identity, lowest similarity
        sim_pos = sim_matrix.clone()
        sim_pos[~labels_equal] = 2.0
        _, hard_pos_idx = sim_pos.min(dim=1)

        # Hard negatives: different identity, highest similarity
        sim_neg = sim_matrix.clone()
        sim_neg[~labels_not_equal] = -2.0
        _, hard_neg_idx = sim_neg.max(dim=1)

        return hard_pos_idx, hard_neg_idx
