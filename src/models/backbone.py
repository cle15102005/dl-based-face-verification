"""
EfficientNetV2 backbone wrapper for face embedding extraction.

Uses timm EfficientNetV2 S/M/L with ImageNet pretrained weights.
Removes the classifier head and outputs 1280-dim L2-normalized embeddings.
Unfreezes the last 30% of layers for fine-tuning.

heyyyy -> previously missing: Dropout(p=0.3) before the L2 normalization layer.
Added to regularize the embedding head before normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetV2Backbone(nn.Module):
    """
    Feature extractor based on EfficientNetV2 (Small / Medium / Large).

    Architecture:
      Input image -> EfficientNetV2 (no classifier) -> Dropout(0.3) -> L2 normalize -> embedding
    """

    def __init__(self, variant="s", unfreeze_ratio=0.3, dropout=0.3, embedding_dim=512):
        """
        Args:
            variant:       's', 'm', or 'l' — selects tf_efficientnetv2_{variant} from timm.
            unfreeze_ratio: Fraction of backbone layers to unfreeze (0.3 = last 30%).
            dropout:       Dropout probability before the projection head.
            embedding_dim: Output size of the projection head (default 512).
                           A smaller, fully-trainable layer between the frozen backbone
                           and ArcFace / pair losses. This layer adapts specifically to
                           face verification regardless of how much backbone is unfrozen.
        """
        super().__init__()
        self.model_name = f"tf_efficientnetv2_{variant}"
        self._embedding_dim = embedding_dim

        # Step 1: Load pretrained EfficientNetV2 with classifier head removed (num_classes=0)
        self.backbone = timm.create_model(self.model_name, pretrained=True, num_classes=0)

        # Step 2: Dropout before projection head
        self.dropout = nn.Dropout(p=dropout)

        # Step 3: Trainable projection head (1280 → embedding_dim).
        # Always fully trainable — gives the model a dedicated component that can adapt
        # to face verification even when most of the backbone is frozen.
        self.embedding_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        # Step 4: Freeze early backbone layers, unfreeze last portion for fine-tuning
        self._freeze_layers(unfreeze_ratio)

    def _freeze_layers(self, unfreeze_ratio):
        """
        Freeze the first (1 - unfreeze_ratio) fraction of parameter groups.
        Only the last `unfreeze_ratio` fraction receives gradient updates.
        """
        parameters = list(self.backbone.parameters())
        total_params = len(parameters)
        freeze_cutoff = int(total_params * (1.0 - unfreeze_ratio))

        for i, param in enumerate(parameters):
            param.requires_grad = i >= freeze_cutoff

        print(f"[INFO] {self.model_name}: Froze the first {freeze_cutoff}/{total_params} layers.")
        print(f"[INFO] {self.model_name}: Unfroze the last {total_params - freeze_cutoff} layers for fine-tuning.")

    @property
    def embedding_dim(self):
        """Output embedding dimension (after projection head)."""
        return self._embedding_dim

    def forward(self, x):
        """
        Forward pass: extract L2-normalized embedding.

        Steps:
          1. Extract raw 1280-dim features from EfficientNetV2 backbone.
          2. Apply dropout.
          3. Project to embedding_dim via trainable linear + BN head.
          4. L2-normalize so cosine similarity equals dot product.
        """
        features = self.backbone(x)
        features = self.dropout(features)
        features = self.embedding_head(features)
        embeddings = F.normalize(features, p=2, dim=1)
        return embeddings

    def set_train_mode(self, freeze_batchnorm: bool = True):
        """Set train mode; optionally keep BatchNorm in eval to preserve pretrained stats."""
        self.train()
        if not freeze_batchnorm:
            return
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
