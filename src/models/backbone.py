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

    def __init__(self, variant="s", unfreeze_ratio=0.3, dropout=0.3):
        """
        Args:
            variant:        's', 'm', or 'l' — selects tf_efficientnetv2_{variant} from timm.
            unfreeze_ratio: Fraction of layers to unfreeze (0.3 = last 30%).
            dropout:        Dropout probability before embedding head (default 0.3).
        """
        super().__init__()
        self.model_name = f"tf_efficientnetv2_{variant}"

        # Step 1: Load pretrained EfficientNetV2 with classifier head removed (num_classes=0)
        self.backbone = timm.create_model(self.model_name, pretrained=True, num_classes=0)

        # Step 2: Dropout before embedding head (added — was missing in earlier version)
        self.dropout = nn.Dropout(p=dropout)

        # Step 3: Freeze early layers, unfreeze last portion for fine-tuning
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
        """Return feature dimension (1280 for all EfficientNetV2 variants)."""
        return self.backbone.num_features

    def forward(self, x):
        """
        Forward pass: extract L2-normalized embedding.

        Steps:
          1. Extract raw features from EfficientNetV2 backbone.
          2. Apply dropout (active during training only).
          3. L2-normalize so cosine similarity equals dot product.
        """
        features = self.backbone(x)
        features = self.dropout(features)
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
