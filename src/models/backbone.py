"""
Backbone wrappers for face embedding extraction.

Two options:
  EfficientNetV2Backbone  — timm EfficientNetV2 S/M/L, ImageNet pretrained.
                            Input: 300/384/480px (variant s/m/l).
  InceptionResnetV1Backbone — facenet-pytorch InceptionResnetV1, face pretrained.
                               Input: 160px (variant i). No embedding head needed —
                               the network already outputs 512-dim embeddings.
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


class InceptionResnetV1Backbone(nn.Module):
    """
    Face embedding backbone based on InceptionResnetV1 (facenet-pytorch).

    Designed for the 'i' variant (160px input). Unlike EfficientNetV2, this
    network was pretrained specifically on face recognition datasets, so it
    converges faster and needs less data for ArcFace fine-tuning.

    Architecture:
      ImageNet-norm input → convert to [-1,1] → InceptionResnetV1 → Dropout → L2 normalize

    The network already outputs 512-dim embeddings — no separate embedding head needed.
    """

    # ImageNet normalization constants (used by our DataLoader)
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        unfreeze_ratio: float = 0.3,
        pretrained: str = "casia-webface",
        dropout: float = 0.1,
    ):
        """
        Args:
            unfreeze_ratio: Fraction of layers to unfreeze for fine-tuning.
            pretrained:     'casia-webface' or 'vggface2' (facenet-pytorch pretrained weights).
                            Pass None to start from random initialization.
            dropout:        Dropout before L2 normalization.
        """
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self.net = InceptionResnetV1(pretrained=pretrained, classify=False)
        self.dropout = nn.Dropout(p=dropout)
        self._embedding_dim = 512
        self._freeze_layers(unfreeze_ratio)

    def _freeze_layers(self, unfreeze_ratio: float):
        params = list(self.net.parameters())
        total = len(params)
        cutoff = int(total * (1.0 - unfreeze_ratio))
        for i, p in enumerate(params):
            p.requires_grad = i >= cutoff

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) with ImageNet normalization (from our DataLoader).

        Returns:
            (B, 512) L2-normalized embeddings.
        """
        # Convert ImageNet normalization → [-1, 1] expected by InceptionResnetV1
        mean = torch.tensor(self._MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = x * std + mean   # → [0, 1]
        x = x * 2.0 - 1.0   # → [-1, 1]

        emb = self.net(x)
        emb = self.dropout(emb)
        return F.normalize(emb, p=2, dim=1)

    def set_train_mode(self, freeze_batchnorm: bool = True):
        """Set train mode; optionally keep BatchNorm in eval."""
        self.train()
        if not freeze_batchnorm:
            return
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
