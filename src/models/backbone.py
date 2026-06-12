"""
Backbone wrapper for face embedding extraction.

  InceptionResnetV1Backbone — facenet-pytorch InceptionResnetV1, pretrained on
                               CASIA-WebFace or VGGFace2 for face recognition.
                               Input: 160px. Outputs 512-dim L2-normalized embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionResnetV1Backbone(nn.Module):
    """
    Face embedding backbone based on InceptionResnetV1 (facenet-pytorch).

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
        pretrained: str = "vggface2",
        dropout: float = 0.1,
    ):
        """
        Args:
            unfreeze_ratio: Fraction of layers to unfreeze for fine-tuning.
            pretrained:     'casia-webface' or 'vggface2'.
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


