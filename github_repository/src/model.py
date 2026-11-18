"""Hybrid ResNet18 + Transformer-encoder classifier.

Pipeline (for a 224x224 RGB input):

    ResNet18 (no avgpool / fc)  ->  F        (512 x 7 x 7)
    1x1 convolution             ->  F'       (256 x 7 x 7)
    flatten + positional emb.   ->  Z0       (49 x 256)
    Transformer encoder x L     ->  Z_L      (49 x 256)
    mean over tokens            ->  z        (256)
    linear                      ->  logits   (3)

The flags ``use_projection``, ``use_pos_embed`` and ``n_layers`` reproduce the
ablation variants of the manuscript (``n_layers=0`` removes the encoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

CLASS_NAMES = ("Benign", "Malignant", "Normal")


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        pretrained: bool = True,
        use_projection: bool = True,
        use_pos_embed: bool = True,
        grid_size: int = 7,
    ) -> None:
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        # Keep conv1 ... layer4; drop global average pooling and the fc layer.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        backbone_channels = 512
        if use_projection:
            self.proj: nn.Module = nn.Conv2d(backbone_channels, d_model, kernel_size=1)
            dim = d_model
        else:
            self.proj = nn.Identity()
            dim = backbone_channels

        self.dim = dim
        self.num_tokens = grid_size * grid_size
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, self.num_tokens, dim)) if use_pos_embed else None
        )
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,  # LayerNorm after each sub-layer
            )
            self.encoder: nn.Module = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False
            )
        else:
            self.encoder = nn.Identity()

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)                       # B x 512 x 7 x 7
        f = self.proj(f)                           # B x d x 7 x 7
        z = f.flatten(2).transpose(1, 2)           # B x 49 x d
        if self.pos_embed is not None:
            z = z + self.pos_embed
        z = self.encoder(z)                        # B x 49 x d
        return self.classifier(z.mean(dim=1))      # B x num_classes


def build_model(cfg: dict | None = None, pretrained: bool = True) -> HybridCNNTransformer:
    """Create the model from the ``model`` section of a config dictionary."""
    cfg = dict(cfg or {})
    cfg.setdefault("pretrained", pretrained)
    return HybridCNNTransformer(**cfg)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
