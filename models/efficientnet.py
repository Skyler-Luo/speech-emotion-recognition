"""EfficientNet-based SER model with single-channel Log-Mel input."""

import os
import torch
import torch.nn as nn

os.environ.setdefault('TORCH_HOME', 'weights/torch')

from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
)
class EfficientNetSER(nn.Module):
    """Single-channel EfficientNet for SER.
    Replaces the first Conv2d with in_channels=1 (weights initialized from 3-ch mean).

    Input:  [B, 1, 224, 224]
    Output: (logits [B, num_classes], embed [B, last_ch])
    """

    _VARIANTS = {
        'b0': (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
        'b5': (efficientnet_b5, EfficientNet_B5_Weights.IMAGENET1K_V1, 2048),
    }

    def __init__(self,
                 num_classes: int = 5,
                 variant: str = 'b5',
                 pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        if variant not in self._VARIANTS:
            raise ValueError(f"variant must be one of {list(self._VARIANTS.keys())}, got '{variant}'")

        build_fn, weights_enum, last_ch = self._VARIANTS[variant]
        backbone = build_fn(weights=weights_enum if pretrained else None)

        # patch first conv: 3ch -> 1ch, average pretrained weights across channels
        orig_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
            if orig_conv.bias is not None:
                new_conv.bias.copy_(orig_conv.bias)
        backbone.features[0][0] = new_conv

        backbone.classifier[1] = nn.Sequential(
            nn.Linear(last_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.backbone = backbone
        self.last_ch = last_ch

    def forward(self, x: torch.Tensor):
        feat = self.backbone.features(x)                # [B, last_ch, H', W']
        embed = self.backbone.avgpool(feat).flatten(1)  # [B, last_ch]
        logits = self.backbone.classifier(embed)        # [B, num_classes]
        return logits, embed


def get_model(num_classes: int = 5,
              variant: str = 'b5',
              pretrained: bool = True,
              dropout: float = 0.3) -> EfficientNetSER:
    """Return an EfficientNetSER instance."""
    return EfficientNetSER(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
        dropout=dropout,
    )
