"""Wav2Vec2 Speech Emotion Recognition."""

import os
import torch
import torch.nn as nn

os.environ.setdefault('HF_HOME', 'weights/huggingface')

try:
    from transformers import Wav2Vec2Model, AutoFeatureExtractor
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class AttentivePool(nn.Module):
    """Attention pooling over time: softmax(fc2(tanh(fc1(h)))).
    Input: [B, T, C]  Output: [B, C]
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = self.fc2(torch.tanh(self.fc1(h))).squeeze(-1)  # [B, T]
        a = torch.softmax(a, dim=1).unsqueeze(-1)           # [B, T, 1]
        return (h * a).sum(dim=1)                           # [B, C]


class MeanPool(nn.Module):
    """Mean pooling with optional padding mask."""

    def forward(self, h: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return h.mean(1)


class Wav2Vec2SER(nn.Module):
    """Wav2Vec2-based speech emotion recognition.

    Args:
        pretrained_path: HF model id or local path for Wav2Vec2
        num_classes:     number of emotion classes
        pool:            pooling strategy: 'attn' | 'mean' | 'stat'
        freeze_feature_extractor: freeze CNN feature extractor weights
        dropout:         classifier dropout rate
    """

    def __init__(self,
                 pretrained_path: str = 'facebook/wav2vec2-base',
                 num_classes: int = 5,
                 pool: str = 'attn',
                 freeze_feature_extractor: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required: pip install transformers")

        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_path)
        if freeze_feature_extractor:
            self.encoder.feature_extractor.requires_grad_(False)

        hid = self.encoder.config.hidden_size  # 768
        self._pool_type = pool

        if pool == 'attn':
            self.pool = AttentivePool(hid)
            pool_out_dim = hid
        elif pool == 'stat':
            self.pool = None
            pool_out_dim = hid * 2
        else:  # 'mean'
            self.pool = MeanPool()
            pool_out_dim = hid

        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_out_dim),
            nn.Linear(pool_out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.hidden_size = hid

    def _pooling(self, h: torch.Tensor,
                 attention_mask: torch.Tensor = None) -> torch.Tensor:
        if self._pool_type == 'stat':
            return torch.cat([h.mean(1), h.std(1)], dim=-1)
        elif self._pool_type == 'attn':
            return self.pool(h)
        else:
            return self.pool(h, attention_mask)

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None):
        """
        Args:
            input_values:   [B, T_wav] normalized 16kHz waveform
            attention_mask: [B, T_wav] padding mask (optional)
        Returns:
            (logits [B, num_classes], embed [B, pool_out_dim])
        """
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h = outputs.last_hidden_state  # [B, T_feat, 768]
        embed = self._pooling(h, attention_mask)
        logits = self.classifier(embed)
        return logits, embed

    def unfreeze_feature_extractor(self):
        """Unfreeze CNN feature extractor for full fine-tuning."""
        self.encoder.feature_extractor.requires_grad_(True)
        print("Wav2Vec2 feature_extractor unfrozen")


def get_feature_extractor(pretrained_path: str = 'facebook/wav2vec2-base'):
    """Return HuggingFace AutoFeatureExtractor for use in collate_fn."""
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers required: pip install transformers")
    return AutoFeatureExtractor.from_pretrained(pretrained_path)


def get_model(num_classes: int = 5,
              pretrained_path: str = 'facebook/wav2vec2-base',
              pool: str = 'attn',
              freeze_feature_extractor: bool = True,
              dropout: float = 0.3) -> Wav2Vec2SER:
    """Return a Wav2Vec2SER instance."""
    return Wav2Vec2SER(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        pool=pool,
        freeze_feature_extractor=freeze_feature_extractor,
        dropout=dropout,
    )
