"""CNN-BiLSTM baseline model for SER.
Features: MFCC + spectral centroid + spectral bandwidth -> [T, feat_dim]
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


def compute_spectral_features(audio: torch.Tensor,
                               sample_rate: int = 44100,
                               n_fft: int = 2048,
                               win_length: int = 2048,
                               hop_length: int = 512,
                               p: int = 2):
    """Compute spectral centroid and bandwidth.

    Args:
        audio:       waveform [1, T]
        sample_rate: sample rate
        n_fft:       FFT size
        win_length:  window length
        hop_length:  frame shift
        p:           bandwidth order

    Returns:
        centroid:  [1, time_frames]
        bandwidth: [1, time_frames]
    """
    window = torch.hann_window(win_length, device=audio.device)
    stft = torch.stft(
        audio.squeeze(0), n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, return_complex=True
    )
    magnitude = stft.abs()  # [freq_bins, time_frames]

    freq_bins = magnitude.size(0)
    freqs = torch.linspace(0, sample_rate / 2, steps=freq_bins,
                           device=audio.device).unsqueeze(1)  # [freq_bins, 1]

    power = magnitude.sum(dim=0, keepdim=False) + 1e-10  # [time_frames]
    centroid = (freqs * magnitude).sum(dim=0) / power  # [time_frames]

    deviation = (freqs - centroid.unsqueeze(0)).abs() ** p
    bandwidth = (magnitude * deviation).sum(dim=0) / power
    bandwidth = bandwidth ** (1.0 / p)  # [time_frames]

    return centroid.unsqueeze(0), bandwidth.unsqueeze(0)  # each [1, time_frames]


def extract_baseline_feature(audio: torch.Tensor,
                              sample_rate: int = 44100,
                              n_mfcc: int = 10,
                              n_fft: int = 2048,
                              hop_length: int = 512,
                              win_length: int = 2048,
                              max_len: int = 300,
                              p: int = 2,
                              mfcc_transform: Optional[T.MFCC] = None) -> torch.Tensor:
    """Extract baseline features: MFCC(n_mfcc) + centroid(1) + bandwidth(1) -> [T, n_mfcc+2].

    Args:
        audio:          waveform [1, T], resampled to sample_rate
        sample_rate:    sample rate
        n_mfcc:         number of MFCC coefficients
        n_fft / hop_length / win_length: STFT parameters
        max_len:        target number of time frames
        p:              spectral bandwidth order
        mfcc_transform: optional pre-built T.MFCC instance to avoid re-initialization
                        on repeated calls (e.g. pass a cached instance in the Dataset)

    Returns:
        feature: [max_len, n_mfcc + 2]
    """
    if mfcc_transform is None:
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': n_fft, 'hop_length': hop_length}
        )
    mfcc_transform = mfcc_transform.to(audio.device)

    mfcc = mfcc_transform(audio).squeeze(0)  # [n_mfcc, T]
    centroid, bandwidth = compute_spectral_features(
        audio, sample_rate, n_fft, win_length, hop_length, p
    )  # each [1, T']

    # align time dim (MFCC and spectral frames may differ by 1)
    min_t = min(mfcc.shape[1], centroid.shape[1])
    feature = torch.cat([mfcc[:, :min_t],
                         centroid[:, :min_t],
                         bandwidth[:, :min_t]], dim=0)  # [n_mfcc+2, T]

    # Z-score normalization
    feature = (feature - feature.mean(dim=1, keepdim=True)) / \
              (feature.std(dim=1, keepdim=True) + 1e-6)

    # pad or truncate to fixed length
    t = feature.shape[1]
    if t < max_len:
        feature = F.pad(feature, (0, max_len - t))
    else:
        feature = feature[:, :max_len]

    return feature.transpose(0, 1)  # [max_len, n_mfcc+2]


class SERBaselineModel(nn.Module):
    """CNN + BiLSTM + Attention baseline model for SER.

    Input:  [B, T, feat_dim]   (T=max_len, feat_dim=n_mfcc+2)
    Output: (logits [B, num_classes], embed [B, 2*lstm_hidden])
    """

    def __init__(self,
                 input_dim: int = 12,
                 num_classes: int = 5,
                 cnn_out_channels: int = 64,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        lstm_input = (input_dim // 2) * cnn_out_channels
        self.bilstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attention = nn.Linear(2 * lstm_hidden, 1)

        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, feat_dim]
        Returns:
            (logits [B, num_classes], embed [B, 2*lstm_hidden])
        """
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.cnn(x)  # [B, C, T', F']
        B, C, T, Fv = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * Fv)

        x, _ = self.bilstm(x)  # [B, T, 2H]
        attn = torch.softmax(self.attention(x), dim=1)  # [B, T, 1]
        embed = (attn * x).sum(dim=1)  # [B, 2H]

        logits = self.classifier(embed)  # [B, num_classes]
        return logits, embed


def get_model(num_classes: int = 5,
              n_mfcc: int = 10,
              cnn_out_channels: int = 64,
              lstm_hidden: int = 128,
              lstm_layers: int = 2,
              dropout: float = 0.5) -> SERBaselineModel:
    """Return a SERBaselineModel instance. input_dim = n_mfcc + 2."""
    return SERBaselineModel(
        input_dim=n_mfcc + 2,
        num_classes=num_classes,
        cnn_out_channels=cnn_out_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )
