import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import librosa

from utils.config import (
    DEFAULT_SR, DEFAULT_MAX_LENGTH,
)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """多声道转单声道，输出形状 [1, T]。"""
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int,
             cache: Optional[dict] = None) -> torch.Tensor:
    """重采样波形"""
    if orig_sr == target_sr:
        return waveform
    device = waveform.device
    y = waveform.cpu().numpy()
    y_res = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    return torch.from_numpy(y_res).to(device)


def pad_or_trim(waveform: torch.Tensor, max_length: int,
                center: bool = True,
                random_offset: bool = False) -> torch.Tensor:
    """裁剪或右填零到 max_length 采样点。

    Args:
        center:        过长时居中截取
        random_offset: 过长时随机起点
    """
    squeeze = waveform.dim() == 1
    if squeeze:
        waveform = waveform.unsqueeze(0)

    L = waveform.shape[1]
    if L > max_length:
        if random_offset:
            start = random.randint(0, L - max_length)
        elif center:
            start = (L - max_length) // 2
        else:
            start = 0
        waveform = waveform[:, start: start + max_length]
    elif L < max_length:
        waveform = F.pad(waveform, (0, max_length - L))

    return waveform.squeeze(0) if squeeze else waveform


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Z-score 归一化，std < eps 时只去均值。"""
    mean = waveform.mean()
    std = waveform.std()
    return (waveform - mean) / std if std >= eps else waveform - mean


def preprocess_waveform(waveform: torch.Tensor,
                        orig_sr: int,
                        target_sr: int = DEFAULT_SR,
                        max_length: int = DEFAULT_MAX_LENGTH,
                        normalize: bool = True,
                        center_crop: bool = True,
                        random_offset: bool = False,
                        resampler_cache: Optional[dict] = None) -> torch.Tensor:
    """单声道 → 重采样 → 定长 → 归一化，返回 [1, max_length]。"""
    wav = to_mono(waveform)
    wav = resample(wav, orig_sr, target_sr, cache=resampler_cache)
    wav = pad_or_trim(wav, max_length, center=center_crop, random_offset=random_offset)
    if normalize:
        wav = normalize_waveform(wav)
    return wav


def load_and_preprocess(file_path: str,
                        target_sr: int = DEFAULT_SR,
                        max_length: int = DEFAULT_MAX_LENGTH,
                        normalize: bool = True,
                        center_crop: bool = True,
                        random_offset: bool = False,
                        resampler_cache: Optional[dict] = None
                        ) -> Tuple[torch.Tensor, bool]:
    """从文件加载并预处理音频。失败时返回 (零张量, False)。"""
    try:
        # 使用 librosa 加载，直接在加载时实现重采样和转单声道
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        wav = torch.from_numpy(y).unsqueeze(0)  # Shape [1, T]
        wav = pad_or_trim(wav, max_length, center=center_crop, random_offset=random_offset)
        if normalize:
            wav = normalize_waveform(wav)
        return wav, True
    except Exception as e:
        print(f"[audio_utils] 加载失败 {file_path}: {e}")
        return torch.zeros(1, max_length), False
