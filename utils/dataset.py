import os
from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

from utils.audio_utils import load_and_preprocess, pad_or_trim
from utils.config import (
    EMOTION_LABEL_MAP,
    DEFAULT_SR, DEFAULT_MAX_LENGTH,
    DEFAULT_N_MELS, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_F_MIN,
)


def collect_wav_files(root_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """扫描 root_dir/{emotion}/*.wav，返回 (file_list, labels, emotion_list)。

    跳过不在 EMOTION_LABEL_MAP 中的子目录，不抛异常。
    """
    file_list, labels, emotion_list = [], [], []
    for emotion, idx in EMOTION_LABEL_MAP.items():
        emotion_dir = os.path.join(root_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        for fname in sorted(os.listdir(emotion_dir)):
            if fname.lower().endswith('.wav'):
                file_list.append(os.path.join(emotion_dir, fname))
                labels.append(idx)
                emotion_list.append(emotion)
    return file_list, labels, emotion_list


class EmotionDataset(Dataset):
    """语音情感识别数据集。

    职责：
      - 扫描目录、构建文件列表
      - 加载并预处理波形（重采样 → 单声道 → 定长 → 归一化）
      - 提取音频特征（mel / mfcc / multi）
      - 将外部 augmenter 的增强调用代理给 apply_augmentation

    不含任何增强实现；增强逻辑由 augmenter 参数注入。

    Args:
        dataset_dir:   数据集根目录（必填）
        target_sr:     目标采样率
        max_length:    最大采样点数
        n_mels:        梅尔频带数
        n_fft:         FFT 窗口大小
        hop_length:    帧移
        feature_type:  特征类型：'mel_spectrogram' | 'mfcc' | 'multi'
        n_mfcc:        MFCC 系数数量
        normalize:     是否对波形 Z-score 归一化
        random_offset: 过长时随机起点截取（训练集用 True）
        return_waveform: True 时 __getitem__ 直接返回波形，跳过特征提取
        augmenter:     可选的 AudioAugmentation 实例，不为 None 且
                       return_waveform=False 时在特征提取前对波形增强
    """

    def __init__(self,
                 dataset_dir: str,
                 target_sr: int = DEFAULT_SR,
                 max_length: int = DEFAULT_MAX_LENGTH,
                 n_mels: int = DEFAULT_N_MELS,
                 n_fft: int = DEFAULT_N_FFT,
                 hop_length: int = DEFAULT_HOP_LENGTH,
                 feature_type: Literal['mel_spectrogram', 'mfcc', 'multi'] = 'mel_spectrogram',
                 n_mfcc: int = 40,
                 normalize: bool = True,
                 random_offset: bool = False,
                 return_waveform: bool = False,
                 augmenter=None):

        self.dataset_dir = dataset_dir
        self.target_sr = target_sr
        self.max_length = max_length
        self.feature_type = feature_type
        self.normalize = normalize
        self.random_offset = random_offset
        self.return_waveform = return_waveform
        self.augmenter = augmenter
        self._resamplers: Dict[Tuple[int, int], T.Resample] = {}

        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=DEFAULT_F_MIN, f_max=target_sr // 2,
        )
        self.mfcc_transform = T.MFCC(
            sample_rate=target_sr, n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft, 'hop_length': hop_length,
                'n_mels': n_mels, 'f_min': DEFAULT_F_MIN, 'f_max': target_sr // 2,
            },
        )

        self.file_list, labels, self.emotion_list = collect_wav_files(dataset_dir)
        if not self.file_list:
            raise FileNotFoundError(f"在 {dataset_dir} 下未找到任何 WAV 文件")

        self.labels = np.array(labels)
        self._print_stats()

    def _print_stats(self):
        counts = Counter(self.emotion_list)
        print(f"[EmotionDataset] {len(self.file_list)} 个文件 | "
              f"{len(counts)} 个类别 | 目录: {self.dataset_dir}")
        for emotion, n in sorted(counts.items()):
            print(f"  {emotion}: {n}")

    def __len__(self) -> int:
        return len(self.file_list)

    def get_labels(self) -> np.ndarray:
        return self.labels

    def _load_waveform(self, idx: int) -> Tuple[torch.Tensor, int]:
        """加载并预处理第 idx 条音频，返回 (waveform [1, T], label)。"""
        wav, ok = load_and_preprocess(
            self.file_list[idx],
            target_sr=self.target_sr,
            max_length=self.max_length,
            normalize=self.normalize,
            center_crop=not self.random_offset,
            random_offset=self.random_offset,
            resampler_cache=self._resamplers,
        )
        return wav, int(self.labels[idx])

    def extract_features(self, waveform: torch.Tensor
                         ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """波形 → 特征张量（或字典）。

        特征提取器自动跟随 waveform 的设备。
        """
        device = waveform.device
        self.mel_transform = self.mel_transform.to(device)
        self.mfcc_transform = self.mfcc_transform.to(device)

        if self.feature_type == 'mfcc':
            return self.mfcc_transform(waveform)

        if self.feature_type == 'multi':
            return {
                'mel_spectrogram': torch.log(self.mel_transform(waveform) + 1e-9),
                'mfcc': self.mfcc_transform(waveform),
            }

        return torch.log(self.mel_transform(waveform) + 1e-9)

    def extract_features_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """批量特征提取，返回 [B, ...]。仅支持 mel / mfcc（非 multi）。"""
        return torch.cat(
            [self.extract_features(waveforms[i:i + 1]) for i in range(waveforms.shape[0])],
            dim=0,
        )

    def __getitem__(self, idx: int):
        waveform, label = self._load_waveform(idx)

        if self.return_waveform:
            return waveform, label

        if self.augmenter is not None:
            waveform = self.augmenter.apply_wave_augmentations(waveform)
            waveform = pad_or_trim(waveform, self.max_length)

        return self.extract_features(waveform), label
