import os
from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.audio_utils import load_and_preprocess, pad_or_trim
from utils.config import (
    EMOTION_LABEL_MAP,
    DEFAULT_SR, DEFAULT_MAX_LENGTH,
    DEFAULT_N_MELS, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_F_MIN,
    SSL_SR,
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
    """统一的语音情感识别数据集，支持多种输出格式。
    
    支持三种模式：
    1. 'spectrogram': 返回频谱图特征（mel/mfcc），用于 CNN 模型
    2. 'cnn_bilstm': 返回 MFCC+谱特征序列，用于 CNN-BiLSTM 模型
    3. 'waveform': 返回原始波形，用于 SSL 模型（HuBERT/Wav2Vec2）
    
    Args:
        dataset_dir: 数据集根目录
        mode: 输出模式 'spectrogram' | 'cnn_bilstm' | 'waveform'
        target_sr: 目标采样率
        max_length: 最大采样点数
        normalize: 是否对波形归一化
        random_offset: 过长时随机起点截取（训练集用 True）
        preload: 是否将所有数据预加载到CPU内存中
        cache_features: 是否缓存已计算的特征（按需模式优化）
        show_progress: 是否显示预加载进度条
        
        # Spectrogram 模式参数
        feature_type: 特征类型 'mel_spectrogram' | 'mfcc' | 'multi'
        n_mels: 梅尔频带数
        n_fft: FFT 窗口大小
        hop_length: 帧移
        n_mfcc: MFCC 系数数量
        augmenter: 数据增强器
        return_waveform: 是否返回波形而非特征
        
        # CNN-BiLSTM 模式参数
        win_length: 窗口长度
        max_frames: 特征时间帧数
    """

    def __init__(self,
                 dataset_dir: str,
                 mode: Literal['spectrogram', 'cnn_bilstm', 'waveform'] = 'spectrogram',
                 target_sr: int = DEFAULT_SR,
                 max_length: int = DEFAULT_MAX_LENGTH,
                 normalize: bool = True,
                 random_offset: bool = False,
                 preload: bool = False,
                 cache_features: bool = True,  # 按需模式下启用特征缓存
                 show_progress: bool = True,
                 num_workers: int = 0,  # 预加载时使用的进程数，0表示自动检测
                 # Spectrogram 模式参数
                 feature_type: Literal['mel_spectrogram', 'mfcc', 'multi'] = 'mel_spectrogram',
                 n_mels: int = DEFAULT_N_MELS,
                 n_fft: int = DEFAULT_N_FFT,
                 hop_length: int = DEFAULT_HOP_LENGTH,
                 n_mfcc: int = 40,
                 augmenter=None,
                 return_waveform: bool = False,
                 # CNN-BiLSTM 模式参数
                 win_length: int = 2048,
                 max_frames: int = 300):

        self.dataset_dir = dataset_dir
        self.mode = mode
        self.target_sr = target_sr
        self.max_length = max_length
        self.normalize = normalize
        self.random_offset = random_offset
        self.preload = preload
        self.cache_features = cache_features and not preload  # 只在按需模式下缓存
        # 使用线程池数量：I/O密集型任务，线程更合适
        self.num_workers = num_workers if num_workers > 0 else min(8, (multiprocessing.cpu_count() or 1) + 4)
        self._resamplers: Dict[Tuple[int, int], T.Resample] = {}
        self._feature_cache: Dict[int, torch.Tensor] = {}  # 特征缓存

        # Spectrogram 模式
        self.feature_type = feature_type
        self.return_waveform = return_waveform
        self.augmenter = augmenter
        
        # CNN-BiLSTM 模式
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_frames = max_frames

        # 初始化特征提取器
        if mode == 'spectrogram':
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
        elif mode == 'cnn_bilstm':
            self.mfcc_transform = T.MFCC(
                sample_rate=target_sr,
                n_mfcc=n_mfcc,
                melkwargs={'n_fft': n_fft, 'hop_length': hop_length},
            )

        # 扫描文件
        self.file_list, labels, self.emotion_list = collect_wav_files(dataset_dir)
        if not self.file_list:
            raise FileNotFoundError(f"在 {dataset_dir} 下未找到任何 WAV 文件")

        self.labels = np.array(labels)
        self._print_stats()
        
        # 预加载数据
        self.preloaded_data = None
        self.failed_indices = []
        if self.preload:
            self._preload_all_data(show_progress)

    def _print_stats(self):
        counts = Counter(self.emotion_list)
        if self.preload:
            mode_str = "预加载模式"
        else:
            cache_str = "+特征缓存" if self.cache_features else ""
            mode_str = f"按需加载模式{cache_str}"
        print(f"[EmotionDataset] {len(self.file_list)} 个文件 | "
              f"模式: {self.mode} | {mode_str} | 目录: {self.dataset_dir}")
        for emotion, n in sorted(counts.items()):
            print(f"  {emotion}: {n}")
    
    def _preload_all_data(self, show_progress: bool = True):
        """预加载所有数据到内存（CPU内存）中，使用线程池加速。
        
        使用线程池而非多进程，原因：
        1. I/O密集型任务（读取音频文件），线程池更高效
        2. 避免 Linux/Windows 服务器上 PyTorch + multiprocessing 的兼容性与卡死问题
        3. 无需序列化数据，更稳定
        """
        print(f"\n[EmotionDataset] 开始预加载 {len(self.file_list)} 个文件到内存...")
        print(f"[EmotionDataset] 使用 {self.num_workers} 个线程并行加载")
        
        # 限制 PyTorch 内部线程数，防止多线程预加载时发生 CPU 资源过度竞争而卡死
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        
        try:
            self.preloaded_data = [None] * len(self.file_list)
            
            def load_single(idx):
                """加载单个文件（线程安全，带超时保护）"""
                try:
                    wav, ok = load_and_preprocess(
                        self.file_list[idx],
                        target_sr=self.target_sr,
                        max_length=self.max_length,
                        normalize=self.normalize,
                        center_crop=True,  # 预加载时使用中心裁剪，更稳定
                        random_offset=False,
                        resampler_cache=None,  # 每个线程独立
                    )
                    return idx, wav, ok, None
                except Exception as e:
                    # 捕获任何异常，避免整个加载过程卡死
                    print(f"\n[警告] 文件 {self.file_list[idx]} 加载失败: {str(e)}")
                    # 返回零张量作为占位
                    dummy_wav = torch.zeros(1, self.max_length, dtype=torch.float32)
                    return idx, dummy_wav, False, str(e)
            
            # 使用线程池并行加载
            if self.num_workers > 1:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # 提交所有任务，设置超时
                    futures = {executor.submit(load_single, idx): idx 
                              for idx in range(len(self.file_list))}
                    
                    # 收集结果，添加超时保护
                    if show_progress:
                        with tqdm(total=len(self.file_list), desc="加载数据") as pbar:
                            for future in as_completed(futures, timeout=None):
                                try:
                                    idx, wav, ok, error = future.result(timeout=30)  # 单个文件最多30秒
                                    if not ok:
                                        self.failed_indices.append(idx)
                                        if error and len(self.failed_indices) <= 5:  # 只打印前5个错误
                                            print(f"\n[错误] 索引 {idx}: {error}")
                                    self.preloaded_data[idx] = wav
                                    pbar.update(1)
                                except TimeoutError:
                                    idx = futures[future]
                                    print(f"\n[超时] 文件 {self.file_list[idx]} 加载超时（>30秒），跳过")
                                    self.failed_indices.append(idx)
                                    self.preloaded_data[idx] = torch.zeros(1, self.max_length, dtype=torch.float32)
                                    pbar.update(1)
                                except Exception as e:
                                    idx = futures[future]
                                    print(f"\n[异常] 索引 {idx} 处理失败: {str(e)}")
                                    self.failed_indices.append(idx)
                                    self.preloaded_data[idx] = torch.zeros(1, self.max_length, dtype=torch.float32)
                                    pbar.update(1)
                    else:
                        for future in as_completed(futures, timeout=None):
                            try:
                                idx, wav, ok, error = future.result(timeout=30)
                                if not ok:
                                    self.failed_indices.append(idx)
                                self.preloaded_data[idx] = wav
                            except (TimeoutError, Exception) as e:
                                idx = futures[future]
                                self.failed_indices.append(idx)
                                self.preloaded_data[idx] = torch.zeros(1, self.max_length, dtype=torch.float32)
            else:
                # 单线程模式
                iterator = tqdm(range(len(self.file_list)), desc="加载数据") if show_progress else range(len(self.file_list))
                for idx in iterator:
                    idx, wav, ok, error = load_single(idx)
                    if not ok:
                        self.failed_indices.append(idx)
                    self.preloaded_data[idx] = wav
        finally:
            torch.set_num_threads(old_threads)
        
        # 根据模式处理数据
        if self.mode == 'cnn_bilstm':
            # 需要提取 MFCC+谱特征
            print("[EmotionDataset] 提取 MFCC 特征...")
            from models.cnn_bilstm import extract_baseline_feature
            
            processed_data = []
            iterator = tqdm(range(len(self.preloaded_data)), desc="提取特征") if show_progress else range(len(self.preloaded_data))
            
            for idx in iterator:
                wav = self.preloaded_data[idx]
                try:
                    feat = extract_baseline_feature(
                        wav,
                        sample_rate=self.target_sr,
                        n_mfcc=self.n_mfcc,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        max_len=self.max_frames,
                        mfcc_transform=self.mfcc_transform,
                    )
                    processed_data.append(feat)
                except Exception as e:
                    print(f"\n[警告] 索引 {idx} 特征提取失败: {str(e)}")
                    # 使用零特征占位
                    dummy_feat = torch.zeros((self.max_frames, self.n_mfcc + 2), dtype=torch.float32)
                    processed_data.append(dummy_feat)
                    if idx not in self.failed_indices:
                        self.failed_indices.append(idx)
            
            self.preloaded_data = processed_data
        
        # 统计信息
        success_count = len(self.file_list) - len(self.failed_indices)
        fail_rate = len(self.failed_indices) / len(self.file_list) * 100 if self.file_list else 0
        
        print(f"[EmotionDataset] 预加载完成！")
        print(f"  成功: {success_count}/{len(self.file_list)} ({100-fail_rate:.2f}%)")
        if self.failed_indices:
            print(f"  失败: {len(self.failed_indices)} 个文件")
            if len(self.failed_indices) <= 10:
                print(f"  失败文件列表:")
                for idx in self.failed_indices:
                    print(f"    - {self.file_list[idx]}")
        
        # 计算内存占用
        total_bytes = sum(d.element_size() * d.nelement() for d in self.preloaded_data if d is not None)
        memory_mb = total_bytes / (1024 ** 2)
        print(f"  内存占用: {memory_mb:.2f} MB")

    def __len__(self) -> int:
        return len(self.file_list)

    def get_labels(self) -> np.ndarray:
        return self.labels
    
    def get_failed_files(self) -> List[str]:
        """返回加载失败的文件路径列表。"""
        return [self.file_list[idx] for idx in self.failed_indices]

    def _load_waveform(self, idx: int) -> Tuple[torch.Tensor, int]:
        """加载波形数据。"""
        if self.preload and self.preloaded_data is not None:
            data = self.preloaded_data[idx].clone()
            label = int(self.labels[idx])
            return data, label
        else:
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
        """波形 → 频谱图特征（spectrogram 模式）。支持单条波形与批量波形特征提取。"""
        device = waveform.device
        if hasattr(self, '_current_device') and self._current_device == device:
            pass
        else:
            self.mel_transform = self.mel_transform.to(device)
            self.mfcc_transform = self.mfcc_transform.to(device)
            self._current_device = device

        if self.feature_type == 'mfcc':
            return self.mfcc_transform(waveform)

        if self.feature_type == 'multi':
            return {
                'mel_spectrogram': torch.log(self.mel_transform(waveform) + 1e-9),
                'mfcc': self.mfcc_transform(waveform),
            }

        return torch.log(self.mel_transform(waveform) + 1e-9)

    def extract_features_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """批量特征提取（spectrogram 模式）"""
        return self.extract_features(waveforms)
    
    def __getitem__(self, idx: int):
        """根据模式返回不同格式的数据。"""
        if self.mode == 'waveform':
            # SSL 模式：返回原始波形
            waveform, label = self._load_waveform(idx)
            return waveform, label
        
        elif self.mode == 'cnn_bilstm':
            # CNN-BiLSTM 模式：返回特征序列
            if self.preload and self.preloaded_data is not None:
                # 预加载时已经提取了特征
                feat = self.preloaded_data[idx].clone()
                label = int(self.labels[idx])
            else:
                # 按需模式：先检查缓存
                if self.cache_features and idx in self._feature_cache:
                    feat = self._feature_cache[idx].clone()
                    label = int(self.labels[idx])
                else:
                    # 实时提取特征
                    from models.cnn_bilstm import extract_baseline_feature
                    wav, label = self._load_waveform(idx)
                    feat = extract_baseline_feature(
                        wav,
                        sample_rate=self.target_sr,
                        n_mfcc=self.n_mfcc,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        max_len=self.max_frames,
                        mfcc_transform=self.mfcc_transform,
                    )
                    # 缓存特征（训练时不缓存，避免随机性问题）
                    if self.cache_features and not self.random_offset:
                        self._feature_cache[idx] = feat.clone()
            return feat, label
        
        else:  # spectrogram 模式
            # 如果启用缓存且没有数据增强，先检查缓存
            if (self.cache_features and not self.preload and 
                self.augmenter is None and not self.return_waveform and 
                idx in self._feature_cache):
                return self._feature_cache[idx].clone(), int(self.labels[idx])
            
            waveform, label = self._load_waveform(idx)

            if self.return_waveform:
                return waveform, label

            # 应用增强
            if self.augmenter is not None:
                waveform = self.augmenter.apply_wave_augmentations(waveform)
                waveform = pad_or_trim(waveform, self.max_length)

            features = self.extract_features(waveform)
            
            # 缓存特征（仅在无增强且非训练模式时）
            if (self.cache_features and not self.preload and 
                self.augmenter is None and not self.random_offset):
                self._feature_cache[idx] = features.clone()
            
            return features, label
