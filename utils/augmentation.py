import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import TensorDataset, DataLoader

from utils.model_utils import batch_extract_features


class EnhancedDataset(TensorDataset):
    """SMOTE 过采样后包装的数据集，保留原始 dataset 的特征提取接口。"""

    def __init__(self, tensors, orig_ds=None):
        super().__init__(*tensors)
        self._orig = orig_ds

    def extract_features(self, waveform):
        if self._orig is not None and hasattr(self._orig, 'extract_features'):
            return self._orig.extract_features(waveform)
        return waveform

    def extract_features_batch(self, waveforms):
        if self._orig is not None and hasattr(self._orig, 'extract_features_batch'):
            return self._orig.extract_features_batch(waveforms)
        return batch_extract_features(waveforms, self)


class AudioAugmentation:
    """波形域与频谱域数据增强。
    wave 增强入口：apply_wave_augmentations(waveform)
    spec 增强入口：apply_spec_augmentations(specs)
    """

    def __init__(self, sample_rate: int = 32000):
        self.sample_rate = sample_rate

    # 波形增强
    def random_speed_change(self, waveform: torch.Tensor,
                            speed_factor_range: tuple = (0.9, 1.1)) -> torch.Tensor:
        """随机改变语音速度（不改变音高），使用 sox tempo 效果。"""
        speed_factor = random.uniform(*speed_factor_range)
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform.cpu(), self.sample_rate, [["tempo", str(speed_factor)]]
            )
            return augmented.to(waveform.device)
        except Exception as e:
            print(f"[augmentation] random_speed_change 失败: {e}")
            return waveform

    def dynamic_range_compression(self, waveform: torch.Tensor,
                                  threshold_db: float = -20,
                                  ratio: float = 4) -> torch.Tensor:
        """动态范围压缩。"""
        eps = 1e-10
        magnitude = torch.abs(waveform)
        db = 20 * torch.log10(magnitude + eps)
        mask = db > threshold_db
        db_c = db.clone()
        db_c[mask] = threshold_db + (db[mask] - threshold_db) / ratio
        mag_c = 10 ** (db_c / 20)
        phase = waveform / (magnitude + eps)
        return mag_c * phase

    def time_shift(self, waveform: torch.Tensor, shift_limit: float = 0.3) -> torch.Tensor:
        if random.random() < 0.5:
            return waveform
        L = waveform.shape[-1]
        max_shift = min(int(shift_limit * L), L // 2)
        if max_shift <= 0:
            return waveform
        shift = random.randint(1, max_shift) * (1 if random.random() > 0.5 else -1)
        result = torch.zeros_like(waveform)
        if shift > 0:
            result[..., shift:] = waveform[..., :-shift]
        else:
            s = abs(shift)
            result[..., :-s] = waveform[..., s:]
        return result

    def add_noise(self, waveform: torch.Tensor,
                  noise_factor_range: tuple = (0.001, 0.02)) -> torch.Tensor:
        if random.random() < 0.5:
            return waveform
        factor = random.uniform(*noise_factor_range)
        return waveform + torch.randn_like(waveform) * factor


    def add_colored_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.7:
            return waveform
        noise_type = random.choice(['pink', 'brown', 'blue'])
        noise_factor = random.uniform(0.001, 0.015)
        white = torch.randn_like(waveform)
        cpu_noise = white.cpu()
        time_len = waveform.shape[-1]
        try:
            noise_fft = torch.fft.rfft(cpu_noise, dim=-1)
            freq = torch.fft.rfftfreq(time_len, d=1 / self.sample_rate)
            if noise_type == 'pink':
                filt = 1 / torch.sqrt(freq + 1e-8)
            elif noise_type == 'brown':
                filt = 1 / (freq + 1e-8)
            else:
                filt = torch.sqrt(freq + 1e-8)
            filt = filt.unsqueeze(0).repeat(cpu_noise.shape[0], 1)
            colored = torch.fft.irfft(noise_fft * filt, n=time_len, dim=-1)
            colored = colored / torch.std(colored, dim=-1, keepdim=True) * noise_factor
            return waveform + colored.to(waveform.device)
        except Exception as e:
            print(f"[augmentation] add_colored_noise 失败: {e}")
            return waveform

    def time_stretch(self, waveform: torch.Tensor,
                     rate_range: tuple = (0.75, 1.25)) -> torch.Tensor:
        if random.random() < 0.5 or waveform.dim() != 2:
            return waveform
        rate = random.uniform(*rate_range)
        device = waveform.device
        try:
            spec = T.Spectrogram()(waveform.cpu())
            stretched = T.TimeStretch(hop_length=256, fixed_rate=rate)(spec)
            result = T.GriffinLim(n_fft=400, hop_length=256)(stretched)
            return result.to(device)
        except Exception:
            return waveform

    def pitch_shift(self, waveform: torch.Tensor,
                    n_steps_range: tuple = (-3, 3)) -> torch.Tensor:
        if random.random() < 0.5 or waveform.dim() != 2:
            return waveform
        n_steps = random.uniform(*n_steps_range)
        device = waveform.device
        try:
            return T.PitchShift(sample_rate=self.sample_rate,
                                n_steps=n_steps)(waveform.cpu()).to(device)
        except Exception:
            return waveform

    def adjust_volume(self, waveform: torch.Tensor,
                      gain_db_range: tuple = (-6, 6)) -> torch.Tensor:
        if random.random() < 0.5:
            return waveform
        gain_db = random.uniform(*gain_db_range)
        return waveform * (10 ** (gain_db / 20))

    def add_reverb(self, waveform: torch.Tensor,
                   reverb_params=None) -> torch.Tensor:
        if random.random() < 0.7:
            return waveform
        if reverb_params is None:
            room_size = random.uniform(0.15, 0.8)
            damping = random.uniform(0.1, 0.6)
            wet_level = random.uniform(0.1, 0.3)
            dry_level = 1.0 - wet_level
        else:
            room_size, damping, wet_level, dry_level = reverb_params
        try:
            delay = min(int(room_size * self.sample_rate * 0.005), 2000)
            if delay <= 0:
                return waveform
            result = waveform * dry_level
            if delay < waveform.shape[-1]:
                result = result.clone()
                result[..., delay:] += waveform[..., :-delay] * wet_level * damping
            return result
        except Exception as e:
            print(f"[augmentation] add_reverb 失败: {e}")
            return waveform


    def random_crop(self, waveform: torch.Tensor,
                    crop_size_range: tuple = (0.8, 1.0)) -> torch.Tensor:
        if random.random() < 0.5:
            return waveform
        length = waveform.shape[-1]
        crop_size = int(length * random.uniform(*crop_size_range))
        if crop_size >= length:
            return waveform
        start = random.randint(0, length - crop_size)
        cropped = waveform[..., start:start + crop_size]
        try:
            result = torch.zeros_like(waveform)
            result[..., :crop_size] = cropped
            for i in range(1, length // crop_size + 1):
                end_idx = min((i + 1) * crop_size, length)
                copy_sz = end_idx - i * crop_size
                if copy_sz > 0:
                    result[..., i * crop_size:end_idx] = cropped[..., :copy_sz]
            return result
        except Exception as e:
            print(f"[augmentation] random_crop 失败: {e}")
            return waveform

    def apply_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.7:
            return waveform
        filter_type = random.choice(['lowpass', 'highpass', 'bandpass'])
        device = waveform.device
        try:
            wf_cpu = waveform.cpu()
            wf_f = torch.fft.rfft(wf_cpu, dim=-1)
            n_freq = wf_f.shape[-1]
            freq_range = torch.linspace(0, 1, n_freq)
            if filter_type == 'lowpass':
                cutoff = random.uniform(0.1, 0.5)
                mask = (freq_range <= cutoff).float()
            elif filter_type == 'highpass':
                cutoff = random.uniform(0.05, 0.3)
                mask = (freq_range >= cutoff).float()
            else:
                low = random.uniform(0.05, 0.3)
                high = random.uniform(low + 0.1, 0.7)
                mask = ((freq_range >= low) & (freq_range <= high)).float()
            ws = int(n_freq * 0.05)
            if ws > 1:
                smoothed = F.avg_pool1d(mask.view(1, 1, -1), ws, 1, ws // 2).squeeze()
                mask = smoothed[:mask.shape[0]]
            mask = mask.to(wf_f.dtype)
            for c in range(wf_f.shape[0]):
                wf_f[c] = wf_f[c] * mask
            return torch.fft.irfft(wf_f, n=wf_cpu.shape[-1], dim=-1).to(device)
        except Exception as e:
            print(f"[augmentation] apply_filter 失败: {e}")
            return waveform

    # 频谱增强
    def spec_augment(self, spec: torch.Tensor,
                     freq_mask_param: int = 20,
                     time_mask_param: int = 20,
                     num_masks: int = 3) -> torch.Tensor:
        if random.random() < 0.5:
            return spec
        result = spec.clone()
        # 支持 [B, C, F, T] 和 [C, F, T]
        if result.dim() == 4:
            for b in range(result.shape[0]):
                for _ in range(num_masks):
                    fw = random.randint(1, min(freq_mask_param, result.shape[2] - 1))
                    fs = random.randint(0, result.shape[2] - fw)
                    result[b, :, fs:fs + fw, :] = 0
                for _ in range(num_masks):
                    tw = random.randint(1, min(time_mask_param, result.shape[3] - 1))
                    ts = random.randint(0, result.shape[3] - tw)
                    result[b, :, :, ts:ts + tw] = 0
        elif result.dim() == 3:
            for _ in range(num_masks):
                fw = random.randint(1, min(freq_mask_param, result.shape[1] - 1))
                fs = random.randint(0, result.shape[1] - fw)
                result[:, fs:fs + fw, :] = 0
            for _ in range(num_masks):
                tw = random.randint(1, min(time_mask_param, result.shape[2] - 1))
                ts = random.randint(0, result.shape[2] - tw)
                result[:, :, ts:ts + tw] = 0
        return result

    def apply_spec_augmentations(self, specs: torch.Tensor,
                                 prob: float = 0.6) -> torch.Tensor:
        """对 batch 频谱随机应用 SpecAugment。"""
        mask = torch.rand(specs.shape[0], device=specs.device) < prob
        for i in range(specs.shape[0]):
            if mask[i]:
                specs[i] = self.spec_augment(specs[i])
        return specs


    # Mixup
    def mixup(self, batch_x: torch.Tensor, batch_y: torch.Tensor,
              alpha: float = 0.3):
        """返回 (mixed_x, y_a, y_b, lam)。"""
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        idx = torch.randperm(batch_x.size(0), device=batch_x.device)
        mixed_x = lam * batch_x + (1 - lam) * batch_x[idx]
        return mixed_x, batch_y, batch_y[idx], lam
    
    def apply_wave_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """对单条波形应用基础增强 + 随机高级增强。"""
        aug = self.time_shift(waveform)
        aug = self.add_noise(aug)
        aug = self.adjust_volume(aug)

        advanced = random.sample(
            ['time_stretch', 'pitch_shift', 'add_colored_noise',
             'add_reverb', 'random_crop', 'apply_filter'],
            random.randint(1, 3),
        )
        try:
            if 'time_stretch'    in advanced: aug = self.time_stretch(aug)
            if 'pitch_shift'     in advanced: aug = self.pitch_shift(aug)
            if 'add_colored_noise' in advanced: aug = self.add_colored_noise(aug)
            if 'add_reverb'      in advanced: aug = self.add_reverb(aug)
            if 'random_crop'     in advanced: aug = self.random_crop(aug)
            if 'apply_filter'    in advanced: aug = self.apply_filter(aug)
        except Exception as e:
            print(f"[augmentation] 高级增强错误: {e}")
        return aug

    def apply_augmentations_to_waveforms(self, waveforms: torch.Tensor,
                                         targets=None,
                                         apply_mixup: bool = False,
                                         mixup_alpha: float = 0.3):
        """对 batch 波形逐条增强，可选 mixup。

        返回：
          有 targets + mixup  → (mixed_x, y_a, y_b, lam)
          有 targets，无 mixup → (waveforms, targets, targets, 1.0)
          无 targets          → waveforms
        """
        augmented = []
        for i in range(waveforms.shape[0]):
            try:
                augmented.append(self.apply_wave_augmentations(waveforms[i:i + 1]))
            except Exception as e:
                print(f"[augmentation] 波形增强错误: {e}")
                augmented.append(waveforms[i:i + 1])
        try:
            waveforms = torch.cat(augmented, dim=0)
        except Exception as e:
            print(f"[augmentation] 合并增强波形错误: {e}")

        if apply_mixup and targets is not None:
            try:
                return self.mixup(waveforms, targets, mixup_alpha)
            except Exception as e:
                print(f"[augmentation] Mixup 错误: {e}")
                return waveforms, targets, targets, 1.0

        if targets is not None:
            return waveforms, targets, targets, 1.0
        return waveforms


def configure_augmentation(augmenter: AudioAugmentation, intensity: str):
    """根据强度（'light' | 'medium' | 'heavy'）重新绑定各方法参数。"""
    configs = {
        'light': dict(time_shift=dict(shift_limit=0.2),
                      add_noise=dict(noise_factor_range=(0.001, 0.01)),
                      time_stretch=dict(rate_range=(0.85, 1.15)),
                      pitch_shift=dict(n_steps_range=(-2, 2)),
                      adjust_volume=dict(gain_db_range=(-3, 3)),
                      spec_augment=dict(freq_mask_param=10, time_mask_param=10, num_masks=2),
                      spec_prob=0.4),
        'medium': dict(time_shift=dict(shift_limit=0.3),
                       add_noise=dict(noise_factor_range=(0.001, 0.02)),
                       time_stretch=dict(rate_range=(0.75, 1.25)),
                       pitch_shift=dict(n_steps_range=(-3, 3)),
                       adjust_volume=dict(gain_db_range=(-6, 6)),
                       spec_augment=dict(freq_mask_param=20, time_mask_param=20, num_masks=3),
                       spec_prob=0.5),
        'heavy': dict(time_shift=dict(shift_limit=0.4),
                      add_noise=dict(noise_factor_range=(0.002, 0.03)),
                      time_stretch=dict(rate_range=(0.7, 1.3)),
                      pitch_shift=dict(n_steps_range=(-4, 4)),
                      adjust_volume=dict(gain_db_range=(-8, 8)),
                      spec_augment=dict(freq_mask_param=30, time_mask_param=30, num_masks=4),
                      spec_prob=0.6),
    }
    cfg = configs.get(intensity, configs['medium'])

    import functools
    augmenter.time_shift = functools.partial(AudioAugmentation.time_shift, augmenter, **cfg['time_shift'])
    augmenter.add_noise = functools.partial(AudioAugmentation.add_noise, augmenter, **cfg['add_noise'])
    augmenter.time_stretch = functools.partial(AudioAugmentation.time_stretch, augmenter, **cfg['time_stretch'])
    augmenter.pitch_shift = functools.partial(AudioAugmentation.pitch_shift, augmenter, **cfg['pitch_shift'])
    augmenter.adjust_volume = functools.partial(AudioAugmentation.adjust_volume, augmenter, **cfg['adjust_volume'])
    augmenter.spec_augment = functools.partial(AudioAugmentation.spec_augment, augmenter, **cfg['spec_augment'])

    augmenter._spec_prob = cfg['spec_prob']
    augmenter.apply_spec_augmentations = functools.partial(
        _apply_spec_augmentations_with_prob, augmenter)

    print(f"[augmentation] 已配置 {intensity} 强度，spec 增强概率: {cfg['spec_prob']:.2f}")


def _apply_spec_augmentations_with_prob(augmenter: 'AudioAugmentation',
                                        specs: torch.Tensor) -> torch.Tensor:
    """模块级函数，可被 pickle，替代 configure_augmentation 内的局部闭包。"""
    prob = getattr(augmenter, '_spec_prob', 0.5)
    mask = torch.rand(specs.shape[0], device=specs.device) < prob
    for i in range(specs.shape[0]):
        if mask[i]:
            specs[i] = augmenter.spec_augment(specs[i])
    return specs


class AudioSMOTE:
    """对音频特征（波形或频谱）进行 SMOTE 过采样，用于缓解类别不平衡。"""

    def __init__(self, sampling_strategy: float = 0.5, k_neighbors: int = 5,
                 random_state=None, device='cpu'):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors

        if isinstance(device, str):
            if (device == 'cuda' or device.startswith('cuda:')) \
                    and not torch.cuda.is_available():
                print("[AudioSMOTE] CUDA 不可用，回退到 CPU")
                device = 'cpu'
        self.device = torch.device(device)

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)
        print(f"[AudioSMOTE] 初始化完成，设备: {self.device}")

    def _find_nn_index(self, X: torch.Tensor, sample: torch.Tensor, k: int):
        try:
            if X.device != sample.device:
                X = X.to(sample.device)
            distances = torch.norm(X - sample.unsqueeze(0), dim=1)
            _, indices = torch.topk(distances, k + 1, largest=False)
            return indices[1:k + 1]
        except Exception as e:
            print(f"[AudioSMOTE] 寻找最近邻失败: {e}，回退到 CPU")
            distances = torch.norm(X.cpu() - sample.cpu().unsqueeze(0), dim=1)
            _, indices = torch.topk(distances, k + 1, largest=False)
            return indices[1:k + 1]

    @staticmethod
    def _interpolate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        alpha = random.random()
        return a + alpha * (b - a)

    def _generate_synthetic(self, samples: torch.Tensor, n_samples: int,
                             log_domain: bool = False) -> torch.Tensor:
        orig_device = samples.device
        n_class = samples.shape[0]
        g_mean = samples.mean()
        g_min, g_max = samples.min(), samples.max()
        flat = samples.view(n_class, -1)
        synthetic = []

        for _ in range(n_samples):
            idx = random.randint(0, n_class - 1)
            sample = flat[idx]
            nn_idx = random.choice(
                self._find_nn_index(flat, sample, min(self.k_neighbors, n_class - 1))
            )
            s1, s2 = samples[idx], samples[nn_idx]
            if torch.isnan(s1).any() or torch.isnan(s2).any():
                continue
            if torch.isinf(s1).any() or torch.isinf(s2).any():
                continue

            if log_domain:
                eps = 1e-5
                s = torch.exp(self._interpolate(torch.log(s1.clamp(min=eps)),
                                                torch.log(s2.clamp(min=eps))))
                s = torch.clamp(s, 0.0, max(g_max.item() * 1.2, 10.0))
            else:
                s = self._interpolate(s1, s2)
                s = torch.clamp(s, min(g_min.item() * 1.2, -2.0),
                                max(g_max.item() * 1.2, 2.0))

            if torch.isnan(s).any() or torch.isinf(s).any():
                s = torch.nan_to_num(s, nan=g_mean.item(),
                                     posinf=g_max.item(), neginf=g_min.item())
            synthetic.append(s)

        if not synthetic:
            return torch.tensor([], device=orig_device)
        result = torch.stack(synthetic)
        return result.to(orig_device)

    def _resample(self, data: torch.Tensor, labels: torch.Tensor,
                  log_domain: bool = False):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        orig_device = data.device
        unique_labels, counts = torch.unique(labels, return_counts=True)
        majority = counts.max().item()

        all_data, all_labels = [], []
        for label in unique_labels:
            idx = torch.where(labels == label)[0]
            cls_data = data[idx]
            all_data.append(cls_data)
            all_labels.append(torch.full((len(idx),), label, device=orig_device))

            current = len(idx)
            if current < majority:
                target = majority if self.sampling_strategy >= 1.0 else \
                    max(current + 1, int(majority * self.sampling_strategy))
                n_new = target - current
                print(f"[AudioSMOTE] 类别 {label.item()}: {current} → {target}（+{n_new}）")
                synth = self._generate_synthetic(cls_data, n_new, log_domain=log_domain)
                if len(synth) > 0:
                    all_data.append(synth.to(orig_device))
                    all_labels.append(
                        torch.full((len(synth),), label, device=orig_device))

        try:
            return torch.cat(all_data), torch.cat(all_labels)
        except Exception as e:
            print(f"[AudioSMOTE] 合并结果失败: {e}，返回原始数据")
            return data, labels

    def fit_resample_wave(self, waves: torch.Tensor,
                          labels: torch.Tensor):
        print(f"[AudioSMOTE] 波形 SMOTE，输入形状: {waves.shape}")
        return self._resample(waves, labels, log_domain=False)

    def fit_resample_spec(self, specs: torch.Tensor,
                          labels: torch.Tensor):
        print(f"[AudioSMOTE] 频谱 SMOTE，输入形状: {specs.shape}")
        return self._resample(specs, labels, log_domain=True)

    def apply_to_imbalanced_classes(self, loader: DataLoader,
                                    feature_type: str = 'mel_spectrogram') -> DataLoader:
        """从 loader 收集全部数据，过采样后返回新的 DataLoader。"""
        all_data, all_labels = [], []
        try:
            for batch_data, batch_labels in loader:
                all_data.append(batch_data)
                all_labels.append(batch_labels)
            all_data = torch.cat(all_data)
            all_labels = torch.cat(all_labels)
            print(f"[AudioSMOTE] 收集数据形状: {all_data.shape}")

            if feature_type.lower() == 'waveform':
                resampled_data, resampled_labels = self.fit_resample_wave(
                    all_data, all_labels)
            else:
                resampled_data, resampled_labels = self.fit_resample_spec(
                    all_data, all_labels)

            print(f"[AudioSMOTE] 过采样后形状: {resampled_data.shape}")

            orig_ds = (getattr(getattr(loader, 'dataset', None), 'dataset', None)
                       or getattr(loader, 'dataset', None))
            dataset = EnhancedDataset((resampled_data, resampled_labels), orig_ds)
            return DataLoader(dataset, batch_size=loader.batch_size, shuffle=True,
                              num_workers=loader.num_workers,
                              worker_init_fn=getattr(loader, 'worker_init_fn', None))
        except Exception as e:
            import traceback
            print(f"[AudioSMOTE] 过采样失败: {e}")
            traceback.print_exc()
            return loader
