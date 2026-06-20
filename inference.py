import torch

import torchaudio.transforms as T

from utils.config import (
    EMOTION_LABEL_MAP, IDX_TO_EMOTION,
    DEFAULT_SR, DEFAULT_MAX_LENGTH,
    DEFAULT_N_MELS, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_F_MIN,
)
from utils.audio_utils import preprocess_waveform
from utils.model_utils import build_model_from_checkpoint, get_logits


DEFAULT_MODEL_PATH = 'runs/EmotionClassification/dymn20_as_weights_best.pt'


_model = None
_mel_transform = None


def load_model(model_path: str = DEFAULT_MODEL_PATH,
               device: str = 'cpu') -> torch.nn.Module:
    """从检查点加载模型"""
    try:
        model, _ = build_model_from_checkpoint(
            model_path, device, num_classes=len(EMOTION_LABEL_MAP))
        return model
    except Exception as e:
        print(f"[inference] 加载模型失败: {e}")
        return None


def predict(audio: torch.Tensor, sr: int,
            model_path: str = DEFAULT_MODEL_PATH) -> str:
    """预测单条音频的情感标签。

    Args:
        audio:      torchaudio.load 返回的波形张量
        sr:         对应采样率
        model_path: 模型权重路径

    Returns:
        情感标签字符串，如 'happy'、'neutral' 等
    """
    global _model, _mel_transform
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if _model is None:
            _model = load_model(model_path, str(device))
        if _model is None:
            raise RuntimeError(f"模型加载失败，请检查权重路径: {model_path}")
        _model = _model.to(device)

        if _mel_transform is None:
            _mel_transform = T.MelSpectrogram(
                sample_rate=DEFAULT_SR, n_fft=DEFAULT_N_FFT,
                hop_length=DEFAULT_HOP_LENGTH, n_mels=DEFAULT_N_MELS,
                f_min=DEFAULT_F_MIN, f_max=DEFAULT_SR // 2,
            )

        wav = preprocess_waveform(
            audio, sr,
            target_sr=DEFAULT_SR,
            max_length=DEFAULT_MAX_LENGTH,
        ).to(device)

        _mel_transform = _mel_transform.to(device)
        feat = torch.log(_mel_transform(wav) + 1e-9).unsqueeze(0)  # [1, 1, n_mels, T]

        with torch.no_grad():
            logits = get_logits(_model(feat))
            idx = torch.argmax(logits, dim=1).item()

        return IDX_TO_EMOTION[idx]

    except Exception as e:
        print(f"[inference] 预测出错: {e}")
        return 'neutral'
