# 全局常量：情感标签、默认音频参数

from typing import Dict

# 情感标签映射（类别名 → 整数索引）
EMOTION_LABEL_MAP: Dict[str, int] = {
    'anger':   0,
    'fear':    1,
    'happy':   2,
    'neutral': 3,
    'sad':     4,
}

# 整数索引 → 类别名（推理时使用）
IDX_TO_EMOTION: Dict[int, str] = {v: k for k, v in EMOTION_LABEL_MAP.items()}

NUM_CLASSES: int = len(EMOTION_LABEL_MAP)

# 默认音频参数（MobileNet / EfficientNet 分支使用 32kHz）
DEFAULT_SR:         int = 32000
DEFAULT_MAX_SEC:    float = 3.0
DEFAULT_MAX_LENGTH: int = int(DEFAULT_MAX_SEC * DEFAULT_SR)  # 96000 samples

# 默认梅尔频谱参数
DEFAULT_N_MELS:     int = 128
DEFAULT_N_FFT:      int = 1024
DEFAULT_HOP_LENGTH: int = 320
DEFAULT_F_MIN:      int = 20

# SSL 模型默认参数（HuBERT / Wav2Vec2 要求 16kHz）
SSL_SR:         int = 16000
SSL_MAX_LENGTH: int = int(DEFAULT_MAX_SEC * SSL_SR)  # 48000 samples
