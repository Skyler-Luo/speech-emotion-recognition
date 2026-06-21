# Speech Emotion Recognition

## 📌 项目概览

本项目针对**语音情感识别（SER）**任务，提供三条从轻量到高精度的完整训练路线，覆盖手工特征、频谱图 CNN 与自监督预训练三种技术范式。

```
输入音频（WAV）
     │
     ├─── 路线 ①  手工特征  ──▶  CNN-BiLSTM + Attention
     │
     ├─── 路线 ②  梅尔频谱图 ──▶  DyMN / MobileNet / EfficientNet
     │
     └─── 路线 ③  原始波形  ──▶  HuBERT / Wav2Vec2（SSL 微调）
                                         │
                                         ▼
                              anger · fear · happy · neutral · sad
```

**核心特性**

- 三条路线统一训练框架，支持 **EMA**、**早停**、**Warmup Cosine 调度**、**梯度裁剪**
- CNN 路线内置 10+ 种数据增强，支持 **Mixup**、**SpecAugment**、**AudioSMOTE**
- SSL 路线使用 **差分学习率** + **OneCycleLR** 微调，特征提取器默认冻结
- 训练日志、TensorBoard 事件与检查点**自动管理**，断点可续训
- 提供独立评估脚本与单行推理 API

## 📂 目录结构

```
speech-emotion-recognition/
│
├── datasets/
│   └── emotion/
│       ├── train/
│       │   ├── anger/          *.wav
│       │   ├── fear/           *.wav
│       │   ├── happy/          *.wav
│       │   ├── neutral/        *.wav
│       │   └── sad/            *.wav
│       └── val/                （同上）
│
├── models/
│   ├── cnn_bilstm.py           CNN-BiLSTM 基线模型
│   ├── mn.py                   MobileNet
│   ├── dymn.py                 Dynamic MobileNet
│   ├── efficientnet.py         EfficientNet
│   ├── hubert_ser.py           HuBERT 微调封装
│   ├── wav2vec2_ser.py         Wav2Vec2 微调封装
│   └── ensemble.py             多模型集成
│
├── utils/
│   ├── config.py               全局常量（标签映射、音频参数）
│   ├── dataset.py              Dataset / DataLoader 工厂
│   ├── audio_utils.py          重采样、裁剪、归一化
│   ├── augmentation.py         AudioAugmentation + AudioSMOTE
│   ├── model_utils.py          EMA、评估循环、推理工具
│   └── logger.py               TrainingLogger + CheckpointManager
│
├── train_baseline.py           路线① 训练入口
├── train_cnn.py                路线② 训练入口
├── train_ssl.py                路线③ 训练入口
├── evaluate.py                 独立评估脚本
├── inference.py                单条推理 API
└── requirements.txt
```

训练产物保存至 `runs/<experiment_name>/`：

```
runs/EmotionClassification/
├── dymn20_as_best.pt           完整检查点（含优化器 / 调度器状态，用于续训）
├── dymn20_as_weights_best.pt   纯模型权重（推理专用）
├── dymn20_as_latest.pt         最新周期检查点
└── events.out.tfevents.*       TensorBoard 日志
```

## ⚡ 快速开始

**1. 安装依赖**

```bash
pip install -r requirements.txt

# SSL 路线额外依赖
pip install transformers
```

> Python ≥ 3.8，GPU 训练请安装与 CUDA 版本匹配的 PyTorch

**2. 准备数据**

将 WAV 文件按情感类别放入对应目录，文件名任意，目录名即标签（详见[数据处理](#-数据处理)）。

**3. 选择路线训练**

```bash
# 路线① 轻量基线（CPU 可跑）
python train_baseline.py --experiment_name Baseline_SER

# 路线② CNN 频谱图（推荐）
python train_cnn.py --model_name dymn20_as --pretrained

# 路线③ HuBERT 微调（最高精度，需 GPU）
python train_ssl.py --model hubert --pretrained_path facebook/hubert-base-ls960
```

**4. 评估与推理**

```bash
# 评估
python evaluate.py \
    --weights runs/EmotionClassification/dymn20_as_weights_best.pt \
    --data_dir datasets/emotion/val \
    --save_results
```

```python
# 推理
import torchaudio
from inference import predict

audio, sr = torchaudio.load("sample.wav")
print(predict(audio, sr))   # → "happy"
```

## 🔧 数据处理

### 数据组织

```
datasets/emotion/<split>/<emotion>/*.wav
```

- **split**：`train` / `val`
- **emotion**：`anger` · `fear` · `happy` · `neutral` · `sad`（目录名即标签）
- WAV 文件名任意；通过 `--train_dir` / `--val_dir` 可指定任意路径

### 音频预处理

| 步骤 | CNN-Mel 路线 | SSL 路线 |
|:----:|:-----------:|:-------:|
| 目标采样率 | 32 kHz | 16 kHz |
| 截断 / 填零 | 3 s（96,000 samples）| 3 s（48,000 samples）|
| 归一化 | 波形 ÷ max abs | 波形 ÷ max abs |
| 随机偏移 | 训练时开启 | 训练时开启 |
| 居中裁剪 | 验证 / 推理时 | 验证 / 推理时 |

### 数据增强（CNN 路线）

增强流水线分三层，强度由 `--aug_intensity light / medium / heavy` 统一控制：

**波形域**（每条样本随机组合 1～3 种）

| 方法 | light | medium | heavy |
|:----:|:-----:|:------:|:-----:|
| 时间偏移 | ±20% | ±30% | ±40% |
| 高斯噪声 | 0.001~0.01 | 0.001~0.02 | 0.002~0.03 |
| 有色噪声（粉/棕/蓝） | ✅ | ✅ | ✅ |
| 时间拉伸 | 0.85~1.15× | 0.75~1.25× | 0.70~1.30× |
| 音高偏移 | ±2 半音 | ±3 半音 | ±4 半音 |
| 音量调整 | ±3 dB | ±6 dB | ±8 dB |
| 混响 / 随机裁剪 / 频率滤波 | ✅ | ✅ | ✅ |

**频谱域**：SpecAugment，频率掩码 + 时间掩码（light: 2 条 / medium: 3 条 / heavy: 4 条）

**批次级**：Mixup（Beta 分布，α=0.3）；可选 AudioSMOTE（少数类 k-NN 特征插值，`--use_smote` 开启）

## 🧩 模型构建

### 路线① · CNN-BiLSTM 基线

从波形中提取手工特征，送入时序网络分类。

**特征提取**

```
WAV(32kHz) → MFCC[n_mfcc=10] + 频谱重心[1] + 频谱带宽[1]
           → Z-Score 归一化 → 填充/截断至 [T=300, D=12]
```

**模型结构**

```
[B, 300, 12]
  → Conv2D(1→64, 3×3) + BN + ReLU + MaxPool(2×2)
  → Reshape → BiLSTM(hidden=128, layers=2, dropout=0.5)
  → Attention Pooling  (score = Linear(tanh) → softmax)
  → FC(128→64) + ReLU + Dropout(0.5) + FC(64→5)
```

**训练配置**：AdamW · lr=1e-3 · WarmupCosine 调度 · EMA(0.999) · 早停(patience=20)

### 路线② · CNN 频谱图模型

将语音转为"图像"，复用音频或视觉预训练的轻量 CNN 主干。

**特征提取**

```
WAV(32kHz) → MelSpectrogram(n_mels=128, n_fft=1024, hop=320, f_min=20Hz)
           → log(·+ 1e-9) → [1, 128, ~300]
```

**可选主干**

| 模型 | 参数量 | 预训练数据集 | 说明 |
|:----:|:------:|:----------:|:----:|
| `mn10_as` | ~3.6 M | AudioSet | 轻量快速 |
| `mn20_as` | ~13 M | AudioSet | 均衡选择 |
| `dymn20_as` | ~13 M | AudioSet | 动态卷积，**默认推荐** |
| `efficientnet_b0` | ~5.3 M | ImageNet | 轻量备选 |
| `efficientnet_b5` | ~30 M | ImageNet | 高精度备选 |

**分类头**：Global Average Pooling → Dropout(0.3) → FC(num_classes)

**训练配置**：AdamW · lr=3e-4 · WarmupCosine(warmup=10) · Focal Loss(γ=1.5) · EMA(0.999) · 早停(patience=15)

### 路线③ · SSL 预训练模型（HuBERT / Wav2Vec2）

冻结底层 CNN 特征提取器，以差分学习率微调 Transformer 编码器。

**模型结构**

```
WAV(16kHz)
  → CNN Feature Extractor（冻结）→ [T, 512]
  → Transformer Encoder（微调，lr=3e-5）→ [T, 768]
  → 池化层（可选）
       ├── Attn：score = softmax(W₂·tanh(W₁·hₜ))，加权求和  ← 默认
       ├── Mean：时间维度均值
       └── Stat：均值 ‖ 标准差 拼接 → [1536]
  → LayerNorm → Linear(256) → GELU → Dropout(0.3) → FC(5)
```

**训练配置**：AdamW 差分学习率（encoder: 3e-5 / head: 1e-4） · OneCycleLR · AMP 混合精度 · EMA(0.999) · 早停(patience=10)

## 📊 性能对比

> 以下结果在本项目数据集验证集上评估，指标为**宏平均 F1**。

| 模型 | 路线 | 参数量 | Val F1 | 单条推理(CPU) | 备注 |
|:----:|:----:|:------:|:------:|:------------:|:----:|
| CNN-BiLSTM | ① Baseline | ~0.5 M | — | < 0.1 s | 无预训练 |
| MobileNet-20 | ② CNN-Mel | ~13 M | — | ~0.2 s | AudioSet 预训练 |
| **DyMN-20** | ② CNN-Mel | ~13 M | — | ~0.2 s | AudioSet 预训练，**推荐** |
| EfficientNet-B5 | ② CNN-Mel | ~30 M | — | ~0.5 s | ImageNet 预训练 |
| Wav2Vec2-base | ③ SSL | ~95 M | — | ~3 s | 16kHz 波形输入 |
| **HuBERT-base** | ③ SSL | ~95 M | — | ~4 s | 16kHz 波形输入，**精度最高** |

> 💡 F1 数值将在完整训练后填入。路线②③支持多模型概率平均集成，集成后 F1 通常可再提升 1~3 个百分点。

**选型建议**

```
资源受限 / 快速验证  →  路线① CNN-BiLSTM
精度与速度均衡       →  路线② DyMN-20（dymn20_as）
追求最高精度         →  路线③ HuBERT + Attn 池化
集成部署             →  路线② × 路线③ 概率平均
```

<div align="center">
<sub>欢迎 ⭐ Star 与交流</sub>
</div>
