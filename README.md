# 🎙️ Speech Emotion Recognition (SER)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](requirements.txt) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](requirements.txt) [![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg?logo=huggingface)](requirements.txt) [![License: MIT](https://img.shields.io/badge/License-MIT-4169E1.svg)](LICENSE)

> 🏆 **本项目荣获 RAICOM2025 省二等奖**

本项目针对**语音情感识别（SER）**任务，提供了一条从轻量级基线到高精度大模型的完整训练路线。项目涵盖了**手工时序特征**、**梅尔频谱图 CNN** 与**自监督预训练大模型（SSL）**三种主流技术范式，并在统一的框架下实现了完整的数据处理、多维度数据增强、状态管理及独立评估推理。

## 📌 项目架构与技术范式

针对不同的计算资源和精度要求，本项目设计了三条典型的语音情感识别技术路线：

```
                    输入音频（WAV）
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
  [ 路线 1 ]          [ 路线 2 ]          [ 路线 3 ]
   手工特征            梅尔频谱图           原始波形
      │                  │                  │
  提取 MFCC+谱特征      时频分析(Mel-Spec)   Wav-to-Vector
      │                  │                  │
      ▼                  ▼                  ▼
  CNN-BiLSTM       轻量主干 CNN         自监督大模型
  + Attention    (DyMN / MobileNet /    (HuBERT / Wav2Vec2)
                 EfficientNet)              │
      │                  │                  ▼
      │                  │            差分微调(Transformer)
      └──────────────────┼──────────────────┘
                         │
                         ▼
             [ 情感分类 (5-Classes) ]
      anger · fear · happy · neutral · sad
```

### 范式对比

| 路线 | 技术范式 | 特征提取 | 核心网络结构 |
| :---: | :---: | :---: | :---: |
| **1** | **时序基线** | 手工特征 (MFCC + 频谱重心/带宽) | Conv2D + 双向 LSTM + Attention Pooling |
| **2** | **时频 CNN (推荐)** | 2D 梅尔频谱图 (Log-Mel Spectrogram) | DyMN / MobileNetV3 / EfficientNet |
| **3** | **自监督微调 (SSL)** | 原始音频波形 (Raw Waveform) | CNN 提取器 + Transformer 编码器 + 差分分类头 |

## 📂 目录结构

```
speech-emotion-recognition/
├── datasets/                   # 数据集目录
│   └── emotion/
│       ├── train/              # 训练集子目录
│       │   ├── anger/          # 愤怒类音频目录 (*.wav)
│       │   ├── fear/           # 恐惧类音频目录
│       │   ├── happy/          # 高兴类音频目录
│       │   ├── neutral/        # 中性类音频目录
│       │   └── sad/            # 悲伤类音频目录
│       └── val/                # 验证集子目录（结构同 train/）
│
├── models/                     # 模型网络定义
│   ├── cnn_bilstm.py           # CNN-BiLSTM-Attention 基线模型
│   ├── mn.py                   # MobileNet 主干模型
│   ├── dymn.py                 # Dynamic MobileNet (动态卷积)
│   ├── efficientnet.py         # EfficientNet 主干模型
│   ├── hubert_ser.py           # HuBERT 情感分类微调封装
│   ├── wav2vec2_ser.py         # Wav2Vec2 情感分类微调封装
│   └── ensemble.py             # 多模型概率集成 (Ensemble)
│
├── utils/                      # 核心工具库
│   ├── config.py               # 全局配置常量（类别映射、采样率、时频分析参数）
│   ├── dataset.py              # 统一的 EmotionDataset 与数据预加载实现
│   ├── audio_utils.py          # 音频底层预处理（采样率转换、截断对齐、归一化）
│   ├── augmentation.py         # 10+ 种波形与频谱数据增强及 AudioSMOTE 插值
│   ├── training.py             # 调度器 (Warmup Cosine)、损失函数 (Focal Loss)
│   ├── model_utils.py          # EMA、多类别评估指标计算、检查点加载、推理接口
│   ├── logger.py               # 命令行日志器与断点续训管理器 CheckpointManager
│   └── utils.py                # 框架通用辅助函数
│
├── train_cnn_bilstm.py         # 时序基线训练入口
├── train_cnn.py                # 时频 CNN 训练入口
├── train_ssl.py                # 自监督大模型训练入口
├── evaluate.py                 # 独立离线多指标评估脚本
├── inference.py                # 生产环境单条推理 API 接口
├── requirements.txt            # 项目依赖声明
└── LICENSE                     # 开源许可证
```

### 训练产物输出格式

每次训练将在指定的 `--run_dir`（默认 `runs/`）下创建以 `--experiment_name` 命名的文件夹：
```
runs/CNN_SER_Experiment/
├── dymn20_as_best.pt           # 包含优化器、调度器、EMA 及迭代次数的完整检查点（可用于断点恢复）
├── dymn20_as_weights_best.pt   # 推理专用模型权重文件（精简体积）
├── dymn20_as_latest.pt         # 最新周期检查点（用于意外中断恢复）
└── events.out.tfevents.*       # TensorBoard 可视化训练事件日志
```

## ⚡ 快速开始

### 1. 环境准备

克隆项目并安装依赖（建议在 Python ≥ 3.8 的虚拟环境中操作）：
```bash
pip install -r requirements.txt
```

### 2. 准备数据集

按照 [📂 目录结构](#-目录结构) 组织数据。数据集根目录下应至少包含 `train` 与 `val` 两个子目录，每个子目录内包含按情感名称命名的文件夹（如 `anger`、`neutral` 等）。

* 默认支持的情感标签：`anger`（愤怒）、`fear`（恐惧）、`happy`（高兴）、`neutral`（中性）、`sad`（悲伤）。
* 支持任意文件名的 `.wav` 格式音频。

### 3. 选择路线开始训练

#### 路线 1：手工特征轻量基线
```bash
python train_cnn_bilstm.py --experiment_name Baseline_SER --epochs 100 --batch_size 64
```

#### 路线 2：梅尔频谱图 CNN
```bash
# 使用 DyMN-20（基于 AudioSet 预训练，动态卷积），开启 EMA 与 Focal Loss
python train_cnn.py --model_name dymn20_as --pretrained --experiment_name DyMN_SER
```

#### 路线 3：自监督预训练大模型微调
```bash
# 基于 HuBERT 预训练模型微调，对底层特征提取器进行冻结，仅微调 Transformer 和注意力池化分类头
python train_ssl.py --model hubert --pretrained_path facebook/hubert-base-ls960 --pool attn --batch_size 32
```

## 🔧 数据预处理与增强方案

### 音频底座预处理参数

项目执行了高标准的归一化和截断/填充流水线：
- **目标采样率**：CNN-Mel 分支采用 **32 kHz**，SSL 分支采用官方预训练标准的 **16 kHz**。
- **长度归一**：默认统一截断或填充零至 **3 秒**（32kHz 对应 96,000 个采样点，16kHz 对应 48,000 个采样点）。
- **动态控制**：训练阶段开启**随机偏移裁剪**以丰富样本多样性；验证与推理阶段自动使用**中心裁剪**以保证确定性评估。
- **预加载模式**：通过指定 `--preload_data` 可启用多线程预加载，所有音频在训练前一次性读入 CPU 内存，可极大消除 I/O 带来的 GPU 等待瓶颈。

### 数据增强流水线 (仅限路线 2)

支持 3 个层次的全面增强，通过 `--aug_intensity` 动态调整强度（`light` / `medium` / `heavy`）：

1. **时域增强 (Waveform Domain)**：
   - **时间偏移 (Time Shift)**：±20% / ±30% / ±40% 随机偏移。
   - **加性噪声**：高斯噪声（标准差 0.001 ~ 0.03 动态调整）和有色噪声（粉红噪声、棕色噪声、蓝色噪声）。
   - **时频缩放**：时间拉伸（0.7x ~ 1.3x）与音高偏移（±2 ~ ±4 半音）。
   - **空间效果**：随机混响、音量随机微调（±3 ~ ±8 dB）、带通/带阻频率滤波器。
2. **频域增强 (Spectral Domain - SpecAugment)**：
   - 包含频率通道掩码和时间掩码，根据增强强度自动增加遮蔽条数（2 至 4 条掩码）。
3. **样本混合 (Batch Domain)**：
   - **Mixup**：训练中默认开启，利用 Beta 分布（$\alpha=0.3$）混合音频特征与标签。
   - **AudioSMOTE**（少数类插值）：针对小样本或不平衡数据集，通过 `--use_smote` 启动 k-NN 特征插值。

---

## 📊 评估与推理

### 多指标评估

利用训练完成后在 `runs/` 下生成的精简版模型权重文件（如 `*_weights_best.pt`）进行评估：
```bash
python evaluate.py \
    --weights runs/DyMN_SER/dymn20_as_weights_best.pt \
    --data_dir datasets/emotion/val \
    --save_results \
    --output_file results_dymn.json
```
评估脚本将输出详细的混淆矩阵统计数据、准确率、宏平均精确率（Precision）、召回率（Recall）以及 **Macro F1-Score**，并将其持久化保存为 JSON。

### 单条音频快速推理 API
```python
import torchaudio
from inference import predict

# 1. 读取单条音频
waveform, sr = torchaudio.load("path/to/test_voice.wav")

# 2. 预测情感（默认加载 runs/EmotionClassification/dymn20_as_weights_best.pt）
emotion_label = predict(waveform, sr, model_path="runs/DyMN_SER/dymn20_as_weights_best.pt")
print(f"预测情感结果为: {emotion_label}")
```

## 📈 性能表现与模型选型建议

| 网络模型 | 路线分类 | 参数量 | 推理耗时 (CPU) | 运行要求 | 推荐指数 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-BiLSTM** | 路线 1 手工时序 | ~0.5 M | < 0.1 s | CPU 可跑，速度极快 | ⭐⭐ |
| **MobileNet-20** | 路线 2 频谱 CNN | ~13 M | ~0.2 s | 需要轻量 GPU 辅助 | ⭐⭐⭐⭐ |
| **DyMN-20** | 路线 2 频谱 CNN | ~13 M | ~0.2 s | 音频表现卓越，**推荐主干** | ⭐⭐⭐⭐⭐ |
| **EfficientNet-B5**| 路线 2 频谱 CNN | ~30 M | ~0.5 s | 适合时频图像精细分析 | ⭐⭐⭐ |
| **HuBERT-Base** | 路线 3 自监督 (Attn) | ~95 M | ~4.0 s | 需中高端 GPU，训练耗时长 | ⭐⭐⭐⭐⭐ (高精度) |

## 🤝 参与贡献与交流

欢迎提交 Issue 和 Pull Request 来帮助完善此项目！

- **代码贡献**：请遵循 PEP8 编码规范，并确保新增功能具有对应单元测试。
- **模型提交**：如果您训练出了在公共数据集上更好的权重，欢迎通过 Issue 分享模型结构与超参数设置。

## 📄 开源协议

本项目基于 **[MIT License](LICENSE)** 协议开源。
