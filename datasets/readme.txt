数据集下载链接：https://drive.google.com/file/d/1h15twUcuNWL_zXcVRUhdUnFSSAra4P99/view?usp=drive_link

数据集目录格式：
emotion/
├── train/              # 训练集
│   ├── anger/          # 愤怒类音频 (*.wav)
│   ├── fear/           # 恐惧类音频 (*.wav)
│   ├── happy/          # 高兴类音频 (*.wav)
│   ├── neutral/        # 中性类音频 (*.wav)
│   └── sad/            # 悲伤类音频 (*.wav)
└── val/                # 验证集 (结构同 train/)
    ├── anger/
    ├── fear/
    ├── happy/
    ├── neutral/
    └── sad/

说明：
- 下载后将数据集解压至 datasets/ 目录下
- 每个情感类别的文件夹包含对应情感的 .wav 格式音频文件