import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from typing import Dict, Tuple

from utils.config import EMOTION_LABEL_MAP

class EMA:
    """指数移动平均，推理时使用更平滑的模型权重。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: dict = {
            n: p.data.clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup: dict = {}

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        return {'decay': self.decay, 'shadow': self.shadow, 'backup': self.backup}

    def load_state_dict(self, sd: dict):
        self.decay  = sd['decay']
        self.shadow = sd['shadow']
        self.backup = sd['backup']


def load_state_from_checkpoint(ckpt: dict, model: nn.Module,
                                prefer_ema: bool = True) -> str:
    """从检查点字典向模型加载权重，优先加载 EMA shadow 权重。

    Returns:
        实际使用的权重键名（用于日志）
    """
    if prefer_ema and 'ema_state_dict' in ckpt:
        state = ckpt['ema_state_dict']
        state = state.get('shadow', state)
        model.load_state_dict(state)
        return 'ema_state_dict'

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        return 'model_state_dict'

    model.load_state_dict(ckpt)
    return 'raw'


def build_model_from_checkpoint(checkpoint_path: str, device,
                                 num_classes: int = None) -> Tuple[nn.Module, str]:
    """从检查点路径完整构建并加载 mn/dymn 模型。

    Args:
        checkpoint_path: .pt 文件路径
        device:          目标设备
        num_classes:     分类头类别数，None 时从 config 读取

    Returns:
        (model, used_key)  model 已置 eval 并移至 device
    """
    from utils.utils import NAME_TO_WIDTH
    from models.mn import get_model as get_mobilenet
    from models.dymn import get_model as get_dymn

    if num_classes is None:
        from utils.config import NUM_CLASSES
        num_classes = NUM_CLASSES

    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'args' in ckpt:
        a = ckpt['args']
        model_name = a.get('model_name', 'dymn20_as')
        pretrained = a.get('pretrained', False)
        width = NAME_TO_WIDTH(model_name) if pretrained else a.get('model_width', 1.0)
        if model_name.startswith('dymn'):
            model = get_dymn(
                width_mult=width, pretrained_name=None,
                pretrain_final_temp=a.get('pretrain_final_temp', 1.0),
                num_classes=num_classes,
            )
        else:
            model = get_mobilenet(
                width_mult=width, pretrained_name=None,
                head_type=a.get('head_type', 'mlp'),
                se_dims=a.get('se_dims', 'c'),
                num_classes=num_classes,
            )
    else:
        model = get_dymn(width_mult=1.0, pretrained_name=None,
                         num_classes=num_classes)

    used_key = load_state_from_checkpoint(ckpt, model)
    return model.to(device).eval(), used_key


def reshape_input(x: torch.Tensor) -> torch.Tensor:
    """将各种维度的输入统一整形为 [B, C, H, W]（频谱图格式）。

    支持：
      [B, 1, 1, H, W] → [B, 1, H, W]
      [B, C, H, W]    → 直接返回
      [B, H, W]       → [B, 1, H, W]
    """
    if x.dim() == 5:
        return x.view(x.shape[0], 1, x.shape[3], x.shape[4])
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        return x.unsqueeze(1) if x.shape[0] > x.shape[1] else x
    raise ValueError(f"reshape_input: 不支持的维度 {x.shape}")


def batch_extract_features(waveforms: torch.Tensor, dataset) -> torch.Tensor:
    """批量从波形提取特征，优先调用 dataset.extract_features_batch。

    Args:
        waveforms: [B, 1, T] 波形张量
        dataset:   含 extract_features / extract_features_batch 方法的数据集
    """
    with torch.no_grad():
        if hasattr(dataset, 'extract_features_batch'):
            return dataset.extract_features_batch(waveforms)
        return torch.cat(
            [dataset.extract_features(waveforms[i:i + 1])
             for i in range(waveforms.shape[0])],
            dim=0,
        )


def get_logits(outputs):
    """从模型输出中提取 logits（兼容 tuple 和直接 tensor 两种格式）。"""
    return outputs[0] if isinstance(outputs, tuple) else outputs


def run_inference(model: nn.Module, loader, criterion, device,
                  prepare_input_fn=None, desc: str = '推理') -> Tuple[np.ndarray, np.ndarray, float]:
    """跑一遍 loader，返回 (targets, preds, avg_loss)。

    Args:
        prepare_input_fn: 可选的输入预处理函数 fn(x) → x，
                          为 None 时直接调用 reshape_input
    """
    all_targets, all_outputs, losses = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc):
            x, y = x.to(device), y.to(device)
            x = prepare_input_fn(x) if prepare_input_fn else reshape_input(x)
            logits = get_logits(model(x))
            losses.append(criterion(logits, y).item())
            all_targets.append(y.cpu().numpy())
            all_outputs.append(logits.cpu().numpy())

    targets = np.concatenate(all_targets)
    preds = np.argmax(np.concatenate(all_outputs), axis=1)
    return targets, preds, float(np.mean(losses))


def evaluate_per_class(model: nn.Module, loader, criterion, device,
                       prepare_input_fn=None,
                       verbose: bool = True) -> Tuple[float, Dict]:
    """完整评估，返回 (f1_macro, metrics_dict)。

    metrics_dict 包含：
      accuracy, loss, precision_macro, recall_macro, f1_macro,
      confusion_matrix (list), class_metrics (dict per emotion)

    Args:
        prepare_input_fn: 同 run_inference
        verbose:          是否打印结果
    """
    model.eval()
    emotion_names = list(EMOTION_LABEL_MAP.keys())

    targets, preds, avg_loss = run_inference(
        model, loader, criterion, device,
        prepare_input_fn=prepare_input_fn, desc='评估'
    )

    acc = metrics.accuracy_score(targets, preds)
    f1 = metrics.f1_score(targets, preds, average='macro')
    prec = metrics.precision_score(targets, preds, average='macro')
    rec = metrics.recall_score(targets, preds, average='macro')
    cm = metrics.confusion_matrix(targets, preds)

    # 用 classification_report 一次性拿所有类别指标，避免逐类重复调用
    report = metrics.classification_report(
        targets, preds,
        labels=list(EMOTION_LABEL_MAP.values()),
        target_names=list(EMOTION_LABEL_MAP.keys()),
        zero_division=0,
        output_dict=True,
    )
    class_metrics: Dict = {
        emotion: {
            'accuracy':  float(((preds == targets) & (targets == idx)).sum()
                               / max((targets == idx).sum(), 1)),
            'precision': float(report[emotion]['precision']),
            'recall':    float(report[emotion]['recall']),
            'f1':        float(report[emotion]['f1-score']),
            'samples':   int(report[emotion]['support']),
        }
        for emotion, idx in EMOTION_LABEL_MAP.items()
        if (targets == idx).sum() > 0
    }

    if verbose:
        print(f"\n===== 评估结果 =====")
        print(f"准确率: {acc*100:.2f}%  F1(macro): {f1*100:.2f}%  "
              f"精确度: {prec*100:.2f}%  召回率: {rec*100:.2f}%\n")
        for emotion, cm_e in class_metrics.items():
            print(f"  {emotion:8s}: acc={cm_e['accuracy']*100:.1f}% "
                  f"p={cm_e['precision']*100:.1f}% r={cm_e['recall']*100:.1f}% "
                  f"f1={cm_e['f1']*100:.1f}% ({cm_e['samples']})")
        print("\n混淆矩阵 (行=实际, 列=预测):")
        header = "          " + " ".join(f"{n:>8}" for n in emotion_names)
        print(header)
        for i, name in enumerate(emotion_names):
            row = f"{name:>10}" + " ".join(
                f"{cm[i, j]:>8}" for j in range(len(emotion_names)))
            print(row)

    return f1, {
        'accuracy':         acc,
        'loss':             avg_loss,
        'precision_macro':  prec,
        'recall_macro':     rec,
        'f1_macro':         f1,
        'confusion_matrix': cm.tolist(),
        'class_metrics':    class_metrics,
    }
