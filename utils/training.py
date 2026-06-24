"""训练相关的工具类 and 函数。

包含学习率调度器、损失函数、混合训练等通用训练组件。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """带预热的余弦退火学习率调度器。
    
    前 warmup_epochs 个 epoch 线性增加学习率，
    之后使用余弦退火降低学习率。
    
    Args:
        optimizer: 优化器
        warmup_epochs: 预热的 epoch 数
        max_epochs: 总 epoch 数
        warmup_start_lr: 预热开始时的学习率
        eta_min: 最小学习率
        last_epoch: 上一个 epoch 的索引
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs,
                 warmup_start_lr=1e-6, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热阶段
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        
        # 余弦退火阶段
        progress = (self.last_epoch - self.warmup_epochs) / (
            self.max_epochs - self.warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.eta_min + cosine_decay * (base_lr - self.eta_min)
                for base_lr in self.base_lrs]


class FocalLoss(nn.Module):
    """Focal Loss 损失函数。
    
    用于解决类别不平衡问题，通过降低易分类样本的权重，
    使模型更关注难分类的样本。
    
    Args:
        alpha: 类别权重，可以是标量或张量
        gamma: 聚焦参数，控制易分类样本的权重降低程度
        reduction: 损失的归约方式 'none' | 'mean' | 'sum'
        class_weights: 每个类别的权重
        
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型预测 logits [B, C]
            targets: 真实标签 [B]
            
        Returns:
            focal loss
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets,
                                  weight=self.class_weights, reduction='none')
        
        # 处理 NaN 和 Inf
        if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
            ce_loss = torch.nan_to_num(ce_loss, nan=1.0, posinf=10.0, neginf=0.0)
        
        # 计算概率
        pt = torch.exp(-ce_loss.clamp(-20, 20))
        
        # 应用 focal 权重
        if self.alpha is not None:
            alpha_t = torch.clamp(self.alpha[targets], 0.05, 0.95)
            focal = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal = (1 - pt) ** self.gamma * ce_loss
        
        # 再次处理 NaN 和 Inf
        focal = torch.nan_to_num(focal, nan=1.0, posinf=10.0, neginf=0.0)
        
        # 归约
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 训练的损失函数。
    
    计算混合样本的损失，作为两个样本损失的加权平均。
    
    Args:
        criterion: 损失函数
        pred: 模型预测
        y_a: 第一个样本的标签
        y_b: 第二个样本的标签
        lam: 混合权重
        
    Returns:
        混合损失
        
    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
