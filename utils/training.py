"""训练相关的工具类和函数。

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


def get_optimizer(model, optimizer_type='adamw', lr=1e-3, weight_decay=1e-4, **kwargs):
    """创建优化器。
    
    Args:
        model: 模型
        optimizer_type: 优化器类型 'adamw' | 'adam' | 'sgd'
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他优化器参数
        
    Returns:
        优化器实例
    """
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        momentum = kwargs.pop('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                              momentum=momentum, **kwargs)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type='cosine', epochs=100, warmup_epochs=0, 
                  warmup_start_lr=1e-6, eta_min=1e-6, **kwargs):
    """创建学习率调度器。
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 'cosine' | 'warmup_cosine' | 'step' | 'onecycle'
        epochs: 总 epoch 数
        warmup_epochs: 预热的 epoch 数（仅用于 warmup_cosine）
        warmup_start_lr: 预热开始学习率
        eta_min: 最小学习率
        **kwargs: 其他调度器参数
        
    Returns:
        调度器实例
    """
    if scheduler_type.lower() == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=max(warmup_epochs, 1),
            max_epochs=epochs,
            warmup_start_lr=warmup_start_lr,
            eta_min=eta_min
        )
    elif scheduler_type.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
    elif scheduler_type.lower() == 'step':
        step_size = kwargs.pop('step_size', 30)
        gamma = kwargs.pop('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type.lower() == 'onecycle':
        max_lr = kwargs.pop('max_lr', 1e-3)
        steps_per_epoch = kwargs.pop('steps_per_epoch', 100)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


def get_criterion(criterion_type='crossentropy', **kwargs):
    """创建损失函数。
    
    Args:
        criterion_type: 损失函数类型 'crossentropy' | 'focal' | 'mse'
        **kwargs: 损失函数参数
        
    Returns:
        损失函数实例
    """
    if criterion_type.lower() == 'crossentropy':
        weight = kwargs.get('weight', None)
        return nn.CrossEntropyLoss(weight=weight)
    elif criterion_type.lower() == 'focal':
        alpha = kwargs.get('alpha', None)
        gamma = kwargs.get('gamma', 2.0)
        class_weights = kwargs.get('class_weights', None)
        return FocalLoss(alpha=alpha, gamma=gamma, class_weights=class_weights)
    elif criterion_type.lower() == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"不支持的损失函数类型: {criterion_type}")
