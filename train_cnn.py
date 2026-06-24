import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from utils.config import EMOTION_LABEL_MAP, NUM_CLASSES
from utils.dataset import EmotionDataset
from utils.augmentation import AudioAugmentation, AudioSMOTE, configure_augmentation
from utils.utils import worker_init_fn, NAME_TO_WIDTH
from utils.training import WarmupCosineScheduler, FocalLoss, mixup_criterion
from utils.model_utils import (
    EMA, reshape_input, batch_extract_features,
    evaluate_per_class, run_inference, get_logits,
)
from utils.logger import TrainingLogger, CheckpointManager
from models.mn import get_model as get_mobilenet
from models.dymn import get_model as get_dymn


def _prepare_input(x, args, dataset):
    """波形 → 频谱特征 → reshape 到 [B, C, H, W]。"""
    if args.augment and x.dim() == 3 and x.shape[1] == 1:
        return batch_extract_features(x, dataset)
    return reshape_input(x)


def _quick_eval(model, loader, criterion, device, args, dataset, logger):
    """验证一轮，返回 (f1_macro, avg_loss, accuracy)。"""
    prepare_fn = lambda x: _prepare_input(x, args, dataset)
    targets, preds, avg_loss = run_inference(
        model, loader, criterion, device,
        prepare_input_fn=prepare_fn, desc='验证'
    )
    acc = metrics.accuracy_score(targets, preds)
    f1 = metrics.f1_score(targets, preds, average='macro')
    logger.log(f"  验证 → 准确率: {acc*100:.2f}%  宏F1: {f1*100:.2f}%  Loss: {avg_loss:.4f}")
    for emotion, idx in EMOTION_LABEL_MAP.items():
        mask = targets == idx
        if mask.sum() > 0:
            cls_acc = metrics.accuracy_score(targets[mask], preds[mask])
            logger.log(f"    {emotion}: {cls_acc*100:.1f}% ({mask.sum()})")
    return f1, avg_loss, acc


def _build_model(args):
    """根据 model_name 构建并返回模型实例"""
    pretrained_name = args.model_name if args.pretrained else None

    if args.model_name.startswith('dymn'):
        width = NAME_TO_WIDTH(args.model_name) if args.pretrained else args.model_width
        return get_dymn(
            width_mult=width,
            pretrained_name=pretrained_name,
            pretrain_final_temp=args.pretrain_final_temp,
            num_classes=NUM_CLASSES,
        )

    if args.model_name.startswith('efficientnet'):
        from models.efficientnet import get_model as get_efficientnet
        # model_name 约定：efficientnet_b0 或 efficientnet_b5
        variant = args.model_name.split('_')[-1] if '_' in args.model_name else 'b5'
        return get_efficientnet(
            num_classes=NUM_CLASSES,
            variant=variant,
            pretrained=args.pretrained,
            dropout=args.dropout,
        )

    # 默认 MobileNet
    width = NAME_TO_WIDTH(args.model_name) if args.pretrained else args.model_width
    return get_mobilenet(
        width_mult=width,
        pretrained_name=pretrained_name,
        head_type=args.head_type,
        se_dims=args.se_dims,
        num_classes=NUM_CLASSES,
    )


def train(args):
    run_dir = os.path.join(args.run_dir, args.experiment_name)
    logger = TrainingLogger(
        run_dir=run_dir,
        experiment_name=args.experiment_name,
    )
    ckpt_mgr = CheckpointManager(
        save_dir=run_dir,
        prefix=args.model_name,
        save_interval=args.save_interval,
        monitor='val_f1',
        logger=logger,
    )

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设备
    if args.cuda and torch.cuda.is_available():
        device = torch.device(
            f'cuda:{args.gpu_id}'
            if 0 <= args.gpu_id < torch.cuda.device_count() else 'cuda:0')
        logger.log(f"使用 GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.log("使用 CPU")

    # 增强器
    augmenter = None
    if args.augment:
        augmenter = AudioAugmentation(sample_rate=32000)
        configure_augmentation(augmenter, args.aug_intensity)

    # 数据集
    need_waveform = args.augment or args.use_smote
    
    # 确定预加载设备
    preload_device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    if args.preload_data:
        logger.log(f"使用预加载数据集模式，目标设备: {preload_device}")
    else:
        logger.log("使用按需加载模式")
    
    train_dataset = EmotionDataset(
        dataset_dir=args.train_dir,
        mode='spectrogram',
        feature_type=args.feature_type,
        max_length=args.max_length,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        normalize=True,
        random_offset=True,
        return_waveform=need_waveform,
        augmenter=augmenter if need_waveform else None,
        preload=args.preload_data,
        preload_device=preload_device,
        show_progress=True,
    )
    val_dataset = EmotionDataset(
        dataset_dir=args.val_dir,
        mode='spectrogram',
        feature_type=args.feature_type,
        max_length=args.max_length,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        normalize=True,
        random_offset=False,
        return_waveform=False,
        augmenter=None,
        preload=args.preload_data,
        preload_device=preload_device,
        show_progress=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    logger.log(f"训练集: {len(train_dataset)}  验证集: {len(val_dataset)}")
    
    # 如果使用预加载，显示加载失败的文件
    if args.preload_data:
        failed_train = train_dataset.get_failed_files()
        failed_val = val_dataset.get_failed_files()
        if failed_train:
            logger.log(f"训练集加载失败文件数: {len(failed_train)}")
        if failed_val:
            logger.log(f"验证集加载失败文件数: {len(failed_val)}")

    # SMOTE（可选）
    if args.use_smote:
        logger.log(f"应用 AudioSMOTE，策略: {args.smote_sampling_strategy}")
        smote = AudioSMOTE(
            sampling_strategy=args.smote_sampling_strategy,
            k_neighbors=args.smote_k_neighbors,
            random_state=args.seed, device=device)
        backup_loader = train_loader
        train_loader = smote.apply_to_imbalanced_classes(
            train_loader, feature_type=args.feature_type)
        try:
            next(iter(train_loader))
            logger.log("SMOTE 应用成功")
        except Exception:
            logger.log("SMOTE 验证失败，恢复原始数据")
            train_loader = backup_loader

    # 模型
    model = _build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"模型: {args.model_name}  可训练参数: {n_params:,}")

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema:
        logger.log(f"EMA 衰减率: {args.ema_decay}")

    # 损失函数
    if args.focal_loss:
        criterion = FocalLoss(alpha=None, gamma=1.5)
        logger.log("损失函数: Focal Loss (gamma=1.5)")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.log("损失函数: CrossEntropyLoss")

    # 优化器 & 调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=max(5, args.warmup_epochs),
            max_epochs=args.epochs,
            warmup_start_lr=args.warmup_start_lr,
            eta_min=args.lr / 200,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr / 200)

    # 恢复检查点
    start_epoch = 0
    if args.resume_checkpoint:
        try:
            start_epoch = ckpt_mgr.resume(
                args.resume_checkpoint, model, device, optimizer, scheduler, ema)
        except Exception as e:
            logger.log(f"恢复失败: {e}，从头开始")

    # 训练循环
    patience_counter = 0
    start_time = time.time()
    logger.log(f"开始训练，epochs={args.epochs}，patience={args.patience}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_loss_list, train_correct, train_total = [], 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                         leave=False):
            x, y = x.to(device), y.to(device)

            # Mixup（仅在波形阶段）
            if args.augment and args.mixup:
                try:
                    x, ya, yb, lam = augmenter.apply_augmentations_to_waveforms(
                        x, y, apply_mixup=True, mixup_alpha=args.mixup_alpha)
                except Exception as e:
                    logger.log(f"Mixup 失败: {e}")
                    ya = yb = y; lam = 1.0
            else:
                ya = yb = y; lam = 1.0

            # 波形 → 频谱图
            x = _prepare_input(x, args, train_dataset)

            # SpecAugment（频谱域）
            if args.augment and augmenter is not None:
                x = augmenter.apply_spec_augmentations(x)

            logits = get_logits(model(x))
            loss = (mixup_criterion(criterion, logits, ya, yb, lam)
                    if args.mixup else criterion(logits, y))

            preds_batch = torch.max(logits, 1)[1]
            if args.mixup:
                train_correct += (
                    lam * (preds_batch == ya).float()
                    + (1 - lam) * (preds_batch == yb).float()
                ).sum().item()
            else:
                train_correct += (preds_batch == y).sum().item()
            train_total += y.size(0)
            train_loss_list.append(loss.detach().cpu().item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = args.clip_grad_norm if args.clip_grad_norm > 0 else 1.0
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if ema:
                ema.update()

        train_acc = train_correct / train_total if train_total else 0
        train_loss = float(np.mean(train_loss_list))
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 验证
        if ema:
            ema.apply_shadow()
        try:
            val_f1, val_loss, val_acc = _quick_eval(
                model, val_loader, criterion, device, args, val_dataset, logger)
        finally:
            if ema:
                ema.restore()

        epoch_metrics = {
            'loss/train': train_loss,
            'loss/val': val_loss,
            'acc/train': train_acc,
            'acc/val': val_acc,
            'f1/val': val_f1,
            'lr': current_lr,
        }
        logger.log_epoch(epoch + 1, epoch_metrics)
        logger.log(f"  耗时: {time.time()-t0:.1f}s")

        is_best = ckpt_mgr.update(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={'val_f1': val_f1, 'val_acc': val_acc, 'val_loss': val_loss},
            ema=ema,
            extra={'args': vars(args)},
            start_time=start_time,
        )

        if is_best:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.log(f"早停！{args.patience} 轮内 F1 无提升")
                break

    # 训练完成
    total = time.time() - start_time
    h, r = divmod(total, 3600); m, s = divmod(r, 60)
    logger.log(
        f"训练结束  用时 {int(h):02d}h{int(m):02d}m{int(s):02d}s | "
        f"最佳 val_f1={ckpt_mgr.best_value:.4f} @ epoch {ckpt_mgr.best_epoch+1}"
    )

    # 最终详细评估
    try:
        ckpt_mgr.load_best(model, device, ema=ema)
        if ema:
            ema.apply_shadow()
        try:
            prepare_fn = lambda x: _prepare_input(x, args, val_dataset)
            final_f1, detailed = evaluate_per_class(
                model, val_loader, criterion, device,
                prepare_input_fn=prepare_fn, verbose=True)
        finally:
            if ema:
                ema.restore()
        logger.log_scalar('final/val_f1', final_f1, 0)
        logger.log_class_metrics(detailed['class_metrics'], step=0)
        logger.log(
            f"最终评估: 准确率={detailed['accuracy']*100:.2f}%  "
            f"宏F1={final_f1*100:.2f}%"
        )
    except Exception as e:
        logger.log(f"最终评估失败: {e}")

    hparams = {k: getattr(args, k) for k in [
        'model_name', 'batch_size', 'lr', 'weight_decay', 'epochs', 'pretrained',
        'augment', 'mixup', 'use_ema', 'warmup', 'focal_loss',
        'n_mels', 'hop_length', 'max_length', 'use_smote',
    ]}
    hparams['aug_intensity'] = args.aug_intensity if args.augment else 'none'
    hparams['total_params'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_hparams(hparams, {'hparam/best_f1': ckpt_mgr.best_value})
    logger.close()


def export_model_weights(checkpoint_path, output_path=None, export_ema=True):
    """从完整检查点中单独导出权重文件。"""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if export_ema and 'ema_state_dict' in ckpt:
            weights = ckpt['ema_state_dict'].get('shadow', ckpt['ema_state_dict'])
            tag = 'ema'
        elif 'model_state_dict' in ckpt:
            weights = ckpt['model_state_dict']
            tag = 'model'
        else:
            weights = ckpt
            tag = 'weights'
        if output_path is None:
            base = os.path.splitext(checkpoint_path)[0]
            output_path = f"{base}_{tag}_exported.pt"
        torch.save(weights, output_path)
        print(f"导出 {tag} 权重 → {output_path}")
        return True, output_path
    except Exception as e:
        print(f"导出失败: {e}")
        return False, None


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(
        description='CNN 频谱图模型训练 (MobileNet / DyMN / EfficientNet)')

    # 基本
    parser.add_argument('--experiment_name', type=str, default='CNN_SER')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--run_dir', type=str, default='runs')
    parser.add_argument('--save_interval', type=int, default=10)

    # 数据
    parser.add_argument('--train_dir', type=str, default='datasets/emotion/train')
    parser.add_argument('--val_dir', type=str, default='datasets/emotion/val')

    # 模型
    parser.add_argument('--model_name', type=str, default='dymn20_as',
                        help='mn10_as | mn20_as | dymn20_as | efficientnet_b0 | efficientnet_b5 ...')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default='mlp')
    parser.add_argument('--se_dims', type=str, default='c')
    parser.add_argument('--dropout', type=float, default=0.3)

    # 训练
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--resume_checkpoint', type=str, default='')

    # 特征
    parser.add_argument('--feature_type', type=str, default='mel_spectrogram',
                        choices=['mel_spectrogram', 'mfcc'])
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--hop_length', type=int, default=320)
    parser.add_argument('--max_length', type=int, default=3 * 32000)

    # 增强
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no_augment', dest='augment', action='store_false')
    parser.add_argument('--aug_intensity', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'])
    parser.add_argument('--mixup', action='store_true', default=True)
    parser.add_argument('--no_mixup', dest='mixup', action='store_false')
    parser.add_argument('--mixup_alpha', type=float, default=0.3)

    # 调度器
    parser.add_argument('--warmup', action='store_true', default=True)
    parser.add_argument('--no_warmup', dest='warmup', action='store_false')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6)

    # EMA & 损失
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--focal_loss', action='store_true', default=True)
    parser.add_argument('--no_focal', dest='focal_loss', action='store_false')

    # SMOTE
    parser.add_argument('--use_smote', action='store_true', default=False)
    parser.add_argument('--smote_sampling_strategy', type=float, default=1.0)
    parser.add_argument('--smote_k_neighbors', type=int, default=5)
    
    # 数据预加载
    parser.add_argument('--preload_data', action='store_true', default=False,
                        help='将所有音频数据预加载到内存中，加速训练但占用更多内存')

    # 权重导出
    parser.add_argument('--export_model', type=str, default='')
    parser.add_argument('--export_output', type=str, default=None)
    parser.add_argument('--export_ema', action='store_true', default=True)

    args = parser.parse_args()

    if args.export_model:
        ok, path = export_model_weights(
            args.export_model, args.export_output, args.export_ema)
        sys.exit(0 if ok else 1)

    train(args)
