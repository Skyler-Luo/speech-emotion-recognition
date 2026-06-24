import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import EMOTION_LABEL_MAP, NUM_CLASSES, DEFAULT_SR
from utils.dataset import EmotionDataset
from utils.utils import worker_init_fn
from utils.training import WarmupCosineScheduler
from utils.model_utils import EMA, evaluate_per_class, run_inference
from utils.logger import TrainingLogger, CheckpointManager
from models.cnn_bilstm import get_model, extract_baseline_feature, CNNBiLSTM


def _get_logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _quick_eval(model, loader, criterion, device, logger):
    """验证一轮，返回 (f1_macro, avg_loss, accuracy)。"""
    model.eval()
    all_targets, all_preds, losses = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='验证', leave=False):
            x, y = x.to(device), y.to(device)
            logits = _get_logits(model(x))
            losses.append(criterion(logits, y).item())
            all_targets.append(y.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    acc = metrics.accuracy_score(targets, preds)
    f1 = metrics.f1_score(targets, preds, average='macro')
    loss = float(np.mean(losses))

    logger.log(f"  验证 → 准确率: {acc*100:.2f}%  宏F1: {f1*100:.2f}%  Loss: {loss:.4f}")
    for emotion, eidx in EMOTION_LABEL_MAP.items():
        mask = targets == eidx
        if mask.sum() > 0:
            cls_acc = metrics.accuracy_score(targets[mask], preds[mask])
            logger.log(f"    {emotion}: {cls_acc*100:.1f}% ({mask.sum()})")
    return f1, loss, acc


def train(args):
    run_dir = os.path.join(args.run_dir, args.experiment_name)
    logger = TrainingLogger(
        run_dir=run_dir,
        experiment_name=args.experiment_name,
    )
    ckpt_mgr = CheckpointManager(
        save_dir=run_dir,
        prefix='cnn_bilstm',
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

    # 数据集
    if args.preload_data:
        logger.log("使用预加载数据集模式")
    else:
        logger.log("使用按需加载模式")
    
    train_dataset = EmotionDataset(
        dataset_dir=args.train_dir,
        mode='cnn_bilstm',
        target_sr=args.sample_rate,
        max_length=args.max_length,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=args.max_frames,
        normalize=True,
        random_offset=True,
        preload=args.preload_data,
        show_progress=True,
    )
    val_dataset = EmotionDataset(
        dataset_dir=args.val_dir,
        mode='cnn_bilstm',
        target_sr=args.sample_rate,
        max_length=args.max_length,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_frames=args.max_frames,
        normalize=True,
        random_offset=False,
        preload=args.preload_data,
        show_progress=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        pin_memory=True if args.cuda else False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        pin_memory=True if args.cuda else False)
    logger.log(f"训练集: {len(train_dataset)}  验证集: {len(val_dataset)}")
    
    # 如果使用预加载，显示加载失败的文件
    if args.preload_data:
        failed_train = train_dataset.get_failed_files()
        failed_val = val_dataset.get_failed_files()
        if failed_train:
            logger.log(f"训练集加载失败文件数: {len(failed_train)}")
        if failed_val:
            logger.log(f"验证集加载失败文件数: {len(failed_val)}")

    # 模型
    model = get_model(
        num_classes=NUM_CLASSES,
        n_mfcc=args.n_mfcc,
        cnn_out_channels=args.cnn_out_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"模型: CNNBiLSTM  可训练参数: {n_params:,}")
    logger.log(f"  n_mfcc={args.n_mfcc}  cnn_out={args.cnn_out_channels}  "
               f"lstm_hidden={args.lstm_hidden}  lstm_layers={args.lstm_layers}  "
               f"dropout={args.dropout}  max_frames={args.max_frames}")

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema:
        logger.log(f"EMA 衰减率: {args.ema_decay}")

    # 损失 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=max(3, args.warmup_epochs),
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
            x, y = x.to(device), y.to(device)  # x: [B, T, D]

            logits = _get_logits(model(x))
            loss = criterion(logits, y)

            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
            train_loss_list.append(loss.detach().cpu().item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
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
                model, val_loader, criterion, device, logger)
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

    # 最终评估（加载最佳模型）
    try:
        ckpt_mgr.load_best(model, device, ema=ema)
        if ema:
            ema.apply_shadow()
        try:
            def _prep(x): return x  # 已是 [B, T, D]
            final_f1, detailed = evaluate_per_class(
                model, val_loader, criterion, device,
                prepare_input_fn=_prep, verbose=True)
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

    hparams = {
        'model': 'cnn_bilstm',
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'n_mfcc': args.n_mfcc,
        'max_frames': args.max_frames,
        'cnn_out_channels': args.cnn_out_channels,
        'lstm_hidden': args.lstm_hidden,
        'lstm_layers': args.lstm_layers,
        'dropout': args.dropout,
        'use_ema': args.use_ema,
    }
    logger.log_hparams(hparams, {'hparam/best_f1': ckpt_mgr.best_value})
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN-BiLSTM 模型训练')

    # 基本
    parser.add_argument('--experiment_name', type=str, default='CNNBiLSTM_SER')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_dir', type=str, default='runs')
    parser.add_argument('--save_interval', type=int, default=10)

    # 数据
    parser.add_argument('--train_dir', type=str, default='datasets/emotion/train')
    parser.add_argument('--val_dir', type=str, default='datasets/emotion/val')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    # 音频 / 特征
    parser.add_argument('--sample_rate', type=int, default=DEFAULT_SR)
    parser.add_argument('--max_length', type=int, default=3 * DEFAULT_SR,
                        help='最大采样点数（波形截断长度）')
    parser.add_argument('--n_mfcc', type=int, default=10)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--win_length', type=int, default=2048)
    parser.add_argument('--max_frames', type=int, default=300,
                        help='特征时间帧数（填充/截断目标长度）')

    # 模型结构
    parser.add_argument('--cnn_out_channels', type=int, default=64)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--resume_checkpoint', type=str, default='')

    # 调度器
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6)

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    # 数据预加载
    parser.add_argument('--preload_data', action='store_true', default=False,
                        help='将所有音频数据预加载到内存中，加速训练但占用更多内存')
    
    args = parser.parse_args()
    train(args)
