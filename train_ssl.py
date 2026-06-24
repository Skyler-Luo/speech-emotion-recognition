import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.config import EMOTION_LABEL_MAP, SSL_SR
from utils.dataset import collect_wav_files, EmotionDataset
from utils.audio_utils import load_and_preprocess
from utils.utils import worker_init_fn
from utils.model_utils import EMA
from utils.logger import TrainingLogger, CheckpointManager


def make_ssl_collate_fn(feature_extractor, target_sr: int = 16000):
    """构建 HuBERT / Wav2Vec2 专用的 DataLoader collate_fn。

    Dataset.__getitem__ 返回 (waveform [1, T], label)。
    输出：(input_values [B, T], attention_mask [B, T], labels [B])
    """
    def collate_fn(batch):
        waveforms, labels = zip(*batch)
        raw = [w.squeeze(0).numpy() for w in waveforms]
        encoded = feature_extractor(
            raw,
            sampling_rate=target_sr,
            padding=True,
            return_tensors='pt',
            return_attention_mask=True,
        )
        return (
            encoded.input_values,
            encoded.attention_mask,
            torch.tensor(labels, dtype=torch.long),
        )
    return collate_fn


def _evaluate(model, loader, criterion, device, logger, desc='评估'):
    """运行验证，返回 (f1_macro, avg_loss, accuracy)。
    batch 格式：(input_values, attention_mask, labels)
    """
    model.eval()
    all_targets, all_preds, losses = [], [], []

    with torch.no_grad(), torch.amp.autocast(device_type=device.type,
                                              enabled=device.type == 'cuda'):
        for wav, attn, y in tqdm(loader, desc=desc, leave=False):
            wav, attn, y = wav.to(device), attn.to(device), y.to(device)
            logits, _ = model(wav, attn)
            losses.append(criterion(logits, y).item())
            all_targets.extend(y.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    acc = metrics.accuracy_score(all_targets, all_preds)
    f1 = metrics.f1_score(all_targets, all_preds, average='macro')
    avg_loss = float(np.mean(losses))

    logger.log(f"  验证 → 准确率: {acc*100:.2f}%  宏F1: {f1*100:.2f}%  Loss: {avg_loss:.4f}")
    for emotion, eidx in EMOTION_LABEL_MAP.items():
        mask = all_targets == eidx
        if mask.sum() > 0:
            cls_acc = metrics.accuracy_score(all_targets[mask], all_preds[mask])
            logger.log(f"    {emotion}: {cls_acc*100:.1f}% ({mask.sum()})")

    return f1, avg_loss, acc


def _train(model, train_loader, val_loader, args, device,
           run_dir: str, logger: TrainingLogger):
    """训练主循环，返回 best_f1。"""
    criterion = nn.CrossEntropyLoss()
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if 'encoder' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr_encoder},
        {'params': head_params, 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_encoder * args.onecycle_max_factor,
                args.lr_head * args.onecycle_max_factor],
        total_steps=total_steps,
        pct_start=0.30,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    ckpt_mgr = CheckpointManager(
        save_dir=run_dir,
        prefix=args.model,
        save_interval=args.save_interval,
        monitor='val_f1',
        logger=logger,
    )

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0

        for wav, attn, y in tqdm(train_loader,
                                  desc=f"Epoch {epoch:02d}/{args.epochs}",
                                  leave=False):
            wav, attn, y = wav.to(device), attn.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=device.type == 'cuda'):
                logits, _ = model(wav, attn)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema:
                ema.update()
            total_loss += loss.item()
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += y.size(0)

        train_acc = n_correct / n_total if n_total else 0
        train_loss = total_loss / len(train_loader)
        cur_lr = scheduler.get_last_lr()[0]

        if ema:
            ema.apply_shadow()
        try:
            val_f1, val_loss, val_acc = _evaluate(
                model, val_loader, criterion, device, logger,
                desc=f"Epoch {epoch:02d} [Val]")
        finally:
            if ema:
                ema.restore()

        epoch_metrics = {
            'loss/train': train_loss,
            'loss/val': val_loss,
            'acc/train': train_acc,
            'acc/val': val_acc,
            'f1/val': val_f1,
            'lr': cur_lr,
        }
        logger.log_epoch(epoch, epoch_metrics)

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

        if not is_best:
            patience_counter = epoch - ckpt_mgr.best_epoch
            if patience_counter >= args.patience:
                logger.log(f"  早停！{args.patience} 轮内 F1 无提升")
                break

    total = time.time() - start_time
    h, r = divmod(total, 3600); m, s = divmod(r, 60)
    logger.log(
        f"训练结束  用时 {int(h):02d}h{int(m):02d}m{int(s):02d}s | "
        f"最佳 val_f1={ckpt_mgr.best_value:.4f} @ epoch {ckpt_mgr.best_epoch}"
    )
    return ckpt_mgr.best_value


def train(args):
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available():
        device = torch.device(
            f'cuda:{args.gpu_id}'
            if 0 <= args.gpu_id < torch.cuda.device_count() else 'cuda:0')
    else:
        device = torch.device('cpu')

    run_dir = os.path.join(args.run_dir, args.experiment_name)
    logger = TrainingLogger(
        run_dir=run_dir,
        experiment_name=args.experiment_name,
    )
    logger.log(f"模型: {args.model}  pool: {args.pool}  设备: {device}")

    # 模型工厂
    if args.model == 'hubert':
        from models.hubert_ser import get_model, get_feature_extractor
    elif args.model == 'wav2vec2':
        from models.wav2vec2_ser import get_model, get_feature_extractor
    else:
        raise ValueError(f"未知模型: {args.model}，支持 'hubert' 或 'wav2vec2'")

    feat_extractor = get_feature_extractor(args.pretrained_path)
    collate_fn = make_ssl_collate_fn(feat_extractor, target_sr=args.target_sr)

    if args.preload_data:
        logger.log("使用预加载数据集模式（数据存储在内存中）")
    else:
        logger.log("使用按需加载模式")

    train_dataset = EmotionDataset(
        dataset_dir=args.train_dir,
        mode='waveform',
        target_sr=args.target_sr,
        max_length=int(args.max_sec * args.target_sr),
        random_offset=True,
        preload=args.preload_data,
        show_progress=True,
    )
    val_dataset = EmotionDataset(
        dataset_dir=args.val_dir,
        mode='waveform',
        target_sr=args.target_sr,
        max_length=int(args.max_sec * args.target_sr),
        random_offset=False,
        preload=args.preload_data,
        show_progress=True,
    )
    num_classes = len(EMOTION_LABEL_MAP)
    logger.log(f"训练集: {len(train_dataset)}  验证集: {len(val_dataset)}  类别数: {num_classes}")
    
    # 如果使用预加载，显示加载失败的文件
    if args.preload_data:
        failed_train = train_dataset.get_failed_files()
        failed_val = val_dataset.get_failed_files()
        if failed_train:
            logger.log(f"训练集加载失败文件数: {len(failed_train)}")
        if failed_val:
            logger.log(f"验证集加载失败文件数: {len(failed_val)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=collate_fn, worker_init_fn=worker_init_fn,
        pin_memory=True if args.cuda else False)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn, worker_init_fn=worker_init_fn,
        pin_memory=True if args.cuda else False)

    model = get_model(
        num_classes=num_classes,
        pretrained_path=args.pretrained_path,
        pool=args.pool,
        freeze_feature_extractor=args.freeze_feature_extractor,
        dropout=args.dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.log(f"参数: 可训练 {trainable:,} / 总计 {total_p:,}")

    best_f1 = _train(model, train_loader, val_loader, args, device, run_dir, logger)

    hparams = {k: getattr(args, k) for k in [
        'model', 'pool', 'batch_size', 'lr_encoder', 'lr_head',
        'weight_decay', 'epochs', 'dropout',
        'freeze_feature_extractor', 'use_ema', 'ema_decay',
    ]}
    logger.log_hparams(hparams, {'hparam/best_f1': best_f1})
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSL 模型语音情感识别训练')

    parser.add_argument('--experiment_name', type=str, default='SSL_SER')
    parser.add_argument('--model', type=str, default='hubert',
                        choices=['hubert', 'wav2vec2'])
    parser.add_argument('--pretrained_path', type=str, default='facebook/hubert-base-ls960')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_dir', type=str, default='runs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='每 N 轮保存一次完整周期检查点，<=0 禁用')

    parser.add_argument('--train_dir', type=str, default='datasets/emotion/train')
    parser.add_argument('--val_dir', type=str, default='datasets/emotion/val')
    parser.add_argument('--target_sr', type=int, default=SSL_SR)
    parser.add_argument('--max_sec', type=float, default=3.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--pool', type=str, default='attn',
                        choices=['attn', 'mean', 'stat'])
    parser.add_argument('--freeze_feature_extractor', action='store_true', default=True)
    parser.add_argument('--no_freeze', dest='freeze_feature_extractor', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_encoder', type=float, default=3e-5)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--onecycle_max_factor', type=float, default=1.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)

    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    parser.add_argument('--preload_data', action='store_true', default=False,
                        help='将所有音频数据预加载到内存中，加速训练但占用更多内存')

    args = parser.parse_args()
    train(args)
