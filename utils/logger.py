from __future__ import annotations

import datetime
import os
import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """将训练日志同时写入控制台、.log 文件和 TensorBoard。"""

    def __init__(
        self,
        log_dir: str,
        save_dir: str,
        experiment_name: str,
        tb_suffix: str = "",
    ):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.save_dir = save_dir
        self.experiment_name = experiment_name

        # .log 文件
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(save_dir, f"train_{ts}.log")

        # TensorBoard
        tb_name = f"{experiment_name}_{ts}"
        if tb_suffix:
            tb_name += f"_{tb_suffix}"
        self.tb_dir = os.path.join(log_dir, tb_name)
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        self.log(f"实验: {experiment_name}")
        self.log(f"日志文件: {self.log_path}")
        self.log(f"TensorBoard: {self.tb_dir}")

    def log(self, msg: str):
        """带时间戳打印到控制台，并追加写入 .log 文件。"""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_scalar(self, tag: str, value: float, step: int):
        """写入单个 TensorBoard 标量。"""
        self.writer.add_scalar(tag, value, step)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """批量写入一组 epoch 级标量，同时打印一行摘要到日志。

        metrics 的 key 直接用作 TensorBoard tag，建议遵循
        ``{group}/{name}`` 格式（如 ``loss/train``、``acc/val``）。
        """
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, epoch)

        # 控制台摘要：只打印"易读"的核心指标
        _fmt = {
            "loss/train": "TrainLoss",
            "loss/val":   "ValLoss",
            "acc/train":  "TrainAcc",
            "acc/val":    "ValAcc",
            "f1/val":     "ValF1",
            "lr":         "LR",
        }
        parts = []
        for key, label in _fmt.items():
            if key in metrics:
                v = metrics[key]
                if key == "lr":
                    parts.append(f"{label}={v:.2e}")
                elif "loss" in key:
                    parts.append(f"{label}={v:.4f}")
                else:
                    parts.append(f"{label}={v*100:.2f}%")
        if parts:
            self.log(f"Epoch {epoch:>4d} | " + " | ".join(parts))

    def log_fold_epoch(self, fold: int, epoch: int, metrics: Dict[str, float],
                       global_step: int):
        """K-Fold 场景：在 ``Fold{N}/`` 命名空间下写入标量。"""
        prefixed = {f"Fold{fold}/{k}": v for k, v in metrics.items()}
        for tag, value in prefixed.items():
            self.writer.add_scalar(tag, value, global_step)

        parts = []
        _fmt = {
            "loss/train": "TrainLoss",
            "loss/val":   "ValLoss",
            "acc/train":  "TrainAcc",
            "acc/val":    "ValAcc",
            "f1/val":     "ValF1",
            "lr":         "LR",
        }
        for key, label in _fmt.items():
            if key in metrics:
                v = metrics[key]
                if key == "lr":
                    parts.append(f"{label}={v:.2e}")
                elif "loss" in key:
                    parts.append(f"{label}={v:.4f}")
                else:
                    parts.append(f"{label}={v*100:.2f}%")
        if parts:
            self.log(
                f"Fold {fold} | Epoch {epoch:>3d} | " + " | ".join(parts)
            )

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """写入超参数面板（训练结束时调用一次）。"""
        self.writer.add_hparams(hparams, metrics)

    def log_class_metrics(self, class_metrics: Dict, step: int = 0):
        """将每类别的指标写入 TensorBoard ``class/`` 命名空间。"""
        for emotion, m in class_metrics.items():
            for k, v in m.items():
                if k != "samples":
                    self.writer.add_scalar(f"class/{emotion}/{k}", v, step)

    def close(self):
        self.writer.close()
        self.log("训练完成，日志已关闭。")


class CheckpointManager:
    """统一的检查点与模型权重保存管理器。

    保存策略
    --------
    best checkpoint  : ``{prefix}_best.pt``
        触发条件：监控指标（默认 val_f1）创新高。
        内容：完整训练状态（model / optimizer / scheduler / ema / 元信息）。

    best weights     : ``{prefix}_weights_best.pt``
        与 best checkpoint 同步保存，仅含纯 state_dict，体积小，用于推理/导出。

    periodic checkpoint : ``{prefix}_epoch_{N}.pt``
        每 ``save_interval`` 轮保存一次完整训练状态（含 optimizer/scheduler），
        便于从任意轮次精确续训。``save_interval <= 0`` 时禁用。

    last checkpoint  : ``{prefix}_last.pt``
        每轮结束都覆盖保存（完整状态），保证训练意外中断后可续训。
    """

    def __init__(
        self,
        save_dir: str,
        prefix: str,
        save_interval: int = 10,
        monitor: str = "val_f1",
        logger: Optional[TrainingLogger] = None,
    ):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.prefix = prefix
        self.save_interval = save_interval
        self.monitor = monitor
        self.logger = logger

        self.best_value: float = 0.0
        self.best_epoch: int = -1

        self.best_ckpt_path = os.path.join(save_dir, f"{prefix}_best.pt")
        self.best_weights_path = os.path.join(save_dir, f"{prefix}_weights_best.pt")
        self.last_ckpt_path = os.path.join(save_dir, f"{prefix}_last.pt")

    def _build_ckpt(
        self,
        epoch: int,
        model: nn.Module,
        optimizer,
        scheduler,
        metrics: Dict[str, float],
        ema=None,
        extra: Optional[Dict] = None,
        start_time: Optional[float] = None,
    ) -> dict:
        import time
        ckpt: Dict[str, Any] = {
            "epoch":              epoch,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            **metrics,
        }
        if ema is not None:
            ckpt["ema_state_dict"] = ema.state_dict()
        if extra:
            ckpt.update(extra)
        if start_time is not None:
            ckpt["training_time"] = time.time() - start_time
        return ckpt

    def _save_weights(self, model: nn.Module, ema=None) -> dict:
        """返回用于 weights_best.pt 的纯 state_dict。
        优先保存 EMA shadow 权重。
        """
        if ema is not None and hasattr(ema, "shadow"):
            return ema.shadow
        return model.state_dict()

    def _log(self, msg: str):
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)

    def update(
        self,
        epoch: int,
        model: nn.Module,
        optimizer,
        scheduler,
        metrics: Dict[str, float],
        ema=None,
        extra: Optional[Dict] = None,
        start_time: Optional[float] = None,
    ) -> bool:
        """保存 last checkpoint；若监控指标创新高则保存 best；
        符合间隔条件则保存 periodic。

        Returns:
            True 表示本轮是新的最佳。
        """
        ckpt = self._build_ckpt(
            epoch, model, optimizer, scheduler, metrics, ema, extra, start_time
        )

        torch.save(ckpt, self.last_ckpt_path)

        if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
            periodic_path = os.path.join(
                self.save_dir, f"{self.prefix}_epoch_{epoch+1}.pt"
            )
            torch.save(ckpt, periodic_path)
            self._log(f"  [ckpt] 周期检查点 → {os.path.basename(periodic_path)}")

        monitor_value = metrics.get(self.monitor, 0.0)
        is_best = monitor_value > self.best_value
        if is_best:
            self.best_value = monitor_value
            self.best_epoch = epoch
            torch.save(ckpt, self.best_ckpt_path)
            torch.save(self._save_weights(model, ema), self.best_weights_path)
            self._log(
                f"  [ckpt] 最佳模型更新 {self.monitor}={monitor_value:.4f} "
                f"@ epoch {epoch+1}  → {os.path.basename(self.best_ckpt_path)}"
            )

        return is_best

    def load_best(
        self,
        model: nn.Module,
        device,
        optimizer=None,
        scheduler=None,
        ema=None,
    ) -> dict:
        """加载 best checkpoint 到 model（及可选的 optimizer/scheduler/ema）。

        Returns:
            checkpoint dict（包含 epoch、metrics 等元信息）。
        """
        if not os.path.exists(self.best_ckpt_path):
            raise FileNotFoundError(f"最佳检查点不存在: {self.best_ckpt_path}")
        ckpt = torch.load(self.best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ema and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        return ckpt

    def load_last(
        self,
        model: nn.Module,
        device,
        optimizer=None,
        scheduler=None,
        ema=None,
    ) -> dict:
        """加载 last checkpoint（用于意外中断后续训）。"""
        if not os.path.exists(self.last_ckpt_path):
            raise FileNotFoundError(f"last 检查点不存在: {self.last_ckpt_path}")
        ckpt = torch.load(self.last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ema and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        return ckpt

    def resume(
        self,
        checkpoint_path: str,
        model: nn.Module,
        device,
        optimizer=None,
        scheduler=None,
        ema=None,
    ) -> int:
        """从任意 checkpoint 文件续训，返回 next_epoch（epoch + 1）。"""
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ema and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        next_epoch = ckpt.get("epoch", 0) + 1
        self.best_value = ckpt.get(self.monitor, 0.0)
        self._log(
            f"从检查点恢复 epoch {next_epoch}，当前最佳 {self.monitor}={self.best_value:.4f}"
        )
        return next_epoch
