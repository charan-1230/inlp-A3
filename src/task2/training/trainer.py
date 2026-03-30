"""
Model-agnostic Trainer — handles the outer training loop, WandB logging,
early stopping, and checkpoint saving / loading.
"""

import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.task2.training.engine import run_epoch
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb

class _Tee:
    """Writes to both the original stdout and a log file."""
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.__stdout__

    def write(self, text: str):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def isatty(self):
        return False

    def fileno(self):
        return self._stdout.fileno()


class Trainer:
    """
    Trains any nn.Module that maps (batch, seq_len) → (batch, seq_len, vocab).

    Parameters
    ----------
    model         : the model to train
    train_loader  : DataLoader for the training split
    val_loader    : DataLoader for the validation split
    criterion     : loss function (CrossEntropyLoss with appropriate ignore_index)
    optimizer     : Adam / SGD / etc.
    device        : torch.device
    config        : full config dict (passed to WandB)
    ckpt_path     : where to save the best model checkpoint
    wandb_name    : run name shown in WandB dashboard
    patience      : early stopping patience (epochs without improvement)
    grad_clip     : max gradient norm (0 to disable)
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        criterion:    nn.CrossEntropyLoss,
        optimizer:    torch.optim.Optimizer,
        device:       torch.device,
        config:       dict,
        ckpt_path:    str,
        wandb_name:   str,
        patience:     int   = 3,
        grad_clip:    float = 1.0,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.device       = device
        self.config       = config
        self.ckpt_path    = ckpt_path
        self.wandb_name   = wandb_name
        self.patience     = patience
        self.grad_clip    = grad_clip

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, epochs: int) -> float:
        """
        Run the full training loop.

        Returns
        -------
        best_val_loss : float
        """
        init_wandb(
            project="inlp-a3-task2",
            config=self.config,
            name=self.wandb_name,
        )

        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir     = self.config.get("log_dir", os.path.join("outputs", "logs"))
        log_path    = os.path.join(log_dir, f"{self.wandb_name}_{timestamp}.txt")
        self._tee   = _Tee(log_path)
        sys.stdout  = self._tee
        print(f"[Trainer] Log file: {log_path}")

        best_val_loss = float("inf")
        no_improve    = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_ppl = run_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.grad_clip,
                desc=f"Epoch {epoch:3d}/{epochs} [Train]"
            )
            val_loss, val_ppl = run_epoch(
                self.model, self.val_loader, self.criterion,
                None, self.device,
                desc=f"Epoch {epoch:3d}/{epochs} [Val  ]"
            )

            elapsed = time.time() - t0
            self._log_epoch(epoch, train_loss, train_ppl, val_loss, val_ppl, elapsed)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, self.ckpt_path
                )
                print(f"    ✓ checkpoint saved  (val_loss={val_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping triggered after {epoch} epochs "
                          f"(patience={self.patience})")
                    break

        finish_wandb()
        sys.stdout = sys.__stdout__
        self._tee.close()
        return best_val_loss

    def evaluate(self) -> tuple[float, float]:
        """
        Load the best checkpoint and run one validation pass.

        Returns
        -------
        (val_loss, val_perplexity)
        """
        info = load_checkpoint(self.ckpt_path, self.model, device=str(self.device))
        print(f"  Loaded checkpoint  epoch={info['epoch']}  "
              f"loss={info['loss']:.4f}")
        self.model.to(self.device)

        val_loss, val_ppl = run_epoch(
            self.model, self.val_loader, self.criterion,
            None, self.device,
            desc="[Eval ]"
        )
        return val_loss, val_ppl

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch:      int,
        train_loss: float,
        train_ppl:  float,
        val_loss:   float,
        val_ppl:    float,
        elapsed:    float,
    ) -> None:
        print(
            f"  epoch={epoch:03d}  "
            f"train_loss={train_loss:.4f}  train_ppl={train_ppl:.2f}  "
            f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  "
            f"({elapsed:.1f}s)"
        )
        log_wandb(
            {
                "train_loss":       train_loss,
                "val_loss":         val_loss,
                "train_perplexity": train_ppl,
                "val_perplexity":   val_ppl,
                "epoch":            epoch,
            },
            step=epoch,
        )