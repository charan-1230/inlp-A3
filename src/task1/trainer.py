"""
Generic Sequence Labeling trainer for Task 1.
Works for both SeqLabelRNN and SeqLabelLSTM.
"""

import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.checkpoints import save_checkpoint, load_checkpoint
from src.utils.hf_wandb import log_wandb


# ---------------------------------------------------------------------------
# Helper: decode token indices → string
# ---------------------------------------------------------------------------

def indices_to_string(indices, idx2char: dict, pad_idx: int) -> str:
    """
    Decodes a sequence of indices back to a string, stopping at <PAD>.
    (No <EOS> in sequence labeling).
    """
    chars = []
    for idx in indices:
        idx = idx.item() if hasattr(idx, "item") else int(idx)
        if idx == pad_idx:
            break
        ch = idx2char.get(idx, "")
        chars.append(ch)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Tee: mirror stdout to a log file
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EarlyStopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Char-accuracy helper (on logits)
# ---------------------------------------------------------------------------

def _char_accuracy_from_logits(
    logits: torch.Tensor,       # (batch, seq_len, vocab)
    targets: torch.Tensor,      # (batch, seq_len)
    pad_idx: int,
) -> tuple[int, int]:
    """Token-level accuracy ignoring pad positions."""
    preds = logits.argmax(dim=-1)          # (batch, seq_len)
    mask  = targets != pad_idx
    correct = ((preds == targets) & mask).sum().item()
    total   = mask.sum().item()
    return correct, total


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        meta: dict,
        checkpoint_path: str,
        use_wandb: bool = True,
    ):
        self.model           = model
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.config          = config
        self.meta            = meta
        self.checkpoint_path = checkpoint_path
        self.use_wandb       = use_wandb

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.run_tag = config.get("run_tag", "run").replace(" ", "_")

        lr = config.get("learning_rate", 1e-3)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        pad_idx = meta.get("pad_idx", 0)
        self.pad_idx   = pad_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.max_epochs = config.get("max_epochs", 25)
        self.grad_clip = config.get("grad_clip", 1.0)

        self.early_stopping = EarlyStopping(
            patience  = config.get("patience",  3),
            min_delta = config.get("min_delta", 1e-4),
        )

        plain_vocab     = meta.get("plain_vocab", {})
        self.idx2plain  = {v: k for k, v in plain_vocab.items()}

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir     = config.get("log_dir", os.path.join("outputs", "logs"))
        log_path    = os.path.join(log_dir, f"{self.run_tag}_{timestamp}.txt")
        self._tee   = _Tee(log_path)
        sys.stdout  = self._tee
        print(f"[Trainer] Log file: {log_path}")

        self._start_epoch = 1


    def maybe_resume(self, hf_repo_id: str | None = None) -> None:
        from src.utils.checkpoints import load_checkpoint
        from src.utils.hf_wandb import pull_from_hub

        ckpt_path = self.checkpoint_path

        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[Trainer] Resuming from local checkpoint: {ckpt_path}")
            info = load_checkpoint(ckpt_path, self.model, self.optimizer, device=str(self.device))
            self._start_epoch = info["epoch"] + 1
            print(f"[Trainer] Resumed at epoch {info['epoch']}  (val_loss={info['loss']:.4f})")
            return

        if hf_repo_id:
            print(f"[Trainer] Local checkpoint not found. Pulling from HuggingFace: {hf_repo_id}")
            try:
                local_dir = os.path.dirname(ckpt_path) if ckpt_path else os.path.join("outputs", "checkpoints")
                filename  = os.path.basename(ckpt_path) if ckpt_path else "checkpoint.pt"
                pulled = pull_from_hub(hf_repo_id, filename, local_dir=local_dir)
                info = load_checkpoint(pulled, self.model, self.optimizer, device=str(self.device))
                self._start_epoch = info["epoch"] + 1
                print(f"[Trainer] Resumed from HuggingFace at epoch {info['epoch']}  (val_loss={info['loss']:.4f})")
                return
            except Exception as e:
                print(f"[Trainer] WARNING: HuggingFace pull failed ({e}). Starting from scratch.")

        print("[Trainer] No checkpoint found — starting from scratch.")


    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens  = 0
        n_batches  = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:3d}/{self.max_epochs} [Train]",
            leave=False,
            dynamic_ncols=True,
            file=sys.__stdout__,
        )

        for cipher, plain, lengths in pbar:
            cipher = cipher.to(self.device)
            plain  = plain.to(self.device)
            lengths  = lengths.to(self.device)

            self.optimizer.zero_grad()
            
            # Sequence labeling forward pass
            logits = self.model(cipher, lengths)

            logits_2d = logits.reshape(-1, logits.size(-1))
            tgt_1d    = plain.reshape(-1)

            loss = self.criterion(logits_2d, tgt_1d)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            c, t = _char_accuracy_from_logits(logits, plain, self.pad_idx)
            total_correct += c
            total_tokens += t
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        return (
            total_loss / max(n_batches, 1),
            acc,
        )


    def _val_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens  = 0
        n_batches  = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:3d}/{self.max_epochs} [Val  ]",
            leave=False,
            dynamic_ncols=True,
            file=sys.__stdout__,
        )

        with torch.no_grad():
            for cipher, plain, lengths in pbar:
                cipher = cipher.to(self.device)
                plain  = plain.to(self.device)
                lengths  = lengths.to(self.device)

                logits    = self.model(cipher, lengths)

                logits_2d = logits.reshape(-1, logits.size(-1))
                tgt_1d    = plain.reshape(-1)

                loss       = self.criterion(logits_2d, tgt_1d)
                total_loss += loss.item()
                c, t = _char_accuracy_from_logits(logits, plain, self.pad_idx)
                total_correct += c
                total_tokens += t
                n_batches  += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        return (
            total_loss / max(n_batches, 1),
            acc,
        )


    def train(self):
        best_val_loss = float("inf")
        best_epoch    = 0
        print(f"[Trainer] Training on {self.device}  |  epochs {self._start_epoch}→{self.max_epochs}")

        for epoch in range(self._start_epoch, self.max_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss,   val_acc   = self._val_epoch(epoch)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"time={elapsed:.1f}s | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
            )

            if self.use_wandb:
                log_wandb(
                    {
                        "train_loss": train_loss,
                        "val_loss":   val_loss,
                        "train_acc":  train_acc,
                        "val_acc":    val_acc,
                    },
                    step=epoch,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    self.checkpoint_path,
                )
                print(f"  ✓ Checkpoint saved  (val_loss={val_loss:.4f})")

            if self.early_stopping.step(val_loss):
                print(f"[Trainer] Early stopping triggered at epoch {epoch}.")
                break

        print(f"\n[Trainer] Training complete. Best val_loss={best_val_loss:.4f} @ epoch {best_epoch}.")

        sys.stdout = sys.__stdout__
        self._tee.close()

        return best_val_loss