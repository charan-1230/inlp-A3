"""
Entry point for `uv run main.py task2_ssm --mode {train,evaluate,both}`.

Wires together:
    preprocessing outputs  →  NWPDataset  →  SSMModel  →  Trainer
"""

import os

import torch
import torch.nn as nn
import yaml

from src.task2.dataset_nwp import NWPDataset
from src.task2.models.ssm_model import SSMModel
from src.task2.training.trainer import Trainer
from src.task2.utils import (
    get_device,
    load_vocab_and_sequences,
    make_dataloader,
    set_seed,
)

OUTPUT_DIR = os.path.join("outputs", "checkpoints")
CKPT_PATH  = os.path.join(OUTPUT_DIR, "task2_ssm.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_components(config: dict):
    """Load data, build model, criterion, optimizer, dataloaders."""
    seed   = config.get("seed", 42)
    device = get_device()
    set_seed(seed)

    # --- data ---
    train_sequences, val_sequences, vocab = load_vocab_and_sequences()
    pad_idx    = vocab.word2idx.get("<PAD>", 0)
    vocab_size = len(vocab.word2idx)

    train_ds             = NWPDataset(train_sequences)
    val_ds               = NWPDataset(val_sequences)
    train_loader         = make_dataloader(train_ds, config["batch_size"], shuffle=True, seed=seed)
    val_loader           = make_dataloader(val_ds,   config["batch_size"], shuffle=False, seed=seed)

    # --- model ---
    model = SSMModel(
        vocab_size = vocab_size,
        embed_dim  = config["embed_dim"],
        hidden_dim = config["hidden_dim"],
        pad_idx    = pad_idx,
        num_layers = config.get("num_layers", 2),
        dropout    = config.get("dropout", 0.1),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))

    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        optimizer    = optimizer,
        device       = device,
        config       = config,
        ckpt_path    = config.get("checkpoint_path", CKPT_PATH),
        wandb_name   = config.get("run_tag", "task2_ssm"),
        patience     = config.get("patience", 3),
        grad_clip    = config.get("grad_clip", 1.0),
    )

    return trainer, device, vocab


# ---------------------------------------------------------------------------
# Public API (called by main.py)
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    print(f"[task2_ssm] Starting training  device={get_device()}")
    trainer, _, _ = _build_components(config)
    best = trainer.train(epochs=config.get("max_epochs", 10))
    print(f"[task2_ssm] Training complete.  Best val_loss={best:.4f}")


import json
def evaluate(config: dict) -> None:
    print("[task2_ssm] Evaluating best checkpoint …")
    trainer, device, vocab = _build_components(config)
    val_loss, val_ppl = trainer.evaluate()
    print(f"[task2_ssm] val_loss={val_loss:.4f}  val_perplexity={val_ppl:.2f}")

    os.makedirs(os.path.join("outputs", "results"), exist_ok=True)
    with open(os.path.join("outputs", "results", "task2_ssm_metrics.json"), "w") as f:
        json.dump({"val_loss": val_loss, "val_ppl": val_ppl}, f, indent=4)
        
    trainer.model.eval()
    with open(os.path.join("outputs", "results", "task2_ssm.txt"), "w") as f:
        with torch.no_grad():
            for inputs, targets in trainer.val_loader:
                inputs = inputs.to(device)
                logits = trainer.model(inputs)
                preds  = logits.argmax(dim=-1)
                for i in range(min(5, len(inputs))):
                    inp_toks = vocab.decode(inputs[i].tolist())
                    tgt_toks = vocab.decode(targets[i].tolist())
                    prd_toks = vocab.decode(preds[i].tolist())
                    f.write(f"Example {i+1}\n")
                    f.write(f"Input:    {' '.join(inp_toks)}\n")
                    f.write(f"Target:   {' '.join(tgt_toks)}\n")
                    f.write(f"Predict:  {' '.join(prd_toks)}\n\n")
                break


def main(config_path: str, mode: str) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if mode == "train":
        train(config)
    elif mode == "evaluate":
        evaluate(config)
    elif mode == "both":
        train(config)
        evaluate(config)
    else:
        raise ValueError(f"Unsupported mode: {mode}")