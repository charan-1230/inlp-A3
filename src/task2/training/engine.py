"""
Reusable train / validation loop for both SSM and BiLSTM.
"""

import math
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.CrossEntropyLoss,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    clip:       float = 1.0,
    desc:       str = "",
) -> tuple[float, float]:
    """
    Run one full pass over `loader`.

    Parameters
    ----------
    optimizer : pass None to run in eval mode (no gradients, no updates)
    clip      : max-norm for gradient clipping (training only)

    Returns
    -------
    avg_loss   : token-weighted average cross-entropy loss
    perplexity : exp(avg_loss)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, file=sys.__stdout__)

    with ctx:
        for inputs, targets in pbar:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)                      # (B, T, V)
            B, T, V = logits.shape

            loss = criterion(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            # Count only non-ignored tokens for a correct token-level loss
            n_tokens     = (targets != criterion.ignore_index).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss   = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))            # clamp to avoid overflow
    return avg_loss, perplexity