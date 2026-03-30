"""
src/task2/dataset_nwp.py
PyTorch Dataset for Next Word Prediction (SSM).

Formulation
-----------
Given a sequence of length 20:
    input  = seq[:-1]   (positions 0..18)
    target = seq[1:]    (positions 1..19)

The model learns to predict the next token at each step.
Loss uses ignore_index=pad_idx so padded positions are excluded.
"""

import torch
from torch.utils.data import Dataset


class NWPDataset(Dataset):
    """
    Wraps the preprocessed sequence tensor into (input, target) pairs
    for Next Word Prediction without copying data.

    Parameters
    ----------
    sequences : torch.Tensor  shape (N, seq_len)
    """

    def __init__(self, sequences: torch.Tensor) -> None:
        super().__init__()
        self.inputs  = sequences[:, :-1]   # (N, seq_len - 1)
        self.targets = sequences[:, 1:]    # (N, seq_len - 1)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]