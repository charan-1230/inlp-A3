"""
Shared helpers: reproducibility, device selection, data loading.
"""

import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join("data", "processed", "task2")


def load_vocab_and_sequences(data_dir: str = DATA_DIR):
    """
    Load preprocessed tensors and vocabulary produced by preprocess_task2.py.

    Returns
    -------
    train_sequences : torch.Tensor  shape (N_train, 20)
    val_sequences   : torch.Tensor  shape (N_val, 20)
    vocab           : VocabTask2    with .word2idx / .idx2word
    """
    dataset_path = os.path.join(data_dir, "dataset_task2.pt")
    vocab_path   = os.path.join(data_dir, "vocab_task2.pkl")

    tensors = torch.load(dataset_path, weights_only=True)
    train_sequences: torch.Tensor = tensors["train_sequences"]
    val_sequences: torch.Tensor   = tensors["val_sequences"]

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    return train_sequences, val_sequences, vocab


def make_dataloader(dataset, batch_size: int, shuffle: bool, seed: int = 42) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        generator=g,
        worker_init_fn=worker_init_fn
    )