"""
Dataset utilities for Task 1: Cipher Decryption.
Loads preprocessed tensors from data/preprocessed/task1/
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.preprocessing.task1.tokenizer import tokenize_cipher_line


class CipherDataset(Dataset):
    """Wraps preprocessed cipher/plain tensor pairs."""

    def __init__(self, cipher_tensors, plain_tensors, lengths):
        self.cipher = cipher_tensors
        self.plain = plain_tensors
        self.lengths = lengths

    def __len__(self):
        return len(self.cipher)

    def __getitem__(self, idx):
        return (
            self.cipher[idx],
            self.plain[idx],
            self.lengths[idx],
        )


def load_vocab(data_dir: str):
    """Load cipher and plain vocabularies from pickled files."""
    cipher_vocab_path = os.path.join(data_dir, "cipher_vocab.pkl")
    plain_vocab_path = os.path.join(data_dir, "plain_vocab.pkl")

    with open(cipher_vocab_path, "rb") as f:
        cipher_vocab = pickle.load(f)
    with open(plain_vocab_path, "rb") as f:
        plain_vocab = pickle.load(f)

    return cipher_vocab, plain_vocab


def load_datasets(config: dict):
    """
    Load preprocessed data and split into train/val sets.
    """
    data_dir = config.get("data_dir", os.path.join("data", "processed", "task1"))
    dataset_path = os.path.join(data_dir, "dataset.pt")

    data = torch.load(dataset_path, weights_only=True)

    cipher = data["input"]
    plain = data["target"]
    lengths = data["lengths"]

    full_dataset = CipherDataset(cipher, plain, lengths)

    n = len(full_dataset)
    val_size = max(1, int(n * config.get("val_split", 0.1)))
    train_size = n - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )

    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    # Vocabulary sizes
    cipher_vocab, plain_vocab = load_vocab(data_dir)

    meta = {
        "cipher_vocab_size": len(cipher_vocab),
        "plain_vocab_size": len(plain_vocab),
        "cipher_vocab": cipher_vocab,
        "plain_vocab": plain_vocab,
        "max_len": data.get("max_len", cipher.shape[1]),
    }

    return train_loader, val_loader, meta


def load_test_file(test_path: str, cipher_vocab, max_len: int, device: str = "cpu"):
    """
    Load an external cipher test file and tokenise it.

    Returns:
        tensors  : LongTensor [M, max_len]
        lengths  : LongTensor [M]
        raw_lines: list[str]
    """
    pad_idx = cipher_vocab.char2idx.get('<PAD>', 0)
    unk_idx = cipher_vocab.char2idx.get('<UNK>', pad_idx)

    raw_lines = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_lines.append(line.rstrip("\n"))

    if len(raw_lines) == 0:
        empty_tensors = torch.empty((0, max_len), dtype=torch.long, device=device)
        empty_lengths = torch.empty((0,), dtype=torch.long, device=device)
        return empty_tensors, empty_lengths, raw_lines    

    tensors = []
    lengths = []
    
    for line in raw_lines:
        tokens = tokenize_cipher_line(line)
        if tokens is None:
            # Fallback if invalid chunk occurs
            tokens = []

        if len(tokens) == 0:
            tokens = ["<PAD>"]
            
        indices = cipher_vocab.encode(tokens)

        length = min(len(indices), max_len)
        indices = indices[:max_len]
        indices += [pad_idx] * (max_len - len(indices))
        
        tensors.append(torch.tensor(indices, dtype=torch.long))
        lengths.append(length)

    tensors = torch.stack(tensors).to(device)
    lengths = torch.tensor(lengths, dtype=torch.long).to(device)
    return tensors, lengths, raw_lines