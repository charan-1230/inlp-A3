"""
Dataset for Masked Language Modeling (Bi-LSTM).
Dynamic masking is applied in __getitem__, NOT cached.

Masking strategy (BERT-style):
    15% of non-PAD tokens selected:
        80% → <MASK>
        10% → random valid token
        10% → unchanged
Labels: original token id at masked positions, -100 elsewhere.
"""

import random

import torch
from torch.utils.data import Dataset


class MLMDataset(Dataset):
    """
    Parameters
    ----------
    tensors   : dict with key "sequences" → Tensor (N, seq_len),
                OR a bare Tensor (N, seq_len)
    vocab     : VocabTask2 instance
    config    : config dict; reads "mask_prob" (default 0.15)
    """

    def __init__(self, tensors, vocab, config: dict) -> None:
        super().__init__()

        if isinstance(tensors, dict) and "sequences" in tensors:
            self.data = tensors["sequences"]
        else:
            self.data = tensors

        self.vocab     = vocab
        self.mask_prob = config.get("mask_prob", 0.15)

        self.pad_idx  = vocab.word2idx.get("<PAD>",  0)
        self.mask_idx = vocab.word2idx.get("<MASK>", 2)

        # Build the pool of valid random-replacement indices
        # (excludes all special tokens)
        special = {"<PAD>", "<UNK>", "<MASK>", "<EOS>"}
        self.valid_indices = [
            idx for token, idx in vocab.word2idx.items()
            if token not in special
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq    = self.data[idx].clone()
        labels = torch.full_like(seq, -100)

        for i in range(seq.size(0)):
            token = seq[i].item()

            if token == self.pad_idx:       # never mask padding
                continue

            if random.random() < self.mask_prob:
                labels[i] = token           # remember original

                prob = random.random()
                if prob < 0.80:
                    seq[i] = self.mask_idx
                elif prob < 0.90:
                    seq[i] = random.choice(self.valid_indices)
                # else: 10% keep unchanged

        return seq, labels