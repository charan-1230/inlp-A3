"""
Sequence Labeling model with a manually implemented RNN (no nn.RNN).

Architecture:
    Embedding → stacked RNN cells → linear projection
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Primitive RNN cell (one layer)
# ---------------------------------------------------------------------------

class RNNCell(nn.Module):
    """
    h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, input_size)
        h : (batch, hidden_size)
        returns h_new : (batch, hidden_size)
        """
        return torch.tanh(self.W_xh(x) + self.W_hh(h))


# ---------------------------------------------------------------------------
# Full Sequence Labeling RNN
# ---------------------------------------------------------------------------

class SeqLabelRNN(nn.Module):
    def __init__(self, config: dict, cipher_vocab_size: int, plain_vocab_size: int,
                 pad_idx: int = 0):
        super().__init__()
        embed_dim  = config.get("embed_dim", 64)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_layers = config.get("num_layers", 1)
        dropout    = config.get("dropout",    0.2)

        self.embedding = nn.Embedding(cipher_vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)

        # Stack of RNN cells
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = embed_dim if i == 0 else self.hidden_dim
            self.cells.append(RNNCell(in_dim, self.hidden_dim))

        # Output projection
        self.fc_out = nn.Linear(self.hidden_dim, plain_vocab_size)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """
        src          : (batch, seq_len)  LongTensor
        src_lengths  : (batch,)          LongTensor

        Returns:
            logits : (batch, seq_len, plain_vocab_size)
        """
        batch_size, src_len = src.shape
        device = src.device

        emb = self.drop(self.embedding(src))  # (batch, seq_len, embed_dim)

        # Initialise hidden states for each layer
        h = [torch.zeros(batch_size, self.hidden_dim, device=device)
             for _ in range(self.num_layers)]
        
        mask = (torch.arange(src_len, device=device)
                .unsqueeze(0)
                .expand(batch_size, src_len)) < src_lengths.unsqueeze(1)

        all_outputs = []
        for t in range(src_len):
            x_t = emb[:, t, :]
            mask_t = mask[:, t].unsqueeze(1)  # (batch, 1)

            for layer_idx, cell in enumerate(self.cells):
                h_new = cell(x_t, h[layer_idx])
                # Apply mask → keep old state if padded
                h[layer_idx] = torch.where(mask_t, h_new, h[layer_idx])
                x_t = h[layer_idx]
                
            all_outputs.append(h[-1].unsqueeze(1))   # top-layer hidden

        outputs = torch.cat(all_outputs, dim=1)       # (batch, seq_len, hidden_dim)

        # Project to vocabulary
        logits = self.fc_out(outputs)                 # (batch, seq_len, plain_vocab_size)

        return logits