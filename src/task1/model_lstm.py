"""
Sequence Labeling model with a manually implemented LSTM (no nn.LSTM).

Gates implemented explicitly:
    i = sigmoid(W_xi x + W_hi h + b_i)   input gate
    f = sigmoid(W_xf x + W_hf h + b_f)   forget gate
    g = tanh   (W_xg x + W_hg h + b_g)   cell gate
    o = sigmoid(W_xo x + W_ho h + b_o)   output gate
    c = f * c_prev + i * g
    h = o * tanh(c)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Primitive LSTM cell (one layer)
# ---------------------------------------------------------------------------

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Combined weight matrix for all 4 gates: [i, f, g, o]
        # Shape: input_size  -> 4 * hidden_size
        self.W_x = nn.Linear(input_size,  4 * hidden_size, bias=True)
        # Shape: hidden_size -> 4 * hidden_size  (no bias – already in W_x)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        x : (batch, input_size)
        h : (batch, hidden_size)
        c : (batch, hidden_size)

        Returns h_new, c_new each (batch, hidden_size)
        """
        gates = self.W_x(x) + self.W_h(h)          # (batch, 4*hidden)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


# ---------------------------------------------------------------------------
# Full Sequence Labeling LSTM
# ---------------------------------------------------------------------------

class SeqLabelLSTM(nn.Module):
    def __init__(self, config: dict, cipher_vocab_size: int, plain_vocab_size: int,
                 pad_idx: int = 0):
        super().__init__()
        embed_dim  = config.get("embed_dim",  64)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_layers = config.get("num_layers", 1)
        dropout    = config.get("dropout",    0.2)

        self.embedding = nn.Embedding(cipher_vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = embed_dim if i == 0 else self.hidden_dim
            self.cells.append(LSTMCell(in_dim, self.hidden_dim))

        self.fc_out = nn.Linear(self.hidden_dim, plain_vocab_size)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """
        src         : (batch, seq_len)
        src_lengths : (batch,)

        Returns:
            logits : (batch, seq_len, plain_vocab_size)
        """
        batch_size, src_len = src.shape
        device = src.device

        emb = self.drop(self.embedding(src))   # (batch, seq_len, embed_dim)

        h = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]

        mask = (torch.arange(src_len, device=device)
                .unsqueeze(0)
                .expand(batch_size, src_len)) < src_lengths.unsqueeze(1)
        
        all_outputs = []
        for t in range(src_len):
            x_t = emb[:, t, :]
            mask_t = mask[:, t].unsqueeze(1) # (batch, 1)

            for layer_idx, cell in enumerate(self.cells):
                h_new, c_new = cell(x_t, h[layer_idx], c[layer_idx])

                h[layer_idx] = torch.where(mask_t, h_new, h[layer_idx])
                c[layer_idx] = torch.where(mask_t, c_new, c[layer_idx])

                x_t = self.drop(h[layer_idx]) if layer_idx < self.num_layers - 1 else h[layer_idx]
                
            all_outputs.append(h[-1].unsqueeze(1))

        outputs = torch.cat(all_outputs, dim=1)         # (batch, seq_len, hidden_dim)

        # Output projection
        logits = self.fc_out(outputs)                   # (batch, seq_len, plain_vocab_size)

        return logits