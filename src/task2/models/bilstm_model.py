"""
Manual Bidirectional LSTM for Masked Language Modeling.

⚠  nn.LSTM / nn.LSTMCell are NOT used anywhere.
   All gate computations are implemented from scratch.

LSTM equations (per timestep):
    i_t = σ(W_i x_t + U_i h_{t-1})
    f_t = σ(W_f x_t + U_f h_{t-1})
    o_t = σ(W_o x_t + U_o h_{t-1})
    g_t = tanh(W_g x_t + U_g h_{t-1})
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)

BiLSTM: forward cell (left → right) + backward cell (right → left),
        outputs concatenated → (batch, seq_len, 2 * hidden_dim).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Manual LSTM Cell
# ---------------------------------------------------------------------------

class LSTMCell(nn.Module):
    """
    Single-step LSTM cell — fully manual implementation.

    Uses fused weight matrices for efficiency:
        W  : (4H, input_dim)   maps x_t
        U  : (4H, hidden_dim)  maps h_{t-1}
    Gate order: i | f | o | g
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Combine all four gate weights into one matrix for speed
        self.W = nn.Linear(input_dim,  4 * hidden_dim, bias=True)
        self.U = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for W; orthogonal for U (standard LSTM init)."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.orthogonal_(self.U.weight)

    def forward(
        self,
        x_t:    torch.Tensor,   # (batch, input_dim)
        h_prev: torch.Tensor,   # (batch, hidden_dim)
        c_prev: torch.Tensor,   # (batch, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_t, c_t), each (batch, hidden_dim)."""
        H     = self.hidden_dim
        gates = self.W(x_t) + self.U(h_prev)   # (batch, 4H)

        i_t = torch.sigmoid(gates[:, :H])           # input  gate
        f_t = torch.sigmoid(gates[:, H  : 2*H])     # forget gate
        o_t = torch.sigmoid(gates[:, 2*H: 3*H])     # output gate
        g_t = torch.tanh(   gates[:, 3*H:      ])   # cell   gate

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# ---------------------------------------------------------------------------
# Manual BiLSTM Layer
# ---------------------------------------------------------------------------

class BiLSTMLayer(nn.Module):
    """
    One bidirectional LSTM layer built from two manual LSTMCells.

    Forward cell  : processes x_0 → x_{T-1}
    Backward cell : processes x_{T-1} → x_0, then outputs are reversed
                    so index t aligns with index t of the forward cell.

    Output : (batch, seq_len, 2 * hidden_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fwd_cell   = LSTMCell(input_dim, hidden_dim)
        self.bwd_cell   = LSTMCell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        out : (batch, seq_len, 2 * hidden_dim)
        """
        batch, seq_len, _ = x.shape
        device = x.device
        H      = self.hidden_dim

        # Initialise hidden / cell states
        h_fwd = torch.zeros(batch, H, device=device)
        c_fwd = torch.zeros(batch, H, device=device)
        h_bwd = torch.zeros(batch, H, device=device)
        c_bwd = torch.zeros(batch, H, device=device)

        fwd_outs = []
        bwd_outs = []

        for t in range(seq_len):
            h_fwd, c_fwd = self.fwd_cell(x[:, t, :],               h_fwd, c_fwd)
            h_bwd, c_bwd = self.bwd_cell(x[:, seq_len - 1 - t, :], h_bwd, c_bwd)
            fwd_outs.append(h_fwd.unsqueeze(1))
            bwd_outs.append(h_bwd.unsqueeze(1))

        fwd = torch.cat(fwd_outs,              dim=1)   # (B, T, H)
        bwd = torch.cat(bwd_outs[::-1],        dim=1)   # (B, T, H) — re-aligned

        return torch.cat([fwd, bwd], dim=2)             # (B, T, 2H)


# ---------------------------------------------------------------------------
# Full BiLSTM Model
# ---------------------------------------------------------------------------

class BiLSTMModel(nn.Module):
    """
    Stacked BiLSTM model for Masked Language Modeling.

    Architecture
    ------------
    Embedding → [BiLSTMLayer × num_layers] → Dropout → Linear(2H → vocab_size)

    Input  : (batch, seq_len)          token ids (some replaced by <MASK>)
    Output : (batch, seq_len, vocab)   logits for every position
    Loss is computed only on masked positions (label = -100 elsewhere).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        pad_idx:    int,
        dropout:    float = 0.2,
        num_layers: int   = 1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout   = nn.Dropout(dropout)

        # Build stacked BiLSTM layers
        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(num_layers):
            layers.append(BiLSTMLayer(in_dim, hidden_dim))
            in_dim = 2 * hidden_dim                     # output of each layer
        self.bilstm_layers = nn.ModuleList(layers)

        self.fc = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.embedding(x))             # (B, T, E)
        for layer in self.bilstm_layers:
            h = self.dropout(layer(h))                  # (B, T, 2H)
        return self.fc(h)                               # (B, T, V)