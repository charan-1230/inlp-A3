"""
State Space Model (SSM) for Next Word Prediction.

Architecture
------------
Embedding → SSMLayer → Dropout → Linear(hidden_dim → vocab_size)

SSM recurrence (per timestep):
    h_t = A h_{t-1} + B x_t
    y_t = C h_t

Matrices
--------
A : (hidden_dim, hidden_dim)   state transition
B : (hidden_dim, embed_dim)    input projection
C : (hidden_dim, hidden_dim)   output projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMLayer(nn.Module):
    """
    Discrete SSM recurrence layer with Skip (D), Diagonal (A), and Gating.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Diagonal Recurrent State (simplifies A from HxH -> 1D vector H)
        self.A = nn.Parameter(torch.empty(hidden_dim))
        
        # Input Projection
        self.B = nn.Parameter(torch.empty(hidden_dim, input_dim))
        
        # Output Projection
        self.C = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        
        # Skip Connection (D matrix)
        self.D = nn.Parameter(torch.empty(hidden_dim, input_dim))
        
        # Gating Mechanism (Mamba style)
        self.Wg = nn.Parameter(torch.empty(hidden_dim, input_dim))

        # Bias Terms
        self.bias_h = nn.Parameter(torch.zeros(hidden_dim))
        self.bias_y = nn.Parameter(torch.zeros(hidden_dim))

        # Initialization
        nn.init.uniform_(self.A, 0.8, 0.95)  # Balanced memory vs adaptation
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.D)
        nn.init.xavier_uniform_(self.Wg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_dim)
        outputs : (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch, self.hidden_dim, device=device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]                                  # (batch, input_dim)
            
            # Inner dropout for sequence robustness
            x_t = F.dropout(x_t, p=0.1, training=self.training)
            
            # Gating value per timestep
            g_t = torch.sigmoid(x_t @ self.Wg.T)              # (batch, hidden_dim)
            g_t = torch.clamp(g_t, min=0.05, max=0.95)        # Avoid pure 0/1 gradient starvation
            
            # Recurrent update proposal (diagonal A is element-wise multiplied)
            h_new = (self.A * h) + (x_t @ self.B.T) + self.bias_h   # (batch, hidden_dim)

            # Improved gated residual update (mathematically equivalent but superior flow representation via difference)
            h = h + g_t * (h_new - h)                         # (batch, hidden_dim)
            
            # Output generation with Skip Connection and Bias
            y_raw = h @ self.C.T + x_t @ self.D.T + self.bias_y   # (batch, hidden_dim)
            y_t = y_raw / math.sqrt(self.hidden_dim)              # Stabilize logit variance
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)                      # (batch, seq_len, hidden_dim)


class SSMModel(nn.Module):
    """
    Full SSM model for Next Word Prediction.

    Input  : (batch, seq_len)          token ids
    Output : (batch, seq_len, vocab)   logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_idx: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.proj_in = nn.Linear(embed_dim, hidden_dim)
        
        # Deep SSM Stack
        self.layers = nn.ModuleList([
            SSMLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))   # (B, T, E)
        h = self.proj_in(emb)                   # (B, T, H)
        
        for layer, norm in zip(self.layers, self.norms):
            # Recurrent SSM Forward
            h_out = layer(h)
            
            # Residual + Normalize
            h = norm(h + self.dropout(h_out))   # (B, T, H)
            
        out = self.proj_out(h)                  # (B, T, E)
        return self.fc(self.dropout(out))       # (B, T, V)