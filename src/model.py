from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    # Input
    input_dim: int          # D, e.g., 21
    seq_len: int = 60       # T, e.g., 60

    # Transformer
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1

    
    pooling: str = "last"   # "last" or "mean"
    out_activation: str = "none"  # "sigmoid" or "none"


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds position information to each time step.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create positional encoding table: one vector per time step
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)PE table
        # Time step indices: 0, 1, 2, ..., T-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1) position index
        # Different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even dimensions, cosine to odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)  # even idx
        pe[:, 1::2] = torch.cos(position * div_term)  # odd idx
        # Add batch dimension for broadcasting
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # Register as buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)# Get the actual sequence length of the current input
        return x + self.pe[:, :T, :]# Add positional encoding to each time step


class TimeSeriesTransformerRegressor(nn.Module):
    """
    Input:  X (B, T, D)
    Output: y_hat (B, 1)  in [0, 1] if sigmoid head is used
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Project raw features (D) -> model dimension (d_model) (21->64)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)

        # Positional encoding for time steps
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=max(5000, cfg.seq_len + 10))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,     # IMPORTANT: expects (B, T, E)
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Output head: convert d_model features into a single regression output
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1)
        )

        if cfg.out_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif cfg.out_activation in ("none", "identity", None, ""):
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation: {cfg.out_activation}")


        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        # 1) Feature projection
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.dropout(x)

        # 2) Add positional encoding (time step info)
        x = self.pos_enc(x)              # (B, T, d_model)

        # 3) Encode sequence: let each time step attend to all other time steps
        z = self.encoder(x, mask=attn_mask)  # (B, T, d_model)

        # 4) Pooling: get a single vector for regression
        if self.cfg.pooling == "last":
            h = z[:, -1, :]              # (B, d_model) use last time step
        elif self.cfg.pooling == "mean":
            h = z.mean(dim=1)            # (B, d_model) average over time
        else:
            raise ValueError(f"Unknown pooling: {self.cfg.pooling}")

        # 5) Regression head
        y_hat = self.head(h)             # (B, 1)
        y_hat = self.out_act(y_hat)      # (B, 1) optionally in [0,1]
        return y_hat


if __name__ == "__main__":
    # Quick shape test
    cfg = ModelConfig(input_dim=21, seq_len=60, d_model=64, nhead=4, num_layers=2)
    model = TimeSeriesTransformerRegressor(cfg)

    x = torch.randn(8, 60, 21)  # (B=8, T=60, D=21)
    y = model(x)
    print("Output shape:", y.shape)
    print("Output range (min,max):", y.min().item(), y.max().item())
