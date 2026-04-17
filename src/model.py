from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """All hyperparameters for the Transformer model."""
    input_dim: int            # number of input features (e.g. 50)
    seq_len: int = 10         # how many past days to look at
    d_model: int = 28         # internal embedding size
    nhead: int = 4            # number of attention heads
    num_layers: int = 2       # number of Transformer encoder layers
    dim_feedforward: int = 32 # hidden size of the feedforward network inside each layer
    dropout: float = 0.2      # dropout rate for regularization
    pooling: str = "last"     # how to aggregate the sequence: "last" day or "mean" of all days
    out_activation: str = "none"  # output activation: "none" outputs raw logit


class SinusoidalPositionalEncoding(nn.Module):
    """
    Add position information to the sequence so the model knows
    which day is day 1, day 2, ... day 20.
    Uses fixed sine/cosine patterns (not learned).
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Even indices get sine, odd indices get cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to the input (same shape, just adds position signal)
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformerRegressor(nn.Module):
    """
    Transformer model for predicting drawdown probability.

    Architecture:
      1. Linear projection: (50 features) -> (d_model=28)
      2. Sinusoidal positional encoding: tells the model which day each row is
      3. Transformer encoder x2: learns temporal patterns via attention
      4. Take the last day's output (pooling="last")
      5. LayerNorm + Linear: compress to a single logit (raw score before sigmoid)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Step 1: project input features to model dimension
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        # Step 2: positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=max(5000, cfg.seq_len + 10))

        # Step 3: stacked Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,     # input shape is (batch, seq, features)
            activation="gelu",
            norm_first=True,      # pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Step 4+5: output head
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1),
        )

        # Optional output activation (usually "none" = raw logit for BCE loss)
        if cfg.out_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif cfg.out_activation in ("none", "identity", None, ""):
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation: {cfg.out_activation}")

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        x = self.dropout(self.input_proj(x))   # project features
        x = self.pos_enc(x)                    # add position signal
        z = self.encoder(x, mask=attn_mask)    # attend over time steps

        # Pool: take the last time step (today) as the summary representation
        if self.cfg.pooling == "last":
            h = z[:, -1, :]
        elif self.cfg.pooling == "mean":
            h = z.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.cfg.pooling}")

        return self.out_act(self.head(h))   # (batch, 1) logit


if __name__ == "__main__":
    # Quick sanity check: create a small model and run a forward pass
    cfg = ModelConfig(input_dim=48, seq_len=10, d_model=28, nhead=4, num_layers=2, dim_feedforward=32)
    model = TimeSeriesTransformerRegressor(cfg)
    x = torch.randn(8, 10, 48)
    print("Output shape:", model(x).shape)
