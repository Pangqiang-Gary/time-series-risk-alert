from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError("Please install PyTorch first: pip install torch")


# ---------------------------
# Config
# ---------------------------
@dataclass
class SplitConfig:
    seq_len: int = 60                  # T: lookback window
    train_end: str = "2017-12-31"      # inclusive
    val_end: str = "2021-12-31"        # inclusive
    test_end: str = "2099-12-31"       # inclusive, effectively "rest"
    target_col: str = "label"
    date_col: str = "Date"
    drop_cols: Tuple[str, ...] = ("worst_ret_H", "max_dd_H")
  # keep or drop


# ---------------------------
# Utilities
# ---------------------------
def time_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by date to avoid leakage (no shuffling).
    """
    train_df = df.loc[:cfg.train_end].copy()
    val_df = df.loc[pd.to_datetime(cfg.train_end) + pd.Timedelta(days=1): cfg.val_end].copy()
    test_df = df.loc[pd.to_datetime(cfg.val_end) + pd.Timedelta(days=1): cfg.test_end].copy()
    return train_df, val_df, test_df


def zscore_fit(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit mean/std on training features only.
    """
    mean = train_x.mean(axis=0) #shape train_x (N_train, D)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # avoid divide-by-zero if some features are not changeable
    return mean, std #shape (D,)
#N_train: number of training samples (days)

#D: number of features


def zscore_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


# ---------------------------
# Dataset
# ---------------------------
class TimeSeriesWindowDataset(Dataset):
    """
    Each item:
        X: (seq_len, D) float32
        y: (1,) float32
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std: Optional[np.ndarray] = None,
        apply_scaling: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.target_col = target_col
        self.feature_cols = feature_cols

        # Extract arrays
        x = df[feature_cols].to_numpy(dtype=np.float32) # x(N,D)
        y = df[target_col].to_numpy(dtype=np.float32)# Y(N,)
        if target_col == "label":
            y = (y > 0).astype(np.float32)

        # Scale features if requested
        self.apply_scaling = apply_scaling
        self.mean = scaler_mean
        self.std = scaler_std
        if apply_scaling:
            if self.mean is None or self.std is None:
                raise ValueError("Scaling is enabled but scaler_mean/std not provided.")
            x = zscore_transform(x, self.mean, self.std).astype(np.float32)

        self.x = x
        self.y = y

        # Number of windows:
        # window t uses x[t-seq_len+1 : t+1], predicts y[t]
        self.n = len(df) - (seq_len - 1)
        if self.n <= 0:
            raise ValueError(f"Not enough rows ({len(df)}) for seq_len={seq_len}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        t = idx + (self.seq_len - 1)
        x_win = self.x[t - self.seq_len + 1: t + 1]  # shape(seq_len, D)
        y_t = self.y[t]                               # scalar

        # Convert to torch
        X = torch.from_numpy(x_win)                   # (T, D)
        y = torch.tensor([y_t], dtype=torch.float32)  # (1,)
        return X, y


# ---------------------------
# Loader function 
# ---------------------------
def load_datasets(
    dataset_csv_path: Path,
    cfg: SplitConfig = SplitConfig(),
) -> Tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset, TimeSeriesWindowDataset, List[str]]:
    """
    Returns train/val/test datasets + feature columns used.
    """
    df = pd.read_csv(dataset_csv_path, parse_dates=[cfg.date_col])
    df = df.sort_values(cfg.date_col).set_index(cfg.date_col)

    # Drop optional columns which don't want as features
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Identify feature columns: exclude target and any non-causal columns (to avoid leakage)
    exclude_cols = {cfg.target_col, "sell_score"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    

    # Split by time
    train_df, val_df, test_df = time_split(df, cfg)

    # Fit scaler on train only (avoid leakage)
    train_x = train_df[feature_cols].to_numpy(dtype=np.float32)
    mean, std = zscore_fit(train_x)

    train_ds = TimeSeriesWindowDataset(train_df, feature_cols, cfg.target_col, cfg.seq_len, mean, std, apply_scaling=True)
    val_ds   = TimeSeriesWindowDataset(val_df,   feature_cols, cfg.target_col, cfg.seq_len, mean, std, apply_scaling=True)
    test_ds  = TimeSeriesWindowDataset(test_df,  feature_cols, cfg.target_col, cfg.seq_len, mean, std, apply_scaling=True)

    return train_ds, val_ds, test_ds, feature_cols


# ---------------------------
# Quick check
# ---------------------------
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[1]
    DATASET_PATH = ROOT_DIR / "data" / "sp500_dataset.csv"

    cfg = SplitConfig(seq_len=60, train_end="2017-12-31", val_end="2021-12-31")
    train_ds, val_ds, test_ds, feats = load_datasets(DATASET_PATH, cfg)

    X, y = train_ds[0]
    print("Feature dim D =", X.shape[1])
    print("X shape =", tuple(X.shape), "y =", y.item())
    print("Train/Val/Test sizes =", len(train_ds), len(val_ds), len(test_ds))
    print("First 10 feature cols:", feats[:10])


