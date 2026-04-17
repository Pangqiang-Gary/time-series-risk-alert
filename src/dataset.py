from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    raise ImportError("Please install PyTorch first: pip install torch")


@dataclass
class SplitConfig:
    """Defines how to split the data into train / val / test sets."""
    seq_len: int = 10              # number of past days per sample window
    train_end: str = "2017-12-31"  # training data ends here (2000-2017)
    val_end: str = "2021-12-31"    # validation data ends here (2018-2021)
    test_end: str = "2099-12-31"   # test data is everything after val (2022-)
    target_col: str = "label"
    date_col: str = "Date"
    ticker_col: str = "ticker_id"


def time_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a single ticker's data into train/val/test by date."""
    train_df = df.loc[:cfg.train_end].copy()
    val_df = df.loc[pd.to_datetime(cfg.train_end) + pd.Timedelta(days=1): cfg.val_end].copy()
    test_df = df.loc[pd.to_datetime(cfg.val_end) + pd.Timedelta(days=1): cfg.test_end].copy()
    return train_df, val_df, test_df


def time_split_multi(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data for multiple tickers independently, then concatenate."""
    train_parts, val_parts, test_parts = [], [], []
    for _, sub in df.groupby(cfg.ticker_col):
        sub = sub.sort_index()
        train, val, test = time_split(sub, cfg)
        if not train.empty:
            train_parts.append(train)
        if not val.empty:
            val_parts.append(val)
        if not test.empty:
            test_parts.append(test)

    def _concat(parts: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(parts, sort=False) if parts else pd.DataFrame()

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


def zscore_fit(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std from training data (used to normalize all splits)."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # avoid division by zero
    return mean, std


def zscore_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization using pre-computed mean and std."""
    return (x - mean) / std


class TimeSeriesWindowDataset(Dataset):
    """
    PyTorch Dataset that produces sliding windows of historical features.

    Each sample is:
      x: (seq_len, num_features) — the past seq_len days of features
      y: scalar label (1 = alert, 0 = normal)

    Windows are built per-ticker so data from different assets never overlap.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int,
        ticker_col: str,
        date_col: str,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std: Optional[np.ndarray] = None,
        apply_scaling: bool = True,
    ) -> None:
        super().__init__()
        if df.empty:
            raise ValueError("Dataset dataframe is empty")

        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.ticker_col = ticker_col

        # Sort rows by ticker then date so windows are contiguous
        df_sorted = df.reset_index().sort_values([ticker_col, date_col])
        self.row_tickers = df_sorted[ticker_col].to_numpy(dtype=np.int64)
        self.row_dates = pd.to_datetime(df_sorted[date_col]).to_numpy(dtype="datetime64[ns]")
        price_arr = df_sorted[feature_cols].to_numpy(dtype=np.float32)
        label_arr = df_sorted[target_col].to_numpy(dtype=np.float32)

        # Convert label: anything > 0 is positive (1), rest is 0
        if target_col == "label":
            label_arr = (label_arr > 0).astype(np.float32)

        # Apply z-score scaling if requested
        if apply_scaling:
            if scaler_mean is None or scaler_std is None:
                raise ValueError("Scaling is enabled but scaler_mean/std not provided.")
            price_arr = zscore_transform(price_arr, scaler_mean, scaler_std).astype(np.float32)

        self.x = price_arr
        self.row_labels = label_arr

        # Find where one ticker ends and the next begins
        # Windows must not cross ticker boundaries
        ticker_changes = np.where(np.diff(self.row_tickers) != 0)[0] + 1
        bounds = np.concatenate(([0], ticker_changes, [len(self.row_tickers)]))
        indices: List[int] = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end - start < seq_len:
                continue  # not enough rows for even one window
            # index points to the LAST row of each window (today)
            indices.extend(range(start + seq_len - 1, end))
        if not indices:
            raise ValueError("Not enough rows to build any sequence windows")

        self.indices = np.array(indices, dtype=np.int64)
        # Store metadata for each window
        self.window_tickers = self.row_tickers[self.indices].astype(np.int64)
        self.window_labels = self.row_labels[self.indices].astype(np.float32)
        self.window_dates = self.row_dates[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        # Return the window ending at index t: rows [t-seq_len+1, ..., t]
        t = self.indices[idx]
        x_win = self.x[t - self.seq_len + 1: t + 1]   # (seq_len, D)
        y_t = self.window_labels[idx]
        return torch.from_numpy(x_win), torch.tensor([y_t], dtype=torch.float32)

    def ticker_counts(self) -> Dict[int, int]:
        """How many windows belong to each ticker."""
        unique, counts = np.unique(self.window_tickers, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}


LABEL_MODES = {"baseline", "event_day_only", "event_day_plus_prev1"}

# Columns that are raw prices — never used as model features
RAW_PRICE_COLS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

# Feature sets organized by engineering stage (used for ablation experiments)
STAGE_FEATURES: Dict[int, List[str]] = {
    0: [   # Basic returns, volatility, drawdown, candle shape, volume
        "log_ret_1", "ret_5", "ret_10", "ret_20",
        "vol_5", "vol_10", "vol_20", "vol_ratio",
        "dd_20", "dd_60", "dd_252",
        "body_pct", "range_pct", "upper_wick_pct", "lower_wick_pct", "is_red",
        "vol_z_20",
    ],
    1: [   # Extended volatility
        "vol_60", "vol_ratio_10_60",
        "downside_vol_20", "upside_vol_20", "down_up_vol_ratio",
    ],
    2: [   # Moving averages and trend
        "price_to_ma20", "price_to_ma60",
        "ma_slope_20", "ma_slope_60",
        "past_max_drawdown_20",
    ],
    3: [   # Technical indicators (RSI, Bollinger, ADX, MFI)
        "rsi_14", "bollinger_pos", "adx_14", "mfi_14",
    ],
    4: [   # Cross-feature ratios and z-scores
        "ret_5_over_vol_20", "drawdown_over_vol",
        "zscore_ret_5", "zscore_vol_10",
    ],
    5: [   # Selected set for stage-ablation experiments
        "vol_5", "downside_vol_20", "down_up_vol_ratio",
        "dd_20", "dd_60", "dd_252",
        "ret_5", "ret_10",
        "ma_slope_60", "price_to_ma20", "past_max_drawdown_20",
        "range_pct", "ret_skew_20", "ret_kurt_20",
    ],
    6: [   # Market state dynamics
        "vol_20_slope", "consec_down_days", "drawdown_duration", "park_vol_20",
        "dd_speed_5", "dist_to_low_252",
    ],
    7: [   # Day-over-day delta features
        "vol5_change_1d", "rsi_change_1d", "dd20_change_1d",
        "ma20_cross_today", "new_low_20_today",
    ],
}


def get_stage_cols(stage: int) -> List[str]:
    """Return all feature columns up to and including the given stage."""
    cols: List[str] = []
    for s in range(stage + 1):
        cols.extend(STAGE_FEATURES.get(s, []))
    return cols


def apply_label_mode(df: pd.DataFrame, cfg: SplitConfig, label_mode: str, ticker_id: int) -> pd.DataFrame:
    """
    Optionally reshape labels for a single ticker.
    - "baseline": keep original labels as-is
    - "event_day_only": only the last positive day of each event is labeled 1
    - "event_day_plus_prev1": last day + one day before
    """
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label_mode={label_mode}")
    if label_mode == "baseline":
        return df

    mask = df[cfg.ticker_col] == ticker_id
    label = df.loc[mask, cfg.target_col].to_numpy()
    pos = label > 0
    next_pos = np.zeros_like(pos, dtype=bool)
    if len(pos) > 1:
        next_pos[:-1] = pos[1:]
    event = pos & (~next_pos)   # last day of each positive run

    if label_mode == "event_day_only":
        new_pos = event
    elif label_mode == "event_day_plus_prev1":
        prev_event = np.roll(event, 1)
        prev_event[0] = False
        new_pos = event | prev_event
    else:
        raise RuntimeError("Unhandled label mode")

    new_label = np.where(new_pos, 1, -1).astype(label.dtype)
    df.loc[mask, cfg.target_col] = new_label
    return df


def load_datasets(
    dataset_csv_paths: Sequence[Path],
    cfg: SplitConfig = SplitConfig(),
    label_mode: str = "baseline",
    feature_stage: Optional[int] = None,
) -> Tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset, TimeSeriesWindowDataset, List[str]]:
    """
    Main entry point: load multiple ticker CSVs, normalize, split, and return datasets.

    Steps:
      1. Load each ticker CSV and find the common set of feature columns.
      2. Stack all tickers into one dataframe.
      3. Apply label mode shaping (optional).
      4. Time-split into train/val/test per ticker.
      5. Z-score normalize using training set statistics per ticker.
      6. Build PyTorch Dataset objects with sliding windows.

    Returns: (train_ds, val_ds, test_ds, feature_cols)
    """
    if not dataset_csv_paths:
        raise ValueError("No dataset paths provided")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}")

    frames: List[pd.DataFrame] = []
    feature_sets: List[set] = []

    for path in dataset_csv_paths:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path, parse_dates=[cfg.date_col])
        df = df.sort_values(cfg.date_col).set_index(cfg.date_col)

        if cfg.ticker_col not in df.columns:
            raise ValueError(f"ticker column '{cfg.ticker_col}' missing from {path}")
        ticker_ids = df[cfg.ticker_col].unique()
        if len(ticker_ids) != 1:
            raise ValueError(f"Dataset {path} must contain exactly one ticker; found {ticker_ids}")
        if cfg.target_col not in df.columns:
            raise ValueError(f"target column '{cfg.target_col}' missing from {path}")

        # Collect feature columns (exclude raw price cols and label/ticker)
        exclude_cols = {cfg.target_col, cfg.ticker_col} | RAW_PRICE_COLS
        feat_cols = [c for c in df.columns if c not in exclude_cols]
        feature_sets.append(set(feat_cols))
        frames.append(df)

    # Use only features that exist in ALL ticker files
    common_feat_set = feature_sets[0]
    for s in feature_sets[1:]:
        common_feat_set = common_feat_set & s

    # Optionally restrict to a specific ablation stage
    if feature_stage is not None:
        allowed = set(get_stage_cols(feature_stage))
        common_feat_set = common_feat_set & allowed

    all_feature_cols = sorted(common_feat_set)

    # Stack all tickers together
    keep_cols = all_feature_cols + [cfg.target_col, cfg.ticker_col]
    stacked = pd.concat([df[keep_cols] for df in frames], sort=False)

    # Apply label mode (optional reshaping)
    for tid in stacked[cfg.ticker_col].unique():
        stacked = apply_label_mode(stacked, cfg, label_mode, int(tid))

    # Split into train/val/test
    train_df, val_df, test_df = time_split_multi(stacked, cfg)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/Val/Test split produced empty partitions. Check date ranges.")

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for split_df in [train_df, val_df, test_df]:
        split_df[all_feature_cols] = split_df[all_feature_cols].astype(np.float32)

    # Normalize each ticker separately: fit on train, apply to val and test
    for tid in stacked[cfg.ticker_col].unique():
        mask_train = train_df[cfg.ticker_col] == tid
        train_x_tid = train_df.loc[mask_train, all_feature_cols].to_numpy(dtype=np.float32)
        if len(train_x_tid) == 0:
            continue
        mean, std = zscore_fit(train_x_tid)
        for split_df in [train_df, val_df, test_df]:
            mask = split_df[cfg.ticker_col] == tid
            if mask.any():
                split_df.loc[mask, all_feature_cols] = zscore_transform(
                    split_df.loc[mask, all_feature_cols].to_numpy(dtype=np.float32), mean, std
                )

    # Build datasets (scaling already done above, so apply_scaling=False)
    train_ds = TimeSeriesWindowDataset(train_df, all_feature_cols, cfg.target_col, cfg.seq_len, cfg.ticker_col, cfg.date_col, apply_scaling=False)
    val_ds = TimeSeriesWindowDataset(val_df, all_feature_cols, cfg.target_col, cfg.seq_len, cfg.ticker_col, cfg.date_col, apply_scaling=False)
    test_ds = TimeSeriesWindowDataset(test_df, all_feature_cols, cfg.target_col, cfg.seq_len, cfg.ticker_col, cfg.date_col, apply_scaling=False)

    return train_ds, val_ds, test_ds, all_feature_cols
