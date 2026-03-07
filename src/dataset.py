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

LOSS_COL = "loss_weight"


@dataclass
class SplitConfig:
    seq_len: int = 60
    train_end: str = "2017-12-31"
    val_end: str = "2021-12-31"
    test_end: str = "2099-12-31"
    target_col: str = "label"
    date_col: str = "Date"
    ticker_col: str = "ticker_id"
    drop_cols: Tuple[str, ...] = tuple()


def time_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[:cfg.train_end].copy()
    val_df = df.loc[pd.to_datetime(cfg.train_end) + pd.Timedelta(days=1): cfg.val_end].copy()
    test_df = df.loc[pd.to_datetime(cfg.val_end) + pd.Timedelta(days=1): cfg.test_end].copy()
    return train_df, val_df, test_df


def time_split_multi(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def zscore_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


class TimeSeriesWindowDataset(Dataset):
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
        self.loss_col = LOSS_COL

        df_sorted = df.reset_index().sort_values([ticker_col, date_col])
        self.row_tickers = df_sorted[ticker_col].to_numpy(dtype=np.int64)
        self.row_weights = df_sorted[self.loss_col].to_numpy(dtype=np.float32)
        self.row_dates = pd.to_datetime(df_sorted[date_col]).to_numpy(dtype="datetime64[ns]")
        price_arr = df_sorted[feature_cols].to_numpy(dtype=np.float32)
        label_arr = df_sorted[target_col].to_numpy(dtype=np.float32)
        if target_col == "label":
            label_arr = (label_arr > 0).astype(np.float32)

        if apply_scaling:
            if scaler_mean is None or scaler_std is None:
                raise ValueError("Scaling is enabled but scaler_mean/std not provided.")
            price_arr = zscore_transform(price_arr, scaler_mean, scaler_std).astype(np.float32)

        self.x = price_arr
        self.row_labels = label_arr

        ticker_changes = np.where(np.diff(self.row_tickers) != 0)[0] + 1
        bounds = np.concatenate(([0], ticker_changes, [len(self.row_tickers)]))
        indices: List[int] = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end - start < seq_len:
                continue
            indices.extend(range(start + seq_len - 1, end))
        if not indices:
            raise ValueError("Not enough rows to build any sequence windows")

        self.indices = np.array(indices, dtype=np.int64)
        self.window_tickers = self.row_tickers[self.indices].astype(np.int64)
        self.window_weights = self.row_weights[self.indices].astype(np.float32)
        self.window_labels = self.row_labels[self.indices].astype(np.float32)
        self.window_dates = self.row_dates[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        start = t - self.seq_len + 1
        x_win = self.x[start: t + 1]
        y_t = self.window_labels[idx]
        w_t = self.window_weights[idx]

        X = torch.from_numpy(x_win)
        y = torch.tensor([y_t], dtype=torch.float32)
        w = torch.tensor([w_t], dtype=torch.float32)
        return X, y, w

    def ticker_counts(self) -> Dict[int, int]:
        unique, counts = np.unique(self.window_tickers, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}


LABEL_MODES = {"baseline", "event_day_only", "event_day_plus_prev1"}
FEATURE_MODES = {"baseline", "spreads", "relation"}


def apply_label_mode(df: pd.DataFrame, cfg: SplitConfig, label_mode: str, primary_ticker_id: int) -> pd.DataFrame:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label_mode={label_mode}")
    if label_mode == "baseline":
        return df

    mask = df[cfg.ticker_col] == primary_ticker_id
    label = df.loc[mask, cfg.target_col].to_numpy()
    pos = label > 0
    next_pos = np.zeros_like(pos, dtype=bool)
    if len(pos) > 1:
        next_pos[:-1] = pos[1:]
    event = pos & (~next_pos)

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


def add_spread_features(df: pd.DataFrame, per_ticker_features: Dict[int, List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    base_feats = ["log_ret_1", "ret_5", "ret_10", "ret_20", "vol_5", "vol_10", "vol_20"]
    ticker_ids = sorted(per_ticker_features.keys())
    if len(ticker_ids) < 2:
        raise ValueError("Spread features require at least two tickers")
    new_cols: List[str] = []
    for i in range(len(ticker_ids)):
        for j in range(i + 1, len(ticker_ids)):
            tid_a = ticker_ids[i]
            tid_b = ticker_ids[j]
            prefix = f"t{tid_a}_minus_t{tid_b}"
            for feat in base_feats:
                col_a = f"t{tid_a}_{feat}"
                col_b = f"t{tid_b}_{feat}"
                if col_a in df.columns and col_b in df.columns:
                    new_name = f"{prefix}_{feat}"
                    df[new_name] = df[col_a] - df[col_b]
                    new_cols.append(new_name)
    return df, new_cols


def add_relation_features(df: pd.DataFrame, primary_id: int, per_ticker_features: Dict[int, List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    ticker_ids = sorted(per_ticker_features.keys())
    if len(ticker_ids) < 2:
        raise ValueError("Relation features require multiple tickers")

    def col_name(tid: int, base: str) -> str:
        return f"t{tid}_{base}"

    new_cols: List[str] = []
    ret_col = "log_ret_1"
    price_col = "Adj Close"
    vol_col = "vol_20"
    corr_window = 20
    beta_window = 60
    z_window = 20

    primary_ret_col = col_name(primary_id, ret_col)
    primary_price_col = col_name(primary_id, price_col)
    primary_vol_col = col_name(primary_id, vol_col)

    if primary_ret_col not in df.columns or primary_price_col not in df.columns:
        return df, new_cols

    ret_primary = df[primary_ret_col]
    price_primary = df[primary_price_col]
    vol_primary = df[primary_vol_col] if primary_vol_col in df.columns else None

    for tid in ticker_ids:
        if tid == primary_id:
            continue
        other_ret_col = col_name(tid, ret_col)
        other_price_col = col_name(tid, price_col)
        other_vol_col = col_name(tid, vol_col)
        if other_ret_col not in df.columns or other_price_col not in df.columns:
            continue
        ret_other = df[other_ret_col]

        corr = ret_primary.rolling(corr_window, min_periods=5).corr(ret_other)
        corr_col = f"rel_corr_{primary_id}_{tid}_{corr_window}"
        df[corr_col] = corr.fillna(0.0)
        new_cols.append(corr_col)

        cov = ret_primary.rolling(beta_window, min_periods=10).cov(ret_other)
        var = ret_other.rolling(beta_window, min_periods=10).var()
        beta = cov / (var + 1e-12)
        beta_col = f"rel_beta_{primary_id}_{tid}_{beta_window}"
        df[beta_col] = beta.fillna(0.0)
        new_cols.append(beta_col)

        price_other = df[other_price_col]
        ratio = np.log(price_primary / price_other.replace(0, np.nan))
        mean = ratio.rolling(z_window, min_periods=5).mean()
        std = ratio.rolling(z_window, min_periods=5).std().replace(0.0, 1.0)
        zscore = (ratio - mean) / std
        z_col = f"rel_logratio_z_{primary_id}_{tid}_{z_window}"
        df[z_col] = zscore.fillna(0.0)
        new_cols.append(z_col)

        if vol_primary is not None and other_vol_col in df.columns:
            vol_ratio = (vol_primary + 1e-6) / (df[other_vol_col] + 1e-6)
            vol_col_name = f"rel_vol_ratio_{primary_id}_{tid}"
            df[vol_col_name] = vol_ratio.fillna(0.0)
            new_cols.append(vol_col_name)

    return df, new_cols


def load_datasets(
    dataset_csv_paths: Sequence[Path],
    cfg: SplitConfig = SplitConfig(),
    primary_ticker_id: int = 0,
    label_mode: str = "baseline",
    feature_mode: str = "baseline",
) -> Tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset, TimeSeriesWindowDataset, List[str]]:
    if not dataset_csv_paths:
        raise ValueError("No dataset paths provided")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
    if not dataset_csv_paths:
        raise ValueError("No dataset paths provided")

    per_ticker_frames: Dict[int, pd.DataFrame] = {}
    per_ticker_features: Dict[int, List[str]] = {}

    for path in dataset_csv_paths:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path, parse_dates=[cfg.date_col])
        df = df.sort_values(cfg.date_col).set_index(cfg.date_col)

        if cfg.ticker_col not in df.columns:
            raise ValueError(f"ticker column '{cfg.ticker_col}' missing from dataset {path}")
        ticker_ids = df[cfg.ticker_col].unique()
        if len(ticker_ids) != 1:
            raise ValueError(f"Dataset {path} must contain exactly one ticker; found {ticker_ids}")
        ticker_id = int(ticker_ids[0])
        if ticker_id in per_ticker_frames:
            raise ValueError(f"Duplicate ticker id {ticker_id} across datasets.")

        for col in cfg.drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        exclude_cols = {cfg.target_col, cfg.ticker_col, LOSS_COL}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        rename_map = {c: f"t{ticker_id}_{c}" for c in feature_cols}
        df = df.rename(columns=rename_map)

        keep_cols = list(rename_map.values())
        if ticker_id == primary_ticker_id:
            if cfg.target_col not in df.columns:
                raise ValueError(f"Primary dataset {path} missing target column '{cfg.target_col}'")
            keep_cols.append(cfg.target_col)
        per_ticker_frames[ticker_id] = df[keep_cols]
        per_ticker_features[ticker_id] = list(rename_map.values())

    if primary_ticker_id not in per_ticker_frames:
        raise ValueError(f"Primary ticker id {primary_ticker_id} not found in provided datasets")

    combined_df = per_ticker_frames[primary_ticker_id].copy()
    for tid, frame in per_ticker_frames.items():
        if tid == primary_ticker_id:
            continue
        combined_df = combined_df.join(frame, how="inner")

    combined_df = combined_df.dropna()
    combined_df[cfg.ticker_col] = primary_ticker_id
    combined_df[LOSS_COL] = 1.0
    combined_df = combined_df.sort_index()

    combined_df = apply_label_mode(combined_df, cfg, label_mode, primary_ticker_id)

    all_feature_cols: List[str] = []
    for tid in sorted(per_ticker_features.keys()):
        all_feature_cols.extend(per_ticker_features[tid])

    extra_cols: List[str] = []
    if feature_mode == "spreads":
        combined_df, extra_cols = add_spread_features(combined_df, per_ticker_features)
        all_feature_cols.extend(extra_cols)
    elif feature_mode == "relation":
        combined_df, extra_cols = add_relation_features(combined_df, primary_ticker_id, per_ticker_features)
        all_feature_cols.extend(extra_cols)

    train_df, val_df, test_df = time_split(combined_df, cfg)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/Val/Test split produced empty partitions. Check date ranges.")

    train_x = train_df[all_feature_cols].to_numpy(dtype=np.float32)
    mean, std = zscore_fit(train_x)

    train_ds = TimeSeriesWindowDataset(train_df, all_feature_cols, cfg.target_col, cfg.seq_len, cfg.ticker_col, cfg.date_col, mean, std, True)
    val_ds = TimeSeriesWindowDataset(val_df, all_feature_cols, cfg.target_col, cfg.seq_len, cfg.ticker_col, cfg.date_col, mean, std, True)
    test_ds = TimeSeriesWindowDataset(test_df, all_feature_cols, cfg.target_col, cfg.seq_len, cfg.ticker_col, cfg.date_col, mean, std, True)

    return train_ds, val_ds, test_ds, all_feature_cols


if __name__ == "__main__":  # pragma: no cover
    ROOT_DIR = Path(__file__).resolve().parents[1]
    paths = [ROOT_DIR / "data" / "sp500_dataset.csv"]
    cfg = SplitConfig(seq_len=60, train_end="2017-12-31", val_end="2021-12-31")
    train_ds, val_ds, test_ds, feats = load_datasets(paths, cfg)
    X, y, w = train_ds[0]
    print("Feature dim D =", X.shape[1])
    print("X shape =", tuple(X.shape), "y =", y.item(), "w =", w.item())
    print("Train/Val/Test sizes =", len(train_ds), len(val_ds), len(test_ds))
    print("First 10 feature cols:", feats[:10])
