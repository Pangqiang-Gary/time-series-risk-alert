from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

import mplfinance as mpf
import matplotlib.pyplot as plt

from dataset import time_split, SplitConfig, zscore_fit, zscore_transform
from model import ModelConfig, TimeSeriesTransformerRegressor


# =====================
# Project paths
# =====================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CKPT_DIR = ROOT_DIR / "checkpoints"
PLOTS_DIR = ROOT_DIR / "plots"

DATASET_PATH = DATA_DIR / "sp500_dataset.csv"
CKPT_PATH = CKPT_DIR / "best_model.pt"
DEFAULT_SAVE_PATH = PLOTS_DIR / "sp500_alerts.png"



def make_windows(df: pd.DataFrame, feature_cols: List[str], seq_len: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Build rolling windows X for all possible end times t.
    Returns:
      X_all: (N, T, D) T:len_seq N :number of window which can use
      end_dates: list of length N, date of each window's last day
    """
    x = df[feature_cols].to_numpy(dtype=np.float32)  # (L, D) L:number of dates be traded
    L, D = x.shape
    N = L - (seq_len - 1)
    X_all = np.zeros((N, seq_len, D), dtype=np.float32)
    end_dates = []

    for i in range(N):
        t = i + (seq_len - 1)
        X_all[i] = x[t - seq_len + 1: t + 1]
        end_dates.append(df.index[t])

    return X_all, end_dates


def find_alert_segments(scores: pd.Series, threshold: float, min_consecutive: int = 3) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Turn score series into alert segments where score >= threshold for at least min_consecutive days.
    Returns list of (start_date, end_date) inclusive.
    """
    is_high = scores >= threshold
    segments = []
    start = None
    run = 0

    dates = scores.index
    for i, d in enumerate(dates):
        if is_high.iloc[i]:
            if start is None:
                start = d
                run = 1
            else:
                run += 1
        else:
            if start is not None and run >= min_consecutive:
                segments.append((start, dates[i - 1]))
            start = None
            run = 0

    # close last segment
    if start is not None and run >= min_consecutive:
        segments.append((start, dates[-1]))

    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))

    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")

    parser.add_argument("--pct", type=float, default=0.90, help="Percentile threshold for alerts, e.g., 0.90 = top 10% risk")
    parser.add_argument("--min_consecutive", type=int, default=3, help="Min consecutive days above threshold to form an alert segment")

    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE_PATH))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_path = Path(args.ckpt).resolve()
    dataset_path = Path(args.dataset).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # 1) Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg_dict = ckpt["model_cfg"]
    split_cfg_dict = ckpt["split_cfg"]
    feature_cols = ckpt["feature_cols"]

    split_cfg = SplitConfig(**split_cfg_dict)
    model_cfg = ModelConfig(**model_cfg_dict)
    

    # 2) Load dataset csv
    df = pd.read_csv(dataset_path, parse_dates=[split_cfg.date_col])
    df = df.sort_values(split_cfg.date_col).set_index(split_cfg.date_col)

    # Keep only needed columns (features + OHLC for plotting + target optional)
    needed = list(set(feature_cols + ["Open", "High", "Low", "Close"]))
    df = df[[c for c in needed if c in df.columns]].dropna()

    # 3) Recompute scaler on TRAIN only (same rule as training: avoid leakage)
    train_df, val_df, test_df = time_split(df, split_cfg)
    train_x = train_df[feature_cols].to_numpy(dtype=np.float32)
    mean, std = zscore_fit(train_x)

    # Apply scaling to full df
    x_all = df[feature_cols].to_numpy(dtype=np.float32)
    x_all = zscore_transform(x_all, mean, std).astype(np.float32) # use in pytorch
    df_scaled = df.copy()
    df_scaled[feature_cols] = x_all

    # 4) Build windows for prediction over whole timeline
    seq_len = model_cfg.seq_len
    X_all, end_dates = make_windows(df_scaled, feature_cols, seq_len)
    X_tensor = torch.from_numpy(X_all).to(device)  # (N, T, D)

    # 5) Load model + predict
    model = TimeSeriesTransformerRegressor(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        # batch prediction (avoid huge memory if needed)
        bs = 512
        preds = []
        for i in range(0, X_tensor.size(0), bs):
            y_hat = model(X_tensor[i:i+bs])  # (b,1)
            preds.append(y_hat.squeeze(1).detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)

    pred_series = pd.Series(preds, index=pd.to_datetime(end_dates), name="pred_sell_score")

    # 6) Choose threshold using TRAIN predictions (more principled than guessing)
    train_pred = pred_series.loc[pred_series.index <= pd.to_datetime(split_cfg.train_end)]
    thr = float(np.quantile(train_pred.values, args.pct))
    print(f"Alert threshold (train {int(args.pct*100)}th pct) =", thr)

    # 7) Build alert segments (continuous risk regimes)
    segments = find_alert_segments(pred_series, thr, min_consecutive=args.min_consecutive)
    print("Num alert segments:", len(segments))
    if segments[:5]:
        print("First segments:", segments[:5])
    

    # 8) Slice to plotting window
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    segments_in_window = [(s,e) for (s,e) in segments if not (e < start or s > end)]
    print("Num segments in plot window:", len(segments_in_window))
    print("First segments in window:", segments_in_window[:5])


    # Candlestick data needs full OHLC indexed by date
    df_plot = df.loc[(df.index >= start) & (df.index <= end)].copy()
    score_plot = pred_series.loc[(pred_series.index >= start) & (pred_series.index <= end)].copy()

    # Align OHLC to pred index: pred starts later (after seq_len-1)
    # For score line, need it indexed by datetime (OK).

    # 9) Prepare mplfinance
    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    add_plots = []
    add_plots.append(mpf.make_addplot(score_plot, panel=1, ylabel="pred sell_score", secondary_y=False))
    add_plots.append(mpf.make_addplot(pd.Series(thr, index=score_plot.index), panel=1))

    # Use a Yahoo-like style
    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle="--")

    fig, axes = mpf.plot(
        df_plot[["Open", "High", "Low", "Close"]],
        type="candle",
        style=style,
        addplot=add_plots,
        panel_ratios=(3, 1),
        figsize=(24, 10),
        returnfig=True,
        datetime_format="%Y",
        xrotation=0,
        ylabel="S&P 500",
        volume=False,
        warn_too_much_data=10000,
    )
    print("mpf.plot done, fig type =", type(fig), "axes len =", len(axes))

# Select price and score axes (mplfinance may add a hidden shared-x axis)
    ax_price = axes[0]
    ax_score = axes[2] if len(axes) > 2 else axes[1]  # mplfinance panel axes indexing

# 10) Overlay alert segments using integer x positions (robust: use searchsorted directly)
    idx = df_plot.index  # DatetimeIndex in plotting window
    segments_in_window = [(s, e) for (s, e) in segments if not (e < start or s > end)]

    dbg = 0
    for (s, e) in segments_in_window:
        s2 = max(s, start)
        e2 = min(e, end)

    # map datetime -> integer positions in df_plot
        xs = int(idx.searchsorted(s2, side="left"))
        xe = int(idx.searchsorted(e2, side="right") - 1)

    # clamp to valid range
        if xs < 0:
            xs = 0
        if xe >= len(idx):
            xe = len(idx) - 1

    # skip empty/invalid spans
        if xe <= xs:
            continue

    # debug: only print first 5 spans
        if dbg < 5:
            print("DRAW:", s2.date(), e2.date(), "->", xs, xe)
            dbg += 1

        ax_price.axvspan(xs, xe, alpha=0.18, color="gray", zorder=5)
        ax_score.axvspan(xs, xe, alpha=0.18, color="gray", zorder=5)
    
    fig.suptitle("S&P 500 Candles with Transformer Risk Alerts", y=0.98)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print("Saved plot to:", save_path)




if __name__ == "__main__":
    main()

