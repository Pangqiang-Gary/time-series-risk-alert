"""
infer.py  -  Run today's sell probability + plot historical probabilities.

Usage:
  python src/infer.py
  python src/infer.py --features data/sp500_features.csv --ckpt checkpoints/best_model.pt
  python src/infer.py --plot_days 365
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import ModelConfig, TimeSeriesTransformerRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]


# Helper functions

def zscore_fit(x: np.ndarray):
    """Compute mean and std from x (used to normalize features the same way as training)."""
    mean = x.mean(axis=0)
    std  = x.std(axis=0)
    std  = np.where(std < 1e-12, 1.0, std)   # avoid division by zero
    return mean, std


def zscore_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization."""
    return (x - mean) / std


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Convert logits to calibrated probabilities using temperature scaling: sigmoid(logit / T)."""
    return torch.sigmoid(torch.tensor(logits, dtype=torch.float32) / T).numpy()


# Load model and run inference

def load_model(ckpt_path: Path, device: torch.device):
    """Load the saved checkpoint and return (model, feature_cols, split_cfg, temperature)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ModelConfig(**ckpt["model_cfg"])
    model = TimeSeriesTransformerRegressor(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["feature_cols"], ckpt["split_cfg"], ckpt.get("temperature", 1.0)


def prepare_data(features_path: Path, feature_cols: list, train_end: str):
    """
    Load feature CSV and normalize using training-set statistics.
    The scaler is fit only on rows up to train_end to avoid data leakage.
    Returns (dates, normalized_feature_array).
    """
    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Features CSV missing columns: {missing}")

    arr = df[feature_cols].to_numpy(dtype=np.float32)

    # Fit scaler on training period only, then apply to all rows
    train_mask = df.index <= pd.Timestamp(train_end)
    if train_mask.sum() == 0:
        raise ValueError(f"No rows on or before train_end={train_end}")
    mean, std = zscore_fit(arr[train_mask])
    arr_scaled = zscore_transform(arr, mean, std).astype(np.float32)

    return df.index, arr_scaled


def run_inference(
    model,
    arr: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_len: int,
    temperature: float,
    device: torch.device,
    start_date: str | None = None,
) -> pd.DataFrame:
    """
    Run sliding-window inference for every day from start_date onward.
    For each day t, feeds the past seq_len days into the model and gets a sell probability.
    If start_date is None, only runs on the last day (today's signal).
    Returns a DataFrame with columns: prob, logit.
    """
    n = len(arr)
    if n < seq_len:
        raise ValueError(f"Not enough data rows ({n}) for seq_len={seq_len}")

    if start_date is not None:
        first_idx = np.searchsorted(dates, pd.Timestamp(start_date))
        first_idx = max(first_idx, seq_len - 1)
    else:
        first_idx = n - 1  # today only

    results = []
    with torch.no_grad():
        for t in range(first_idx, n):
            window = arr[t - seq_len + 1: t + 1]          # (seq_len, D)
            x = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, seq_len, D)
            logit = model(x).item()
            prob  = apply_temperature(np.array([logit]), temperature)[0]
            results.append({"date": dates[t], "prob": float(prob), "logit": float(logit)})

    return pd.DataFrame(results).set_index("date")


def detect_events(labels: pd.Series, gap_days: int = 5) -> list[dict]:
    """Group consecutive label=1 days into discrete drawdown events.
    Two positive days more than gap_days apart become separate events."""
    pos_dates = labels[labels == 1].index.sort_values()
    if len(pos_dates) == 0:
        return []
    events = []
    start = pos_dates[0]
    prev  = pos_dates[0]
    for d in pos_dates[1:]:
        if (d - prev).days > gap_days:
            events.append({"start": start, "end": prev})
            start = d
        prev = d
    events.append({"start": start, "end": prev})
    return events


def plot_results(
    prob_df: pd.DataFrame,
    price_df: pd.DataFrame,
    threshold: float,
    out_path: Path,
    title: str = "SP500 Crisis Early-Warning Probability",
    events: list | None = None,
):
    """
    Save a two-panel chart:
      Top panel: price chart with red dots for alert days and orange shading for drawdown events.
      Bottom panel: model sell probability over time with threshold line.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.2]})

    # Top panel: price chart
    price_col = "Adj Close" if "Adj Close" in price_df.columns else "Close"
    shared_idx = prob_df.index.intersection(price_df.index)
    price_plot = price_df.loc[shared_idx, price_col]

    ax1.plot(price_plot.index, price_plot.values, color="steelblue", linewidth=1.2, label="Price")

    # Shade drawdown event windows in orange (if events provided)
    # Only draw events that fall within the displayed date range
    if events:
        plot_start = prob_df.index.min()
        plot_end   = prob_df.index.max()
        visible_events = [ev for ev in events if ev["end"] >= plot_start and ev["start"] <= plot_end]
        for ev in visible_events:
            ax1.axvspan(ev["start"], ev["end"], alpha=0.20, color="orange")
            ax2.axvspan(ev["start"], ev["end"], alpha=0.20, color="orange")
        if visible_events:
            # Empty plot as a proxy artist for the legend (no data, won't affect x-axis range)
            ax1.plot([], [], color="orange", alpha=0.6, linewidth=8, label="Drawdown event")

    # Mark alert days as red dots
    alert_mask = prob_df.loc[shared_idx, "prob"] >= threshold
    alert_dates = shared_idx[alert_mask]
    if len(alert_dates) > 0:
        ax1.scatter(alert_dates, price_plot.loc[alert_dates],
                    color="red", zorder=5, s=20, label=f"Alert (thr={threshold:.2f})")

    # Force x-axis to only show the probability date range
    ax1.set_xlim(prob_df.index.min(), prob_df.index.max())

    ax1.set_ylabel("Price")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title(title, fontsize=12)

    # Bottom panel: probability
    ax2.fill_between(prob_df.index, prob_df["prob"], alpha=0.35, color="tomato")
    ax2.plot(prob_df.index, prob_df["prob"], color="tomato", linewidth=0.8)
    ax2.axhline(threshold, color="black", linestyle="--", linewidth=1.0,
                label=f"Threshold {threshold:.2f}")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Sell probability")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved -> {out_path}")


# CLI

def main():
    parser = argparse.ArgumentParser(description="Run sell probability inference")
    parser.add_argument("--features",  type=str,
                        default=str(ROOT_DIR / "data" / "sp500_features.csv"))
    parser.add_argument("--ckpt",      type=str,
                        default=str(ROOT_DIR / "checkpoints" / "best_model.pt"))
    parser.add_argument("--threshold", type=float, default=0.64,
                        help="Alert threshold")
    parser.add_argument("--plot_days", type=int, default=500,
                        help="How many past days to plot (0 = skip plot)")
    parser.add_argument("--out_plot",  type=str, default="",
                        help="Output path for the plot (default: artifacts/plots/infer_<date>.png)")
    parser.add_argument("--dataset",   type=str, default="",
                        help="Optional: path to sp500_dataset.csv to shade drawdown events on plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_path = Path(args.features).resolve()
    ckpt_path     = Path(args.ckpt).resolve()

    print(f"Device   : {device}")
    print(f"Features : {features_path}")
    print(f"Checkpoint: {ckpt_path}")

    model, feature_cols, split_cfg, temperature = load_model(ckpt_path, device)
    seq_len   = split_cfg["seq_len"]
    train_end = split_cfg["train_end"]
    print(f"seq_len={seq_len}  train_end={train_end}  T={temperature:.4f}  features={len(feature_cols)}")

    dates, arr = prepare_data(features_path, feature_cols, train_end)

    # Run inference for today only (single prediction)
    today_df = run_inference(model, arr, dates, seq_len, temperature, device, start_date=None)
    today    = today_df.index[-1]
    prob_today = float(today_df["prob"].iloc[-1])

    print("\n" + "=" * 45)
    print(f"  Date       : {today.date()}")
    print(f"  Sell prob  : {prob_today:.3f}")
    print(f"  Threshold  : {args.threshold:.3f}")
    decision = "[ALERT] Sell / Reduce" if prob_today >= args.threshold else "[HOLD ] Hold position"
    print(f"  Decision   : {decision}")
    print("=" * 45)

    # Run inference over the historical window and save a plot
    if args.plot_days > 0:
        start_date = (today - pd.Timedelta(days=args.plot_days)).strftime("%Y-%m-%d")
        hist_df = run_inference(model, arr, dates, seq_len, temperature, device,
                                start_date=start_date)

        # Load price series for the top panel of the chart
        raw_df = pd.read_csv(features_path, parse_dates=["Date"]).set_index("Date").sort_index()

        out_plot = args.out_plot
        if not out_plot:
            plots_dir = ROOT_DIR / "artifacts" / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            out_plot = str(plots_dir / f"infer_{today.date()}.png")

        # Load events for shading if dataset provided
        events = None
        if args.dataset:
            ds = pd.read_csv(args.dataset, parse_dates=["Date"]).set_index("Date")
            ds = ds[ds["ticker_id"] == 0].sort_index()
            ds = ds[ds.index >= hist_df.index[0]]
            ds["label_bin"] = (ds["label"] == 1).astype(int)
            events = detect_events(ds["label_bin"])

        plot_results(hist_df, raw_df, args.threshold, Path(out_plot), events=events)

        # Print recent alert dates
        alert_rows = hist_df[hist_df["prob"] >= args.threshold]
        if len(alert_rows) > 0:
            print(f"\nAlerts in last {args.plot_days} days (total={len(alert_rows)}):")
            for d, row in alert_rows.tail(10).iterrows():
                print(f"  {d.date()}  prob={row['prob']:.3f}")
        else:
            print(f"\nNo alerts in last {args.plot_days} days.")


if __name__ == "__main__":
    main()
