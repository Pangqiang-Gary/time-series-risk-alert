from pathlib import Path
import argparse

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Defaults for backward compatibility
DEFAULT_FEATS = DATA_DIR / "sp500_features.csv"
DEFAULT_OUT = DATA_DIR / "sp500_dataset.csv"

# ====== hyperparameters ======
H = 10         # look 10 days into the future
DD_TH = -0.05  # drawdown threshold (e.g., -5%)


def make_early_warning_label(df: pd.DataFrame, ew: int, horizon: int, dd_th: float) -> pd.Series:
    price = df["Adj Close"].astype(float).to_numpy()
    n = len(price)

    event = np.zeros(n, dtype=np.int8)
    last_t = n - horizon - 1
    for t in range(last_t + 1):
        base = price[t]
        future_slice = price[t + 1: t + horizon + 1]
        min_future = np.min(future_slice)
        dd = (min_future / base) - 1.0
        if dd <= dd_th:
            event[t] = 1

    label = -np.ones(n, dtype=np.int8)
    event_idxs = np.where(event == 1)[0]
    for t in event_idxs:
        start = max(0, t - ew)
        label[start: t + 1] = 1

    return pd.Series(label, index=df.index, name="label")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=str(DEFAULT_FEATS), help="Input feature CSV path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT), help="Output labeled dataset path")
    parser.add_argument("--ticker_id", type=int, default=0, help="Numeric ticker id for this asset")
    parser.add_argument("--primary_ticker_id", type=int, default=0, help="Ticker id treated as primary (weight=1.0)")
    parser.add_argument("--aux_loss_weight", type=float, default=0.0, help="Loss weight for non-primary tickers")
    parser.add_argument("--ew", type=int, default=1, help="Early-warning window length (days before event)")
    parser.add_argument("--h", type=int, default=H, help="Forecast horizon length (days into the future)")
    parser.add_argument("--dd_th", type=float, default=DD_TH, help="Drawdown threshold for event labeling")
    args = parser.parse_args()

    feat_path = Path(args.features).resolve()
    out_path = Path(args.output).resolve()

    df = pd.read_csv(feat_path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    label = make_early_warning_label(df, args.ew, args.h, args.dd_th)
    df["label"] = label
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    ticker_id = int(args.ticker_id)
    df["ticker_id"] = ticker_id

    loss_weight = 1.0 if ticker_id == args.primary_ticker_id else float(args.aux_loss_weight)
    df["loss_weight"] = loss_weight

    df.to_csv(out_path)

    pos_count = int((df["label"] == 1).sum())
    print(f"Saved to: {out_path}")
    print("Rows:", len(df), "Positive labels:", pos_count, "Loss weight:", loss_weight)


if __name__ == "__main__":
    main()
