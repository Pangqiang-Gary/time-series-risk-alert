from pathlib import Path
import argparse

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Default input/output paths
DEFAULT_FEATS = DATA_DIR / "sp500_features.csv"
DEFAULT_OUT = DATA_DIR / "sp500_dataset.csv"

# Label settings: look 7 days ahead, trigger if price drops >= 2.8%
H = 7
DD_TH = -0.028


def make_early_warning_label(
    df: pd.DataFrame, ew: int, horizon: int, dd_th: float, cooldown: int = 0
) -> pd.Series:
    """
    Create a binary early-warning label for each day.

    Logic:
      - For each day t, check if the minimum price in the next `horizon` days
        falls >= `dd_th` below today's price (i.e. a drawdown event).
      - If yes, mark day t as an event day (label=1).
      - `ew` extends the label 1 day earlier so the model has more warning time.
      - Days without enough future data get label=-1 (ignored in training).
    """
    price = df["Adj Close"].astype(float).to_numpy()
    n = len(price)

    # Step 1: Mark event days (drawdown >= threshold within next horizon days)
    event = np.zeros(n, dtype=np.int8)
    last_t = n - horizon - 1
    for t in range(last_t + 1):
        base = price[t]
        future_slice = price[t + 1: t + horizon + 1]
        dd = (np.min(future_slice) / base) - 1.0   # fractional drawdown
        if dd <= dd_th:
            event[t] = 1

    # Step 2: Apply cooldown (optional: ignore events too close together)
    event_idxs = np.where(event == 1)[0]
    if cooldown > 0 and len(event_idxs) > 0:
        kept = [event_idxs[0]]
        for t in event_idxs[1:]:
            if t - kept[-1] >= cooldown:
                kept.append(t)
        event_idxs = np.array(kept)

    # Step 3: Build final label array
    # Default is -1 (ignore). Event window gets label=1.
    label = -np.ones(n, dtype=np.int8)
    for t in event_idxs:
        start = max(0, t - ew)     # extend one day earlier for early warning
        label[start: t + 1] = 1

    return pd.Series(label, index=df.index, name="label")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",  type=str, default=str(DEFAULT_FEATS))
    parser.add_argument("--output",    type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--ticker_id", type=int, default=0)
    parser.add_argument("--ew",        type=int, default=1)
    parser.add_argument("--h",         type=int, default=H)
    parser.add_argument("--dd_th",     type=float, default=DD_TH)
    parser.add_argument("--cooldown",  type=int, default=0)
    args = parser.parse_args()

    feat_path = Path(args.features).resolve()
    out_path  = Path(args.output).resolve()

    # Load feature file and compute labels
    df = pd.read_csv(feat_path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    label = make_early_warning_label(df, args.ew, args.h, args.dd_th, args.cooldown)
    df["label"] = label

    # Remove rows with no label (end of series where future data is missing)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    pos_count = int((df["label"] == 1).sum())

    # Add ticker ID column so multi-asset datasets can be stacked
    df["ticker_id"] = int(args.ticker_id)
    df.to_csv(out_path)

    print(f"Saved to: {out_path}")
    print("Rows:", len(df), "Positive labels:", pos_count)


if __name__ == "__main__":
    main()
