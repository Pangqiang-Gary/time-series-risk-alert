from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

FEAT_PATH = DATA_DIR / "sp500_features.csv"
OUT_PATH = DATA_DIR / "sp500_dataset.csv"

# ====== hyperparameters ======
H = 15          # see in 15 days in future
LOW_WIN = 20    # low point in windows

DD_TH = -0.08   # drawdown threshold, e.g. -8%
EW = 10        # early-warning window length (days before event)


def make_sell_score(df: pd.DataFrame) -> pd.DataFrame:
    price = df["Adj Close"].astype(float)

    # ===== 1.Future cumulative return=====
    future_price = price.shift(-H)  #price after H days / today price
    cum_ret = (future_price / price) - 1.0        # (Price(t+H)/Price(t))-1 if < 0 it will go down
    # normalized so that a 5% drawdown maps to a score of 1.
    cum_score = (-cum_ret / 0.05).clip(0, 1)

    # ===== 2. rate of days go down =====
    future_ret = price.shift(-1).pct_change() #future_ret[t] = (price[t+1] / price[t]) - 1
    down_days = (future_ret < 0).rolling(H).sum() #future_ret[t-H+1] to future_ret[t] to count the numeber of days go down
    down_ratio = (down_days / H).clip(0, 1)

    # ===== 3. Whether the price breaks below a recent local low. =====
    recent_low = price.rolling(LOW_WIN).min()
    future_min = price.shift(-1).rolling(H).min()
    break_low = (future_min < recent_low).astype(float)

    # ===== 4 trend=====
    trend = price / price.rolling(30).mean() - 1.0
    #When the price is in a strong uptrend, the sell-risk label is down-weighted; once the price falls below the moving average, the risk is no longer suppressed.
    #  strong uptrend，gate close to 0；below the moving average, gate → 1
    trend_gate = (1 -0.5* trend.clip(lower=0))



    # ===== 5. sell_score =====
    sell_score = (
        0.5 * cum_score +
        0.3 * down_ratio +
        0.2 * break_low
    )* trend_gate
    sell_score = sell_score.clip(0, 1)

    df_out = df.copy()
    df_out["sell_score"] = sell_score
    df_out = df_out.dropna(subset=["sell_score"])
    return df_out


def make_early_warning_label(df: pd.DataFrame) -> pd.Series:
    """
    Early-warning label definition:
    1) Define an event at day t if within the next H days, the maximum drawdown
       from price[t] to min(price[t+1..t+H]) is <= DD_TH.
    2) Convert event to early-warning labels by marking the EW days BEFORE an event as +1.
       All other days are -1.
    """
    price = df["Adj Close"].astype(float).to_numpy()
    n = len(price)

    # event[t] = 1 if future max drawdown within next H days <= DD_TH
    event = np.zeros(n, dtype=np.int8)

    last_t = n - H - 1  # last index where still have a full future window
    for t in range(last_t + 1):
        base = price[t]
        future_slice = price[t + 1 : t + H + 1]  # next H days
        min_future = np.min(future_slice)
        dd = (min_future / base) - 1.0
        if dd <= DD_TH:
            event[t] = 1

    # Initialize all labels as -1
    label = -np.ones(n, dtype=np.int8)

    # For each event time t, label [t-EW, ..., t] as +1
    event_idxs = np.where(event == 1)[0]
    for t in event_idxs:
        start = max(0, t - EW)
        label[start : t + 1] = 1

    return pd.Series(label, index=df.index, name="label")


def main():
    # 1) Load feature file
    df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    # 2) Compute rule-based sell_score (baseline)
    df2 = make_sell_score(df)

    # 3) Compute early-warning label (align to df2's index)
    label = make_early_warning_label(df2)
    df2["label"] = label

    # 4) Drop rows with missing label (typically last H days)
    df2 = df2.dropna(subset=["label"])

    # 5) Keep label as int (-1 / +1) for now
    df2["label"] = df2["label"].astype(int)

    # 6) Save
    df2.to_csv(OUT_PATH)

    print("Saved to:", OUT_PATH)
    print("sell_score range:", float(df2["sell_score"].min()), float(df2["sell_score"].max()))
    print("label counts:", df2["label"].value_counts().to_dict())




if __name__ == "__main__":
    main()
