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


def main():
    df = pd.read_csv(FEAT_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    df2 = make_sell_score(df)
    df2.to_csv(OUT_PATH)

    print("Saved to:", OUT_PATH)
    print("sell_score range:", df2["sell_score"].min(), df2["sell_score"].max())


if __name__ == "__main__":
    main()
