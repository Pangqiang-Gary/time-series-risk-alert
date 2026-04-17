from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Each asset has a raw OHLCV file (input) and a features file (output)
ASSETS = {
    "sp500": {
        "raw": DATA_DIR / "sp500_raw.csv",
        "features": DATA_DIR / "sp500_features.csv",
    },
    "qqq": {
        "raw": DATA_DIR / "qqq_raw.csv",
        "features": DATA_DIR / "qqq_features.csv",
    },
    "dji": {
        "raw": DATA_DIR / "dji_raw.csv",
        "features": DATA_DIR / "dji_features.csv",
    },
}


def load_yahoo_csv(raw_path: Path) -> pd.DataFrame:
    """Load a raw Yahoo Finance CSV, handling different file formats."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    # Some Yahoo downloads have extra header rows — skip them if present
    skiprows = None
    with raw_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip().lower()
        second_line = f.readline().strip().lower()
        if first_line.startswith("price") and second_line.startswith("ticker"):
            skiprows = [1, 2]

    df = pd.read_csv(raw_path, skiprows=skiprows)

    # Ensure the first column is named "Date"
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Drop the "Price" column if present (duplicate of Close in some formats)
    if "Price" in df.columns:
        df = df.drop(columns=["Price"])

    df.columns = [c.strip() for c in df.columns]

    # Keep only standard OHLCV columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 50+ technical features from raw OHLCV data.
    Organized into 8 stages, each adding more features.
    """
    out = df.copy()
    # Use adjusted close price as the main price series
    price = out["Adj Close"] if "Adj Close" in out.columns else out["Close"]

    # --- Stage 0: Basic return and volatility features ---
    out["log_ret_1"] = np.log(price / price.shift(1))   # daily log return
    out["ret_5"]  = price.pct_change(5)                  # 5-day return
    out["ret_10"] = price.pct_change(10)
    out["ret_20"] = price.pct_change(20)

    out["vol_5"]  = out["log_ret_1"].rolling(5).std()    # short-term volatility
    out["vol_10"] = out["log_ret_1"].rolling(10).std()
    out["vol_20"] = out["log_ret_1"].rolling(20).std()
    out["vol_ratio"] = out["vol_5"] / (out["vol_20"] + 1e-12)  # short vs long vol ratio

    # Drawdown from recent peaks
    rolling_max_20  = price.rolling(20).max()
    rolling_max_60  = price.rolling(60).max()
    rolling_max_252 = price.rolling(252).max()
    out["dd_20"]  = (price - rolling_max_20)  / rolling_max_20   # % below 20-day high
    out["dd_60"]  = (price - rolling_max_60)  / rolling_max_60
    out["dd_252"] = (price - rolling_max_252) / rolling_max_252  # % below 1-year high

    # Candlestick shape features
    if all(c in out.columns for c in ["Open", "High", "Low", "Close"]):
        out["body_pct"]       = (out["Close"] - out["Open"]) / price    # candle body size
        out["range_pct"]      = (out["High"]  - out["Low"])  / price    # daily range
        out["upper_wick_pct"] = (out["High"]  - np.maximum(out["Open"], out["Close"])) / price
        out["lower_wick_pct"] = (np.minimum(out["Open"], out["Close"]) - out["Low"])   / price
        out["is_red"] = (out["Close"] < out["Open"]).astype(int)  # 1 if price fell today

    # Volume spike: how unusual is today's volume?
    if "Volume" in out.columns:
        v = out["Volume"]
        out["vol_z_20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()

    # --- Stage 1: Extended volatility features ---
    log_ret = out["log_ret_1"]
    out["vol_60"] = log_ret.rolling(60).std()                # 60-day volatility
    out["vol_ratio_10_60"] = out["vol_10"] / (out["vol_60"] + 1e-12)

    # Downside vs upside volatility (asymmetry signal)
    out["downside_vol_20"] = np.sqrt((log_ret.clip(upper=0) ** 2).rolling(20).mean())
    out["upside_vol_20"]   = np.sqrt((log_ret.clip(lower=0) ** 2).rolling(20).mean())
    out["down_up_vol_ratio"] = out["downside_vol_20"] / (out["upside_vol_20"] + 1e-12)

    # --- Stage 2: Trend / moving average features ---
    ma20 = price.rolling(20).mean()
    ma60 = price.rolling(60).mean()
    out["price_to_ma20"] = (price / (ma20 + 1e-12)) - 1    # % above/below 20-day MA
    out["price_to_ma60"] = (price / (ma60 + 1e-12)) - 1
    out["ma_slope_20"] = (ma20 - ma20.shift(5)) / (ma20.shift(5) + 1e-12)  # MA trend direction
    out["ma_slope_60"] = (ma60 - ma60.shift(5)) / (ma60.shift(5) + 1e-12)
    out["past_max_drawdown_20"] = (price.rolling(20).min() - rolling_max_20) / (rolling_max_20 + 1e-12)

    # --- Stage 3: Technical indicators ---
    # RSI: momentum oscillator (0=oversold, 1=overbought)
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 1 - (1 / (1 + gain / (loss + 1e-12)))

    # Bollinger band position: how far price is from the 20-day mean in std units
    bb_std = price.rolling(20).std()
    out["bollinger_pos"] = (price - ma20) / (2 * bb_std + 1e-12)

    # ADX: trend strength indicator (high = strong trend, low = ranging market)
    if all(c in out.columns for c in ["High", "Low", "Close"]):
        high, low, close = out["High"], out["Low"], out["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        up_move   = high.diff()
        down_move = -low.diff()
        plus_dm  = pd.Series(np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0), index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0), index=high.index)
        p = 14
        tr_s  = tr.ewm(alpha=1/p, adjust=False).mean()
        pdi   = 100 * plus_dm.ewm(alpha=1/p,  adjust=False).mean() / (tr_s + 1e-12)
        mdi   = 100 * minus_dm.ewm(alpha=1/p, adjust=False).mean() / (tr_s + 1e-12)
        dx    = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
        out["adx_14"] = dx.ewm(alpha=1/p, adjust=False).mean() / 100

    # MFI: money flow index (like RSI but also considers volume)
    if all(c in out.columns for c in ["High", "Low", "Close", "Volume"]):
        tp  = (out["High"] + out["Low"] + out["Close"]) / 3   # typical price
        rmf = tp * out["Volume"]
        tp_diff = tp.diff()
        pos_mf = rmf.where(tp_diff > 0, 0.0).rolling(14).sum()
        neg_mf = rmf.where(tp_diff < 0, 0.0).rolling(14).sum()
        out["mfi_14"] = 1 - 1 / (1 + pos_mf / (neg_mf + 1e-12))

    # --- Stage 4: Cross-feature combinations ---
    out["ret_5_over_vol_20"]  = out["ret_5"] / (out["vol_20"] + 1e-12)   # risk-adjusted return
    out["drawdown_over_vol"]  = out["dd_20"] / (out["vol_20"] + 1e-12)   # drawdown severity relative to vol
    out["zscore_ret_5"]  = (out["ret_5"]  - out["ret_5"].rolling(60).mean())  / (out["ret_5"].rolling(60).std()  + 1e-12)
    out["zscore_vol_10"] = (out["vol_10"] - out["vol_10"].rolling(60).mean()) / (out["vol_10"].rolling(60).std() + 1e-12)

    # --- Stage 5: Return distribution shape ---
    out["ret_skew_20"] = out["log_ret_1"].rolling(20).skew()   # negative skew = more crashes
    out["ret_kurt_20"] = out["log_ret_1"].rolling(20).kurt()   # fat tails = extreme moves more likely

    # --- Stage 6: Market state dynamics ---
    # How fast is volatility changing?
    out["vol_20_slope"] = (out["vol_20"] - out["vol_20"].shift(10)) / (out["vol_20"].shift(10) + 1e-12)

    # Consecutive down days streak
    sign = (out["log_ret_1"] < 0).astype(int)
    groups = (sign != sign.shift()).cumsum()
    out["consec_down_days"] = sign.groupby(groups).cumsum() * sign

    # How long since the last all-time high?
    at_high = price >= price.rolling(252, min_periods=1).max()
    out["drawdown_duration"] = (~at_high).astype(int).groupby(at_high.cumsum()).cumsum()

    # Parkinson volatility: uses high/low spread (more efficient than close-to-close vol)
    if all(c in out.columns for c in ["High", "Low"]):
        log_hl = np.log(out["High"] / out["Low"])
        out["park_vol_20"] = np.sqrt((log_hl ** 2).rolling(20).mean() / (4 * np.log(2)))

    # Speed of drawdown: is the drawdown accelerating?
    out["dd_speed_5"] = out["dd_20"] - out["dd_20"].shift(5)

    # Distance from 1-year low (low = near bottom, might mean oversold or more to fall)
    rolling_min_252 = price.rolling(252, min_periods=1).min()
    out["dist_to_low_252"] = (price - rolling_min_252) / (rolling_min_252 + 1e-12)

    # --- Stage 7: Day-over-day delta features ---
    # How much did each key indicator change since yesterday?
    out["vol5_change_1d"]   = out["vol_5"]  - out["vol_5"].shift(1)
    out["rsi_change_1d"]    = out["rsi_14"] - out["rsi_14"].shift(1)
    out["dd20_change_1d"]   = out["dd_20"]  - out["dd_20"].shift(1)
    # Did price just cross below the 20-day moving average?
    out["ma20_cross_today"] = (
        (out["price_to_ma20"] < 0) & (out["price_to_ma20"].shift(1) >= 0)
    ).astype(int)
    # Did price hit a new 20-day low today?
    out["new_low_20_today"] = (
        price == price.rolling(20, min_periods=1).min()
    ).astype(int)

    # --- Stage 8: Fragility features (quiet-before-storm signals) ---
    # Unusually low volatility (calm before a storm)
    out["vol_percentile_60"] = out["vol_20"].rolling(60).rank(pct=True)
    # Short-term vol vs long-term baseline: >1 means vol is breaking out
    out["vol_regime_shift"] = out["vol_5"] / (out["vol_60"] + 1e-12)

    # Drop any rows that have NaN values (from rolling window warm-up)
    return out.dropna()


def process_asset(name: str, raw_path: Path, features_path: Path) -> None:
    """Load raw data, compute features, and save to CSV."""
    df_raw = load_yahoo_csv(raw_path)
    df_feat = add_features(df_raw)
    df_feat.to_csv(features_path)
    print(f"[{name.upper()}] Saved features to: {features_path} (rows={len(df_feat)}, cols={len(df_feat.columns)})")


def main():
    # Process all three assets
    for name, paths in ASSETS.items():
        process_asset(name, paths["raw"], paths["features"])


if __name__ == "__main__":
    main()
