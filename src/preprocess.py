from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------
# Paths
# ---------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_PATH = DATA_DIR / "sp500_raw.csv"
OUT_PATH = DATA_DIR / "sp500_features.csv"

# ---------------------------
# Load raw Yahoo Finance CSV
# ---------------------------
def load_yahoo_csv(raw_path: Path) -> pd.DataFrame:
    """
    Load Yahoo Finance CSV contain extra header rows:
    - Price,Adj Close,Close,...
    - Ticker,^GSPC,^GSPC,...
    - Date,,,,,,
    Then data rows start with YYYY-MM-DD.
    """
    # Read all lines, find where actual data starts (first line beginning with a date)
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line[:4].isdigit() and line[4:5] == "-" and line[7:8] == "-":
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Cannot find the first data row (YYYY-MM-DD) in the CSV.")

    df = pd.read_csv(
        raw_path,
        skiprows=[1, 2],  
    )

    # Ensure Date exists and is datetime index
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Drop'Price' column
    if "Price" in df.columns:
        df = df.drop(columns=["Price"])

    # Standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Keep only expected columns if available
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()

    return df


# ---------------------------
# Feature engineering
# ---------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create risk-related features suitable for time-series models.
    """
    out = df.copy()

    # Use Adj Close for returns (standard practice)
    price = out["Adj Close"] if "Adj Close" in out.columns else out["Close"]

    # Log return (1-day)（today/yesterday）
    out["log_ret_1"] = np.log(price / price.shift(1))

    # Rolling returns (momentum)
    out["ret_5"] = price.pct_change(5)#P(t)​−P(t−5)​​/P(t−5)
    out["ret_10"] = price.pct_change(10)
    out["ret_20"] = price.pct_change(20)

    # Rolling volatility (risk)
    out["vol_5"] = out["log_ret_1"].rolling(5).std()#log_ret_1(t) to log_ret_1(t-4)
    out["vol_10"] = out["log_ret_1"].rolling(10).std()
    out["vol_20"] = out["log_ret_1"].rolling(20).std()

    # Drawdown (risk)
    rolling_max_20 = price.rolling(20).max()#max price in recent 20 days
    out["dd_20"] = (price - rolling_max_20) / rolling_max_20# rate to the peak in 20 days

    rolling_max_60 = price.rolling(60).max()
    out["dd_60"] = (price - rolling_max_60) / rolling_max_60

    # Candlestick geometry features ( "4D vector" idea)
    # Body, range, upper/lower wick
    if all(c in out.columns for c in ["Open", "High", "Low", "Close"]):
        out["body"] = out["Close"] - out["Open"]
        out["range"] = out["High"] - out["Low"]
        out["upper_wick"] = out["High"] - np.maximum(out["Open"], out["Close"])
        out["lower_wick"] = np.minimum(out["Open"], out["Close"]) - out["Low"]
        out["is_red"] = (out["Close"] < out["Open"]).astype(int)  # optional binary signal

    # Volume features
    if "Volume" in out.columns:
        out["vol_z_20"] = (out["Volume"] - out["Volume"].rolling(20).mean()) / out["Volume"].rolling(20).std() # (today volume- mean volume in 20 days)/ volume std in 20 days

    # Clean NaNs from rolling computations
    out = out.dropna()

    return out


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df_raw = load_yahoo_csv(RAW_PATH)
    df_feat = add_features(df_raw)

    df_feat.to_csv(OUT_PATH)
    print(f"Saved features to: {OUT_PATH}")
    print("Columns:", list(df_feat.columns))
    print("Rows:", len(df_feat))


if __name__ == "__main__":
    main()
