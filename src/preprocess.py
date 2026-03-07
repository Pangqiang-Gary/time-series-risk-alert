from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

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
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    skiprows = None
    with raw_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip().lower()
        second_line = f.readline().strip().lower()
        if first_line.startswith("price") and second_line.startswith("ticker"):
            skiprows = [1, 2]

    df = pd.read_csv(raw_path, skiprows=skiprows)

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    if "Price" in df.columns:
        df = df.drop(columns=["Price"])

    df.columns = [c.strip() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price = out["Adj Close"] if "Adj Close" in out.columns else out["Close"]

    out["log_ret_1"] = np.log(price / price.shift(1))
    out["ret_5"] = price.pct_change(5)
    out["ret_10"] = price.pct_change(10)
    out["ret_20"] = price.pct_change(20)

    out["vol_5"] = out["log_ret_1"].rolling(5).std()
    out["vol_10"] = out["log_ret_1"].rolling(10).std()
    out["vol_20"] = out["log_ret_1"].rolling(20).std()

    rolling_max_20 = price.rolling(20).max()
    out["dd_20"] = (price - rolling_max_20) / rolling_max_20
    rolling_max_60 = price.rolling(60).max()
    out["dd_60"] = (price - rolling_max_60) / rolling_max_60

    if all(c in out.columns for c in ["Open", "High", "Low", "Close"]):
        out["body"] = out["Close"] - out["Open"]
        out["range"] = out["High"] - out["Low"]
        out["upper_wick"] = out["High"] - np.maximum(out["Open"], out["Close"])
        out["lower_wick"] = np.minimum(out["Open"], out["Close"]) - out["Low"]
        out["is_red"] = (out["Close"] < out["Open"]).astype(int)

    if "Volume" in out.columns:
        vol = out["Volume"]
        out["vol_z_20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

    return out.dropna()


def process_asset(name: str, raw_path: Path, features_path: Path) -> None:
    df_raw = load_yahoo_csv(raw_path)
    df_feat = add_features(df_raw)
    df_feat.to_csv(features_path)
    print(f"[{name.upper()}] Saved features to: {features_path} (rows={len(df_feat)}, cols={len(df_feat.columns)})")


def main():
    for name, paths in ASSETS.items():
        process_asset(name, paths["raw"], paths["features"])


if __name__ == "__main__":
    main()
