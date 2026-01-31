import yfinance as yf
import pandas as pd
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
TICKER = "^GSPC"
START_DATE = "2000-01-01"
DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "sp500_raw.csv"

# ---------------------------
# Download data
# ---------------------------
def download_data():
    df = yf.download(
        TICKER,
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    df = df.dropna()
    df = df.sort_index()

    df.to_csv(OUTPUT_PATH)
    print(f"Saved raw data to {OUTPUT_PATH}")

if __name__ == "__main__":
    download_data()
