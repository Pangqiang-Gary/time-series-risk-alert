"""
Download all raw price data used in this project.

Assets:
  ^GSPC  → sp500_raw.csv
  QQQ    → qqq_raw.csv
  ^DJI   → dji_raw.csv

Run:
  python data/download_data.py
"""
import yfinance as yf
from pathlib import Path

# Start date for historical data
START_DATE = "2000-01-01"
DATA_DIR   = Path(__file__).parent

# Map each ticker symbol to its output filename
TICKERS = {
    "^GSPC": "sp500_raw.csv",   # S&P 500 index
    "QQQ":   "qqq_raw.csv",     # Nasdaq ETF
    "^DJI":  "dji_raw.csv",     # Dow Jones index
}


def download_ticker(ticker: str, filename: str) -> None:
    # Download daily OHLCV data from Yahoo Finance
    df = yf.download(
        ticker,
        start=START_DATE,
        interval="1d",
        auto_adjust=False,   # keep raw prices (not adjusted for splits/dividends automatically)
        progress=False,
    )
    df = df.dropna().sort_index()
    out = DATA_DIR / filename
    df.to_csv(out)
    print(f"Saved {ticker} -> {out}  ({len(df)} rows)")


def main() -> None:
    # Download each ticker one by one
    for ticker, filename in TICKERS.items():
        download_ticker(ticker, filename)
    print("\nDone. All raw data saved.")


if __name__ == "__main__":
    main()
