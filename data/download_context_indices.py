import yfinance as yf
from pathlib import Path

TICKERS = {
    "QQQ": "qqq_raw.csv",
    "^DJI": "dji_raw.csv",
}

START_DATE = "2000-01-01"
DATA_DIR = Path(__file__).parent


def download_ticker(ticker: str, filename: str) -> None:
    df = yf.download(
        ticker,
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    df = df.dropna().sort_index()
    output_path = DATA_DIR / filename
    df.to_csv(output_path)
    print(f"Saved {ticker} data to {output_path}")


def main():
    for ticker, filename in TICKERS.items():
        download_ticker(ticker, filename)


if __name__ == "__main__":
    main()
