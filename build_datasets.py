"""
Build all datasets for training.

Pipeline:
  1. Feature engineering: raw CSVs -> feature CSVs (src/preprocess.py)
  2. SP500 labels: computed from SP500 price (make_labels.py)
  3. QQQ / DJI labels: each computed independently from their own price

Run:
  python build_datasets.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SRC = ROOT / "src"

# Input feature files (produced by preprocess.py)
SP500_FEATURES = DATA / "sp500_features.csv"
QQQ_FEATURES   = DATA / "qqq_features.csv"
DJI_FEATURES   = DATA / "dji_features.csv"

# Output dataset files (features + labels, ready for training)
SP500_DATASET  = DATA / "sp500_dataset.csv"
QQQ_DATASET    = DATA / "qqq_dataset.csv"
DJI_DATASET    = DATA / "dji_dataset.csv"

# Label parameters: look 7 days ahead, alert if price drops >= 2.8%
H      = 7
DD_TH  = "-0.028"
EW     = 1    # extend label 1 day earlier (early warning)


def run(cmd: list) -> None:
    # Run a python script as a subprocess and stop if it fails
    print("$", " ".join(str(c) for c in cmd))
    result = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    # Step 1: Build feature files from raw OHLCV data
    print("\n=== Step 1: Feature engineering ===")
    run([SRC / "preprocess.py"])

    # Step 2: Build SP500 dataset (ticker_id=0)
    print("\n=== Step 2: SP500 labels ===")
    run([
        SRC / "make_labels.py",
        "--features", SP500_FEATURES,
        "--output",   SP500_DATASET,
        "--ticker_id", "0",
        "--ew",   str(EW),
        "--h",    str(H),
        "--dd_th", DD_TH,
    ])

    # Step 3: Build QQQ dataset (ticker_id=1)
    print("\n=== Step 3: QQQ labels ===")
    run([
        SRC / "make_labels.py",
        "--features",  QQQ_FEATURES,
        "--output",    QQQ_DATASET,
        "--ticker_id", "1",
        "--ew",   str(EW),
        "--h",    str(H),
        "--dd_th", DD_TH,
    ])

    # Step 4: Build DJI dataset (ticker_id=2)
    print("\n=== Step 4: DJI labels ===")
    run([
        SRC / "make_labels.py",
        "--features",  DJI_FEATURES,
        "--output",    DJI_DATASET,
        "--ticker_id", "2",
        "--ew",   str(EW),
        "--h",    str(H),
        "--dd_th", DD_TH,
    ])

    print("\nDone. Datasets ready:")
    for p in [SP500_DATASET, QQQ_DATASET, DJI_DATASET]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
