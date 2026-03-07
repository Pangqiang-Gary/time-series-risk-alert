# Minimal Baseline Pipeline (split_b, DD_TH = -0.028, H = 7)

This README documents the exact commands that currently run the SP500 early-warning baseline with SP500 as the primary ticker and QQQ/DJI integrated as context features (as implemented in `dataset.py`).

## 1. Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(Bash)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Stage Commands

### Stage 1 – Data download → features → labeled datasets
Run the existing scripts in order (all paths are relative to repo root):

```powershell
python data\download_sp500.py
python data\download_context_indices.py                # downloads QQQ + ^DJI
python src\preprocess.py                               # builds sp500/qqq/dji feature CSVs
python src\make_labels.py --features data\sp500_features.csv --output data\sp500_dataset_h7_neg0p028.csv --ticker_id 0 --primary_ticker_id 0 --ew 1 --h 7 --dd_th -0.028
python src\make_labels.py --features data\qqq_features.csv --output data\qqq_dataset.csv --ticker_id 1 --primary_ticker_id 0 --aux_loss_weight 0.0 --ew 1 --h 7 --dd_th -0.028
python src\make_labels.py --features data\dji_features.csv --output data\dji_dataset.csv --ticker_id 2 --primary_ticker_id 0 --aux_loss_weight 0.0 --ew 1 --h 7 --dd_th -0.028
```

Outputs after Stage 1:
```
/data
  sp500_raw.csv, qqq_raw.csv, dji_raw.csv
  sp500_features.csv, qqq_features.csv, dji_features.csv
  sp500_dataset_h7_neg0p028.csv (primary, ticker_id=0)
  qqq_dataset.csv (ticker_id=1, aux weight 0)
  dji_dataset.csv (ticker_id=2, aux weight 0)
```

### Stage 2 – Train + evaluate on split_b

```powershell
python src\train.py `
  --datasets data/sp500_dataset_h7_neg0p028.csv,data/qqq_dataset.csv,data/dji_dataset.csv `
  --train_end 2020-12-31 `
  --val_end 2022-12-31 `
  --run_name split_b_ddth028_h7 `
  --save_dir checkpoints/split_b_ddth028_h7 `
  --threshold_policy constrained_f1 `
  --min_alert_rate 0.05 `
  --max_alert_rate 0.20 `
  --alert_rate 0.12 `
  --batch_size 64 `
  --epochs 30 `
  --patience 10
```

Key outputs:
```
/checkpoints/split_b_ddth028_h7/best_model.pt
/artifacts/evals/split_b_ddth028_h7_summary.json
/artifacts/evals/split_b_ddth028_h7_probabilities.npz
```

### Stage 3 – Threshold policy report CSV

```powershell
python analyze_threshold_policy.py `
  --summaries artifacts/evals/split_b_ddth028_h7_summary.json `
  --output artifacts/threshold_policy_metrics.csv
```

Output:
```
/artifacts/threshold_policy_metrics.csv
```

## 3. Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `ModuleNotFoundError` for torch/sklearn/etc. | Ensure the virtualenv is active and re-run `pip install -r requirements.txt`. |
| `FileNotFoundError: data/...csv` during Stage 2 | Rerun Stage 1 commands so the raw/features/datasets exist. |
| `No validation samples for primary ticker` | Make sure Stage 1 produced up-to-date SP500 dataset covering 2000–2025 and you’re passing `--train_end 2020-12-31 --val_end 2022-12-31`. |
| Yahoo Finance throttling downloads | Delete the partially-downloaded `*_raw.csv` files and rerun the download scripts after a short wait/VPN. |
| Reusing an existing checkpoint | Keep `--save_dir checkpoints/...` pointing to the run you want so training resumes from saved weights. |

## 4. Notes

- `dataset.load_datasets` performs the SP500+QQQ+DJI merge by renaming each ticker’s engineered columns (e.g., `t1_*`, `t2_*`) and joining them on aligned trading days before building windows.
- Thresholding experiments operate on the VAL-derived threshold per `--threshold_policy`; Stage 3 simply reformats the JSON metrics into a CSV for presentation.
- All non-baseline experiments remain accessible under `/archive` so nothing is lost, but the root workflow stays focused on the baseline pipeline.
