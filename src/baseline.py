"""
baseline.py - Traditional ML baselines for comparison with the Transformer model.

Models trained:
  - Logistic Regression
  - SVM (RBF kernel)
  - XGBoost
  - AdaBoost

Features: same 50 features, same train/val/test split, same event-based evaluation.
Two feature representations:
  - flat_1: last-day only (50 features) - tests if temporal context helps
  - flat_seq: flattened seq_len=20 window (1000 features) - fair comparison with Transformer

Usage:
  python src/baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from dataset import SplitConfig, load_datasets
from eval_events import detect_events

# Settings matching the Transformer model (for fair comparison)
SEQ_LEN    = 20
TRAIN_END  = "2017-12-31"
VAL_END    = "2021-12-31"
TEST_START = "2022-01-01"
GAP_DAYS   = 5         # gap between label=1 days to separate events
ALERT_THR  = 0.61      # same threshold family as Transformer
LEAD_WINDOW = 7        # days BEFORE event start to look for an alert
LAG_ALLOW   = 3        # days AFTER event start still counted as a valid (late but ok) hit


# ── helpers ──────────────────────────────────────────────────────────────────

def build_windows(ds, seq_len: int, flat: bool = True):
    """
    Convert a TimeSeriesWindowDataset into numpy arrays.
    flat=True  → (N, seq_len * D)  for sklearn models
    flat=False → (N, seq_len, D)   3-D for manual feature extraction
    """
    X_list, y_list = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        X_list.append(x.numpy())
        y_list.append(float(y.item()))
    X = np.stack(X_list)                         # (N, seq_len, D)
    y = np.array(y_list)
    if flat:
        X = X.reshape(len(X), -1)
    return X, y


def last_day_only(X_seq: np.ndarray) -> np.ndarray:
    """From (N, seq_len, D) take only the last time-step → (N, D).
    Used for "last-day" baselines that ignore history."""
    return X_seq[:, -1, :]


def threshold_at_rate(prob: np.ndarray, rate: float) -> float:
    """Set threshold so that the top `rate` fraction of predictions become alerts."""
    return float(np.quantile(prob, 1.0 - rate))


def evaluate_day_level(y: np.ndarray, prob: np.ndarray, thr: float):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    prauc = float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    alert_rate = float(pred.mean())
    return {"prec": prec, "rec": rec, "pr_auc": prauc, "alert_rate": alert_rate}


def evaluate_events(prob_series: pd.Series, events: list, thr: float):
    """
    Check how many events were caught by at least one alert within the evaluation window.
    Window: from 7 days before event start to 3 days after event start.
    Also computes average lead time (positive = early, negative = late).
    """
    hit = 0
    lead_times = []
    for ev in events:
        ev_start = ev["start"]
        win_start = ev_start - pd.Timedelta(days=LEAD_WINDOW)
        win_end   = ev_start + pd.Timedelta(days=LAG_ALLOW)
        window = prob_series[(prob_series.index >= win_start) & (prob_series.index <= win_end)]
        alerts = window[window >= thr]
        if len(alerts) > 0:
            first = alerts.index[0]
            lead  = (ev_start - first).days
            hit  += 1
            lead_times.append(lead)
    hit_rate = hit / len(events) if events else float("nan")
    avg_lead = float(np.mean(lead_times)) if lead_times else float("nan")
    return {"hit": hit, "total": len(events), "hit_rate": hit_rate, "avg_lead": avg_lead}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load datasets using the exact same pipeline as the Transformer ──────
    dataset_paths = [
        ROOT_DIR / "data" / "sp500_dataset.csv",
        ROOT_DIR / "data" / "qqq_dataset.csv",
        ROOT_DIR / "data" / "dji_dataset.csv",
    ]
    cfg = SplitConfig(seq_len=SEQ_LEN, train_end=TRAIN_END, val_end=VAL_END)
    train_ds, val_ds, test_ds, feature_cols = load_datasets(dataset_paths, cfg)

    print(f"Feature dim    : {len(feature_cols)}")
    print(f"Train windows  : {len(train_ds)}")
    print(f"Val windows    : {len(val_ds)}")
    print(f"Test windows   : {len(test_ds)}")

    # ── 2. Build arrays and filter to SP500 only (ticker_id=0) ───────────────
    # SP500-only masks
    train_mask = train_ds.window_tickers == 0
    val_mask   = val_ds.window_tickers   == 0
    test_mask  = test_ds.window_tickers  == 0

    # 3-D arrays: (N, seq_len, D)
    X_train_seq, y_train = build_windows(train_ds, SEQ_LEN, flat=False)
    X_val_seq,   y_val   = build_windows(val_ds,   SEQ_LEN, flat=False)
    X_test_seq,  y_test  = build_windows(test_ds,  SEQ_LEN, flat=False)

    # SP500 only
    X_tr_seq = X_train_seq[train_mask];  y_tr = y_train[train_mask]
    X_va_seq = X_val_seq[val_mask];      y_va = y_val[val_mask]
    X_te_seq = X_test_seq[test_mask];    y_te = y_test[test_mask]

    # Flat representations
    X_tr_last = last_day_only(X_tr_seq);  X_tr_flat = X_tr_seq.reshape(len(X_tr_seq), -1)
    X_va_last = last_day_only(X_va_seq);  X_va_flat = X_va_seq.reshape(len(X_va_seq), -1)
    X_te_last = last_day_only(X_te_seq);  X_te_flat = X_te_seq.reshape(len(X_te_seq), -1)

    test_dates = test_ds.window_dates[test_mask]

    # ── 3. Ground-truth events in test set ───────────────────────────────────
    label_df = pd.read_csv(ROOT_DIR / "data" / "sp500_dataset.csv",
                           parse_dates=["Date"]).set_index("Date")
    label_df = label_df[label_df["ticker_id"] == 0].sort_index()
    label_df["label_bin"] = (label_df["label"] == 1).astype(int)
    label_test = label_df[label_df.index >= pd.Timestamp(TEST_START)]
    events = detect_events(label_test["label_bin"], gap_days=GAP_DAYS)
    print(f"\nTest events    : {len(events)}\n")

    # ── 4. Set the same alert rate as the Transformer (4.1%) for fair comparison ─
    TARGET_ALERT_RATE = 0.041

    # ── 5. Model definitions ──────────────────────────────────────────────────
    # Re-scale: features are already z-scored per ticker, but flattening
    # mixes time steps together, so we apply an extra StandardScaler pass.
    # XGBoost doesn't need scaling — it's tree-based.
    scaler_last = StandardScaler()
    scaler_flat = StandardScaler()
    X_tr_last_s = scaler_last.fit_transform(X_tr_last)
    X_va_last_s = scaler_last.transform(X_va_last)
    X_te_last_s = scaler_last.transform(X_te_last)
    X_tr_flat_s = scaler_flat.fit_transform(X_tr_flat)
    X_va_flat_s = scaler_flat.transform(X_va_flat)
    X_te_flat_s = scaler_flat.transform(X_te_flat)

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    models = {
        "LogReg (last-day)": {
            "model": LogisticRegression(C=0.1, class_weight="balanced",
                                        max_iter=1000, random_state=42),
            "X_tr": X_tr_last_s, "X_va": X_va_last_s, "X_te": X_te_last_s,
        },
        "LogReg (seq-flat)": {
            "model": LogisticRegression(C=0.01, class_weight="balanced",
                                        max_iter=1000, random_state=42),
            "X_tr": X_tr_flat_s, "X_va": X_va_flat_s, "X_te": X_te_flat_s,
        },
        "SVM (last-day)": {
            "model": SVC(C=1.0, kernel="rbf", class_weight="balanced",
                         probability=True, random_state=42),
            "X_tr": X_tr_last_s, "X_va": X_va_last_s, "X_te": X_te_last_s,
        },
        "XGBoost (last-day)": {
            "model": XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                scale_pos_weight=pos_weight,
                use_label_encoder=False, eval_metric="aucpr",
                random_state=42, verbosity=0,
            ),
            "X_tr": X_tr_last, "X_va": X_va_last, "X_te": X_te_last,
        },
        "XGBoost (seq-flat)": {
            "model": XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                scale_pos_weight=pos_weight,
                use_label_encoder=False, eval_metric="aucpr",
                random_state=42, verbosity=0,
            ),
            "X_tr": X_tr_flat, "X_va": X_va_flat, "X_te": X_te_flat,
        },
        "AdaBoost (last-day)": {
            "model": AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                        random_state=42),
            "X_tr": X_tr_last_s, "X_va": X_va_last_s, "X_te": X_te_last_s,
        },
    }

    # ── 6. Train and evaluate each model ──────────────────────────────────────
    results = []

    for name, cfg_m in models.items():
        print(f"Training {name} ...", end=" ", flush=True)
        m = cfg_m["model"]
        m.fit(cfg_m["X_tr"], y_tr)
        print("done")

        val_prob  = m.predict_proba(cfg_m["X_va"])[:, 1]
        test_prob = m.predict_proba(cfg_m["X_te"])[:, 1]

        # Set threshold on val set so alert rate = 4.1%, then apply same threshold to test
        thr = threshold_at_rate(val_prob, TARGET_ALERT_RATE)
        val_metrics  = evaluate_day_level(y_va, val_prob,  thr)
        test_metrics = evaluate_day_level(y_te, test_prob, thr)

        # Event evaluation requires matching dates
        prob_series = pd.Series(test_prob, index=pd.to_datetime(test_dates))
        ev_metrics = evaluate_events(prob_series, events, thr)

        results.append({
            "model": name,
            "val_prec":  val_metrics["prec"],
            "val_prauc": val_metrics["pr_auc"],
            "test_prec": test_metrics["prec"],
            "test_rec":  test_metrics["rec"],
            "test_prauc":test_metrics["pr_auc"],
            "alert_rate":test_metrics["alert_rate"],
            "event_hit":  ev_metrics["hit_rate"],
            "avg_lead":   ev_metrics["avg_lead"],
            "n_hit":      ev_metrics["hit"],
        })

    # ── 7. Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  BASELINE COMPARISON  (all at ~4.1% alert rate, SP500 test set 2022-)")
    print("=" * 100)
    print(f"  {'Model':<26} {'ValPrec':>8} {'ValPRAUC':>9} {'TestPrec':>9} {'TestRec':>8} "
          f"{'TestPRAUC':>10} {'EventHit':>10} {'AvgLead':>9}")
    print("-" * 100)

    for r in results:
        lead_str = f"{r['avg_lead']:+.1f}d" if not np.isnan(r["avg_lead"]) else "   -"
        print(f"  {r['model']:<26} {r['val_prec']:8.3f} {r['val_prauc']:9.3f} "
              f"{r['test_prec']:9.3f} {r['test_rec']:8.3f} {r['test_prauc']:10.3f} "
              f"  {r['n_hit']}/{len(events)}={r['event_hit']:.0%}   {lead_str}")

    # Transformer reference row (hardcoded from last full run for quick comparison)
    print("-" * 100)
    print(f"  {'Transformer (ours)':<26} {'0.488':>8} {'0.338':>9} {'0.429':>9} "
          f"{'0.130':>8} {'0.367':>10}    7/29=24%   +5.7d")
    print("=" * 100)
    print("\nNote: last-day = only today's features; seq-flat = 20-day window flattened")
    print(f"      Transformer uses seq_len=20 with temporal attention (d_model=28, 11,937 params)")


if __name__ == "__main__":
    main()
