"""
case_analysis.py - Case-by-case analysis of model alerts on major drawdown events.

For each event:
  - Caught events: examine features on the first alert day
  - Missed events: examine features on t-1 (day before event start)
Then compare feature values between caught vs missed to find diagnostic patterns.

Also reports:
  - Event severity classification (Major/Moderate/Minor by actual max drawdown)
  - Capturable drawdown: price drop from first alert to event trough
  - Per-period PR-AUC breakdown by market phase

Usage:
  python src/case_analysis.py
  python src/case_analysis.py --test_start 2018-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from infer import load_model, prepare_data, run_inference

import torch


def detect_events(labels: pd.Series, gap_days: int = 5) -> list[dict]:
    """
    Group consecutive label=1 days into discrete drawdown events.
    If two positive days are more than gap_days apart, they are separate events.
    Returns a list of {start, end, n_days} dicts.
    """
    pos_dates = labels[labels == 1].index.sort_values()
    if len(pos_dates) == 0:
        return []
    events = []
    start = pos_dates[0]
    prev  = pos_dates[0]
    for d in pos_dates[1:]:
        if (d - prev).days > gap_days:
            events.append({"start": start, "end": prev,
                           "n_days": (prev - start).days + 1})
            start = d
        prev = d
    events.append({"start": start, "end": prev,
                   "n_days": (prev - start).days + 1})
    return events

# Features to compare on alert day (caught) vs day before event (missed)
# These help diagnose why the model caught or missed each event
DIAG_FEATURES = [
    "vol_20",          # volatility level
    "vol_ratio",       # short vs long vol
    "dd_20",           # drawdown from 20-day high
    "rsi_14",          # momentum
    "price_to_ma20",   # price vs trend
    "ma_slope_20",     # trend direction
    "adx_14",          # trend strength
    "log_ret_1",       # yesterday's return
]

# Market regime by year (manually labeled)
REGIME = {
    2018: "bear",
    2019: "bull",
    2020: "volatile",
    2021: "bull",
    2022: "bear",
    2023: "bull",
    2024: "bull",
    2025: "bull",
    2026: "?",
}

# Named market phases for per-period PR-AUC
PERIODS = [
    ("2018 Bear",       "2018-01-01", "2018-12-31"),
    ("2019 Bull",       "2019-01-01", "2019-12-31"),
    ("2020 Volatile",   "2020-01-01", "2020-12-31"),
    ("2021 Bull",       "2021-01-01", "2021-12-31"),
    ("2022 Bear",       "2022-01-01", "2022-12-31"),
    ("2023-24 Bull",    "2023-01-01", "2024-12-31"),
    ("2025+",           "2025-01-01", "2099-12-31"),
]

# Classify events by how bad the actual drawdown was
SEVERITY_MAJOR    = -0.05   # >5% drop = Major
SEVERITY_MODERATE = -0.03   # 3-5% drop = Moderate; <3% = Minor


def assign_regime(date: pd.Timestamp) -> str:
    return REGIME.get(date.year, "?")


def classify_severity(actual_dd: float) -> str:
    if actual_dd <= SEVERITY_MAJOR:
        return "Major"
    elif actual_dd <= SEVERITY_MODERATE:
        return "Moderate"
    else:
        return "Minor"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   type=str, default=str(ROOT_DIR / "data" / "sp500_features.csv"))
    parser.add_argument("--dataset",    type=str, default=str(ROOT_DIR / "data" / "sp500_dataset.csv"))
    parser.add_argument("--ckpt",       type=str, default=str(ROOT_DIR / "checkpoints" / "best_model.pt"))
    parser.add_argument("--test_start", type=str, default="2018-01-01")
    parser.add_argument("--threshold",  type=float, default=0.61)
    parser.add_argument("--lead_days",  type=int, default=7)
    parser.add_argument("--gap_days",   type=int, default=5)
    parser.add_argument("--output",     type=str, default=str(ROOT_DIR / "artifacts" / "evals" / "case_analysis.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and run sliding-window inference from test_start to today
    model, feature_cols, split_cfg, temperature = load_model(Path(args.ckpt), device)
    dates, arr = prepare_data(Path(args.features), feature_cols, split_cfg["train_end"])
    prob_series = run_inference(model, arr, dates, split_cfg["seq_len"], temperature, device,
                                start_date=args.test_start)

    # Load ground-truth labels for SP500 (ticker_id=0)
    label_df = pd.read_csv(args.dataset, parse_dates=["Date"]).set_index("Date")
    label_df = label_df[label_df["ticker_id"] == 0].sort_index()
    label_df = label_df[label_df.index >= pd.Timestamp(args.test_start)]

    # Load price series for computing actual drawdown depths
    price_df = pd.read_csv(args.features, parse_dates=["Date"]).set_index("Date").sort_index()
    price_col = "Adj Close" if "Adj Close" in price_df.columns else "Close"
    price = price_df[price_col]

    # Load raw feature values for the diagnostic comparison section
    feat_df = price_df[price_df.index >= pd.Timestamp(args.test_start)].copy()
    diag_cols = [c for c in DIAG_FEATURES if c in feat_df.columns]

    # Join model probabilities with labels into one table
    merged = prob_series.join(label_df["label"], how="left").fillna(-1)

    # Group label=1 days into discrete events
    events = detect_events(label_df["label"], gap_days=args.gap_days)

    # Per-event analysis
    records = []
    caught_features = []
    missed_features = []

    print(f"\n{'='*90}")
    print(f"  CASE-BY-CASE ANALYSIS  (test_start={args.test_start}, thr={args.threshold})")
    print(f"{'='*90}")
    print(f"\n{'#':<4} {'Start':<12} {'End':<12} {'Days':>4}  {'Severity':<10} {'Result':<12} {'Lead':>7}  {'MaxProb':>7}  {'CapturDD':>9}  {'Regime':<8}")
    print("-" * 90)

    for i, ev in enumerate(events):
        ev_start = ev["start"]
        ev_end   = ev["end"]
        regime   = assign_regime(ev_start)

        # Actual max drawdown within the event window (start price -> trough)
        ev_prices = price[ev_start:ev_end]
        if len(ev_prices) > 0:
            start_price = float(price[price.index <= ev_start].iloc[-1])
            trough_price = float(ev_prices.min())
            actual_dd = (trough_price / start_price) - 1.0
        else:
            actual_dd = float("nan")
        severity = classify_severity(actual_dd) if not np.isnan(actual_dd) else "?"

        # Evaluation window: 7 days before event start to 3 days after.
        # An alert anywhere in this window counts as a hit.
        window_start = ev_start - pd.Timedelta(days=args.lead_days)
        window_end   = ev_start + pd.Timedelta(days=3)
        window = merged[(merged.index >= window_start) & (merged.index <= window_end)]["prob"]
        max_prob = float(window.max()) if len(window) > 0 else float("nan")

        # Find first alert above threshold
        alerts = window[window >= args.threshold]
        capturable_dd = float("nan")

        if len(alerts) > 0:
            first_alert = alerts.index[0]
            lead = (ev_start - first_alert).days

            # Classify how early or late the first alert was
            if lead > 3:
                result = "Early"       # alert came more than 3 days before event start
            elif lead >= 0:
                result = "On-time"     # alert on the event day or up to 3 days before
            elif lead >= -3:
                result = "Late(ok)"    # alert up to 3 days after event start (teacher allows this)
            else:
                result = "Too-late"    # alert came too late to be useful

            lead_str = f"{lead:+d}d"

            # Capturable drawdown: from alert price to trough within event
            if first_alert in price.index:
                alert_price = float(price[first_alert])
                future_prices = price[first_alert:ev_end]
                if len(future_prices) > 0:
                    capturable_dd = (future_prices.min() / alert_price) - 1.0

            # Features on first alert day
            if first_alert in feat_df.index:
                row = feat_df.loc[first_alert, diag_cols].to_dict()
                row["prob"] = float(alerts.iloc[0])
                row["event_start"] = str(ev_start.date())
                row["alert_date"] = str(first_alert.date())
                row["lead"] = lead
                row["result"] = result
                row["regime"] = regime
                caught_features.append(row)
        else:
            result = "Missed"
            lead_str = "-"

            # Features on t-1 (day before event start)
            t_minus_1 = ev_start - pd.Timedelta(days=1)
            candidates = feat_df[feat_df.index <= t_minus_1]
            if len(candidates) > 0:
                t1_date = candidates.index[-1]
                row = feat_df.loc[t1_date, diag_cols].to_dict()
                row["prob"] = float(merged.loc[t1_date, "prob"]) if t1_date in merged.index else float("nan")
                row["event_start"] = str(ev_start.date())
                row["alert_date"] = str(t1_date.date()) + " (t-1)"
                row["lead"] = None
                row["result"] = "Missed"
                row["regime"] = regime
                missed_features.append(row)

        cap_str = f"{capturable_dd:+.1%}" if not np.isnan(capturable_dd) else "  -"
        print(f"{i+1:<4} {str(ev_start.date()):<12} {str(ev_end.date()):<12} {ev['n_days']:>4}  "
              f"{severity:<10} {result:<12} {lead_str:>7}  {max_prob:>7.3f}  {cap_str:>9}  {regime:<8}")

        records.append({
            "event_num": i + 1,
            "start": str(ev_start.date()),
            "end": str(ev_end.date()),
            "n_days": ev["n_days"],
            "severity": severity,
            "actual_dd": round(actual_dd, 4) if not np.isnan(actual_dd) else None,
            "result": result,
            "lead_days": lead if len(alerts) > 0 else None,
            "max_prob": round(max_prob, 3),
            "capturable_dd": round(capturable_dd, 4) if not np.isnan(capturable_dd) else None,
            "regime": regime,
        })

    # ── Summary by result category ──────────────────────────────────────────
    result_counts = {}
    for r in records:
        result_counts[r["result"]] = result_counts.get(r["result"], 0) + 1

    total = len(records)
    useful_hits = sum(v for k, v in result_counts.items() if k not in ("Missed", "Too-late"))

    print(f"\n{'='*90}")
    print("  RESULT SUMMARY")
    print(f"{'='*90}")
    for cat in ["Early", "On-time", "Late(ok)", "Too-late", "Missed"]:
        n = result_counts.get(cat, 0)
        print(f"  {cat:<12}: {n:>3} / {total}  ({n/total:.1%})")
    print(f"  {'Useful hits':<12}: {useful_hits:>3} / {total}  ({useful_hits/total:.1%})  [Early + On-time + Late(ok)]")

    # ── Summary by severity ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  PERFORMANCE BY EVENT SEVERITY  (actual max drawdown within event)")
    print(f"{'='*90}")
    print(f"  {'Severity':<10} {'Threshold':<14} {'Events':>7} {'Useful':>7} {'HitRate':>9} {'AvgLead':>9} {'AvgCapDD':>10}")
    print("  " + "-" * 65)
    sev_def = [("Major", ">5%"), ("Moderate", "3-5%"), ("Minor", "<3%")]
    for sev, sev_label in sev_def:
        evs = [r for r in records if r["severity"] == sev]
        if not evs:
            continue
        useful = [e for e in evs if e["result"] in ("Early", "On-time", "Late(ok)")]
        leads = [e["lead_days"] for e in useful if e["lead_days"] is not None]
        caps  = [e["capturable_dd"] for e in useful if e["capturable_dd"] is not None]
        avg_lead = f"{np.mean(leads):+.1f}d" if leads else "  -"
        avg_cap  = f"{np.mean(caps):+.1%}" if caps else "  -"
        print(f"  {sev:<10} {sev_label:<14} {len(evs):>7} {len(useful):>7} "
              f"{len(useful)/len(evs):>9.1%} {avg_lead:>9} {avg_cap:>10}")

    # ── Summary by regime ───────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  PERFORMANCE BY MARKET REGIME")
    print(f"{'='*90}")
    print(f"  {'Regime':<10} {'Events':>7} {'Useful':>7} {'HitRate':>9} {'AvgLead':>9} {'AvgCapDD':>10}")
    print("  " + "-" * 58)
    regime_groups: dict[str, list] = {}
    for r in records:
        regime_groups.setdefault(r["regime"], []).append(r)
    for reg in ["bear", "volatile", "bull", "?"]:
        evs = regime_groups.get(reg, [])
        if not evs:
            continue
        useful = [e for e in evs if e["result"] in ("Early", "On-time", "Late(ok)")]
        leads = [e["lead_days"] for e in useful if e["lead_days"] is not None]
        caps  = [e["capturable_dd"] for e in useful if e["capturable_dd"] is not None]
        avg_lead = f"{np.mean(leads):+.1f}d" if leads else "  -"
        avg_cap  = f"{np.mean(caps):+.1%}" if caps else "  -"
        print(f"  {reg:<10} {len(evs):>7} {len(useful):>7} {len(useful)/len(evs):>9.1%} {avg_lead:>9} {avg_cap:>10}")

    # ── Per-period PR-AUC ───────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  PR-AUC BY MARKET PHASE")
    print(f"{'='*90}")
    print(f"  {'Period':<18} {'Samples':>8} {'PosRate':>8} {'PR-AUC':>8}  Note")
    print("  " + "-" * 60)

    # Build binary label series aligned with prob_series
    label_binary = (merged["label"] == 1).astype(int)

    for period_name, p_start, p_end in PERIODS:
        mask = (merged.index >= pd.Timestamp(p_start)) & (merged.index <= pd.Timestamp(p_end))
        if mask.sum() < 10:
            continue
        y_true = label_binary[mask].values
        y_prob = merged.loc[mask, "prob"].values
        pos_rate = y_true.mean()
        if pos_rate == 0 or pos_rate == 1:
            note = "skip (no variance)"
            print(f"  {period_name:<18} {mask.sum():>8} {pos_rate:>8.1%} {'  -':>8}  {note}")
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        # Baseline PR-AUC = positive rate (random classifier)
        lift = pr_auc / pos_rate
        note = f"lift={lift:.2f}x over random"
        print(f"  {period_name:<18} {mask.sum():>8} {pos_rate:>8.1%} {pr_auc:>8.3f}  {note}")

    # ── Feature comparison: caught vs missed ────────────────────────────────
    print(f"\n{'='*90}")
    print("  FEATURE COMPARISON: Caught (first alert day) vs Missed (t-1)")
    print(f"{'='*90}")
    if caught_features and missed_features:
        caught_df = pd.DataFrame(caught_features)
        missed_df = pd.DataFrame(missed_features)
        print(f"\n  {'Feature':<20} {'Caught mean':>12} {'Missed mean':>12} {'Diff':>10}")
        print("  " + "-" * 58)
        for col in diag_cols + ["prob"]:
            if col in caught_df.columns and col in missed_df.columns:
                c_mean = caught_df[col].mean()
                m_mean = missed_df[col].mean()
                diff = c_mean - m_mean
                marker = " <--" if abs(diff) > 0.05 * abs(m_mean + 1e-9) else ""
                print(f"  {col:<20} {c_mean:>12.4f} {m_mean:>12.4f} {diff:>+10.4f}{marker}")

    # ── Lead-time analysis: how does the model probability build up before events? ──
    # Shows average probability at t+7, t+5, t+3, t+1, t=0, t-1, t-3 relative to event start.
    # A rising probability approaching t=0 confirms the model gives early warning.
    print(f"\n{'='*90}")
    print("  LEAD-TIME ANALYSIS  (avg model probability N days before each event start)")
    print(f"{'='*90}")
    for offset in [7, 5, 3, 1, 0, -1, -3]:
        probs = []
        for ev in events:
            target_date = ev["start"] - pd.Timedelta(days=offset)
            candidates = merged.index[merged.index <= target_date]
            if len(candidates) == 0:
                continue
            closest = candidates[-1]
            if (target_date - closest).days <= 3:
                probs.append(float(merged.loc[closest, "prob"]))
        if probs:
            label_str = f"t=0 (event start)" if offset == 0 else f"t{offset:+d}"
            print(f"  {label_str:<22}: avg={np.mean(probs):.3f}  n={len(probs)}")

    # ── Save CSV ────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\n  Saved event records to: {out_path}")


if __name__ == "__main__":
    main()
