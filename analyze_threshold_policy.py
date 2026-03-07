from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate threshold policy metrics")
    parser.add_argument("--summaries", nargs="+", required=True, help="List of summary json files")
    parser.add_argument("--output", type=str, default="artifacts/threshold_policy_metrics.csv")
    args = parser.parse_args()

    rows = []
    for item in args.summaries:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(path)
        summary = load_summary(path)
        policy = summary.get("threshold_policy", "unknown")
        val_metrics = summary.get("val_alert_metrics", {})
        test_metrics = summary.get("test_alert_metrics", {})
        val_year = summary.get("val_year_pred_pos_rate", {})
        test_year = summary.get("test_year_pred_pos_rate", {})
        row = {
            "summary_path": str(path),
            "policy_name": policy,
            "thr_value": summary.get("thr_primary", summary.get("alert_threshold")),
            "derived_alert_rate": summary.get("derived_alert_rate"),
            "alert_rate_used": summary.get("alert_rate_used"),
            "best_thr_constrained": summary.get("best_thr_constrained"),
            "val_f1": val_metrics.get("f1"),
            "val_pr_auc": val_metrics.get("pr_auc"),
            "test_f1": test_metrics.get("f1"),
            "test_pr_auc": test_metrics.get("pr_auc"),
            "val_pred_pos_rate": summary.get("val_pred_pos_rate"),
            "test_pred_pos_rate": summary.get("test_pred_pos_rate"),
            "val_year_min": min(val_year.values()) if val_year else None,
            "val_year_max": max(val_year.values()) if val_year else None,
            "test_year_min": min(test_year.values()) if test_year else None,
            "test_year_max": max(test_year.values()) if test_year else None,
            "val_test_alert_gap": summary.get("val_test_alert_gap"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(df)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
