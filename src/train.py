from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score


from dataset import SplitConfig, load_datasets
from model import ModelConfig, TimeSeriesTransformerRegressor

ARTIFACTS_DIR = Path("artifacts")
EVAL_DIR = ARTIFACTS_DIR / "evals"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    # Fix random seeds for reproducible results
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_probs(model, loader, device, ticker_seq: np.ndarray):
    """Run the model on a full dataloader and collect probabilities, logits, labels, ticker IDs."""
    model.eval()
    probs: List[torch.Tensor] = []
    logits_all: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ticker_chunks: List[np.ndarray] = []
    offset = 0
    ticker_seq = np.asarray(ticker_seq)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float().view(-1, 1)
        logits = model(X)
        prob = torch.sigmoid(logits)
        batch_size = X.size(0)

        probs.append(prob.detach().cpu())
        logits_all.append(logits.detach().cpu())
        ys.append(y.detach().cpu())
        ticker_chunks.append(ticker_seq[offset: offset + batch_size])
        offset += batch_size

    prob_all = torch.cat(probs, dim=0).numpy().reshape(-1)
    logit_all = torch.cat(logits_all, dim=0).numpy().reshape(-1)
    y_all = torch.cat(ys, dim=0).numpy().reshape(-1)
    ticker_all = np.concatenate(ticker_chunks) if ticker_chunks else np.array([], dtype=int)
    return prob_all, logit_all, y_all, ticker_all


def metrics_from_probs(y: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    """Compute precision, recall, F1, accuracy from predicted probabilities and a threshold."""
    if len(y) == 0:
        return {
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "pred_pos_rate": float("nan"),
            "true_pos_rate": float("nan"),
            "cm": [[0, 0], [0, 0]],
        }

    pred = (prob >= thr).astype(np.float32)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))

    acc = (tp + tn) / max(len(y), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    pred_pos_rate = float(pred.mean())
    true_pos_rate = float(y.mean())

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_pos_rate": float(pred_pos_rate),
        "true_pos_rate": float(true_pos_rate),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def safe_average_precision(y: np.ndarray, prob: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    if np.all(y == y[0]):
        return float("nan")
    return float(average_precision_score(y, prob))


def evaluate_subset(prob: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, float]:
    metrics = metrics_from_probs(y, prob, thr)
    metrics["pr_auc"] = safe_average_precision(y, prob)
    metrics["positive_rate"] = metrics["true_pos_rate"]
    return metrics


def threshold_sweep(prob: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> Dict[str, object]:
    best_thr = thresholds[0]
    best_f1 = -float("inf")
    sweep_rows = []
    for thr in thresholds:
        metrics = metrics_from_probs(y, prob, thr)
        sweep_rows.append({"thr": float(thr), **metrics})
        if metrics["f1"] > best_f1 + 1e-6:
            best_f1 = metrics["f1"]
            best_thr = thr
    return {"best_threshold": float(best_thr), "best_f1": float(best_f1), "rows": sweep_rows}


def select_constrained_best_f1(
    sweep_rows: List[Dict[str, float]],
    min_rate: float,
    max_rate: float,
) -> Tuple[float, Dict[str, float], bool]:
    best_row: Optional[Dict[str, float]] = None
    for row in sweep_rows:
        rate = row.get("pred_pos_rate", 0.0)
        if rate < min_rate or rate > max_rate:
            continue
        if best_row is None or row["f1"] > best_row["f1"] + 1e-6:
            best_row = row
    constrained = True
    if best_row is None:
        # No row meets the alert rate bounds; pick the closest one
        constrained = False
        best_row = min(
            sweep_rows,
            key=lambda r: min(abs(r.get("pred_pos_rate", 0.0) - min_rate), abs(r.get("pred_pos_rate", 0.0) - max_rate)),
        )
    return float(best_row["thr"]), best_row, constrained


def threshold_by_alert_rate(prob: np.ndarray, alert_rate: float) -> float:
    """Set threshold so that exactly `alert_rate` fraction of samples get flagged as alerts."""
    alert_rate = float(np.clip(alert_rate, 1e-6, 1.0 - 1e-6))
    return float(np.quantile(prob, 1.0 - alert_rate))


def select_precision_target(
    sweep_rows: List[Dict[str, float]],
    min_precision: float,
) -> Tuple[float, Dict[str, float]]:
    """Pick the lowest threshold where precision >= min_precision (maximizes recall)."""
    candidates = [r for r in sweep_rows if r.get("precision", 0.0) >= min_precision and r.get("recall", 0.0) > 0]
    if not candidates:
        # No row meets min_precision; fall back to highest precision row
        best_row = max(sweep_rows, key=lambda r: r.get("precision", 0.0))
    else:
        best_row = max(candidates, key=lambda r: r.get("recall", 0.0))
    return float(best_row["thr"]), best_row


def class_stats(prob: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats = {}
    for label in (0, 1):
        mask = y == label
        if mask.any():
            stats[str(label)] = {
                "mean": float(prob[mask].mean()),
                "median": float(np.median(prob[mask])),
                "count": int(mask.sum()),
            }
        else:
            stats[str(label)] = {"mean": float("nan"), "median": float("nan"), "count": 0}
    return stats


def yearly_pred_pos_rate(prob: np.ndarray, thr: float, dates: Optional[np.ndarray]) -> Dict[str, float]:
    if dates is None or len(dates) == 0:
        return {}
    df = pd.DataFrame({"prob": prob, "date": pd.to_datetime(dates)})
    df["pred"] = (df["prob"] >= thr).astype(float)
    df["year"] = df["date"].dt.year
    grouped = df.groupby("year")["pred"].mean().to_dict()
    return {str(int(k)): float(v) for k, v in grouped.items()}


def yearly_metrics(
    prob: np.ndarray,
    y: np.ndarray,
    thr: float,
    dates: Optional[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute precision/recall/PR-AUC for each calendar year. Returns {year: metrics}."""
    if dates is None or len(dates) == 0:
        return {}
    df = pd.DataFrame({"prob": prob, "y": y, "date": pd.to_datetime(dates)})
    df["year"] = df["date"].dt.year
    result: Dict[str, Dict[str, float]] = {}
    for yr, grp in df.groupby("year"):
        m = evaluate_subset(grp["prob"].to_numpy(), grp["y"].to_numpy(), thr)
        result[str(int(yr))] = m
    return result


# SP500 market regime per year.
# bear: annual return < -10%. volatile: large swings (e.g. 2020: +16% but -34% crash). bull: > +10%.
YEAR_REGIME: Dict[int, str] = {
    2000: "bear", 2001: "bear", 2002: "bear",
    2003: "bull", 2004: "bull", 2005: "bull", 2006: "bull", 2007: "bull",
    2008: "bear", 2009: "bull",
    2010: "bull", 2011: "volatile", 2012: "bull", 2013: "bull",
    2014: "bull", 2015: "volatile", 2016: "bull", 2017: "bull",
    2018: "bear", 2019: "bull",
    2020: "volatile", 2021: "bull", 2022: "bear",
    2023: "bull", 2024: "bull", 2025: "bull",
}


def print_yearly_regime_table(
    prob: np.ndarray,
    y: np.ndarray,
    thr: float,
    dates: Optional[np.ndarray],
    label: str = "TEST",
) -> Dict[str, Dict[str, float]]:
    """Print a table of per-year metrics with the market regime label."""
    ym = yearly_metrics(prob, y, thr, dates)
    if not ym:
        return ym
    print(f"\n[{label}] Per-year metrics (thr={thr:.3f})")
    print(f"  {'Year':<6} {'Regime':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'PR-AUC':>7} {'AlertR':>7} {'N':>5}")
    for yr_str in sorted(ym.keys()):
        m = ym[yr_str]
        yr_int = int(yr_str)
        regime = YEAR_REGIME.get(yr_int, "?")
        prec   = m.get("precision", float("nan"))
        rec    = m.get("recall",    float("nan"))
        f1     = m.get("f1",        float("nan"))
        prauc  = m.get("pr_auc",    float("nan"))
        alert  = m.get("pred_pos_rate", float("nan"))
        n_pos  = int(round(m.get("positive_rate", 0) * (m["cm"][0][0] + m["cm"][0][1] + m["cm"][1][0] + m["cm"][1][1]))) if "cm" in m else 0
        total  = m["cm"][0][0] + m["cm"][0][1] + m["cm"][1][0] + m["cm"][1][1] if "cm" in m else 0
        print(
            f"  {yr_str:<6} {regime:<10} {prec:6.3f} {rec:6.3f} {f1:6.3f} "
            f"{prauc:7.3f} {alert:7.3f} {total:5d}"
        )
    return ym



def probability_hist(prob: np.ndarray, bins: np.ndarray) -> List[int]:
    counts, _ = np.histogram(prob, bins=bins)
    return counts.astype(int).tolist()


def reliability_table(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(prob)
    table = []
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (prob >= lower) & (prob <= upper)
        else:
            mask = (prob >= lower) & (prob < upper)
        count = int(mask.sum())
        if count > 0:
            conf = float(prob[mask].mean())
            acc = float(y[mask].mean())
        else:
            conf = float("nan")
            acc = float("nan")
        weight = count / max(total, 1)
        if count > 0:
            ece += abs(acc - conf) * weight
        table.append({
            "bin": i,
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "confidence": conf,
            "accuracy": acc,
        })
    return {"bins": bins.tolist(), "table": table, "ece": float(ece)}


def format_metrics(label: str, metrics: Dict[str, float]) -> str:
    return (
        f"{label}: prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} "
        f"f1={metrics['f1']:.3f} pr_auc={metrics['pr_auc']:.3f} pos={metrics['positive_rate']:.3f}"
    )


def dataset_counts(dataset) -> Dict[str, int]:
    counts = dataset.ticker_counts()
    return {f"ticker_{tid}": count for tid, count in sorted(counts.items())}


def save_eval_artifacts(run_name: str, summary: Dict[str, object], arrays: Dict[str, np.ndarray]) -> None:
    ensure_dir(EVAL_DIR)
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = dict(summary)
    summary["run_name"] = run_name
    summary["timestamp_utc"] = timestamp
    json_path = EVAL_DIR / f"{run_name}_summary.json"
    npz_path = EVAL_DIR / f"{run_name}_probabilities.npz"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez(npz_path, **arrays)


def find_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Find temperature T that minimizes NLL on the validation set.
    Calibrated probability = sigmoid(logit / T).
    T > 1 shifts probabilities toward 0.5 (reduces overconfidence).
    """
    logits_t = torch.tensor(logits, dtype=torch.float32).unsqueeze(1)
    labels_t = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    T = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logits_t / T.clamp(min=0.1), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.item())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return torch.sigmoid(torch.tensor(logits, dtype=torch.float32) / T).numpy()


def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """
    Focal Loss: like BCE but down-weights easy examples.
    gamma=2 means the model focuses more on hard-to-classify samples.
    alpha balances the positive class weight (~25% positive in our data).
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = targets * probs + (1 - targets) * (1 - probs)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    loss = alpha_t * ((1 - p_t) ** gamma) * bce
    return loss


def train_one_epoch(model, loader, optimizer, loss_fn, device, loss_type: str, focal_alpha: float, focal_gamma: float) -> float:
    """Run one epoch of training: forward pass, compute loss, backpropagate, update weights."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        if loss_type == "focal":
            loss = focal_loss_with_logits(logits, y, focal_alpha, focal_gamma).mean()
        else:
            loss = loss_fn(logits, y).mean()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_training(args) -> Dict[str, float]:
    """
    Full training pipeline:
      1. Load datasets (SP500 + QQQ + DJI)
      2. Build Transformer model
      3. Train with Focal Loss + AdamW, save best checkpoint by val PR-AUC
      4. Load best checkpoint, apply temperature scaling to calibrate probabilities
      5. Choose alert threshold using the selected threshold policy
      6. Print and save final metrics
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_paths = [Path(p.strip()) for p in args.datasets.split(",") if p.strip()]
    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    split_cfg = SplitConfig(seq_len=args.seq_len, train_end=args.train_end, val_end=args.val_end)
    primary_id = args.primary_ticker_id
    train_ds, val_ds, test_ds, feature_cols = load_datasets(
        dataset_paths,
        split_cfg,
        label_mode=args.label_mode,
        feature_stage=args.feature_stage,
    )
    current_ts = datetime.now(timezone.utc)
    run_name = args.run_name or f"run_{current_ts.strftime('%Y%m%d_%H%M%S')}"

    train_counts = dataset_counts(train_ds)
    val_counts = dataset_counts(val_ds)
    test_counts = dataset_counts(test_ds)

    D = len(feature_cols)
    print(f"Feature dim D = {D}")
    print(f"Train/Val/Test window counts = {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    print("Train counts:", train_counts)
    print("Val counts:", val_counts)
    print("Test counts:", test_counts)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    y_train = train_ds.window_labels
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    train_pos_ratio = float(pos / max(pos + neg, 1.0))
    print(f"[DEBUG] train_pos_ratio = {train_pos_ratio:.4f}")

    model_cfg = ModelConfig(
        input_dim=D,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        pooling=args.pooling,
        out_activation="none",
    )
    model = TimeSeriesTransformerRegressor(model_cfg).to(device)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {count_parameters(model):,}")

    neg = max(len(y_train) - pos, 0.0)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    focal_alpha = float(np.clip(1.0 - train_pos_ratio, 0.05, 0.95)) if args.focal_alpha <= 0 else float(args.focal_alpha)
    print("[DEBUG] pos_weight =", float(pos_weight.item()))
    print("[DEBUG] focal_alpha =", focal_alpha)
    print("[DEBUG] alert_rate =", float(args.alert_rate))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best_model.pt"

    best_val_pr_auc = -float("inf")
    bad_epochs = 0

    print("\nTraining start...")
    print("ModelConfig:", asdict(model_cfg))
    print("SplitConfig:", asdict(split_cfg))
    print("Checkpoint:", ckpt_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.loss_type,
            focal_alpha,
            args.focal_gamma,
        )

        val_prob, val_logit, val_y, val_ticker = collect_probs(model, val_loader, device, val_ds.window_tickers)
        mask_primary = val_ticker == primary_id
        if not mask_primary.any():
            raise ValueError("No validation samples for primary ticker; adjust date splits.")

        thr_primary = threshold_by_alert_rate(val_prob[mask_primary], args.alert_rate)
        thr_pooled = threshold_by_alert_rate(val_prob, args.alert_rate)
        val_primary_metrics = evaluate_subset(val_prob[mask_primary], val_y[mask_primary], thr_primary)
        val_pooled_metrics = evaluate_subset(val_prob, val_y, thr_primary)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | thr_primary={thr_primary:.3f} "
            f"(thr_all={thr_pooled:.3f})"
        )
        print("    ", format_metrics("VAL[SP500]", val_primary_metrics))
        print("    ", format_metrics("VAL[pooled]", val_pooled_metrics))

        val_pr = val_primary_metrics["pr_auc"]
        if val_pr > best_val_pr_auc + 1e-6:
            best_val_pr_auc = val_pr
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "split_cfg": asdict(split_cfg),
                    "feature_cols": feature_cols,
                },
                ckpt_path,
            )
            print(f"      -> Saved best model (val_pr_auc={best_val_pr_auc:.6f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered (patience={args.patience}).")
                break

    print("\nLoading best checkpoint and evaluating on validation/test...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_prob, val_logit, val_y, val_ticker = collect_probs(model, val_loader, device, val_ds.window_tickers)
    mask_primary = val_ticker == primary_id
    if not mask_primary.any():
        raise ValueError("No validation samples for primary ticker; adjust date splits.")
    val_prob_primary = val_prob[mask_primary]
    val_logit_primary = val_logit[mask_primary]
    val_y_primary = val_y[mask_primary]
    val_dates = getattr(val_ds, "window_dates", None)
    val_dates_primary = val_dates[mask_primary] if val_dates is not None else None

    test_prob, test_logit, test_y, test_ticker = collect_probs(model, test_loader, device, test_ds.window_tickers)
    mask_primary_test = test_ticker == primary_id
    test_prob_primary = test_prob[mask_primary_test]
    test_logit_primary = test_logit[mask_primary_test]
    test_y_primary = test_y[mask_primary_test]
    test_dates = getattr(test_ds, "window_dates", None)
    test_dates_primary = test_dates[mask_primary_test] if test_dates is not None else None

    val_pos_ratio = float(val_y_primary.mean())
    test_pos_ratio = float(test_y_primary.mean())

    # Temperature scaling: calibrate probabilities on the validation set.
    # This finds a scalar T so that sigmoid(logit/T) is better calibrated.
    # T > 1 softens overconfident predictions toward 0.5.
    ece_before = reliability_table(val_prob_primary, val_y_primary)["ece"]
    temperature = find_temperature(val_logit_primary, val_y_primary)
    val_prob_primary = apply_temperature(val_logit_primary, temperature)
    test_prob_primary = apply_temperature(test_logit_primary, temperature)
    val_prob = apply_temperature(val_logit, temperature)
    test_prob = apply_temperature(test_logit, temperature)
    ece_after = reliability_table(val_prob_primary, val_y_primary)["ece"]
    print(f"[Temperature Scaling] T={temperature:.4f} | VAL ECE: {ece_before:.3f} -> {ece_after:.3f}")

    # Save temperature into checkpoint so inference uses the same calibration
    ckpt_data = torch.load(ckpt_path, map_location=device)
    ckpt_data["temperature"] = temperature
    torch.save(ckpt_data, ckpt_path)

    thresholds = np.linspace(0.01, 0.99, 99)
    sweep = threshold_sweep(val_prob_primary, val_y_primary, thresholds)
    best_thr = sweep["best_threshold"]
    best_thr_constrained, constrained_row, constraint_active = select_constrained_best_f1(
        sweep["rows"], args.min_alert_rate, args.max_alert_rate
    )

    # Choose the alert threshold using the selected policy:
    #   "precision_target" : lowest threshold where val precision >= min_precision (maximizes recall)
    #   "constrained_f1"   : best F1 within allowed alert rate range
    #   "derived_alert"    : alert_rate = val_positive_rate * multiplier
    #   "fixed_alert"      : use the exact --alert_rate value
    derived_alert_rate: Optional[float] = None
    alert_rate_used: Optional[float] = None
    if args.threshold_policy == "derived_alert":
        base_rate = float(val_pos_ratio * args.derived_alert_multiplier)
        alert_rate_used = float(np.clip(base_rate, args.min_alert_rate, args.max_alert_rate))
        derived_alert_rate = alert_rate_used
        thr_primary = threshold_by_alert_rate(val_prob_primary, alert_rate_used)
    elif args.threshold_policy == "constrained_f1":
        thr_primary = best_thr_constrained
        alert_rate_used = float(constrained_row.get("pred_pos_rate", float("nan")))
    elif args.threshold_policy == "precision_target":
        thr_primary, pt_row = select_precision_target(sweep["rows"], args.min_precision)
        alert_rate_used = float(pt_row.get("pred_pos_rate", float("nan")))
    else:
        base_rate = float(args.alert_rate)
        alert_rate_used = float(np.clip(base_rate, args.min_alert_rate, args.max_alert_rate))
        thr_primary = threshold_by_alert_rate(val_prob_primary, alert_rate_used)

    val_primary_metrics = evaluate_subset(val_prob_primary, val_y_primary, thr_primary)
    val_pooled_metrics = evaluate_subset(val_prob, val_y, thr_primary)
    test_primary_metrics = evaluate_subset(test_prob_primary, test_y_primary, thr_primary)
    test_pooled_metrics = evaluate_subset(test_prob, test_y, thr_primary)

    val_pred_pos_rate = val_primary_metrics["pred_pos_rate"]
    test_pred_pos_rate = test_primary_metrics["pred_pos_rate"]
    val_year_pred_pos_rate = yearly_pred_pos_rate(val_prob_primary, thr_primary, val_dates_primary)
    test_year_pred_pos_rate = yearly_pred_pos_rate(test_prob_primary, thr_primary, test_dates_primary)
    val_test_alert_gap = abs(val_pred_pos_rate - test_pred_pos_rate)

    print(
        f"[DEBUG] Threshold policy={args.threshold_policy} -> thr={thr_primary:.3f}, "
        f"alert_rate_used={alert_rate_used if alert_rate_used is not None else float('nan'):.3f}"
    )
    print("VAL primary:", format_metrics("SP500", val_primary_metrics))
    print("VAL pooled:", format_metrics("All", val_pooled_metrics))
    print("TEST primary:", format_metrics("SP500", test_primary_metrics))
    print("TEST pooled:", format_metrics("All", test_pooled_metrics))

    # Per-year breakdown by market regime
    print_yearly_regime_table(val_prob_primary,  val_y_primary,  thr_primary, val_dates_primary,  label="VAL")
    print_yearly_regime_table(test_prob_primary, test_y_primary, thr_primary, test_dates_primary, label="TEST")

    val_best_metrics = evaluate_subset(val_prob_primary, val_y_primary, best_thr)
    test_best_metrics = evaluate_subset(test_prob_primary, test_y_primary, best_thr)

    print("\n[Diagnostics] Threshold sweep (VAL best F1)")
    print(f"    best_thr={best_thr:.3f} -> {format_metrics('VAL_best', val_best_metrics)}")
    print(f"    TEST@best_thr -> {format_metrics('TEST_best', test_best_metrics)}")

    hist_bins = np.linspace(0.0, 1.0, 11)
    val_hist = probability_hist(val_prob_primary, hist_bins)
    test_hist = probability_hist(test_prob_primary, hist_bins)
    val_class = class_stats(val_prob_primary, val_y_primary)
    test_class = class_stats(test_prob_primary, test_y_primary)
    val_rel = reliability_table(val_prob_primary, val_y_primary, n_bins=10)
    test_rel = reliability_table(test_prob_primary, test_y_primary, n_bins=10)

    print("[Diagnostics] Probability histogram (VAL)")
    print(f"    bins={hist_bins.tolist()} counts={val_hist}")
    print("[Diagnostics] Probability histogram (TEST)")
    print(f"    bins={hist_bins.tolist()} counts={test_hist}")
    print("[Diagnostics] Class stats (VAL)", val_class)
    print("[Diagnostics] Class stats (TEST)", test_class)
    val_gap  = val_class.get("1", {}).get("mean", float("nan"))  - val_class.get("0", {}).get("mean", float("nan"))
    test_gap = test_class.get("1", {}).get("mean", float("nan")) - test_class.get("0", {}).get("mean", float("nan"))
    print(f"[Diagnostics] Class mean gap => VAL: {val_gap:.4f}  TEST: {test_gap:.4f}  (target >0.15)")
    print(f"[Diagnostics] Reliability (VAL) ECE={val_rel['ece']:.3f}")
    for row in val_rel["table"]:
        print("    ", row)
    print(f"[Diagnostics] Reliability (TEST) ECE={test_rel['ece']:.3f}")
    for row in test_rel["table"]:
        print("    ", row)

    arg_snapshot: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            arg_snapshot[key] = str(value)
        else:
            arg_snapshot[key] = value

    derived_alert_rate_value = None if derived_alert_rate is None else float(derived_alert_rate)
    alert_rate_used_value = None
    if alert_rate_used is not None and not np.isnan(alert_rate_used):
        alert_rate_used_value = float(alert_rate_used)

    summary = {
        "temperature": float(temperature),
        "ece_before_calibration": float(ece_before),
        "ece_after_calibration": float(ece_after),
        "feature_dim": D,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "train_pos_ratio": train_pos_ratio,
        "val_pos_ratio": val_pos_ratio,
        "test_pos_ratio": test_pos_ratio,
        "threshold_policy": args.threshold_policy,
        "alert_threshold": float(thr_primary),
        "thr_primary": float(thr_primary),
        "derived_alert_rate": derived_alert_rate_value,
        "alert_rate_used": alert_rate_used_value,
        "best_thr_constrained": float(best_thr_constrained),
        "best_thr_constrained_active": bool(constraint_active),
        "val_pred_pos_rate": float(val_pred_pos_rate),
        "test_pred_pos_rate": float(test_pred_pos_rate),
        "val_year_pred_pos_rate": val_year_pred_pos_rate,
        "test_year_pred_pos_rate": test_year_pred_pos_rate,
        "val_test_alert_gap": float(val_test_alert_gap),
        "val_alert_metrics": val_primary_metrics,
        "test_alert_metrics": test_primary_metrics,
        "best_threshold": float(best_thr),
        "val_best_metrics": val_best_metrics,
        "test_best_metrics": test_best_metrics,
        "hist_bins": hist_bins.tolist(),
        "val_hist_counts": val_hist,
        "test_hist_counts": test_hist,
        "val_class_stats": val_class,
        "test_class_stats": test_class,
        "val_reliability": val_rel,
        "test_reliability": test_rel,
        "threshold_sweep": sweep["rows"],
        "args": arg_snapshot,
    }

    arrays = {
        "val_prob": val_prob_primary,
        "val_logit": val_logit_primary,
        "val_label": val_y_primary,
        "test_prob": test_prob_primary,
        "test_logit": test_logit_primary,
        "test_label": test_y_primary,
    }
    save_eval_artifacts(run_name, summary, arrays)

    return {
        "val_primary": val_primary_metrics,
        "test_primary": test_primary_metrics,
        "thr": thr_primary,
        "feature_dim": D,
        "run_name": run_name,
        "best_threshold": best_thr,
        "val_best": val_best_metrics,
        "test_best": test_best_metrics,
        "train_pos_ratio": train_pos_ratio,
        "val_pos_ratio": val_pos_ratio,
        "test_pos_ratio": test_pos_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str,
                        default="data/sp500_dataset.csv,data/qqq_dataset.csv,data/dji_dataset.csv")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--train_end", type=str, default="2017-12-31")
    parser.add_argument("--val_end", type=str, default="2021-12-31")
    parser.add_argument("--primary_ticker_id", type=int, default=0)
    parser.add_argument("--label_mode", type=str, default="baseline",
                        choices=["baseline", "event_day_only", "event_day_plus_prev1"],
                        help="Label shaping strategy")
    parser.add_argument("--feature_stage", type=int, default=None,
                        help="Ablation stage (0-4). None = all features in CSV")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"],
                        help="Loss function: weighted BCE or focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.0,
                        help="Set >0 to override automatic focal alpha (pos-class weight)")

    parser.add_argument("--d_model", type=int, default=28)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])

    parser.add_argument("--threshold_policy", type=str, default="precision_target",
                        choices=["fixed_alert", "derived_alert", "constrained_f1", "precision_target"],
                        help="Threshold selection strategy")
    parser.add_argument("--min_precision", type=float, default=0.45,
                        help="Minimum precision required (used by precision_target policy)")
    parser.add_argument("--min_alert_rate", type=float, default=0.05,
                        help="Lower bound for acceptable alert rate")
    parser.add_argument("--max_alert_rate", type=float, default=0.20,
                        help="Upper bound for acceptable alert rate")
    parser.add_argument("--derived_alert_multiplier", type=float, default=1.0,
                        help="Scale factor applied to VAL positive ratio when deriving alert rate")

    parser.add_argument("--alert_rate", type=float, default=0.12,
                        help="Target predicted positive rate on VAL (primary ticker)")
    parser.add_argument("--save_dir", type=str, default=str(Path("checkpoints")))
    parser.add_argument("--run_name", type=str, default="", help="Optional identifier for eval artifacts")

    args = parser.parse_args()

    results = run_training(args)
    print("Final SP500 metrics => VAL:", format_metrics("SP500", results["val_primary"]))
    print("Final SP500 metrics => TEST:", format_metrics("SP500", results["test_primary"]))


if __name__ == "__main__":
    main()
