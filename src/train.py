from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from dataset import SplitConfig, load_datasets
from model import ModelConfig, TimeSeriesTransformerRegressor


# ---------------------------
# Repro
# ---------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Prob collection
# ---------------------------
@torch.no_grad()
def collect_probs(model, loader, device): # model output porb after sigmoid
    """
    Return:
      prob_all: (N,) numpy float in [0,1]
      y_all:    (N,) numpy float in {0,1}
    """
    model.eval()
    probs = []
    ys = []
    for X, y in loader:
        X = X.to(device)                           # (B,T,D)
        y = y.to(device).float().view(-1, 1)       # (B,1)

        logits = model(X)                          # (B,1)
        prob = torch.sigmoid(logits)               # (B,1)

        probs.append(prob.detach().cpu())#sum batch
        ys.append(y.detach().cpu())

    prob_all = torch.cat(probs, dim=0).numpy().reshape(-1) # reduce dimention
    y_all = torch.cat(ys, dim=0).numpy().reshape(-1)
    return prob_all, y_all


# ---------------------------
# Metrics
# ---------------------------
def metrics_from_probs(y, prob, thr):
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

# sort ability ≈ pos ratio
def pr_auc(y, prob) -> float:
    # Average precision == area under Precision-Recall curve
    return float(average_precision_score(y, prob))


# ---------------------------
# Threshold selection by alert rate
# ---------------------------
def threshold_by_alert_rate(prob: np.ndarray, alert_rate: float) -> float:
    """
    Choose threshold so that predicted positive rate ~= alert_rate.
    We do this by picking the (1 - alert_rate) quantile of prob.
    """
    alert_rate = float(np.clip(alert_rate, 1e-6, 1.0 - 1e-6))
    thr = float(np.quantile(prob, 1.0 - alert_rate))
    return thr


# ---------------------------
# Train loop
# ---------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_n = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float().view(-1, 1)   # (B,1)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X)                      # (B,1)
        loss = loss_fn(logits, y)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    # data split
    parser.add_argument("--dataset", type=str, default=str(Path("data") / "sp500_dataset.csv"))
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--train_end", type=str, default="2017-12-31")
    parser.add_argument("--val_end", type=str, default="2021-12-31")

    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # model
    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])

    # control alert frequency
    parser.add_argument("--alert_rate", type=float, default=0.12,
                        help="Target predicted positive rate on VAL, e.g. 0.10~0.20")

    # save
    parser.add_argument("--save_dir", type=str, default=str(Path("checkpoints")))

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load datasets
    split_cfg = SplitConfig(seq_len=args.seq_len, train_end=args.train_end, val_end=args.val_end)
    train_ds, val_ds, test_ds, feature_cols = load_datasets(dataset_path, split_cfg)

    D = len(feature_cols)
    print(f"Feature dim D = {D}")
    print(f"Train/Val/Test sizes = {len(train_ds)} {len(val_ds)} {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Build model
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
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    # pos_weight for imbalance
    y_train = np.array([train_ds[i][1] for i in range(len(train_ds))], dtype=np.float32).reshape(-1)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("[DEBUG] pos_weight =", float(pos_weight.item()))
    print("[DEBUG] alert_rate =", float(args.alert_rate))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # checkpoint
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
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip=args.grad_clip)

        # VAL: collect probs
        val_prob, val_y = collect_probs(model, val_loader, device)

        # threshold chosen ONLY by alert_rate (not by F1)
        thr = threshold_by_alert_rate(val_prob, args.alert_rate)

        # metrics at that threshold
        val_metrics = metrics_from_probs(val_y, val_prob, thr)
        val_pr = pr_auc(val_y, val_prob)
        baseline = float(val_y.mean())  # random PR-AUC baseline is roughly the positive rate

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | thr*={thr:.3f} | "
            f"val_pr_auc={val_pr:.6f} (baseline={baseline:.3f}) | "
            f"val_acc={val_metrics['acc']:.6f} "
            f"val_prec={val_metrics['precision']:.6f} "
            f"val_rec={val_metrics['recall']:.6f} "
            f"val_f1={val_metrics['f1']:.6f} "
            f"pred_pos={val_metrics['pred_pos_rate']:.3f} "
            f"true_pos={val_metrics['true_pos_rate']:.3f}"
        )

        # Early stopping on PR-AUC (threshold-free)
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
            print(f"  -> Saved best model (val_pr_auc={best_val_pr_auc:.6f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered (patience={args.patience}).")
                break

    # TEST evaluation
    print("\nLoading best checkpoint and evaluating on test...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # re-compute threshold on VAL using alert_rate, then apply to TEST
    val_prob, val_y = collect_probs(model, val_loader, device)
    test_thr = threshold_by_alert_rate(val_prob, args.alert_rate)

    test_prob, test_y = collect_probs(model, test_loader, device)
    test_metrics = metrics_from_probs(test_y, test_prob, test_thr)
    test_pr = pr_auc(test_y, test_prob)

    print(f"[DEBUG] Using test threshold from VAL alert_rate={args.alert_rate:.3f} -> thr={test_thr:.3f}")
    print("Test metrics:", test_metrics, "| test_pr_auc=", test_pr)


if __name__ == "__main__":
    main()