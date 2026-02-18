from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import your modules
from dataset import SplitConfig, load_datasets
from model import ModelConfig, TimeSeriesTransformerRegressor


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)#gpu


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5, debug_prob=False):
    pos60 = pos70 = pos80 = 0

    model.eval()
    correct = 0
    total = 0

    tp = fp = fn = 0    #calculate matrix trur pos,fal pos,fal neg
    pred_pos = 0
    true_pos = 0

    prob_min = float("inf")
    prob_max = float("-inf")
    prob_sum = 0.0
    prob_cnt = 0    #Debug

    for X, y in loader:
        X = X.to(device)    #(B,T,D)
        y = y.to(device)    #(B, )

        logits = model(X)
        prob = torch.sigmoid(logits)
        pred = (prob >= thr).float()

        if debug_prob:
            prob_min = min(prob_min, prob.min().item())
            prob_max = max(prob_max, prob.max().item())
            prob_sum += prob.sum().item()
            prob_cnt += prob.numel()

            pos60 += (prob >= 0.60).sum().item() # the number of postive in different thr
            pos70 += (prob >= 0.70).sum().item()
            pos80 += (prob >= 0.80).sum().item()

        correct += (pred == y).sum().item()
        total += y.numel()

        pred_pos += pred.sum().item()
        true_pos += y.sum().item()

        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    pred_pos_rate = pred_pos / max(total, 1)
    true_pos_rate = true_pos / max(total, 1)

    out = {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_pos_rate": float(pred_pos_rate),
        "true_pos_rate": float(true_pos_rate),
    }

    if debug_prob and prob_cnt > 0:
        out["prob_min"] = prob_min
        out["prob_max"] = prob_max
        out["prob_mean"] = prob_sum / prob_cnt
        out["pos_rate@0.60"] = pos60 / prob_cnt
        out["pos_rate@0.70"] = pos70 / prob_cnt
        out["pos_rate@0.80"] = pos80 / prob_cnt
        print("Prob stats:", {"min": out["prob_min"], "max": out["prob_max"], "mean": out["prob_mean"]})
        print("Pos rate @ thr:", {"0.60": out["pos_rate@0.60"], "0.70": out["pos_rate@0.70"], "0.80": out["pos_rate@0.80"]})

    return out





def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for X, y in loader:
        X = X.to(device)  # (B, T, D)
        y = y.to(device)  # (B, 1)

        optimizer.zero_grad(set_to_none=True)

        y_hat = model(X)               # (B, 1)
        loss = loss_fn(y_hat, y)       # scalar

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) ## Gradient clipping for training stability

        optimizer.step() # update model parameters

        bs = y.size(0)      # batch size
        total_loss += float(loss.item()) * bs       # accumulate total loss (sum over samples)
        total_n += bs

    return total_loss / max(total_n, 1)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(Path("data") / "sp500_dataset.csv"))
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--train_end", type=str, default="2017-12-31")
    parser.add_argument("--val_end", type=str, default="2021-12-31")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2) #L2
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])

   #parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--patience", type=int, default=6)  #early stopping tolerant round
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_dir", type=str, default=str(Path("checkpoints")))
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # 1) Load datasets (train/val/test)
    split_cfg = SplitConfig(seq_len=args.seq_len, train_end=args.train_end, val_end=args.val_end)
    train_ds, val_ds, test_ds, feature_cols = load_datasets(dataset_path, split_cfg)

    D = len(feature_cols)
    print(f"Feature dim D = {D}")
    print(f"Train/Val/Test sizes = {len(train_ds)} {len(val_ds)} {len(test_ds)}")

    # 2) DataLoaders (IMPORTANT: shuffle=False for time series)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 3) Build model
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

    # 4) Loss function
    loss_fn = nn.BCEWithLogitsLoss()



    # 5) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #L2

    # 6) Train with early stopping on val f1
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best_model.pt"

    best_val_f1 = -float("inf")
    bad_epochs = 0

    print("\nTraining start...")
    print("ModelConfig:", asdict(model_cfg))
    print("SplitConfig:", asdict(split_cfg))
    print("Checkpoint:", ckpt_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=args.grad_clip,
        )

        # ---- choose best threshold on VAL (maximize F1) ----
        thr = 0.4
        val_metrics = evaluate(model, val_loader, device, thr=thr)
        val_f1 = val_metrics["f1"]



# ----------------------------------------------------


        print(
    f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | thr={thr:.2f} | "
    f"val_acc={val_metrics['acc']:.6f} "
    f"val_prec={val_metrics['precision']:.6f} "
    f"val_rec={val_metrics['recall']:.6f} "
    f"val_f1={val_metrics['f1']:.6f} "
    f"pred_pos={val_metrics['pred_pos_rate']:.3f} "
    f"true_pos={val_metrics['true_pos_rate']:.3f}"
)




        # Early stopping
        # Early stopping (classification: higher f1 is better)
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
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
            print(f"  -> Saved best model (val_f1={best_val_f1:.6f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered (patience={args.patience}).")
                break

    # 7) Load best model and evaluate on test
    print("\nLoading best checkpoint and evaluating on test...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Re-tune threshold on VAL using the best checkpoint
    test_thr = 0.4
    print(f"[DEBUG] Using test threshold = {test_thr:.2f}")
    test_metrics = evaluate(model, test_loader, device, thr=test_thr, debug_prob=True)


    print("Test metrics:", test_metrics)
    



if __name__ == "__main__":
    main()
