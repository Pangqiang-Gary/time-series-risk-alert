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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    mse = 0.0
    mae = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device)  # (B, T, D)
        y = y.to(device)  # (B, 1)

        y_hat = model(X)  # (B, 1)
        err = y_hat - y

        mse += float((err ** 2).sum().item()) #Sum of Squared Errors
        mae += float(err.abs().sum().item())#sum of absolute errors
        n += y.numel()#number of samples

    mse /= max(n, 1)
    mae /= max(n, 1)
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse, "mae": mae}


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
    parser.add_argument("--dataset", type=str, default=str(Path("..") / "data" / "sp500_dataset.csv"))
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--train_end", type=str, default="2017-12-31")
    parser.add_argument("--val_end", type=str, default="2021-12-31")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2) #L2
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])

    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--patience", type=int, default=6)  #early stopping tolerant round
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_dir", type=str, default=str(Path("..") / "checkpoints"))
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
        out_activation="sigmoid",
    )
    model = TimeSeriesTransformerRegressor(model_cfg).to(device)

    # 4) Loss function
    if args.loss == "mse":
        loss_fn = nn.MSELoss()
    else:
        # Huber is often more robust to noisy labels/outliers
        loss_fn = nn.HuberLoss(delta=0.5)

    # 5) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6) Train with early stopping on val RMSE
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best_model.pt"

    best_val_rmse = float("inf")
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

        val_metrics = evaluate(model, val_loader, device)
        val_rmse = val_metrics["rmse"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_rmse={val_rmse:.6f} val_mae={val_metrics['mae']:.6f}"
        )

        # Early stopping
        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
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
            print(f"  -> Saved best model (val_rmse={best_val_rmse:.6f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered (patience={args.patience}).")
                break

    # 7) Load best model and evaluate on test
    print("\nLoading best checkpoint and evaluating on test...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, device)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
