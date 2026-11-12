"""
Training script for the 2D Spatio-Temporal Fourier Neural Operator (FNO)
surrogate for 1D Saint-Venant Open Channel Hydraulics.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import get_dataloaders
from fno import FNO2d
from metrics import field_metrics, format_metrics


class RelativeL2Loss(nn.Module):
    """
    Relative L2 Loss function commonly used for training neural operators.
    Computes ||y_pred - y_true||_2 / ||y_true||_2 averaged across the batch.

    This runs on z-scored tensors and is a training objective only; reported
    accuracy figures come from ``metrics.field_metrics`` in physical units.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, nx, nt, C)
        diff = pred - target
        diff_norm = torch.norm(diff.reshape(diff.shape[0], -1), p=2, dim=1)
        target_norm = torch.norm(target.reshape(target.shape[0], -1), p=2, dim=1) + self.eps
        return torch.mean(diff_norm / target_norm)


def safe_save_checkpoint(
    checkpoint: dict,
    target_path: str,
    max_retries: int = 5,
    retry_delay: float = 0.2
):
    """
    Safely saves a PyTorch checkpoint on Windows/Linux by writing to a temporary file
    and atomically replacing the target to avoid ERROR_SHARING_VIOLATION (error 32).
    """
    tmp_path = target_path + ".tmp"
    torch.save(checkpoint, tmp_path)

    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, target_path)
            return
        except (PermissionError, OSError):
            if attempt == max_retries - 1:
                # Fallback: if replace fails, try direct overwrite or leave tmp
                try:
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.rename(tmp_path, target_path)
                except Exception:
                    # As a last resort, torch.save directly
                    torch.save(checkpoint, target_path)
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                return
            time.sleep(retry_delay * (attempt + 1))


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Mean training-objective loss over a loader, in normalised space."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred_norm = model(x_batch)
            # Compute loss in normalized space
            loss = criterion(pred_norm, y_batch)
            total_loss += loss.item() * x_batch.shape[0]
            count += x_batch.shape[0]

    return total_loss / max(count, 1)


def train_fno(
    data_path: str = "data/saint_venant_dataset.npz",
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    modes_x: int = 16,
    modes_t: int = 24,
    width: int = 64,
    num_layers: int = 4,
    padding_frac: float = 0.2,
    save_dir: str = "models",
    save_name: str = "fno_saint_venant.pt"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Training 2D Spatio-Temporal Fourier Neural Operator (FNO)")
    print("=" * 70)
    print(f"Device               : {device}")
    print(f"Dataset Path         : {data_path}")
    print(f"Epochs               : {epochs}")
    print(f"Batch Size           : {batch_size}")
    print(f"Learning Rate        : {lr}")
    print(f"Fourier Modes (x, t) : ({modes_x}, {modes_t})")
    print(f"Hidden Width / Layers: {width} / {num_layers}")
    print(f"Domain Padding Frac  : {padding_frac}")
    print("-" * 70)

    # 1. Load Data
    train_loader, val_loader, test_loader, x_normalizer, y_normalizer, meta = get_dataloaders(
        data_path=data_path,
        batch_size=batch_size
    )

    # 2. Instantiate Model
    model_args = {
        "in_channels": 6,
        "out_channels": 2,
        "modes_x": modes_x,
        "modes_t": modes_t,
        "width": width,
        "num_layers": num_layers,
        "padding_frac": padding_frac,
    }
    model = FNO2d(**model_args).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters : {total_params:,}")
    print("-" * 70)

    # 3. Setup Optimizer, Scheduler, and Loss
    criterion = RelativeL2Loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    save_path = os.path.join(save_dir, save_name)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred_norm = model(x_batch)
            loss = criterion(pred_norm, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.shape[0]
            train_count += x_batch.shape[0]

        train_loss /= max(train_count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "x_normalizer": x_normalizer.state_dict(),
                "y_normalizer": y_normalizer.state_dict(),
                "val_loss": val_loss,
                "meta": meta,
                "model_args": model_args
            }
            safe_save_checkpoint(checkpoint, save_path)
            saved_indicator = "*"
        else:
            saved_indicator = " "

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Train Rel-L2 (norm): {train_loss:.4e} | "
                f"Val Rel-L2 (norm): {val_loss:.4e} {saved_indicator} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # 4. Final Evaluation on Test Set using Best Model
    print("-" * 70)
    best_checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)

    # Physical denormalized error evaluation
    model.eval()
    H_pred_all, Q_pred_all, H_true_all, Q_true_all = [], [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred_phys = y_normalizer.decode(model(x_batch))
            target_phys = y_normalizer.decode(y_batch)

            H_pred_all.append(pred_phys[..., 0].cpu().numpy())
            Q_pred_all.append(pred_phys[..., 1].cpu().numpy())
            H_true_all.append(target_phys[..., 0].cpu().numpy())
            Q_true_all.append(target_phys[..., 1].cpu().numpy())

    metrics = field_metrics(
        np.concatenate(H_pred_all), np.concatenate(Q_pred_all),
        np.concatenate(H_true_all), np.concatenate(Q_true_all)
    )

    print(f"Training Complete! Best model saved to: {save_path}")
    print(f"Test Set Training Objective (normalised Rel-L2): {test_loss:.4e}")
    print(f"Test Set Physical Accuracy ({metrics['n_samples']} held-out trajectories):")
    print(format_metrics(metrics))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fourier Neural Operator for 1D Saint-Venant.")
    parser.add_argument("--data-path", type=str, default="data/saint_venant_dataset.npz", help="Path to dataset NPZ file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--modes-x", type=int, default=16, help="Fourier modes along space (x).")
    parser.add_argument("--modes-t", type=int, default=24, help="Fourier modes along time (t).")
    parser.add_argument("--width", type=int, default=64, help="Hidden channel width.")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of spectral convolution layers.")
    parser.add_argument("--padding-frac", type=float, default=0.2, help="Non-periodic domain padding as a fraction of grid size.")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save checkpoints.")
    parser.add_argument("--save-name", type=str, default="fno_saint_venant.pt", help="Checkpoint filename.")
    args = parser.parse_args()

    train_fno(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        modes_x=args.modes_x,
        modes_t=args.modes_t,
        width=args.width,
        num_layers=args.num_layers,
        padding_frac=args.padding_frac,
        save_dir=args.save_dir,
        save_name=args.save_name
    )
