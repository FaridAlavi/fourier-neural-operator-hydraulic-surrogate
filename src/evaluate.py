"""
Evaluation and inference script for the trained 1D Saint-Venant FNO surrogate.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import get_dataloaders
from fno import FNO2d
from metrics import field_metrics, format_metrics
from train import RelativeL2Loss, evaluate


def evaluate_checkpoint(
    data_path: str = "data/saint_venant_dataset.npz",
    checkpoint_path: str = "models/fno_saint_venant.pt",
    force_device: str = None
):
    if force_device:
        device = torch.device(force_device)
    else:
        # Check cuda availability safely
        try:
            if torch.cuda.is_available():
                _ = torch.zeros(1, device="cuda")
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        except Exception:
            device = torch.device("cpu")

    print("=" * 70)
    print("Evaluating Trained FNO Checkpoint on Test Set")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Dataset Path    : {data_path}")
    print(f"Checkpoint Path : {checkpoint_path}")
    print("-" * 70)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 1. Load Checkpoint (load to CPU first to avoid broken CUDA runtime driver crashes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["model_args"]
    print(f"Loaded Checkpoint from Epoch: {checkpoint['epoch']} (Best Val Loss: {checkpoint['val_loss']:.4e})")

    # 2. Reconstruct Model
    model = FNO2d(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load Data & Test Set
    _, _, test_loader, x_normalizer, y_normalizer, meta = get_dataloaders(
        data_path=data_path,
        batch_size=16
    )

    criterion = RelativeL2Loss()
    test_loss = evaluate(model, test_loader, criterion, device)

    # 4. Compute Physical Metrics through the shared definition in metrics.py
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

    print("-" * 70)
    print(f"Test Set Evaluation Results ({metrics['n_samples']} held-out trajectories):")
    print(format_metrics(metrics))
    print(f"  Training objective (normalised Rel-L2) : {test_loss:.4e}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved FNO checkpoint.")
    parser.add_argument("--data-path", type=str, default="data/saint_venant_dataset.npz")
    parser.add_argument("--checkpoint-path", type=str, default="models/fno_saint_venant.pt")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    evaluate_checkpoint(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        force_device=args.device
    )
