"""
Plotting and Visualization Utilities for 1D Saint-Venant FNO Benchmark.

Generates:
  1. 2D Spatio-temporal heatmaps (Ground Truth vs. FNO vs. Error for H and Q)
  2. Boundary Hydrograph Tracking Curves (x=0 and x=L)
  3. Spatial Water Surface Profile Snapshots across channel length
  4. Mass Balance Residual curves over time
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import UnitGaussianNormalizer, get_split_indices
from fno import FNO2d


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 2.0+ compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))


def plot_trajectory_comparison(
    sample_idx: int = 0,
    checkpoint_path: str = "models/fno_saint_venant.pt",
    data_path: str = "data/saint_venant_dataset.npz",
    save_dir: str = "figures"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check paths
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join("src", checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not os.path.exists(data_path):
        alt_data = os.path.join("src", data_path)
        if os.path.exists(alt_data):
            data_path = alt_data
        else:
            raise FileNotFoundError(f"Dataset not found: {data_path}")

    # 1. Load Checkpoint and Data
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FNO2d(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_normalizer = UnitGaussianNormalizer()
    y_normalizer = UnitGaussianNormalizer()
    x_normalizer.load_state_dict(checkpoint["x_normalizer"])
    y_normalizer.load_state_dict(checkpoint["y_normalizer"])

    raw = np.load(data_path)
    total_samples = raw["H"].shape[0]
    _, _, test_indices = get_split_indices(total_samples)
    if sample_idx < 0 or sample_idx >= len(test_indices):
        raise IndexError(f"sample_idx must be between 0 and {len(test_indices) - 1}.")
    test_idx = int(test_indices[sample_idx])

    H_true = raw["H"][test_idx]
    Q_true = raw["Q"][test_idx]
    q_up = raw["q_up"][test_idx]
    h_down = raw["h_down"][test_idx]
    H0 = raw["H0"][test_idx]
    Q0 = raw["Q0"][test_idx]
    x_grid = raw["x"] / 1000.0  # convert to km
    t_grid = raw["t"] / 60.0    # convert to min

    nx = len(x_grid)
    nt = len(t_grid)
    length = float(raw["channel_length"]) if "channel_length" in raw else 10000.0
    total_time = float(raw["t"][-1])

    # 2. Predict with FNO
    inp = np.zeros((1, nx, nt, 6), dtype=np.float32)
    inp[0, :, :, 0] = H0[:, None]
    inp[0, :, :, 1] = Q0[:, None]
    inp[0, :, :, 2] = q_up[None, :]
    inp[0, :, :, 3] = h_down[None, :]
    inp[0, :, :, 4] = (raw["x"] / length).reshape(nx, 1)
    inp[0, :, :, 5] = (raw["t"] / total_time).reshape(1, nt)

    with torch.no_grad():
        inp_t = torch.from_numpy(inp).to(device)
        pred_norm = model(x_normalizer.encode(inp_t))
        pred_phys = y_normalizer.decode(pred_norm).cpu().numpy()[0]

    H_pred = pred_phys[:, :, 0]
    Q_pred = pred_phys[:, :, 1]

    # Calculate errors
    H_err = np.abs(H_pred - H_true)
    Q_err = np.abs(Q_pred - Q_true)

    # -------------------------------------------------------------
    # Plot 1: 2D Spatio-Temporal Heatmaps for Water Level (H) and Flow (Q)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    T_mesh, X_mesh = np.meshgrid(t_grid, x_grid)

    # Water Level H
    vmin_h, vmax_h = np.min(H_true), np.max(H_true)
    im0 = axes[0, 0].pcolormesh(T_mesh, X_mesh, H_true, shading="auto", cmap="viridis", vmin=vmin_h, vmax=vmax_h)
    axes[0, 0].set_title("Ground Truth Water Level $H(x, t)$ [m]", fontsize=11, fontweight="bold")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(T_mesh, X_mesh, H_pred, shading="auto", cmap="viridis", vmin=vmin_h, vmax=vmax_h)
    axes[0, 1].set_title("FNO Predicted Water Level $H(x, t)$ [m]", fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].pcolormesh(T_mesh, X_mesh, H_err, shading="auto", cmap="magma")
    axes[0, 2].set_title("Absolute Error $|H_{pred} - H_{true}|$ [m]", fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=axes[0, 2])

    # Discharge Q
    vmin_q, vmax_q = np.min(Q_true), np.max(Q_true)
    im3 = axes[1, 0].pcolormesh(T_mesh, X_mesh, Q_true, shading="auto", cmap="coolwarm", vmin=vmin_q, vmax=vmax_q)
    axes[1, 0].set_title("Ground Truth Discharge $Q(x, t)$ [$\mathrm{m^3/s}$]", fontsize=11, fontweight="bold")
    fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(T_mesh, X_mesh, Q_pred, shading="auto", cmap="coolwarm", vmin=vmin_q, vmax=vmax_q)
    axes[1, 1].set_title("FNO Predicted Discharge $Q(x, t)$ [$\mathrm{m^3/s}$]", fontsize=11, fontweight="bold")
    fig.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].pcolormesh(T_mesh, X_mesh, Q_err, shading="auto", cmap="magma")
    axes[1, 2].set_title("Absolute Error $|Q_{pred} - Q_{true}|$ [$\mathrm{m^3/s}$]", fontsize=11, fontweight="bold")
    fig.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Distance along Channel (km)")

    heatmap_path = os.path.join(save_dir, f"heatmap_sample_{sample_idx}.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"[Saved] Spatio-temporal heatmaps: {heatmap_path}")

    # -------------------------------------------------------------
    # Plot 2: Boundary Tracking Hydrographs & Spatial Snapshots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    # Upstream & Downstream Discharges over time
    axes[0, 0].plot(t_grid, Q_true[0, :], "k-", lw=2, label="PDE Inflow $Q(0, t)$")
    axes[0, 0].plot(t_grid, Q_pred[0, :], "r--", lw=1.5, label="FNO Inflow $Q(0, t)$")
    axes[0, 0].plot(t_grid, Q_true[-1, :], "b-", lw=2, label="PDE Outflow $Q(L, t)$")
    axes[0, 0].plot(t_grid, Q_pred[-1, :], "c--", lw=1.5, label="FNO Outflow $Q(L, t)$")
    axes[0, 0].set_title("Discharge Hydrographs at Channel Boundaries", fontweight="bold")
    axes[0, 0].set_xlabel("Time (min)")
    axes[0, 0].set_ylabel("Discharge $Q$ ($\mathrm{m^3/s}$)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Upstream & Downstream Water Levels over time
    axes[0, 1].plot(t_grid, H_true[0, :], "k-", lw=2, label="PDE Upstream $H(0, t)$")
    axes[0, 1].plot(t_grid, H_pred[0, :], "r--", lw=1.5, label="FNO Upstream $H(0, t)$")
    axes[0, 1].plot(t_grid, H_true[-1, :], "b-", lw=2, label="PDE Downstream $H(L, t)$")
    axes[0, 1].plot(t_grid, H_pred[-1, :], "c--", lw=1.5, label="FNO Downstream $H(L, t)$")
    axes[0, 1].set_title("Water Level Profiles at Channel Boundaries", fontweight="bold")
    axes[0, 1].set_xlabel("Time (min)")
    axes[0, 1].set_ylabel("Water Level $H$ (m)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Spatial Profiles at Discrete Time Snapshots
    time_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(time_fractions)))

    for frac, color in zip(time_fractions, colors):
        t_idx = int(frac * (nt - 1))
        t_val = t_grid[t_idx]
        axes[1, 0].plot(x_grid, H_true[:, t_idx], color=color, linestyle="-", lw=2, label=f"PDE t={t_val:.0f}m")
        axes[1, 0].plot(x_grid, H_pred[:, t_idx], color=color, linestyle="--", lw=1.5, label=f"FNO t={t_val:.0f}m")

    axes[1, 0].set_title("Spatial Water Surface Profiles $H(x)$ across Time", fontweight="bold")
    axes[1, 0].set_xlabel("Distance along Channel (km)")
    axes[1, 0].set_ylabel("Water Level $H$ (m)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(ncol=2, fontsize=8)

    # Mass Conservation Residual over Time
    dx_m = raw["x"][1] - raw["x"][0]
    dt_s = raw["t"][1] - raw["t"][0]
    B = float(raw["channel_width"]) if "channel_width" in raw else 10.0

    storage_pred_t = np.array([trapezoid_integral(B * H_pred[:, i], x=raw["x"]) - trapezoid_integral(B * H_pred[:, 0], x=raw["x"]) for i in range(nt)])
    flux_pred_t = np.cumsum(Q_pred[0, :] - Q_pred[-1, :]) * dt_s
    mass_residual_fno = np.abs(storage_pred_t - flux_pred_t)

    storage_true_t = np.array([trapezoid_integral(B * H_true[:, i], x=raw["x"]) - trapezoid_integral(B * H_true[:, 0], x=raw["x"]) for i in range(nt)])
    flux_true_t = np.cumsum(Q_true[0, :] - Q_true[-1, :]) * dt_s
    mass_residual_pde = np.abs(storage_true_t - flux_true_t)

    axes[1, 1].plot(t_grid, mass_residual_pde, "k-", lw=2, label="Preissmann PDE Solver")
    axes[1, 1].plot(t_grid, mass_residual_fno, "r--", lw=2, label="FNO Surrogate")
    axes[1, 1].set_title("Mass Conservation Residual $|\Delta V_{\mathrm{storage}} - \Delta V_{\mathrm{flux}}|$", fontweight="bold")
    axes[1, 1].set_xlabel("Time (min)")
    axes[1, 1].set_ylabel("Volume Residual ($\mathrm{m^3}$)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    hydro_path = os.path.join(save_dir, f"profiles_sample_{sample_idx}.png")
    plt.savefig(hydro_path, dpi=300)
    plt.close()
    print(f"[Saved] Profiles & Hydrograph curves: {hydro_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark visualization plots for FNO.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Test sample index to visualize.")
    parser.add_argument("--checkpoint-path", type=str, default="models/fno_saint_venant.pt")
    parser.add_argument("--data-path", type=str, default="data/saint_venant_dataset.npz")
    parser.add_argument("--save-dir", type=str, default="figures")
    args = parser.parse_args()

    plot_trajectory_comparison(
        sample_idx=args.sample_idx,
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        save_dir=args.save_dir
    )
