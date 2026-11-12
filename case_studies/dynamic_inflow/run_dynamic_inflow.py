"""
Case Study: Dynamic Inflow Comparison between 1D Saint-Venant PDE Solver and FNO Surrogate.

Scenario:
  - Prismatic open channel reach (L = 10 km, B = 10 m, S0 = 0.0002, n = 0.025).
  - Realistic multi-peak dynamic inflow hydrograph featuring baseflow, progressive flood rising limbs,
    peak attenuation, and recession periods.
  - Generates spatio-temporal water level H(x, t) and discharge Q(x, t) using:
      1) The physics-based Preissmann implicit 1D Saint-Venant PDE solver.
      2) The trained Fourier Neural Operator (FNO2d) surrogate model.
  - Benchmarks execution time, speedup, point-wise RMSE, MAE, and relative L2 errors.
  - Exports summary comparison figures (.png) and a statistical summary (.json).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root and src to python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from dataset import UnitGaussianNormalizer
from fno import FNO2d
from metrics import relative_l2
from saint_venant import ChannelParameters, SaintVenant1D


def resolve_checkpoint_path(custom_path: str = "models/fno_saint_venant.pt") -> str:
    """Finds the model checkpoint across possible relative directories."""
    candidates = [
        custom_path,
        str(project_root / custom_path),
        str(project_root / "src" / custom_path),
        str(project_root / "models" / "fno_saint_venant.pt"),
        str(project_root / "src" / "models" / "fno_saint_venant.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return str(Path(c).resolve())
    raise FileNotFoundError(f"Could not locate checkpoint at any of: {candidates}")


def build_dynamic_boundary_conditions(
    nx: int = 51,
    nt: int = 120,
    dt: float = 60.0,
    channel_length: float = 10000.0,
    base_flow: float = 12.0,
    h_down_val: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a multi-event dynamic inflow hydrograph and time arrays.
    
    Includes:
      - Steady initial baseflow period.
      - First moderate wave peak (t ~ 25 min).
      - Second major steep flood wave peak (t ~ 65 min).
      - Smooth recession back towards baseline.
    """
    t_grid = np.arange(nt) * dt
    x_grid = np.linspace(0.0, channel_length, nx)

    t_min = t_grid / 60.0  # minutes

    # Multi-component dynamic hydrograph
    # 1. Baseflow
    q_up = np.full(nt, base_flow, dtype=np.float32)

    # 2. First moderate wave (Gaussian pulse centered around 25 min)
    pulse1 = 14.0 * np.exp(-((t_min - 25.0) / 7.0) ** 2)

    # 3. Second larger steep flood pulse (Skewed peak around 65 min)
    pulse2 = 22.0 * np.exp(-((t_min - 65.0) / 10.0) ** 2)

    # 4. Low-frequency natural oscillation
    wave_osc = 3.0 * np.sin(2.0 * np.pi * t_min / 90.0) * np.clip((t_min - 10.0) / 20.0, 0.0, 1.0)

    q_up = q_up + pulse1 + pulse2 + wave_osc
    q_up = np.maximum(q_up, 5.0).astype(np.float32)

    # Downstream stage: constant or slight backwater level
    h_down = np.full(nt, h_down_val, dtype=np.float32)

    return x_grid, t_grid, q_up, h_down


def run_pde_simulation(
    channel_params: ChannelParameters,
    nx: int,
    dt: float,
    q_up: np.ndarray,
    h_down: np.ndarray,
    timing_runs: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Simulates Saint-Venant PDE using the Preissmann implicit scheme with timing.
    """
    solver = SaintVenant1D(params=channel_params, nx=nx)
    H0, Q0 = solver.compute_steady_state(q_initial=float(q_up[0]), h_downstream=float(h_down[0]))

    # Execution and timing
    t0 = time.perf_counter()
    H_pde, Q_pde, _ = solver.simulate(q_up_series=q_up, h_down_series=h_down, dt=dt, H_init=H0, Q_init=Q0)
    pde_time_ms = (time.perf_counter() - t0) * 1000.0

    return H_pde, Q_pde, H0, pde_time_ms


def run_fno_inference(
    checkpoint_path: str,
    H0: np.ndarray,
    Q0: np.ndarray,
    q_up: np.ndarray,
    h_down: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    channel_length: float,
    timing_runs: int = 20,
    force_device: str = None
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """
    Performs inference with the trained FNO surrogate with precise timing.
    """
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

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FNO2d(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_normalizer = UnitGaussianNormalizer()
    y_normalizer = UnitGaussianNormalizer()
    x_normalizer.load_state_dict(checkpoint["x_normalizer"])
    y_normalizer.load_state_dict(checkpoint["y_normalizer"])

    nx = len(x_grid)
    nt = len(t_grid)
    total_time = float(t_grid[-1])

    # Assemble 6-channel input tensor
    inp = np.zeros((1, nx, nt, 6), dtype=np.float32)
    inp[0, :, :, 0] = H0[:, None]
    inp[0, :, :, 1] = Q0[:, None]
    inp[0, :, :, 2] = q_up[None, :]
    inp[0, :, :, 3] = h_down[None, :]
    inp[0, :, :, 4] = (x_grid / channel_length).reshape(nx, 1)
    inp[0, :, :, 5] = (t_grid / total_time).reshape(1, nt)

    inp_tensor = torch.from_numpy(inp).to(device)

    # Warmup
    with torch.no_grad():
        inp_norm = x_normalizer.encode(inp_tensor)
        for _ in range(5):
            _ = model(inp_norm)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(timing_runs):
            pred_norm = model(inp_norm)
            pred_phys = y_normalizer.decode(pred_norm)
    if device.type == "cuda":
        torch.cuda.synchronize()

    fno_time_ms = ((time.perf_counter() - t0) / timing_runs) * 1000.0

    pred_arr = pred_phys.cpu().numpy()[0]
    H_fno = pred_arr[:, :, 0]
    Q_fno = pred_arr[:, :, 1]

    return H_fno, Q_fno, fno_time_ms, str(device)


def compute_comparison_statistics(
    H_pde: np.ndarray,
    Q_pde: np.ndarray,
    H_fno: np.ndarray,
    Q_fno: np.ndarray,
    pde_time_ms: float,
    fno_time_ms: float
) -> Dict:
    """Computes comparison error metrics and speedup."""
    h_diff = H_fno - H_pde
    q_diff = Q_fno - Q_pde

    h_rmse = float(np.sqrt(np.mean(h_diff ** 2)))
    q_rmse = float(np.sqrt(np.mean(q_diff ** 2)))
    h_mae = float(np.mean(np.abs(h_diff)))
    q_mae = float(np.mean(np.abs(q_diff)))
    h_max_err = float(np.max(np.abs(h_diff)))
    q_max_err = float(np.max(np.abs(q_diff)))

    h_rel_l2 = relative_l2(H_fno, H_pde)
    q_rel_l2 = relative_l2(Q_fno, Q_pde)
    overall_rel_l2 = float(0.5 * (h_rel_l2 + q_rel_l2))

    speedup = float(pde_time_ms / max(fno_time_ms, 1e-6))

    return {
        "pde_time_ms": round(pde_time_ms, 2),
        "fno_time_ms": round(fno_time_ms, 2),
        "speedup": round(speedup, 1),
        "pde_hardware": "CPU, single-threaded banded Newton",
        "fno_hardware": "GPU" if torch.cuda.is_available() else "CPU",
        "h_rmse_m": round(h_rmse, 5),
        "h_mae_m": round(h_mae, 5),
        "h_max_error_m": round(h_max_err, 5),
        "h_relative_l2": round(h_rel_l2, 5),
        "q_rmse_m3s": round(q_rmse, 4),
        "q_mae_m3s": round(q_mae, 4),
        "q_max_error_m3s": round(q_max_err, 4),
        "q_relative_l2": round(q_rel_l2, 5),
        "overall_relative_l2": round(overall_rel_l2, 5),
    }


def plot_summary_figures(
    H_pde: np.ndarray,
    Q_pde: np.ndarray,
    H_fno: np.ndarray,
    Q_fno: np.ndarray,
    q_up: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    stats: Dict,
    out_dir: Path
):
    """Generates comprehensive summary comparison figures."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    t_min = t_grid / 60.0
    x_km = x_grid / 1000.0

    # 1. Spatio-temporal Contour Comparisons
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # Water level H
    vmin_h = min(np.min(H_pde), np.min(H_fno))
    vmax_h = max(np.max(H_pde), np.max(H_fno))
    im0 = axes[0, 0].pcolormesh(t_min, x_km, H_pde, shading="auto", cmap="viridis", vmin=vmin_h, vmax=vmax_h)
    axes[0, 0].set_title("PDE: Water Level $H(x, t)$ [m]", fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], label="$H$ [m]")

    im1 = axes[0, 1].pcolormesh(t_min, x_km, H_fno, shading="auto", cmap="viridis", vmin=vmin_h, vmax=vmax_h)
    axes[0, 1].set_title("FNO: Water Level $H(x, t)$ [m]", fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 1], label="$H$ [m]")

    err_h = np.abs(H_fno - H_pde)
    im2 = axes[0, 2].pcolormesh(t_min, x_km, err_h, shading="auto", cmap="inferno")
    axes[0, 2].set_title(f"Absolute Error $|H_{{FNO}} - H_{{PDE}}|$ (RMSE: {stats['h_rmse_m']*100:.2f} cm)", fontweight="bold")
    plt.colorbar(im2, ax=axes[0, 2], label="Error [m]")

    # Discharge Q
    vmin_q = min(np.min(Q_pde), np.min(Q_fno))
    vmax_q = max(np.max(Q_pde), np.max(Q_fno))
    im3 = axes[1, 0].pcolormesh(t_min, x_km, Q_pde, shading="auto", cmap="plasma", vmin=vmin_q, vmax=vmax_q)
    axes[1, 0].set_title("PDE: Discharge $Q(x, t)$ [m$^3$/s]", fontweight="bold")
    plt.colorbar(im3, ax=axes[1, 0], label="$Q$ [m$^3$/s]")

    im4 = axes[1, 1].pcolormesh(t_min, x_km, Q_fno, shading="auto", cmap="plasma", vmin=vmin_q, vmax=vmax_q)
    axes[1, 1].set_title("FNO: Discharge $Q(x, t)$ [m$^3$/s]", fontweight="bold")
    plt.colorbar(im4, ax=axes[1, 1], label="$Q$ [m$^3$/s]")

    err_q = np.abs(Q_fno - Q_pde)
    im5 = axes[1, 2].pcolormesh(t_min, x_km, err_q, shading="auto", cmap="inferno")
    axes[1, 2].set_title(f"Absolute Error $|Q_{{FNO}} - Q_{{PDE}}|$ (RMSE: {stats['q_rmse_m3s']:.2f} m$^3$/s)", fontweight="bold")
    plt.colorbar(im5, ax=axes[1, 2], label="Error [m$^3$/s]")

    for ax in axes.flat:
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Reach Distance [km]")

    fig.suptitle(
        f"Dynamic Inflow Hydrograph Case Study: PDE Solver vs. FNO Surrogate\n"
        f"Speedup: {stats['speedup']:.1f}x | Overall Rel L2: {stats['overall_relative_l2']*100:.2f}%",
        fontsize=14, fontweight="bold"
    )
    fig.savefig(figures_dir / "dynamic_inflow_spatiotemporal.png", dpi=200)
    plt.close(fig)

    # 2. Station Hydrographs at Multiple Sections
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    stations = [(0, "Upstream Inlet (x = 0 km)"), (len(x_km)//2, "Mid-Reach (x = 5 km)"), (-1, "Downstream Outlet (x = 10 km)")]

    for ax, (idx, name) in zip(axes, stations):
        ax.plot(t_min, Q_pde[idx, :], "b-", lw=2.5, label="PDE Solver")
        ax.plot(t_min, Q_fno[idx, :], "r--", lw=2.0, label="FNO Surrogate")
        if idx == 0:
            ax.plot(t_min, q_up, "k:", lw=1.5, alpha=0.7, label="Prescribed Inflow")
        ax.set_title(f"Discharge Hydrograph at {name}", fontweight="bold", fontsize=11)
        ax.set_ylabel("Discharge $Q$ [m$^3$/s]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [minutes]")
    fig.savefig(figures_dir / "dynamic_inflow_station_hydrographs.png", dpi=200)
    plt.close(fig)
    print(f"Summary figures saved in: {figures_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Run Dynamic Inflow Case Study (PDE vs. FNO).")
    parser.add_argument("--checkpoint-path", type=str, default="models/fno_saint_venant.pt")
    parser.add_argument("--output-dir", type=str, default="case_studies/dynamic_inflow")
    parser.add_argument("--nx", type=int, default=51)
    parser.add_argument("--nt", type=int, default=120)
    parser.add_argument("--dt", type=float, default=60.0)
    parser.add_argument("--base-flow", type=float, default=12.0)
    parser.add_argument("--h-down", type=float, default=3.0)
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint_path(args.checkpoint_path)
    print("=" * 75)
    print("  CASE STUDY: Dynamic Inflow Hydrograph (PDE vs. FNO)")
    print("=" * 75)
    print(f"Checkpoint Path : {ckpt_path}")
    print(f"Output Dir      : {out_dir}")
    print(f"Grid Dimensions : nx = {args.nx}, nt = {args.nt}, dt = {args.dt}s")

    channel_params = ChannelParameters(
        length=10000.0,
        width=10.0,
        bed_slope=0.0002,
        manning_n=0.025,
        h_bed_down=0.0
    )

    # 1. Build Dynamic Conditions
    x_grid, t_grid, q_up, h_down = build_dynamic_boundary_conditions(
        nx=args.nx, nt=args.nt, dt=args.dt,
        channel_length=channel_params.length,
        base_flow=args.base_flow, h_down_val=args.h_down
    )

    # 2. Simulate PDE
    print("\n[1/3] Simulating Saint-Venant PDE (Preissmann Scheme)...")
    H_pde, Q_pde, H0, pde_time_ms = run_pde_simulation(
        channel_params=channel_params, nx=args.nx, dt=args.dt,
        q_up=q_up, h_down=h_down
    )
    print(f"  -> Completed in {pde_time_ms:.1f} ms")

    # 3. Simulate FNO
    print("\n[2/3] Simulating FNO Surrogate Inference...")
    H0_fno = H0
    Q0_fno = np.full_like(H0, float(q_up[0]))
    H_fno, Q_fno, fno_time_ms, dev_name = run_fno_inference(
        checkpoint_path=ckpt_path,
        H0=H0_fno, Q0=Q0_fno,
        q_up=q_up, h_down=h_down,
        x_grid=x_grid, t_grid=t_grid,
        channel_length=channel_params.length,
        force_device=args.device
    )
    print(f"  -> Completed on {dev_name} in {fno_time_ms:.2f} ms")

    # 4. Compare & Save
    print("\n[3/3] Computing Error Metrics & Saving Artifacts...")
    stats = compute_comparison_statistics(
        H_pde=H_pde, Q_pde=Q_pde,
        H_fno=H_fno, Q_fno=Q_fno,
        pde_time_ms=pde_time_ms,
        fno_time_ms=fno_time_ms
    )

    with open(out_dir / "results_summary.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Generate Figures
    plot_summary_figures(
        H_pde, Q_pde, H_fno, Q_fno, q_up, x_grid, t_grid, stats, out_dir
    )

    print("\n" + "=" * 75)
    print("RESULTS SUMMARY:")
    print(f"  PDE Time           : {stats['pde_time_ms']:.1f} ms")
    print(f"  FNO Time           : {stats['fno_time_ms']:.2f} ms")
    print(f"  Inference Speedup  : {stats['speedup']:.1f}x"
          f" ({stats['pde_hardware']} vs FNO on {stats['fno_hardware']})")
    print(f"  Water Level RMSE   : {stats['h_rmse_m']*100:.2f} cm")
    print(f"  Discharge RMSE     : {stats['q_rmse_m3s']:.3f} m^3/s")
    print(f"  Overall Rel L2     : {stats['overall_relative_l2']*100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
