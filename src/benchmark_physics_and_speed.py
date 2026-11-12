"""
Physics Fidelity, Mass Conservation, Wave Dynamics & Latency Benchmarks
for the 1D Saint-Venant FNO Surrogate.

Evaluates:
  1. Accuracy against the Preissmann solution (shared metric definition)
  2. Global mass / volume conservation residuals
  3. Flood wave propagation (peak attenuation & travel time) on flood-wave cases only
  4. End-to-end inference latency, measured like-for-like

Timing notes
------------
The Preissmann baseline assembles its Newton system analytically and solves it as
a banded system, which is the standard way to implement the scheme. Comparing the
surrogate against a dense finite-difference Jacobian implementation would inflate
the speedup by roughly three orders of magnitude and say nothing about the method.

Both sides are timed end to end for the same task: given (H0, Q0, q_up, h_down),
produce H(x, t) and Q(x, t) as NumPy arrays. For the surrogate that includes input
tensor assembly, normalisation, the forward pass, denormalisation and the device
transfer back to host memory. The CPU comparison is also reported with PyTorch
restricted to a single thread, because the solver baseline is single-threaded.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import UnitGaussianNormalizer, get_split_indices
from fno import FNO2d
from metrics import field_metrics, format_metrics
from saint_venant import SaintVenant1D, ChannelParameters


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 2.0+ compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))


def cumulative_trapezoid_integral(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Integrate a time series cumulatively, with an exact zero at the initial time."""
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(increments)))


def volume_balance(
    H: np.ndarray,
    Q: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    bed_elevation: np.ndarray,
    channel_width: float
) -> Dict[str, float]:
    """
    Global volume conservation residual of a single trajectory.

        dV_storage(t) = int_0^L B * (h(x, t) - h(x, 0)) dx
        dV_flux(t)    = int_0^t (Q(0, tau) - Q(L, tau)) dtau
        residual(t)   = |dV_storage(t) - dV_flux(t)|

    Normalised by the volume that actually entered the reach over the horizon.
    """
    depth = H - bed_elevation[:, None]
    storage = np.array([
        trapezoid_integral(channel_width * depth[:, k], x=x_grid) for k in range(len(t_grid))
    ])
    storage -= storage[0]
    flux = cumulative_trapezoid_integral(Q[0, :] - Q[-1, :], t_grid)
    residual = np.abs(storage - flux)
    throughput = trapezoid_integral(np.abs(Q[0, :]), x=t_grid) + 1e-9
    return {
        "max_residual_m3": float(np.max(residual)),
        "mean_residual_m3": float(np.mean(residual)),
        "rel_error": float(np.max(residual) / throughput),
    }


def wave_dynamics(Q_pred: np.ndarray, Q_true: np.ndarray, t_grid: np.ndarray) -> Dict[str, float]:
    """Flood peak amplitude and arrival-time error at the downstream boundary (x = L)."""
    q_out_pred = Q_pred[-1, :]
    q_out_true = Q_true[-1, :]

    peak_pred = float(np.max(q_out_pred))
    peak_true = float(np.max(q_out_true))
    t_peak_pred = float(t_grid[np.argmax(q_out_pred)])
    t_peak_true = float(t_grid[np.argmax(q_out_true)])

    return {
        "peak_true_q": peak_true,
        "peak_pred_q": peak_pred,
        "rel_peak_amp_error": abs(peak_pred - peak_true) / (peak_true + 1e-9),
        "timing_error_sec": abs(t_peak_pred - t_peak_true),
    }


def assemble_input(
    H0: np.ndarray,
    Q0: np.ndarray,
    q_up: np.ndarray,
    h_down: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    channel_length: float
) -> np.ndarray:
    """6-channel space-time input grid of shape (1, nx, nt, 6)."""
    nx, nt = len(x_grid), len(t_grid)
    inp = np.zeros((1, nx, nt, 6), dtype=np.float32)
    inp[0, :, :, 0] = H0[:, None]
    inp[0, :, :, 1] = Q0[:, None]
    inp[0, :, :, 2] = q_up[None, :]
    inp[0, :, :, 3] = h_down[None, :]
    inp[0, :, :, 4] = (x_grid / channel_length).reshape(nx, 1)
    inp[0, :, :, 5] = (t_grid / t_grid[-1]).reshape(1, nt)
    return inp


def time_fno_end_to_end(
    model: FNO2d,
    x_normalizer: UnitGaussianNormalizer,
    y_normalizer: UnitGaussianNormalizer,
    raw_inputs: np.ndarray,
    device: torch.device,
    runs: int
) -> float:
    """Milliseconds per trajectory, including host transfer and denormalisation."""
    def one_pass() -> np.ndarray:
        tensor = torch.from_numpy(raw_inputs).to(device)
        with torch.no_grad():
            out = y_normalizer.decode(model(x_normalizer.encode(tensor)))
        return out.cpu().numpy()

    for _ in range(5):
        one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / runs * 1000.0
    return elapsed_ms / raw_inputs.shape[0]


def benchmark_speed_and_physics(
    checkpoint_path: str = "models/fno_saint_venant.pt",
    data_path: str = "data/saint_venant_dataset.npz",
    num_test_samples: int = 20,
    timing_runs: int = 50,
    results_path: Optional[str] = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("Physics Fidelity, Mass Conservation & Latency Benchmarks")
    print("=" * 78)
    print(f"Device               : {device}")
    print(f"Checkpoint Path      : {checkpoint_path}")
    print(f"Dataset Path         : {data_path}")
    print(f"Test Samples         : {num_test_samples}")
    print("-" * 78)

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

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FNO2d(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_normalizer = UnitGaussianNormalizer()
    y_normalizer = UnitGaussianNormalizer()
    x_normalizer.load_state_dict(checkpoint["x_normalizer"])
    y_normalizer.load_state_dict(checkpoint["y_normalizer"])

    # 2. Load Test Data
    raw = np.load(data_path)
    total_samples = raw["H"].shape[0]
    if num_test_samples <= 0:
        raise ValueError("num_test_samples must be positive.")
    _, _, test_indices_tensor = get_split_indices(total_samples)
    test_indices = test_indices_tensor[:num_test_samples].numpy()

    patterns = raw["pattern"] if "pattern" in raw else None
    if patterns is None:
        print("[warning] dataset has no 'pattern' labels; wave metrics fall back to all samples")

    x_grid = raw["x"]
    t_grid = raw["t"]
    nx, nt = len(x_grid), len(t_grid)
    dt = float(t_grid[1] - t_grid[0])

    channel_params = ChannelParameters(
        length=float(raw["channel_length"]) if "channel_length" in raw else 10000.0,
        width=float(raw["channel_width"]) if "channel_width" in raw else 10.0,
        bed_slope=float(raw["bed_slope"]) if "bed_slope" in raw else 0.0002,
        manning_n=float(raw["manning_n"]) if "manning_n" in raw else 0.025
    )
    bed_elevation = channel_params.bed_elevation(x_grid)

    # 3. Accuracy, conservation and wave dynamics
    H_pred_all, Q_pred_all, H_true_all, Q_true_all = [], [], [], []
    mass_fno, mass_pde = [], []
    peak_errors, timing_errors = [], []
    flood_cases = 0
    sample_inputs = []

    for idx in test_indices:
        H_true, Q_true = raw["H"][idx], raw["Q"][idx]
        inp = assemble_input(
            raw["H0"][idx], raw["Q0"][idx], raw["q_up"][idx], raw["h_down"][idx],
            x_grid, t_grid, channel_params.length
        )
        sample_inputs.append(inp)

        with torch.no_grad():
            tensor = torch.from_numpy(inp).to(device)
            pred_phys = y_normalizer.decode(model(x_normalizer.encode(tensor))).cpu().numpy()[0]

        H_pred, Q_pred = pred_phys[:, :, 0], pred_phys[:, :, 1]

        H_pred_all.append(H_pred)
        Q_pred_all.append(Q_pred)
        H_true_all.append(H_true)
        Q_true_all.append(Q_true)

        mass_fno.append(volume_balance(H_pred, Q_pred, x_grid, t_grid, bed_elevation, channel_params.width)["rel_error"])
        mass_pde.append(volume_balance(H_true, Q_true, x_grid, t_grid, bed_elevation, channel_params.width)["rel_error"])

        # A downstream "peak arrival time" is only meaningful for a travelling flood wave.
        if patterns is None or str(patterns[idx]) == "flood_wave":
            wave = wave_dynamics(Q_pred, Q_true, t_grid)
            peak_errors.append(wave["rel_peak_amp_error"])
            timing_errors.append(wave["timing_error_sec"])
            flood_cases += 1

    accuracy = field_metrics(
        np.stack(H_pred_all), np.stack(Q_pred_all), np.stack(H_true_all), np.stack(Q_true_all)
    )

    # 4. Latency: identical task, identical output, for both methods
    print("\n[Benchmarking inference latency...]")
    solver = SaintVenant1D(params=channel_params, nx=nx)
    pde_runs = min(10, len(test_indices))

    for i in range(3):  # warmup
        idx = test_indices[i % pde_runs]
        solver.simulate(q_up_series=raw["q_up"][idx], h_down_series=raw["h_down"][idx],
                        dt=dt, H_init=raw["H0"][idx], Q_init=raw["Q0"][idx])

    start = time.perf_counter()
    for i in range(pde_runs):
        idx = test_indices[i]
        solver.simulate(q_up_series=raw["q_up"][idx], h_down_series=raw["h_down"][idx],
                        dt=dt, H_init=raw["H0"][idx], Q_init=raw["Q0"][idx])
    pde_ms = (time.perf_counter() - start) / pde_runs * 1000.0

    single = sample_inputs[0]
    batch = np.concatenate(sample_inputs[:min(16, len(sample_inputs))], axis=0)

    cpu = torch.device("cpu")
    model_cpu = FNO2d(**checkpoint["model_args"]).to(cpu)
    model_cpu.load_state_dict(checkpoint["model_state_dict"])
    model_cpu.eval()

    default_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    fno_cpu_1t = time_fno_end_to_end(model_cpu, x_normalizer, y_normalizer, single, cpu, timing_runs)
    torch.set_num_threads(default_threads)
    fno_cpu = time_fno_end_to_end(model_cpu, x_normalizer, y_normalizer, single, cpu, timing_runs)
    fno_cpu_batch = time_fno_end_to_end(model_cpu, x_normalizer, y_normalizer, batch, cpu, timing_runs)

    timings = {
        "pde_solver_cpu_1_thread_ms": pde_ms,
        "fno_cpu_1_thread_ms": fno_cpu_1t,
        f"fno_cpu_{default_threads}_threads_ms": fno_cpu,
        f"fno_cpu_{default_threads}_threads_batch16_ms": fno_cpu_batch,
    }
    if device.type == "cuda":
        timings["fno_gpu_ms"] = time_fno_end_to_end(model, x_normalizer, y_normalizer, single, device, timing_runs)
        timings["fno_gpu_batch16_ms"] = time_fno_end_to_end(model, x_normalizer, y_normalizer, batch, device, timing_runs)

    # 4b. Latency scaling across spatial (Nx) and temporal (Nt) discretisations
    print("[Benchmarking latency scaling across (Nx, Nt) discretisations...]")
    grid_resolutions = [(51, 120), (51, 240), (101, 120), (101, 240), (201, 120), (201, 240)]
    grid_latency_scaling: List[Dict] = []
    n_grid_cases = min(16, len(test_indices))
    total_time = float(t_grid[-1])

    for nx_g, nt_g in grid_resolutions:
        x_g = np.linspace(0.0, channel_params.length, nx_g)
        t_g = np.linspace(0.0, total_time, nt_g)
        dt_g = float(t_g[1] - t_g[0])
        solver_g = SaintVenant1D(params=channel_params, nx=nx_g)

        # Prepare boundary series and initial conditions on mesh (nx_g, nt_g)
        grid_cases = []
        grid_inputs = []
        for k in range(n_grid_cases):
            idx = test_indices[k]
            f_q = interp1d(t_grid, raw["q_up"][idx], kind="cubic", fill_value="extrapolate")
            f_h = interp1d(t_grid, raw["h_down"][idx], kind="cubic", fill_value="extrapolate")
            q_up_g = np.asarray(f_q(t_g), dtype=float)
            h_down_g = np.asarray(f_h(t_g), dtype=float)
            H0_g, Q0_g = solver_g.compute_steady_state(q_initial=float(q_up_g[0]), h_downstream=float(h_down_g[0]))
            grid_cases.append((q_up_g, h_down_g, H0_g, Q0_g))
            grid_inputs.append(assemble_input(H0_g, Q0_g, q_up_g, h_down_g, x_g, t_g, channel_params.length))

        # Time Preissmann solver on (nx_g, nt_g)
        n_pde_g = min(5, len(grid_cases))
        for k in range(min(2, n_pde_g)):
            q_up_g, h_down_g, H0_g, Q0_g = grid_cases[k]
            solver_g.simulate(q_up_series=q_up_g, h_down_series=h_down_g, dt=dt_g, H_init=H0_g, Q_init=Q0_g)

        start = time.perf_counter()
        for k in range(n_pde_g):
            q_up_g, h_down_g, H0_g, Q0_g = grid_cases[k]
            solver_g.simulate(q_up_series=q_up_g, h_down_series=h_down_g, dt=dt_g, H_init=H0_g, Q_init=Q0_g)
        pde_g_ms = (time.perf_counter() - start) / n_pde_g * 1000.0

        single_g = grid_inputs[0]
        batch_g = np.concatenate(grid_inputs, axis=0)
        runs_g = max(10, timing_runs // 2)

        fno_cpu_g_ms = time_fno_end_to_end(model_cpu, x_normalizer, y_normalizer, single_g, cpu, runs_g)
        entry: Dict[str, float] = {
            "nx": nx_g,
            "nt": nt_g,
            "dx_m": float(channel_params.length / (nx_g - 1)),
            "dt_s": dt_g,
            "pde_solver_cpu_1t_ms": pde_g_ms,
            f"fno_cpu_{default_threads}t_single_ms": fno_cpu_g_ms,
        }
        if device.type == "cuda":
            entry["fno_gpu_single_ms"] = time_fno_end_to_end(model, x_normalizer, y_normalizer, single_g, device, runs_g)
            entry["fno_gpu_batch16_ms"] = time_fno_end_to_end(model, x_normalizer, y_normalizer, batch_g, device, runs_g)
            entry["speedup_gpu_single_vs_pde"] = pde_g_ms / entry["fno_gpu_single_ms"]
            entry["speedup_gpu_batch16_vs_pde"] = pde_g_ms / entry["fno_gpu_batch16_ms"]
        grid_latency_scaling.append(entry)

    # 5. Summary Report
    print("=" * 78)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 78)
    print(f"1. Accuracy vs. Preissmann solution ({accuracy['n_samples']} held-out trajectories):")
    print(format_metrics(accuracy, indent="   "))
    print("\n2. Global volume conservation (max residual / inflow volume):")
    print(f"   - FNO surrogate                      : {np.mean(mass_fno) * 100:.2f}% mean, {np.max(mass_fno) * 100:.2f}% max")
    print(f"   - Preissmann solver                  : {np.mean(mass_pde) * 100:.2f}% mean, {np.max(mass_pde) * 100:.2f}% max")
    if peak_errors:
        scope = "flood-wave cases" if patterns is not None else "all cases (no pattern labels)"
        print(f"\n3. Flood wave tracking at x = L ({flood_cases} {scope}):")
        print(f"   - Downstream peak discharge error     : {np.mean(peak_errors) * 100:.2f}% mean")
        print(f"   - Peak arrival timing error           : {np.mean(timing_errors):.1f} s "
              f"({np.mean(timing_errors) / dt:.2f} time steps)")
    else:
        print("\n3. Flood wave tracking: no flood-wave cases in this test subset.")
    print("\n4. End-to-end latency per trajectory on base grid (51 x 120):")
    print(f"   - Preissmann solver, CPU 1 thread     : {pde_ms:8.2f} ms   (baseline)")
    print(f"   - FNO, CPU 1 thread                   : {fno_cpu_1t:8.2f} ms   ({pde_ms / fno_cpu_1t:6.2f}x)")
    print(f"   - FNO, CPU {default_threads} threads  : {fno_cpu:8.2f} ms   ({pde_ms / fno_cpu:6.2f}x)")
    print(f"   - FNO, CPU {default_threads} threads, batch 16 : {fno_cpu_batch:8.2f} ms   ({pde_ms / fno_cpu_batch:6.2f}x)")
    if "fno_gpu_ms" in timings:
        print(f"   - FNO, GPU single                     : {timings['fno_gpu_ms']:8.2f} ms   ({pde_ms / timings['fno_gpu_ms']:6.2f}x)")
        print(f"   - FNO, GPU batch 16                   : {timings['fno_gpu_batch16_ms']:8.2f} ms   ({pde_ms / timings['fno_gpu_batch16_ms']:6.2f}x)")

    print("\n5. Latency scaling across spatial (Nx) and temporal (Nt) discretisations:")
    if device.type == "cuda":
        print(
            f"   {'Grid (Nx x Nt)':<14} | {'dx':>6} | {'dt':>5} | "
            f"{'Solver (CPU)':>12} | {'FNO (CPU)':>10} | {'FNO (GPU 1)':>11} | "
            f"{'FNO (GPU B16)':>13} | {'Speedup (B16)':>13}"
        )
        print("   " + "-" * 106)
        for row in grid_latency_scaling:
            grid_lbl = f"{int(row['nx'])} x {int(row['nt'])}"
            cpu_key = f"fno_cpu_{default_threads}t_single_ms"
            print(
                f"   {grid_lbl:<14} | {row['dx_m']:5.0f}m | {row['dt_s']:4.0f}s | "
                f"{row['pde_solver_cpu_1t_ms']:9.2f} ms | "
                f"{row[cpu_key]:7.2f} ms | "
                f"{row['fno_gpu_single_ms']:8.2f} ms | "
                f"{row['fno_gpu_batch16_ms']:10.2f} ms | "
                f"{row['speedup_gpu_batch16_vs_pde']:12.1f}x"
            )
    else:
        print(
            f"   {'Grid (Nx x Nt)':<14} | {'dx':>6} | {'dt':>5} | "
            f"{'Solver (CPU)':>12} | {'FNO (CPU)':>10} | {'Speedup (CPU)':>13}"
        )
        print("   " + "-" * 72)
        for row in grid_latency_scaling:
            grid_lbl = f"{int(row['nx'])} x {int(row['nt'])}"
            cpu_key = f"fno_cpu_{default_threads}t_single_ms"
            spd = row["pde_solver_cpu_1t_ms"] / row[cpu_key]
            print(
                f"   {grid_lbl:<14} | {row['dx_m']:5.0f}m | {row['dt_s']:4.0f}s | "
                f"{row['pde_solver_cpu_1t_ms']:9.2f} ms | "
                f"{row[cpu_key]:7.2f} ms | "
                f"{spd:12.2f}x"
            )
    print("=" * 78)

    if results_path:
        payload = {
            "accuracy": accuracy,
            "mass_conservation": {
                "fno_mean_rel": float(np.mean(mass_fno)),
                "fno_max_rel": float(np.max(mass_fno)),
                "pde_mean_rel": float(np.mean(mass_pde)),
                "pde_max_rel": float(np.max(mass_pde)),
            },
            "wave_tracking": {
                "n_flood_cases": flood_cases,
                "mean_rel_peak_error": float(np.mean(peak_errors)) if peak_errors else None,
                "mean_timing_error_sec": float(np.mean(timing_errors)) if timing_errors else None,
            },
            "latency_ms": timings,
            "latency_scaling_by_grid": grid_latency_scaling,
        }
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results written to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics and latency benchmarking for FNO.")
    parser.add_argument("--checkpoint-path", type=str, default="models/fno_saint_venant.pt")
    parser.add_argument("--data-path", type=str, default="data/saint_venant_dataset.npz")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--results-path", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    benchmark_speed_and_physics(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        num_test_samples=args.num_samples,
        results_path=args.results_path
    )
