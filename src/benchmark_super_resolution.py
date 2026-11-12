"""
Zero-Shot Super-Resolution & Mesh Invariance Benchmark for 1D Saint-Venant FNO Surrogate.

Evaluates the trained FNO across spatial (nx) and temporal (nt) meshes without
retraining. Two methodological points matter for the result to mean anything:

1. Initial conditions are recomputed with ``compute_steady_state`` on *each*
   evaluation mesh. Interpolating the coarse steady state onto a finer mesh does
   not produce a steady state of that mesh, and the resulting spurious start-up
   transient would otherwise be charged to the surrogate.
2. The reference is a single mesh-converged solution computed once on a fine grid
   and resampled onto each evaluation mesh. The Preissmann solver run *at* each
   evaluation mesh is scored against the same reference, so the table separates
   the surrogate's error from the reference discretisation error. A mesh-invariance
   claim is only meaningful where the former is not swamped by the latter.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import UnitGaussianNormalizer, get_split_indices
from fno import FNO2d
from metrics import field_metrics
from saint_venant import SaintVenant1D, ChannelParameters


def boundary_condition_interpolants(q_up: np.ndarray, h_down: np.ndarray, t_base: np.ndarray):
    """Continuous-in-time boundary forcing, so every mesh samples the same functions."""
    f_q_up = interp1d(t_base, q_up, kind="cubic", fill_value="extrapolate")
    f_h_down = interp1d(t_base, h_down, kind="cubic", fill_value="extrapolate")
    return f_q_up, f_h_down


def resample_field(
    field: np.ndarray,
    x_src: np.ndarray,
    t_src: np.ndarray,
    x_dst: np.ndarray,
    t_dst: np.ndarray
) -> np.ndarray:
    """Resample an (nx, nt) field onto a different space-time mesh."""
    along_x = interp1d(x_src, field, axis=0, kind="cubic", fill_value="extrapolate")(x_dst)
    return interp1d(t_src, along_x, axis=1, kind="cubic", fill_value="extrapolate")(t_dst)


def build_input_tensor_for_grid(
    H0: np.ndarray,
    Q0: np.ndarray,
    q_up: np.ndarray,
    h_down: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    channel_length: float,
    total_time: float
) -> torch.Tensor:
    """
    Assembles the 6-channel spatio-temporal input tensor on an arbitrary (nx, nt) grid.
    Input shape: (1, nx, nt, 6)
    """
    nx = len(x_grid)
    nt = len(t_grid)

    inp = np.zeros((1, nx, nt, 6), dtype=np.float32)
    inp[0, :, :, 0] = H0[:, None]
    inp[0, :, :, 1] = Q0[:, None]
    inp[0, :, :, 2] = q_up[None, :]
    inp[0, :, :, 3] = h_down[None, :]
    inp[0, :, :, 4] = (x_grid / channel_length).reshape(nx, 1)
    inp[0, :, :, 5] = (t_grid / total_time).reshape(1, nt)

    return torch.from_numpy(inp)


def simulate_on_grid(
    channel_params: ChannelParameters,
    nx: int,
    t_grid: np.ndarray,
    f_q_up,
    f_h_down
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the Preissmann solver on a given mesh from that mesh's own steady state."""
    solver = SaintVenant1D(params=channel_params, nx=nx)
    q_up = np.asarray(f_q_up(t_grid), dtype=float)
    h_down = np.asarray(f_h_down(t_grid), dtype=float)
    H0, Q0 = solver.compute_steady_state(q_initial=float(q_up[0]), h_downstream=float(h_down[0]))
    dt = float(t_grid[1] - t_grid[0])
    H, Q, _ = solver.simulate(q_up_series=q_up, h_down_series=h_down, dt=dt, H_init=H0, Q_init=Q0)
    return H, Q, H0, Q0


def benchmark_mesh_invariance(
    checkpoint_path: str = "models/fno_saint_venant.pt",
    data_path: str = "data/saint_venant_dataset.npz",
    spatial_resolutions: List[int] = [51, 101, 201],
    temporal_resolutions: List[int] = [120, 240],
    num_test_samples: int = 15,
    nx_reference: int = 801,
    nt_reference: int = 960
) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 104)
    print("Zero-Shot Super-Resolution & Mesh Invariance Benchmark")
    print("=" * 104)
    print(f"Device               : {device}")
    print(f"Checkpoint Path      : {checkpoint_path}")
    print(f"Test Dataset Path    : {data_path}")
    print(f"Spatial Grids (Nx)   : {spatial_resolutions} nodes")
    print(f"Temporal Grids (Nt)  : {temporal_resolutions} steps")
    print(f"Reference Mesh       : {nx_reference} x {nt_reference} (Preissmann)")
    print(f"Number of Test Cases : {num_test_samples}")
    print("-" * 104)

    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join("src", checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    if not os.path.exists(data_path):
        alt_data = os.path.join("src", data_path)
        if os.path.exists(alt_data):
            data_path = alt_data
        else:
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FNO2d(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_normalizer = UnitGaussianNormalizer()
    y_normalizer = UnitGaussianNormalizer()
    x_normalizer.load_state_dict(checkpoint["x_normalizer"])
    y_normalizer.load_state_dict(checkpoint["y_normalizer"])

    # 2. Load Base Test Dataset
    raw = np.load(data_path)
    total_samples = raw["H"].shape[0]
    if num_test_samples <= 0:
        raise ValueError("num_test_samples must be positive.")
    _, _, test_indices_tensor = get_split_indices(total_samples)
    test_indices = test_indices_tensor[:num_test_samples].numpy()

    channel_length = float(raw["channel_length"]) if "channel_length" in raw else 10000.0
    channel_params = ChannelParameters(
        length=channel_length,
        width=float(raw["channel_width"]) if "channel_width" in raw else 10.0,
        bed_slope=float(raw["bed_slope"]) if "bed_slope" in raw else 0.0002,
        manning_n=float(raw["manning_n"]) if "manning_n" in raw else 0.025
    )

    t_base = raw["t"]
    total_time = float(t_base[-1])
    x_ref = np.linspace(0.0, channel_length, nx_reference)
    t_ref = np.linspace(0.0, total_time, nt_reference)

    # 3. Mesh-converged reference solutions, computed once per test case
    print("[Computing mesh-converged reference solutions...]")
    references = {}
    usable_indices = []
    for idx in test_indices:
        f_q_up, f_h_down = boundary_condition_interpolants(raw["q_up"][idx], raw["h_down"][idx], t_base)
        try:
            H_ref, Q_ref, _, _ = simulate_on_grid(channel_params, nx_reference, t_ref, f_q_up, f_h_down)
        except (RuntimeError, ValueError) as exc:
            print(f"  sample {idx}: reference solve failed ({exc}); excluded")
            continue
        references[idx] = (H_ref, Q_ref, f_q_up, f_h_down)
        usable_indices.append(idx)

    if not usable_indices:
        raise RuntimeError("No reference solutions could be computed.")
    print(f"  {len(usable_indices)}/{len(test_indices)} reference solutions available")
    print("-" * 104)

    print(
        f"{'Grid (Nx x Nt)':<16} | {'dx':>7} | {'dt':>6} | "
        f"{'FNO relL2':>10} | {'FNO H RMSE':>11} | {'FNO Q RMSE':>12} | "
        f"{'Solver relL2':>13} | {'Solver ms':>10} | {'FNO ms':>8} | {'Status':<10}"
    )
    print("-" * 128)

    results: List[Dict] = []
    for nx_eval in spatial_resolutions:
        for nt_eval in temporal_resolutions:
            x_eval = np.linspace(0.0, channel_length, nx_eval)
            t_eval = np.linspace(0.0, total_time, nt_eval)

            fno_H, fno_Q, solver_H, solver_Q, ref_H, ref_Q = [], [], [], [], [], []
            solver_times_ms: List[float] = []
            fno_times_ms: List[float] = []

            for idx in usable_indices:
                H_ref, Q_ref, f_q_up, f_h_down = references[idx]

                try:
                    t0_sol = time.perf_counter()
                    H_solver, Q_solver, H0, Q0 = simulate_on_grid(
                        channel_params, nx_eval, t_eval, f_q_up, f_h_down
                    )
                    solver_times_ms.append((time.perf_counter() - t0_sol) * 1000.0)
                except (RuntimeError, ValueError):
                    continue

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0_fno = time.perf_counter()
                inp_tensor = build_input_tensor_for_grid(
                    H0=H0,
                    Q0=Q0,
                    q_up=np.asarray(f_q_up(t_eval), dtype=float),
                    h_down=np.asarray(f_h_down(t_eval), dtype=float),
                    x_grid=x_eval,
                    t_grid=t_eval,
                    channel_length=channel_length,
                    total_time=total_time
                ).to(device)

                with torch.no_grad():
                    pred_phys = y_normalizer.decode(model(x_normalizer.encode(inp_tensor))).cpu().numpy()[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fno_times_ms.append((time.perf_counter() - t0_fno) * 1000.0)

                fno_H.append(pred_phys[:, :, 0])
                fno_Q.append(pred_phys[:, :, 1])
                solver_H.append(H_solver)
                solver_Q.append(Q_solver)
                ref_H.append(resample_field(H_ref, x_ref, t_ref, x_eval, t_eval))
                ref_Q.append(resample_field(Q_ref, x_ref, t_ref, x_eval, t_eval))

            if not fno_H:
                continue

            ref_H_arr, ref_Q_arr = np.stack(ref_H), np.stack(ref_Q)
            fno_metrics = field_metrics(np.stack(fno_H), np.stack(fno_Q), ref_H_arr, ref_Q_arr)
            solver_metrics = field_metrics(np.stack(solver_H), np.stack(solver_Q), ref_H_arr, ref_Q_arr)
            mean_solver_ms = float(np.mean(solver_times_ms)) if solver_times_ms else float("nan")
            mean_fno_ms = float(np.mean(fno_times_ms)) if fno_times_ms else float("nan")

            tag = "Trained" if (nx_eval == 51 and nt_eval == len(t_base)) else "Zero-Shot"
            dx = channel_length / (nx_eval - 1)
            dt = total_time / (nt_eval - 1)
            print(
                f"{f'{nx_eval} x {nt_eval}':<16} | {dx:6.1f}m | {dt:5.1f}s | "
                f"{fno_metrics['rel_l2_overall'] * 100:9.2f}% | "
                f"{fno_metrics['rmse_H_m']:8.4f} m | {fno_metrics['rmse_Q_m3s']:8.3f} m3/s | "
                f"{solver_metrics['rel_l2_overall'] * 100:12.2f}% | "
                f"{mean_solver_ms:7.2f} ms | {mean_fno_ms:5.2f} ms | {tag:<10}"
            )

            results.append({
                "nx": nx_eval,
                "nt": nt_eval,
                "fno": fno_metrics,
                "solver": solver_metrics,
                "solver_ms": mean_solver_ms,
                "fno_ms": mean_fno_ms,
                "tag": tag,
            })

    print("-" * 104)
    print("Relative L2 is normalised by the reference dynamics (deviation from the initial profile),")
    print("not by the raw field, so the several-metre datum offset in H does not deflate the numbers.")
    print("'Solver relL2' is the Preissmann scheme's own discretisation error against the same reference.")
    if results:
        base = next((r for r in results if r["tag"] == "Trained"), results[0])
        worst = max(results, key=lambda r: r["fno"]["rel_l2_overall"])
        ratio = worst["fno"]["rel_l2_overall"] / max(base["fno"]["rel_l2_overall"], 1e-12)
        print(
            f"Surrogate error changes {ratio:.2f}x from the training mesh "
            f"({base['nx']}x{base['nt']}) to the worst mesh ({worst['nx']}x{worst['nt']})."
        )
    print("=" * 104)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot mesh super-resolution benchmark for FNO.")
    parser.add_argument("--checkpoint-path", type=str, default="models/fno_saint_venant.pt")
    parser.add_argument("--data-path", type=str, default="data/saint_venant_dataset.npz")
    parser.add_argument("--num-samples", type=int, default=15, help="Number of test scenarios to evaluate.")
    parser.add_argument("--nx-reference", type=int, default=801, help="Spatial nodes of the converged reference mesh.")
    parser.add_argument("--nt-reference", type=int, default=960, help="Time steps of the converged reference mesh.")
    args = parser.parse_args()

    benchmark_mesh_invariance(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        num_test_samples=args.num_samples,
        nx_reference=args.nx_reference,
        nt_reference=args.nt_reference
    )
