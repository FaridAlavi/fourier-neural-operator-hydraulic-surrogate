"""
Dataset Generation Script for 1D Saint-Venant Open Channel Hydraulics.

Generates diverse spatio-temporal training and testing trajectories
for training a Fourier Neural Operator (FNO) surrogate model.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Add src folder to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from saint_venant import SaintVenant1D, ChannelParameters


def sample_gaussian_process(
    rng: np.random.Generator,
    nt: int,
    dt: float,
    mean: float,
    std: float,
    length_scale: float,
    lower_bound: float
) -> np.ndarray:
    """Generate a smooth 1D trajectory using a Gaussian Process with RBF kernel."""
    t = np.arange(nt) * dt
    # RBF covariance matrix
    dist_sq = (t[:, None] - t[None, :]) ** 2
    cov = (std ** 2) * np.exp(-0.5 * dist_sq / (length_scale ** 2)) + 1e-6 * np.eye(nt)
    
    # Sample Gaussian Process
    sample = rng.multivariate_normal(np.zeros(nt), cov)
    trajectory = mean + sample
    return np.maximum(trajectory, lower_bound)


def sample_step_ramp(
    rng: np.random.Generator,
    nt: int,
    dt: float,
    base_val: float,
    delta_range: Tuple[float, float],
    lower_bound: float
) -> np.ndarray:
    """
    Generate a multi-event control hydrograph using smooth raised-cosine (S-curve)
    transitions and pulses rather than instantaneous single-step discontinuities.
    """
    traj = np.full(nt, base_val, dtype=np.float64)
    
    num_events = int(rng.integers(1, 4))
    event_times = np.sort(rng.choice(np.arange(int(0.1 * nt), int(0.80 * nt)), size=num_events, replace=False))
    
    for t_idx in event_times:
        delta = float(rng.uniform(*delta_range))
        event_type = rng.choice(["step", "ramp", "pulse"])
        
        # Ensure downward transitions do not cross lower_bound and create hard clip corners
        min_headroom = float(np.min(traj[t_idx:])) - lower_bound
        if delta < -min_headroom:
            delta = -0.9 * max(min_headroom, 0.0)
        
        if event_type == "step":
            # Fast gate/pump transition: smooth S-curve over 6 to 15 minutes (at dt=60s)
            min_len = max(6, int(round(360.0 / max(dt, 1.0))))
            max_len = max(min_len + 1, int(round(900.0 / max(dt, 1.0))) + 1)
            trans_len = int(rng.integers(min_len, max_len))
            end_idx = min(t_idx + trans_len, nt)
            s_curve = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, end_idx - t_idx)))
            traj[t_idx:end_idx] += delta * s_curve
            traj[end_idx:] += delta
        elif event_type == "ramp":
            # Gradual operational ramp: smooth S-curve over 15 to 30 minutes (at dt=60s)
            min_len = max(12, int(round(900.0 / max(dt, 1.0))))
            max_len = max(min_len + 1, int(round(1800.0 / max(dt, 1.0))) + 1)
            ramp_len = int(rng.integers(min_len, max_len))
            end_idx = min(t_idx + ramp_len, nt)
            s_curve = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, end_idx - t_idx)))
            traj[t_idx:end_idx] += delta * s_curve
            traj[end_idx:] += delta
        elif event_type == "pulse":
            pulse_len = int(rng.integers(max(12, int(0.1 * nt)), max(13, int(0.3 * nt))))
            end_idx = min(t_idx + pulse_len, nt)
            window = np.sin(np.linspace(0.0, np.pi, end_idx - t_idx)) ** 2
            traj[t_idx:end_idx] += delta * window

    return np.maximum(traj, lower_bound)


def generate_single_sample(
    sample_id: int,
    channel_params: ChannelParameters,
    nx: int,
    nt: int,
    dt: float,
    q_min: float = 2.0,
    q_max: float = 60.0,
    q_var_scale: float = 1.0,
    base_seed: int = 20240917
) -> Tuple[int, Optional[Dict[str, np.ndarray]], str]:
    """Generate a single trajectory simulation with randomized boundary dynamics."""
    # Independent, non-overlapping random streams per sample id.
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, sample_id]))
    solver = SaintVenant1D(params=channel_params, nx=nx)
    
    # Randomize baseline operating point across wider range
    q_base = rng.uniform(q_min, q_max)         # m^3/s
    h_down_base = rng.uniform(1.8, 5.0)         # m above downstream bed
    
    pattern_type = rng.choice(["gp", "step_ramp", "sinusoid", "flood_wave"], p=[0.4, 0.3, 0.15, 0.15])
    q_floor = max(0.5, q_min * 0.25)
    
    if pattern_type == "gp":
        std_q = rng.uniform(0.15, 0.40) * q_base * q_var_scale
        q_up = sample_gaussian_process(
            rng=rng, nt=nt, dt=dt, mean=q_base, std=max(std_q, 1.0),
            length_scale=rng.uniform(1800.0, 7200.0), lower_bound=q_floor
        )
        h_down = sample_gaussian_process(
            rng=rng, nt=nt, dt=dt, mean=h_down_base, std=rng.uniform(0.1, 0.4),
            length_scale=rng.uniform(3600.0, 10800.0), lower_bound=1.0
        )
    elif pattern_type == "step_ramp":
        delta_low = -0.5 * q_base * q_var_scale
        delta_high = 0.8 * q_base * q_var_scale
        q_up = sample_step_ramp(
            rng=rng, nt=nt, dt=dt, base_val=q_base, delta_range=(delta_low, delta_high), lower_bound=q_floor
        )
        h_down = sample_step_ramp(
            rng=rng, nt=nt, dt=dt, base_val=h_down_base, delta_range=(-0.4, 0.4), lower_bound=1.0
        )
    elif pattern_type == "sinusoid":
        t = np.arange(nt) * dt
        period_q = rng.uniform(3600.0, 14400.0)
        amp_q = rng.uniform(0.2, 0.6) * q_base * q_var_scale
        q_up = np.maximum(q_base + amp_q * np.sin(2.0 * np.pi * t / period_q + rng.uniform(0, 2*np.pi)), q_floor)
        
        period_h = rng.uniform(7200.0, 21600.0)
        amp_h = rng.uniform(0.1, 0.35)
        h_down = np.maximum(h_down_base + amp_h * np.sin(2.0 * np.pi * t / period_h), 1.0)
    else:  # flood_wave
        t = np.arange(nt) * dt
        t_peak = rng.uniform(0.2 * nt * dt, 0.5 * nt * dt)
        duration = rng.uniform(0.15 * nt * dt, 0.4 * nt * dt)
        flood_peak = rng.uniform(0.5, 1.5) * q_base * q_var_scale
        # Smooth flood pulse
        pulse = np.exp(-0.5 * ((t - t_peak) / (0.5 * duration)) ** 2)
        q_up = q_base + flood_peak * pulse
        # Smooth downstream stage drift rather than step-to-step white noise
        h_down = sample_gaussian_process(
            rng=rng, nt=nt, dt=dt, mean=h_down_base, std=rng.uniform(0.02, 0.08),
            length_scale=rng.uniform(3600.0, 10800.0), lower_bound=1.0
        )

    try:
        H_traj, Q_traj, t_grid = solver.simulate(
            q_up_series=q_up,
            h_down_series=h_down,
            dt=dt
        )
        
        # Check for unphysical NaNs or severe divergences
        if np.isnan(H_traj).any() or np.isnan(Q_traj).any():
            return sample_id, None, "non-finite state"
            
        data = {
            "H": H_traj.astype(np.float32),          # (nx, nt)
            "Q": Q_traj.astype(np.float32),          # (nx, nt)
            "q_up": q_up.astype(np.float32),        # (nt,)
            "h_down": h_down.astype(np.float32),    # (nt,)
            "H0": H_traj[:, 0].astype(np.float32),  # (nx,)
            "Q0": Q_traj[:, 0].astype(np.float32),  # (nx,)
            "pattern": str(pattern_type),
        }
        return sample_id, data, ""
    except (RuntimeError, ValueError) as exc:
        return sample_id, None, str(exc).split("\n")[0][:120]


def generate_dataset(
    num_samples: int = 200,
    nx: int = 51,
    nt: int = 120,
    dt: float = 60.0,
    q_min: float = 2.0,
    q_max: float = 60.0,
    q_var_scale: float = 1.0,
    num_workers: int = 4,
    output_dir: str = "data",
    output_filename: str = "saint_venant_dataset.npz"
):
    """
    Generate and save a multi-sample dataset for Fourier Neural Operator training.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    
    channel_params = ChannelParameters(
        length=10000.0,
        width=10.0,
        bed_slope=0.0002,
        manning_n=0.025,
        h_bed_down=0.0
    )
    
    print("=" * 70)
    print("1D Saint-Venant Dataset Generator for Fourier Neural Operator (FNO)")
    print("=" * 70)
    print(f"Total Target Samples : {num_samples}")
    print(f"Spatial Grid Nodes nx: {nx} (dx = {channel_params.length/(nx-1):.2f} m)")
    print(f"Time Horizon nt      : {nt} steps (dt = {dt:.1f} s, Total T = {nt*dt/3600:.2f} h)")
    print(f"Discharge Q Range    : [{q_min:.1f}, {q_max:.1f}] m^3/s (variation scale: {q_var_scale:.2f}x)")
    print(f"Output File          : {out_path}")
    print(f"Parallel Workers     : {num_workers}")
    print("-" * 70)

    results_H = []
    results_Q = []
    results_q_up = []
    results_h_down = []
    results_H0 = []
    results_Q0 = []
    results_pattern = []

    # Every candidate id is evaluated and the accepted set is the num_samples
    # lowest ids that converged. The dataset therefore does not depend on worker
    # scheduling, which is what makes the train/val/test split reproducible.
    max_attempts = int(num_samples * 1.5)
    collected: Dict[int, Dict[str, np.ndarray]] = {}
    rejections: List[Tuple[int, str]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                generate_single_sample,
                sample_id=i,
                channel_params=channel_params,
                nx=nx,
                nt=nt,
                dt=dt,
                q_min=q_min,
                q_max=q_max,
                q_var_scale=q_var_scale
            ) for i in range(max_attempts)
        ]

        done = 0
        for future in as_completed(futures):
            sample_id, data, reason = future.result()
            if data is not None:
                collected[sample_id] = data
            else:
                rejections.append((sample_id, reason))
            done += 1
            if done % max(1, max_attempts // 10) == 0 or done == max_attempts:
                print(f"[{done:4d}/{max_attempts}] candidates simulated, {len(collected)} converged...")

    if len(collected) < num_samples:
        raise RuntimeError(
            f"Only {len(collected)} of {max_attempts} candidate trajectories converged; "
            f"{num_samples} were requested. Widen the boundary-condition ranges or raise the attempt budget."
        )

    accepted_ids = sorted(collected)[:num_samples]
    for sample_id in accepted_ids:
        data = collected[sample_id]
        results_H.append(data["H"])
        results_Q.append(data["Q"])
        results_q_up.append(data["q_up"])
        results_h_down.append(data["h_down"])
        results_H0.append(data["H0"])
        results_Q0.append(data["Q0"])
        results_pattern.append(data["pattern"])

    acceptance_rate = 100.0 * len(collected) / max_attempts
    print("-" * 70)
    print(f"Candidate acceptance rate: {acceptance_rate:.1f}% ({len(collected)}/{max_attempts} converged)")
    if rejections:
        reasons = {}
        for _, reason in rejections:
            key = reason.split("(")[0].strip() or "unspecified"
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  rejected x{count:4d}: {reason}")

    # Stack into numpy arrays
    H_arr = np.stack(results_H, axis=0)          # (N, nx, nt)
    Q_arr = np.stack(results_Q, axis=0)          # (N, nx, nt)
    q_up_arr = np.stack(results_q_up, axis=0)    # (N, nt)
    h_down_arr = np.stack(results_h_down, axis=0)# (N, nt)
    H0_arr = np.stack(results_H0, axis=0)        # (N, nx)
    Q0_arr = np.stack(results_Q0, axis=0)        # (N, nx)
    pattern_arr = np.array(results_pattern)      # (N,)

    x_grid = np.linspace(0.0, channel_params.length, nx).astype(np.float32)
    t_grid = (np.arange(nt) * dt).astype(np.float32)

    # Save to compressed NPZ
    np.savez_compressed(
        out_path,
        H=H_arr,
        Q=Q_arr,
        q_up=q_up_arr,
        h_down=h_down_arr,
        H0=H0_arr,
        Q0=Q0_arr,
        pattern=pattern_arr,
        x=x_grid,
        t=t_grid,
        channel_length=channel_params.length,
        channel_width=channel_params.width,
        bed_slope=channel_params.bed_slope,
        manning_n=channel_params.manning_n,
        gravity=channel_params.gravity
    )

    print("-" * 70)
    print(f"Dataset successfully saved to: {out_path}")
    print(f"Array shapes:")
    print(f"  H (water levels)  : {H_arr.shape}")
    print(f"  Q (discharges)    : {Q_arr.shape}")
    print(f"  q_up (upstream)   : {q_up_arr.shape}")
    print(f"  h_down (downstream): {h_down_arr.shape}")
    print(f"  H0 (initial level): {H0_arr.shape}")
    print(f"  Q0 (initial flow) : {Q0_arr.shape}")
    unique, counts = np.unique(pattern_arr, return_counts=True)
    print(f"  Hydrograph mix    : " + ", ".join(f"{u}={c}" for u, c in zip(unique, counts)))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 1D Saint-Venant training data for FNO.")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of simulation trajectories to generate.")
    parser.add_argument("--nx", type=int, default=51, help="Spatial discretization nodes (n_level_nodes).")
    parser.add_argument("--nt", type=int, default=120, help="Number of time steps per trajectory.")
    parser.add_argument("--dt", type=float, default=60.0, help="Time step size in seconds.")
    parser.add_argument("--q-min", type=float, default=2.0, help="Minimum baseline discharge Q (m^3/s).")
    parser.add_argument("--q-max", type=float, default=60.0, help="Maximum baseline discharge Q (m^3/s).")
    parser.add_argument("--q-var-scale", type=float, default=1.0, help="Scaling factor for Q perturbation amplitudes.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory path.")
    parser.add_argument("--output-filename", type=str, default="saint_venant_dataset.npz", help="Output .npz file name.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Complete output .npz path; overrides --output-dir and --output-filename."
    )
    args = parser.parse_args()

    if args.output_path is not None:
        if args.output_path.suffix.lower() != ".npz":
            parser.error("--output-path must end with .npz")
        output_dir = str(args.output_path.parent)
        output_filename = args.output_path.name
    else:
        output_dir = args.output_dir
        output_filename = args.output_filename

    generate_dataset(
        num_samples=args.num_samples,
        nx=args.nx,
        nt=args.nt,
        dt=args.dt,
        q_min=args.q_min,
        q_max=args.q_max,
        q_var_scale=args.q_var_scale,
        num_workers=args.num_workers,
        output_dir=output_dir,
        output_filename=output_filename
    )
