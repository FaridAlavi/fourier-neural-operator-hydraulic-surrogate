"""
GIF Animation Generator for Dynamic Inflow Case Study.

Creates an animated GIF visualizing:
  1. Top Panel: Spatial Water Surface Profile H(x, t) along the channel reach (Bed, PDE, FNO, Water Column).
  2. Middle Panel: Spatial Discharge Profile Q(x, t) along the reach showing wave propagation & attenuation.
  3. Bottom Panel: Dynamic Boundary Inflow Q(0, t) vs. Outflow Q(L, t) with moving time cursor & real-time metrics.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Add project root and src to python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from saint_venant import ChannelParameters
from run_dynamic_inflow import (
    resolve_checkpoint_path,
    build_dynamic_boundary_conditions,
    run_pde_simulation,
    run_fno_inference,
)


def run_simulation(
    checkpoint_path: str,
    nx: int = 51,
    nt: int = 120,
    dt: float = 60.0,
    base_flow: float = 12.0,
    h_down_val: float = 3.0,
    device: str = None
):
    """Runs the PDE and FNO dynamic inflow simulation."""
    channel_params = ChannelParameters(
        length=10000.0,
        width=10.0,
        bed_slope=0.0002,
        manning_n=0.025,
        h_bed_down=0.0
    )

    print("[Running PDE and FNO dynamic simulation...]")
    resolved_ckpt = resolve_checkpoint_path(checkpoint_path)
    x_grid, t_grid, q_up, h_down = build_dynamic_boundary_conditions(
        nx=nx, nt=nt, dt=dt,
        channel_length=channel_params.length,
        base_flow=base_flow, h_down_val=h_down_val
    )

    H_pde, Q_pde, H0_profile, _ = run_pde_simulation(
        channel_params=channel_params, nx=nx, dt=dt, q_up=q_up, h_down=h_down
    )
    H0_fno = H0_profile
    Q0_fno = np.full_like(H0_profile, float(q_up[0]))

    H_fno, Q_fno, _, _ = run_fno_inference(
        checkpoint_path=resolved_ckpt,
        H0=H0_fno, Q0=Q0_fno,
        q_up=q_up, h_down=h_down,
        x_grid=x_grid, t_grid=t_grid,
        channel_length=channel_params.length,
        force_device=device
    )

    return H_pde, Q_pde, H_fno, Q_fno, x_grid, t_grid, q_up, h_down, channel_params


def render_dynamic_gif(
    H_pde: np.ndarray,
    Q_pde: np.ndarray,
    H_fno: np.ndarray,
    Q_fno: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    q_up: np.ndarray,
    h_down: np.ndarray,
    channel_params: ChannelParameters,
    output_filepath: str = "case_studies/dynamic_inflow/dynamic_inflow_comparison.gif",
    fps: int = 15,
    dpi: int = 120
):
    """Generates a GIF animation comparing PDE and FNO under dynamic inflow."""
    nx, nt = H_pde.shape
    x_km = x_grid / 1000.0
    t_min = t_grid / 60.0

    # Channel Bed Elevation: H_b(x) = h_bed_down + S0 * (L - x)
    H_b = channel_params.h_bed_down + channel_params.bed_slope * (channel_params.length - x_grid)

    # 3-panel figure layout
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(14, 10), height_ratios=[1.3, 1.0, 1.0], constrained_layout=True
    )

    # --- 1. Top Panel: Channel Water Surface Profile ---
    y_min_bed = float(np.min(H_b)) - 0.5
    y_max_water = float(max(np.max(H_pde), np.max(H_fno))) + 0.6

    ax_top.fill_between(x_km, y_min_bed, H_b, color="#8B5A2B", alpha=0.5, label="Channel Bed $H_b(x)$")
    ax_top.plot(x_km, H_b, "k-", lw=2)

    line_water_pde, = ax_top.plot([], [], color="#1f77b4", lw=2.5, label="Saint-Venant PDE $H_{\\mathrm{PDE}}(x, t)$")
    line_water_fno, = ax_top.plot([], [], color="#d62728", linestyle="--", lw=2.0, label="FNO Surrogate $H_{\\mathrm{FNO}}(x, t)$")
    fill_water = [None]

    ax_top.set_xlim(0, x_km[-1])
    ax_top.set_ylim(y_min_bed, y_max_water)
    ax_top.set_title("Dynamic Wave Propagation: Water Surface Profile $H(x, t)$ along 10 km Reach", fontsize=12, fontweight="bold")
    ax_top.set_xlabel("Distance along Channel [km]", fontsize=10)
    ax_top.set_ylabel("Elevation [m a.s.l.]", fontsize=10)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right", framealpha=0.9)

    time_text = ax_top.text(
        0.02, 0.86, "", transform=ax_top.transAxes,
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
    )

    # --- 2. Middle Panel: Spatial Discharge Profile along Reach ---
    q_all_max = float(max(np.max(Q_pde), np.max(Q_fno), np.max(q_up))) * 1.15
    line_q_reach_pde, = ax_mid.plot([], [], color="#1f77b4", lw=2.5, label="PDE $Q_{\\mathrm{PDE}}(x, t)$")
    line_q_reach_fno, = ax_mid.plot([], [], color="#d62728", linestyle="--", lw=2.0, label="FNO $Q_{\\mathrm{FNO}}(x, t)$")

    ax_mid.set_xlim(0, x_km[-1])
    ax_mid.set_ylim(0, q_all_max)
    ax_mid.set_title("Spatial Discharge Profile $Q(x, t)$ across Channel Reach", fontsize=11, fontweight="bold")
    ax_mid.set_xlabel("Distance along Channel [km]", fontsize=10)
    ax_mid.set_ylabel("Discharge $Q$ [m$^3$/s]", fontsize=10)
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(loc="upper right", framealpha=0.9)

    # --- 3. Bottom Panel: Hydrographs & Time Tracking ---
    ax_bot.plot(t_min, q_up, "k:", lw=2.0, label="Dynamic Upstream Inflow $Q(0, t)$")
    line_q_out_pde, = ax_bot.plot([], [], color="#1f77b4", lw=2.5, label="PDE Outflow $Q(L, t)$")
    line_q_out_fno, = ax_bot.plot([], [], color="#d62728", linestyle="--", lw=2.0, label="FNO Outflow $Q(L, t)$")
    time_cursor = ax_bot.axvline(0, color="#2c3e50", linestyle="-", lw=1.8, alpha=0.85)

    ax_bot.set_xlim(0, t_min[-1])
    ax_bot.set_ylim(0, q_all_max)
    ax_bot.set_title("Boundary Hydrographs: Inflow vs. Attenuated Outflow", fontsize=11, fontweight="bold")
    ax_bot.set_xlabel("Time [minutes]", fontsize=10)
    ax_bot.set_ylabel("Discharge $Q$ [m$^3$/s]", fontsize=10)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="upper left", fontsize=9, framealpha=0.9)

    metric_text = ax_bot.text(
        0.58, 0.65, "", transform=ax_bot.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fdfefe", ec="#bdc3c7", alpha=0.92)
    )

    def init():
        line_water_pde.set_data([], [])
        line_water_fno.set_data([], [])
        line_q_reach_pde.set_data([], [])
        line_q_reach_fno.set_data([], [])
        line_q_out_pde.set_data([], [])
        line_q_out_fno.set_data([], [])
        time_text.set_text("")
        metric_text.set_text("")
        return (line_water_pde, line_water_fno, line_q_reach_pde,
                line_q_reach_fno, line_q_out_pde, line_q_out_fno)

    def update(frame):
        t_curr = t_min[frame]
        h_pde_curr = H_pde[:, frame]
        h_fno_curr = H_fno[:, frame]
        q_pde_curr = Q_pde[:, frame]
        q_fno_curr = Q_fno[:, frame]

        # 1. Update Top Panel
        line_water_pde.set_data(x_km, h_pde_curr)
        line_water_fno.set_data(x_km, h_fno_curr)

        if fill_water[0] is not None:
            fill_water[0].remove()
        fill_water[0] = ax_top.fill_between(x_km, H_b, h_pde_curr, color="#1f77b4", alpha=0.25)

        time_text.set_text(
            f"Simulation Time: {t_curr:5.1f} min\n"
            f"Inflow $Q_{{in}}(t)$: {q_up[frame]:.1f} m$^3$/s"
        )

        # 2. Update Middle Panel
        line_q_reach_pde.set_data(x_km, q_pde_curr)
        line_q_reach_fno.set_data(x_km, q_fno_curr)

        # 3. Update Bottom Panel
        line_q_out_pde.set_data(t_min[:frame + 1], Q_pde[-1, :frame + 1])
        line_q_out_fno.set_data(t_min[:frame + 1], Q_fno[-1, :frame + 1])
        time_cursor.set_xdata([t_curr, t_curr])

        rmse_h_cm = float(np.sqrt(np.mean((h_fno_curr - h_pde_curr) ** 2))) * 100.0
        rmse_q = float(np.sqrt(np.mean((q_fno_curr - q_pde_curr) ** 2)))
        q_out_pde = float(Q_pde[-1, frame])
        q_out_fno = float(Q_fno[-1, frame])

        metric_text.set_text(
            f"Instantaneous Metrics (t = {t_curr:.1f} min):\n"
            f"  Reach Water Level RMSE(H) : {rmse_h_cm:4.2f} cm\n"
            f"  Reach Discharge RMSE(Q)   : {rmse_q:4.2f} m$^3$/s\n"
            f"  Outflow PDE $Q(L, t)$     : {q_out_pde:4.2f} m$^3$/s\n"
            f"  Outflow FNO $Q(L, t)$     : {q_out_fno:4.2f} m$^3$/s\n"
            f"  Outflow Error             : {abs(q_out_fno - q_out_pde):4.2f} m$^3$/s"
        )

        return (line_water_pde, line_water_fno, line_q_reach_pde,
                line_q_reach_fno, line_q_out_pde, line_q_out_fno,
                time_text, metric_text)

    anim = FuncAnimation(fig, update, frames=nt, init_func=init, blit=False, interval=1000 / fps)

    out_path = Path(output_filepath).with_suffix(".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Rendering GIF ({nt} frames @ {fps} fps)...]")
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    print(f"[SUCCESS] Animation successfully saved as GIF: {out_path.resolve()}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate GIF animation of dynamic inflow case study.")
    parser.add_argument("--checkpoint-path", type=str, default="models/fno_saint_venant.pt")
    parser.add_argument("--output-dir", type=str, default="case_studies/dynamic_inflow")
    parser.add_argument("--filename", type=str, default="dynamic_inflow_comparison.gif")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    case_dir = Path(args.output_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    out_file = case_dir / args.filename

    print("=" * 75)
    print("  CREATING GIF ANIMATION FOR DYNAMIC INFLOW CASE STUDY")
    print("=" * 75)

    (
        H_pde, Q_pde, H_fno, Q_fno,
        x_grid, t_grid, q_up, h_down,
        channel_params
    ) = run_simulation(
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    render_dynamic_gif(
        H_pde=H_pde,
        Q_pde=Q_pde,
        H_fno=H_fno,
        Q_fno=Q_fno,
        x_grid=x_grid,
        t_grid=t_grid,
        q_up=q_up,
        h_down=h_down,
        channel_params=channel_params,
        output_filepath=str(out_file),
        fps=args.fps,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
