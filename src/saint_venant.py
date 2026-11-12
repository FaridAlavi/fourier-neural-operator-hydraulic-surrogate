"""
1D Saint-Venant (Shallow Water) Equation Solver for Rectangular Open Channels.

Implements the Preissmann 4-point implicit finite difference scheme with
Newton-Raphson iterations and steady-state backwater curve initialization.

The Newton system is block-tridiagonal; it is assembled analytically and solved
with a banded LAPACK routine (kl = ku = 2) in O(nx) work per iteration.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import root_scalar

MIN_DEPTH = 0.05  # Wetted-depth floor (m); below it the state is treated as non-physical.


@dataclass
class ChannelParameters:
    """Physical and geometric parameters of a rectangular open channel."""
    length: float = 10000.0        # Channel length (m)
    width: float = 10.0           # Bottom width B (m)
    bed_slope: float = 0.0002     # Bed slope S_0 (m/m)
    manning_n: float = 0.025      # Manning roughness coefficient (s/m^(1/3))
    h_bed_down: float = 0.0       # Bed elevation at downstream boundary (m)
    gravity: float = 9.81         # Gravitational acceleration (m/s^2)

    def bed_elevation(self, x: np.ndarray) -> np.ndarray:
        """Bed elevation H_b(x) along channel from x=0 (upstream) to x=L (downstream)."""
        # Downstream bed is h_bed_down, upstream bed is h_bed_down + bed_slope * (length - x)
        return self.h_bed_down + self.bed_slope * (self.length - x)


class SaintVenant1D:
    """
    Solves 1D unsteady non-linear Saint-Venant equations for a prismatic rectangular channel.
    
    Equations:
        Mass Conservation:
            B * dH/dt + dQ/dx = q_lat
        Momentum Conservation:
            dQ/dt + d(Q^2/A)/dx + g * A * dH/dx + g * A * S_f = 0
            
    where:
        h(x, t) = H(x, t) - H_b(x)
        A(x, t) = B * h(x, t)
        P(x, t) = B + 2 * h(x, t)
        R(x, t) = A(x, t) / P(x, t)
        S_f(x, t) = (n^2 * |Q| * Q) / (A^2 * R^(4/3))
    """

    def __init__(
        self,
        params: Optional[ChannelParameters] = None,
        nx: int = 51,
        theta: float = 0.60,
        max_iter: int = 25,
        tol: float = 1e-6
    ):
        """
        Parameters
        ----------
        params : ChannelParameters
            Channel geometry and physical constants.
        nx : int
            Number of spatial nodes along the reach (n_level_nodes).
        theta : float
            Preissmann time-weighting parameter (0.5 < theta <= 1.0). Default 0.60.
        max_iter : int
            Max Newton-Raphson iterations per time step.
        tol : float
            Convergence tolerance for Newton-Raphson.
        """
        self.params = params or ChannelParameters()
        self.nx = nx
        self.theta = theta
        self.max_iter = max_iter
        self.tol = tol

        # Spatial discretization
        self.x = np.linspace(0.0, self.params.length, self.nx)
        self.dx = self.params.length / (self.nx - 1)
        self.H_b = self.params.bed_elevation(self.x)

    def compute_steady_state(self, q_initial: float, h_downstream: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute steady-state initial water level H(x) and discharge Q(x)
        by solving the stationary discrete momentum equation from downstream to upstream.

        This is the discrete counterpart of
        dH/dx = (Fr^2 * S0 - Sf) / (1 - Fr^2), including the bed-slope term,
        and is exactly consistent with the Preissmann spatial discretization.
        """
        B = self.params.width
        n = self.params.manning_n
        g = self.params.gravity
        H = np.zeros(self.nx)
        Q = np.full(self.nx, q_initial)

        H[-1] = h_downstream
        h_down = H[-1] - self.H_b[-1]
        if h_down <= MIN_DEPTH:
            raise ValueError(f"Downstream water depth {h_down:.3f} m is too small / unphysical.")

        critical_depth = (q_initial**2 / (g * B**2)) ** (1.0 / 3.0)
        if h_down <= 1.001 * critical_depth:
            raise ValueError("Downstream boundary must be subcritical for a prescribed water level.")

        def hydraulic_values(stage: float, bed_elevation: float) -> Tuple[float, float, float]:
            depth = stage - bed_elevation
            A = B * depth
            P = B + 2.0 * depth
            R = A / P
            FQ = q_initial**2 / A
            Sf = n**2 * abs(q_initial) * q_initial / (A**2 * R**(4.0 / 3.0))
            return A, FQ, Sf

        for i in range(self.nx - 2, -1, -1):
            A_right, FQ_right, Sf_right = hydraulic_values(H[i + 1], self.H_b[i + 1])

            def stationary_residual(stage_left: float) -> float:
                A_left, FQ_left, Sf_left = hydraulic_values(stage_left, self.H_b[i])
                A_bar = 0.5 * (A_left + A_right)
                Sf_bar = 0.5 * (Sf_left + Sf_right)
                return (
                    (FQ_right - FQ_left) / self.dx
                    + g * A_bar * (H[i + 1] - stage_left) / self.dx
                    + g * A_bar * Sf_bar
                )

            lower = self.H_b[i] + max(1.0001 * MIN_DEPTH, 1.001 * critical_depth)
            upper = max(lower + 1.0, H[i + 1] + 2.0)
            lower_residual = stationary_residual(lower)
            upper_residual = stationary_residual(upper)

            for _ in range(20):
                if lower_residual * upper_residual <= 0.0:
                    break
                upper = lower + 2.0 * (upper - lower)
                upper_residual = stationary_residual(upper)
            else:
                raise RuntimeError(f"Could not bracket a subcritical steady-state solution at cell {i}.")

            solution = root_scalar(
                stationary_residual,
                bracket=(lower, upper),
                method="brentq",
                xtol=1e-10
            )
            if not solution.converged:
                raise RuntimeError(f"Steady-state solve failed to converge at cell {i}.")
            H[i] = solution.root

        return H, Q

    def _hydraulics(self, H: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute area A, momentum flux F_Q, and friction slope S_f."""
        B = self.params.width
        n = self.params.manning_n
        depth = np.maximum(H - self.H_b, MIN_DEPTH)
        A = B * depth
        P = B + 2.0 * depth
        R = A / P
        F_Q = (Q**2) / A
        Sf = (n**2 * np.abs(Q) * Q) / (A**2 * (R**(4.0 / 3.0)))
        return A, F_Q, Sf

    def _hydraulics_and_derivatives(self, H: np.ndarray, Q: np.ndarray):
        """
        Nodal hydraulics together with their exact partial derivatives w.r.t. H and Q.

        Derivatives vanish where the depth floor is active, so the Jacobian stays
        consistent with the clamped residual instead of silently disagreeing with it.
        """
        B = self.params.width
        n = self.params.manning_n

        raw_depth = H - self.H_b
        wet = raw_depth > MIN_DEPTH
        depth = np.where(wet, raw_depth, MIN_DEPTH)

        A = B * depth
        P = B + 2.0 * depth
        R = A / P
        R43 = R**(4.0 / 3.0)

        F_Q = (Q**2) / A
        Sf = (n**2 * np.abs(Q) * Q) / (A**2 * R43)

        dA_dH = np.where(wet, B, 0.0)
        dF_dH = -(Q**2) / (A**2) * dA_dH
        dF_dQ = 2.0 * Q / A
        # dSf/dH = Sf * (-2/A * dA/dH - (4/3)/R * dR/dH), with dR/dH = B^2 / P^2
        dSf_dH = np.where(wet, Sf * (-2.0 * B / A - (4.0 / 3.0) * B**2 / (P * A)), 0.0)
        dSf_dQ = (n**2 * 2.0 * np.abs(Q)) / (A**2 * R43)

        return A, F_Q, Sf, dA_dH, dF_dH, dF_dQ, dSf_dH, dSf_dQ

    def step(
        self,
        H_n: np.ndarray,
        Q_n: np.ndarray,
        q_up_next: float,
        h_down_next: float,
        dt: float,
        q_lat: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance solution by one time step dt using the Preissmann implicit scheme
        with Newton-Raphson iteration on the analytically assembled banded Jacobian.
        """
        g = self.params.gravity
        theta = self.theta
        dx = self.dx
        nx = self.nx
        ndof = 2 * nx

        # Frozen contributions from the known time level n
        A_n, FQ_n, Sf_n = self._hydraulics(H_n, Q_n)
        A_sum_n = A_n[1:] + A_n[:-1]
        Q_sum_n = Q_n[1:] + Q_n[:-1]
        dQ_dx_n = (Q_n[1:] - Q_n[:-1]) / dx
        dFQ_dx_n = (FQ_n[1:] - FQ_n[:-1]) / dx
        dH_dx_n = (H_n[1:] - H_n[:-1]) / dx
        A_bar_n = 0.5 * (A_n[1:] + A_n[:-1])
        Sf_bar_n = 0.5 * (Sf_n[1:] + Sf_n[:-1])

        # Initial guess for state at n+1
        H_next = H_n.copy()
        Q_next = Q_n.copy()
        H_next[-1] = h_down_next
        Q_next[0] = q_up_next

        # Column slices for the four unknowns touched by cell i: H_i, Q_i, H_i+1, Q_i+1
        col_H_left = slice(0, ndof - 2, 2)
        col_Q_left = slice(1, ndof - 1, 2)
        col_H_right = slice(2, ndof, 2)
        col_Q_right = slice(3, ndof + 1, 2)

        for iteration in range(self.max_iter + 1):
            (A, F_Q, Sf, dA_dH, dF_dH, dF_dQ, dSf_dH, dSf_dQ) = \
                self._hydraulics_and_derivatives(H_next, Q_next)

            A_bar = theta * 0.5 * (A[1:] + A[:-1]) + (1.0 - theta) * A_bar_n
            Sf_bar = theta * 0.5 * (Sf[1:] + Sf[:-1]) + (1.0 - theta) * Sf_bar_n
            dH_dx = theta * (H_next[1:] - H_next[:-1]) / dx + (1.0 - theta) * dH_dx_n

            residual = np.empty(ndof)
            residual[0] = Q_next[0] - q_up_next
            residual[1:ndof - 2:2] = (
                ((A[1:] + A[:-1]) - A_sum_n) / (2.0 * dt)
                + theta * (Q_next[1:] - Q_next[:-1]) / dx
                + (1.0 - theta) * dQ_dx_n
                - q_lat
            )
            residual[2:ndof - 1:2] = (
                ((Q_next[1:] + Q_next[:-1]) - Q_sum_n) / (2.0 * dt)
                + theta * (F_Q[1:] - F_Q[:-1]) / dx
                + (1.0 - theta) * dFQ_dx_n
                + g * A_bar * dH_dx
                + g * A_bar * Sf_bar
            )
            residual[ndof - 1] = H_next[-1] - h_down_next

            norm_res = np.linalg.norm(residual, ord=np.inf)
            if norm_res < self.tol:
                break
            if iteration == self.max_iter:
                raise RuntimeError(
                    f"Newton-Raphson failed to converge after {self.max_iter} iterations "
                    f"(residual infinity norm={norm_res:.3e}, tolerance={self.tol:.3e})."
                )

            # Banded Jacobian in LAPACK storage: ab[ku + r - c, c] = J[r, c], kl = ku = 2
            ab = np.zeros((5, ndof))
            ab[1, 1] = 1.0            # upstream discharge BC
            ab[3, ndof - 2] = 1.0     # downstream stage BC

            inv_2dt = 1.0 / (2.0 * dt)
            theta_dx = theta / dx

            ab[3, col_H_left] = dA_dH[:-1] * inv_2dt
            ab[2, col_Q_left] = -theta_dx
            ab[1, col_H_right] = dA_dH[1:] * inv_2dt
            ab[0, col_Q_right] = theta_dx

            dAbar_dH_left = theta * 0.5 * dA_dH[:-1]
            dAbar_dH_right = theta * 0.5 * dA_dH[1:]
            gA_theta_half = g * A_bar * theta * 0.5
            transport = dH_dx + Sf_bar

            ab[4, col_H_left] = (
                -theta_dx * dF_dH[:-1]
                + g * dAbar_dH_left * transport
                - g * A_bar * theta_dx
                + gA_theta_half * dSf_dH[:-1]
            )
            ab[3, col_Q_left] = inv_2dt - theta_dx * dF_dQ[:-1] + gA_theta_half * dSf_dQ[:-1]
            ab[2, col_H_right] = (
                theta_dx * dF_dH[1:]
                + g * dAbar_dH_right * transport
                + g * A_bar * theta_dx
                + gA_theta_half * dSf_dH[1:]
            )
            ab[1, col_Q_right] = inv_2dt + theta_dx * dF_dQ[1:] + gA_theta_half * dSf_dQ[1:]

            try:
                delta = solve_banded((2, 2), ab, -residual, check_finite=False)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise RuntimeError(f"Banded Newton solve failed: {exc}") from exc

            delta_H = delta[0::2]
            delta_Q = delta[1::2]

            # Backtrack until the update stays in the wet, physically admissible region.
            damping = 1.0
            for _ in range(12):
                H_trial = H_next + damping * delta_H
                if np.isfinite(H_trial).all() and np.min(H_trial - self.H_b) > MIN_DEPTH:
                    break
                damping *= 0.5
            else:
                raise RuntimeError("Newton-Raphson could not find an admissible damped update.")

            H_next = H_next + damping * delta_H
            Q_next = Q_next + damping * delta_Q

        if not np.isfinite(H_next).all() or not np.isfinite(Q_next).all():
            raise RuntimeError("Newton-Raphson produced non-finite state values.")
        min_depth = float(np.min(H_next - self.H_b))
        if min_depth <= MIN_DEPTH:
            raise RuntimeError(f"Newton-Raphson produced an invalid minimum depth of {min_depth:.3e} m.")

        return H_next, Q_next

    def simulate(
        self,
        q_up_series: np.ndarray,
        h_down_series: np.ndarray,
        dt: float,
        H_init: Optional[np.ndarray] = None,
        Q_init: Optional[np.ndarray] = None,
        q_lat_series: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate full trajectory over time.

        Parameters
        ----------
        q_up_series : np.ndarray
            Upstream discharge hydrograph of shape (nt,).
        h_down_series : np.ndarray
            Downstream stage hydrograph of shape (nt,).
        dt : float
            Time step duration in seconds.
        H_init : np.ndarray, optional
            Initial water level profile of shape (nx,). If None, computed from steady state.
        Q_init : np.ndarray, optional
            Initial discharge profile of shape (nx,). If None, computed from steady state.
        q_lat_series : np.ndarray, optional
            Lateral inflow series of shape (nt,).

        Returns
        -------
        H_traj : np.ndarray
            Water levels of shape (nx, nt).
        Q_traj : np.ndarray
            Discharges of shape (nx, nt).
        t_grid : np.ndarray
            Time array of shape (nt,).
        """
        nt = len(q_up_series)
        t_grid = np.arange(nt) * dt

        if H_init is None or Q_init is None:
            H_curr, Q_curr = self.compute_steady_state(q_up_series[0], h_down_series[0])
        else:
            H_curr, Q_curr = H_init.copy(), Q_init.copy()

        H_traj = np.zeros((self.nx, nt))
        Q_traj = np.zeros((self.nx, nt))

        H_traj[:, 0] = H_curr
        Q_traj[:, 0] = Q_curr

        for step_idx in range(1, nt):
            q_lat_val = q_lat_series[step_idx] if q_lat_series is not None else 0.0
            try:
                H_next, Q_next = self.step(
                    H_n=H_curr,
                    Q_n=Q_curr,
                    q_up_next=q_up_series[step_idx],
                    h_down_next=h_down_series[step_idx],
                    dt=dt,
                    q_lat=q_lat_val
                )
            except RuntimeError as exc:
                raise RuntimeError(f"Simulation failed at time step {step_idx}: {exc}") from exc
            H_traj[:, step_idx] = H_next
            Q_traj[:, step_idx] = Q_next
            H_curr, Q_curr = H_next, Q_next

        return H_traj, Q_traj, t_grid
