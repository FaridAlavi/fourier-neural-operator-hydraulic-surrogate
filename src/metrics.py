"""
Single shared definition of the accuracy metrics reported for the FNO surrogate.

All relative L2 errors are computed in *physical* units and normalised by the
**dynamic** part of the reference signal, i.e. its deviation from the initial
profile. The water surface elevation H carries a several-metre datum offset; if
the raw field norm is used as the denominator, that offset dominates and deflates
the reported relative error by roughly an order of magnitude. Normalising by the
dynamics measures what the surrogate is actually asked to predict.

Every reporting script in this repository imports from here, so a single number
definition is used across evaluation, benchmarks and case studies. The relative
L2 loss used for *training* (``train.RelativeL2Loss``) operates in normalised
space and is an optimisation objective, not a reported accuracy figure.
"""

from typing import Dict

import numpy as np


def _as_batch(field: np.ndarray) -> np.ndarray:
    """Promote an (nx, nt) field to a (1, nx, nt) batch."""
    array = np.asarray(field, dtype=np.float64)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected a field of shape (nx, nt) or (N, nx, nt), got {array.shape}.")


def relative_l2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    """
    Relative L2 error of a single space-time field, normalised by the reference
    dynamics ``true - true[:, 0]`` rather than by the raw reference field.
    """
    pred_b = _as_batch(pred)
    true_b = _as_batch(true)
    dynamics = true_b - true_b[:, :, :1]
    return float(np.linalg.norm(pred_b - true_b) / max(float(np.linalg.norm(dynamics)), eps))


def field_metrics(
    H_pred: np.ndarray,
    Q_pred: np.ndarray,
    H_true: np.ndarray,
    Q_true: np.ndarray
) -> Dict[str, float]:
    """
    Accuracy of a predicted (H, Q) trajectory set against a reference solution.

    Accepts either a single sample of shape (nx, nt) or a batch (N, nx, nt).
    Relative L2 errors are averaged per sample; RMSE/MAE/max are pooled over all
    points so they remain directly interpretable in metres and m^3/s.
    """
    H_pred_b, Q_pred_b = _as_batch(H_pred), _as_batch(Q_pred)
    H_true_b, Q_true_b = _as_batch(H_true), _as_batch(Q_true)

    rel_h = [relative_l2(H_pred_b[i], H_true_b[i]) for i in range(H_true_b.shape[0])]
    rel_q = [relative_l2(Q_pred_b[i], Q_true_b[i]) for i in range(Q_true_b.shape[0])]

    h_diff = H_pred_b - H_true_b
    q_diff = Q_pred_b - Q_true_b

    return {
        "rel_l2_H": float(np.mean(rel_h)),
        "rel_l2_Q": float(np.mean(rel_q)),
        "rel_l2_overall": float(0.5 * (np.mean(rel_h) + np.mean(rel_q))),
        "rmse_H_m": float(np.sqrt(np.mean(h_diff**2))),
        "mae_H_m": float(np.mean(np.abs(h_diff))),
        "max_abs_H_m": float(np.max(np.abs(h_diff))),
        "rmse_Q_m3s": float(np.sqrt(np.mean(q_diff**2))),
        "mae_Q_m3s": float(np.mean(np.abs(q_diff))),
        "max_abs_Q_m3s": float(np.max(np.abs(q_diff))),
        "n_samples": int(H_true_b.shape[0]),
    }


def format_metrics(metrics: Dict[str, float], indent: str = "  ") -> str:
    """Human-readable block used by the evaluation and benchmark scripts."""
    return "\n".join([
        f"{indent}Relative L2 (H, dynamics-normalised) : {metrics['rel_l2_H']:.4e} ({metrics['rel_l2_H'] * 100:.2f}%)",
        f"{indent}Relative L2 (Q, dynamics-normalised) : {metrics['rel_l2_Q']:.4e} ({metrics['rel_l2_Q'] * 100:.2f}%)",
        f"{indent}Relative L2 (overall)                : {metrics['rel_l2_overall']:.4e} ({metrics['rel_l2_overall'] * 100:.2f}%)",
        f"{indent}Water level  RMSE / MAE / max (H)    : {metrics['rmse_H_m']:.4f} / {metrics['mae_H_m']:.4f} / {metrics['max_abs_H_m']:.4f} m",
        f"{indent}Discharge    RMSE / MAE / max (Q)    : {metrics['rmse_Q_m3s']:.4f} / {metrics['mae_Q_m3s']:.4f} / {metrics['max_abs_Q_m3s']:.4f} m^3/s",
    ])
