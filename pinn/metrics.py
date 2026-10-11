"""
Error metrics used throughout the report.

Relative L2 error is the standard accuracy number in the PINN literature
(Raissi et al. report it for every example). It is scale-free:
1e-2 means "about 1% wrong" whatever the units of u.
Max absolute error is reported alongside it because a small average can hide
a large local error, which is exactly what happens near the boundary in the
heat problem and near the origin in the consolidation problem.
"""
from __future__ import annotations

import numpy as np
import torch


def _to_numpy(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def relative_l2(pred, ref) -> float:
    """||pred - ref||_2 / ||ref||_2 over all points."""
    p, r = _to_numpy(pred).ravel(), _to_numpy(ref).ravel()
    return float(np.linalg.norm(p - r) / np.linalg.norm(r))


def max_abs_error(pred, ref) -> float:
    """Largest point-wise |pred - ref|."""
    p, r = _to_numpy(pred).ravel(), _to_numpy(ref).ravel()
    return float(np.max(np.abs(p - r)))


def relative_error(estimate: float, true: float) -> float:
    """|estimate - true| / |true|, for scalar parameters such as c and k."""
    return abs(float(estimate) - float(true)) / abs(float(true))
