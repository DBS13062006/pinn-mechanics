"""
Paper section 4.1: one-dimensional (Terzaghi) consolidation, a forward problem.

Dimensionless form (paper Eq. 4.4), with x = depth / layer thickness and
t = c t_phys / L^2:

    p_t - p_xx = 0          0 < x < 1,  0 < t <= 0.5
    p = 0                   x = 0       (drained top: water escapes here)
    p_x = 0                 x = 1       (impermeable base: no flow)
    p = 1                   t = 0       (load applied: pore pressure jumps to 1)

Physical picture: a load squeezes a saturated soil layer, the water pressure
jumps to 1 everywhere, then bleeds out through the drained top. Near the top
the pressure must drop from 1 to 0 instantly at t = 0+, so at the corner
(x = 0, t = 0) the solution has an infinitely steep gradient. That corner is
where the paper's PINN makes its largest error.

The paper's network setup is reproduced exactly:
  * output multiplied by x, so p = 0 at x = 0 holds exactly ("hard" BC),
  * 500 collocation points, 250 points on the x boundary, 125 on t = 0,
  * one boundary loss averaged over all 375 boundary points, weight 1/4.

Reference solution: two independent formulas are implemented and checked
against each other (tests/test_consolidation.py):
  * the paper's Fourier series (Eq. 4.5), accurate for larger t but needing
    very many terms near t = 0, and
  * the method-of-images (erfc) solution, which converges in a handful of
    terms everywhere on this domain and is exact near the corner.
The images formula is used as the reference.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import erfc as np_erfc

from pinn.derivatives import grad

T_MAX = 0.5
N_COLLOCATION = 500
N_BOUNDARY_X = 250   # placed on x = 1 (x = 0 is already exact through the output transform)
N_INITIAL = 125
BANDS = ((0.0, 0.01), (0.01, 0.05), (0.05, T_MAX + 1e-12))   # time bands for error diagnostics


# ------------------------------------------------------------ exact solutions
def exact_series(x, t, n_terms: int = 2000):
    """Paper Eq. (4.5): sum over odd m of 4/(m pi) sin(m pi x / 2) exp(-m^2 pi^2 t / 4).

    numpy only. Converges slowly for small t (and not at all at t = 0, where
    it is the Fourier series of a step): used only to cross-check exact().
    """
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    out = np.zeros_like(x)
    for j in range(n_terms):
        m = 2 * j + 1
        out += 4.0 / (m * math.pi) * np.sin(m * math.pi * x / 2) * np.exp(-(m * math.pi) ** 2 * t / 4)
    return out


def exact(x, t, n_images: int = 20):
    """Method-of-images solution. Works on numpy arrays or torch tensors.

    With q = 1 - p, q = 1 at x = 0, q_x = 0 at x = 1, q = 0 at t = 0, and
        q = sum_{n>=0} (-1)^n [ erfc((2n + x) / (2 sqrt t)) + erfc((2n + 2 - x) / (2 sqrt t)) ].
    The first term, erfc(x / (2 sqrt t)), is the classic solution for a
    half-space suddenly drained at its surface; the others are mirror images
    that enforce the no-flow base at x = 1. For t <= 0.5 the terms shrink
    like erfc(n / 0.7), so 20 terms are far more than enough.
    At t = 0 the limit is used: p = 1 for x > 0 and p = 0 at x = 0.
    """
    if isinstance(t, torch.Tensor):
        erfc, sqrt, where, lib = torch.special.erfc, torch.sqrt, torch.where, torch
    else:
        x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
        erfc, sqrt, where, lib = np_erfc, np.sqrt, np.where, np
    positive = t > 0
    safe_t = where(positive, t, lib.ones_like(t))     # avoid dividing by zero; masked below
    s = 2.0 * sqrt(safe_t)
    q = lib.zeros_like(x * safe_t)
    for n in range(n_images):
        q = q + (-1) ** n * (erfc((2 * n + x) / s) + erfc((2 * n + 2 - x) / s))
    p_t0 = where(x > 0, lib.ones_like(q), lib.zeros_like(q))
    return where(positive, 1.0 - q, p_t0)


# ------------------------------------------------------------ PINN pieces
def pressure(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Paper's output transform: p = x * network(x, t), so p(0, t) = 0 exactly."""
    return x * model(x, t)


def half_space(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """erf(x / (2 sqrt t)): the exact pressure in a layer with NO base (a half-space)
    that is suddenly drained at x = 0. It satisfies the PDE, p = 0 at x = 0 and
    p = 1 at t = 0, and it contains the corner jump exactly. Near the corner the
    real layer cannot yet "feel" its base at x = 1, so this is the true local
    behaviour there.
    """
    t_safe = t.clamp_min(1e-30)          # t = 0 gives erf(+inf) = 1 for x > 0, erf(0) = 0 at x = 0
    return torch.special.erf(x / (2.0 * torch.sqrt(t_safe)))


def pressure_enriched(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Corner-enriched output: p = erf(x / 2 sqrt t) + x * t * network(x, t).

    The erf term carries the jump; the x * t factor keeps p = 0 at x = 0 and
    p = 1 at t = 0 exact for ANY network, so the network only has to learn a
    smooth correction (the effect of the impermeable base) and the loss has
    just two terms: the PDE and the no-flow condition at x = 1.
    """
    return half_space(x, t) + x * t * model(x, t)


def pde_residual(model, x: torch.Tensor, t: torch.Tensor, output=pressure) -> torch.Tensor:
    """p_t - p_xx at (x, t). Zero wherever the consolidation equation holds."""
    p = output(model, x, t)
    p_t = grad(p, t)
    p_xx = grad(grad(p, x), x)
    return p_t - p_xx


def neumann_residual(model, t: torch.Tensor, output=pressure) -> torch.Tensor:
    """p_x at x = 1 (should be 0: impermeable base)."""
    x = torch.ones_like(t).requires_grad_(True)
    return grad(output(model, x, t), x)


def initial_residual(model, x: torch.Tensor, output=pressure) -> torch.Tensor:
    """p(x, 0) - 1 (should be 0)."""
    return output(model, x, torch.zeros_like(x)) - 1.0


# ------------------------------------------------------------ point sets
def collocation_points(n: int = N_COLLOCATION, device=None):
    """Uniform random points in (0, 1) x (0, T_MAX), as in the paper."""
    x = torch.rand(n, 1, device=device).requires_grad_(True)
    t = (T_MAX * torch.rand(n, 1, device=device)).requires_grad_(True)
    return x, t


def boundary_times(n: int = N_BOUNDARY_X, device=None) -> torch.Tensor:
    return T_MAX * torch.rand(n, 1, device=device)


def initial_positions(n: int = N_INITIAL, device=None) -> torch.Tensor:
    return torch.rand(n, 1, device=device)


def test_grid(nx: int = 201, nt: int = 201, t_max: float = T_MAX, device=None):
    """Regular (x, t) grid, flattened to columns; also returns the grid shape."""
    x = torch.linspace(0, 1, nx, device=device)
    t = torch.linspace(0, t_max, nt, device=device)
    T, X = torch.meshgrid(t, x, indexing="ij")          # rows = time, columns = depth
    return X.reshape(-1, 1), T.reshape(-1, 1), (nt, nx)


# ------------------------------------------------------------ diagnostics
def error_by_band(err: np.ndarray, t: np.ndarray) -> list[dict]:
    """How the error is spread over time: max error and share of squared error per band."""
    err, t = np.abs(np.ravel(err)), np.ravel(t)
    total = float(np.sum(err**2))
    out = []
    for a, b in BANDS:
        m = (t >= a) & (t < b)
        out.append({"band": f"[{a:g}, {min(b, T_MAX):g}]",
                    "max_err": float(err[m].max()) if m.any() else 0.0,
                    "share_sq_err": float(np.sum(err[m] ** 2) / total) if total > 0 else 0.0,
                    "share_of_domain": (min(b, T_MAX) - a) / T_MAX})
    return out
