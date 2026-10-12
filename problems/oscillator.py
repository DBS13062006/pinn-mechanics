"""
Paper section 4.3: find the damping c and stiffness k of a vibrating mass.

    m u'' + c u' + k u = 0,     u(0) = u0,  u'(0) = v0,     t in [0, 15]

m = u0 = v0 = 1 are known. c and k are unknown and must be found from 50
measurements of the displacement u(t). The measurements are generated from
the exact solution with the true values c = 0.4, k = 4 (paper Eq. 4.9).

Physical meaning:
  k (stiffness) sets how fast it oscillates: natural frequency wn = sqrt(k/m).
  c (damping)   sets how fast the oscillation dies out: the envelope decays
                like exp(-c t / (2m)).
So k is read from the PERIOD of the wiggles and c from how fast they SHRINK.

Everything in this module follows the paper's setup exactly. Settings the
paper does not state (time window, starting guesses) were read off its
Figure 4: t runs from 0 to 15, and c and k start at 0.2 and 2.0.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from pinn.derivatives import grad

# Known quantities
M = 1.0
U0 = 1.0
V0 = 1.0
T_MAX = 15.0

# Ground truth (used only to make the data and to score the result)
C_TRUE = 0.4
K_TRUE = 4.0

# Starting guesses, read off the paper's Figure 4: half the true values.
C_START = 0.2
K_START = 2.0


def exact(t, c: float = C_TRUE, k: float = K_TRUE, m: float = M,
          u0: float = U0, v0: float = V0):
    """Paper Eq. (4.9): underdamped free vibration. Works on numpy or torch."""
    lib = torch if isinstance(t, torch.Tensor) else np
    wn = math.sqrt(k / m)                 # natural frequency
    xi = c / (2.0 * m * wn)               # damping ratio, < 1 for underdamped
    if xi >= 1.0:
        raise ValueError(f"damping ratio {xi:.3f} >= 1: not underdamped")
    wd = wn * math.sqrt(1.0 - xi**2)      # damped frequency
    return lib.exp(-xi * wn * t) * (u0 * lib.cos(wd * t)
                                    + (v0 + xi * wn * u0) / wd * lib.sin(wd * t))


def residual(model, t: torch.Tensor, c, k, m: float = M) -> torch.Tensor:
    """m u'' + c u' + k u at times t (each (N, 1), requires_grad). Zero if the physics holds."""
    u = model(t)
    u_t = grad(u, t)
    u_tt = grad(u_t, t)
    return m * u_tt + c * u_t + k * u


def initial_condition_loss(model, device) -> torch.Tensor:
    """Paper Eq. (3.5) for the two initial conditions:
    (u(0) - u0)^2 + (u'(0) - v0)^2.

    The paper lists "50 random points" for this boundary set. For an ODE the
    only boundary is t = 0, so all 50 would be the same point and give the
    same value; we evaluate it once.
    """
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)
    u = model(t0)
    v = grad(u, t0)
    return ((u - U0) ** 2 + (v - V0) ** 2).mean()


def observation_times(n: int = 50, device=None) -> torch.Tensor:
    """n equally spaced measurement times on [0, T_MAX] (paper: 50)."""
    return torch.linspace(0.0, T_MAX, n, device=device).reshape(-1, 1)


def collocation_times(n: int = 100, device=None) -> torch.Tensor:
    """n random times where the ODE is enforced (paper: 100)."""
    return (T_MAX * torch.rand(n, 1, device=device)).requires_grad_(True)


def test_times(n: int = 10_000, device=None) -> torch.Tensor:
    """Dense grid for measuring error (paper: 10000 test points)."""
    return torch.linspace(0.0, T_MAX, n, device=device).reshape(-1, 1)


# ---------------------------------------------------------------- diagnostics
def implied_parameters(model, t: torch.Tensor, m: float = M) -> tuple[float, float]:
    """The (c*, k*) that best explain the network's CURRENT curve.

    The ODE  m u'' + c u' + k u = 0  is linear in c and k. So for any curve
    u(t), the c and k that make it "most like" a solution are found by
    ordinary least squares:   [u'  u] [c k]^T  ~=  -m u''.
    Comparing these with the trainable c and k tells us whether the
    parameters are just following the network's fit (they match) or are
    lagging behind it (they don't).

    t: (N, 1) times. Gradients are needed, so call this with grad enabled.
    """
    t = t.detach().clone().requires_grad_(True)
    u = model(t)
    u_t = grad(u, t)
    u_tt = grad(u_t, t)
    A = torch.cat([u_t, u], dim=1).detach()
    b = (-m * u_tt).detach()
    # 2x2 normal equations (A^T A) x = A^T b
    x = torch.linalg.solve(A.T @ A, A.T @ b)
    return x[0].item(), x[1].item()


WINDOWS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0))


# ---------------------------------------------------------------- Step 8: imperfect data
# Noise standard deviation as a fraction of the initial displacement u0 = 1.
# For scale: the true oscillation has decayed to exp(-c T_MAX / 2m) = 5% of its
# starting size by t = 15, so 3-10% noise buries the late tail, which is where
# the damping c is visible.
NOISE_LEVELS = (0.0, 0.01, 0.03, 0.10)
# name -> (number of measurements, last measurement time). Measurements are
# equally spaced on [0, t_end]; the ODE is still enforced on all of [0, T_MAX].
DATA_SETTINGS = {
    "full50": (50, T_MAX),          # the paper's setting
    "full15": (15, T_MAX),          # sparse: fewer than 3 points per oscillation
    "early50": (50, T_MAX / 3),     # short record: only the first third is measured
}


def noisy_observations(setting: str, noise: float, draw: int, device=None):
    """Measurement times and noisy displacements for one data setting.

    The noise is Gaussian with standard deviation noise * u0. `draw` picks the
    random numbers; the same draw at different noise levels uses the SAME
    standard-normal numbers, only scaled, so curves over noise level are smooth
    and differences between methods are not luck of the draw.
    """
    n, t_end = DATA_SETTINGS[setting]
    t = torch.linspace(0.0, t_end, n, device=device).reshape(-1, 1)
    gen = torch.Generator().manual_seed(10_000 + draw)
    eps = torch.randn(n, 1, generator=gen).to(device)
    return t, exact(t) + noise * U0 * eps


def solve_numerically(t, c: float, k: float, m: float = M, u0: float = U0, v0: float = V0) -> np.ndarray:
    """u(t) by a standard ODE solver (no closed form needed). t: 1D numpy array, sorted."""
    from scipy.integrate import solve_ivp
    t = np.asarray(t, float)
    sol = solve_ivp(lambda _, y: [y[1], -(c * y[1] + k * y[0]) / m], (0.0, float(t[-1])), [u0, v0],
                    t_eval=t, method="DOP853", rtol=1e-10, atol=1e-12)
    return sol.y[0]


def classical_fit(t, u, c0: float = C_START, k0: float = K_START) -> dict:
    """Classical baseline: ODE-constrained least squares.

    Adjust c and k until the numerical ODE solution passes as close as possible
    to the measurements (scipy least_squares, from the same starting guess as
    the PINN). It uses exactly the information the PINN has: the ODE, the
    initial conditions and the data; it does not use the closed-form solution.

    Also returns the textbook standard errors, sigma^2 (J^T J)^-1, with sigma
    estimated from the leftover misfit (used for uncertainty in Step 8b).
    """
    from scipy.optimize import least_squares
    t = np.asarray(t, float).ravel()
    u = np.asarray(u, float).ravel()
    fit = least_squares(lambda p: solve_numerically(t, p[0], p[1]) - u, x0=[c0, k0],
                        bounds=([0.0, 1e-6], [np.inf, np.inf]), x_scale=[0.1, 1.0],
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    c, k = fit.x
    dof = max(len(t) - 2, 1)
    sigma2 = 2 * fit.cost / dof
    try:
        cov = sigma2 * np.linalg.inv(fit.jac.T @ fit.jac)
        se_c, se_k = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se_c = se_k = float("nan")
    return {"c": float(c), "k": float(k), "se_c": float(se_c), "se_k": float(se_k),
            "nfev": int(fit.nfev), "converged": bool(fit.success)}


def cramer_rao_std(setting: str, noise: float) -> tuple[float, float]:
    """The smallest standard deviation ANY unbiased method can achieve for c and
    k (Cramer-Rao bound), as a fraction of the true values.

    With Gaussian noise sigma, the measurements carry Fisher information
    J^T J / sigma^2 about (c, k), where J[i] = d u(t_i) / d(c, k) at the truth.
    Its inverse bounds the covariance of every unbiased estimator. A typical
    (median) error of an ideal method is about 0.67 times this std.
    """
    if noise == 0:
        return 0.0, 0.0
    n, t_end = DATA_SETTINGS[setting]
    t = np.linspace(0.0, t_end, n)
    hc, hk = 1e-6 * C_TRUE, 1e-6 * K_TRUE
    J = np.column_stack([(exact(t, C_TRUE + hc, K_TRUE) - exact(t, C_TRUE - hc, K_TRUE)) / (2 * hc),
                         (exact(t, C_TRUE, K_TRUE + hk) - exact(t, C_TRUE, K_TRUE - hk)) / (2 * hk)])
    cov = (noise * U0) ** 2 * np.linalg.inv(J.T @ J)
    return float(np.sqrt(cov[0, 0]) / C_TRUE), float(np.sqrt(cov[1, 1]) / K_TRUE)


def amplitude_ratios(u_pred: torch.Tensor, t: torch.Tensor) -> list[float]:
    """For each time window, max|network| / max|true solution|.

    1.0 means the network has picked up the full oscillation in that window;
    near 0 means the network is flat there (it has not learned it yet).
    """
    u_true = exact(t)
    out = []
    for a, b in WINDOWS:
        mask = (t >= a) & (t < b)
        out.append((u_pred[mask].abs().max() / u_true[mask].abs().max()).item())
    return out
