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
