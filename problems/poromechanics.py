"""
Paper section 5 (future work): the coupled poromechanics equations, Eqs. (5.1)-(5.2).

    (1/M) p_t - (k/mu_f) lap(p) + alpha d(div u)/dt = 0      fluid mass balance  (5.1)
    G lap(u) + G/(1-2 nu) grad(div u) - alpha grad(p) = 0    solid equilibrium   (5.2)

(The body force rho_b g only sets the initial geostatic state and drops out
of the problem for the extra load, so it is left out.)

Setting: Terzaghi's oedometer, the same soil column as section 4.1, but now
solved as the FULL coupled system for BOTH the soil displacement u and the
pore pressure p, instead of the single pressure equation that section 4.1
obtains by eliminating u by hand. In 1D (depth z, positive down, z = 0 at
the drained, loaded top, z = L at a rigid impermeable base) Eq. (5.2)
becomes K_v u_zz - alpha p_z = 0 with K_v = 2 G (1 - nu) / (1 - 2 nu).

Dimensionless form, with x = z / L, tau = c t / L^2, P = p / p0,
U = u K_v / (p0 L) and s = K_v / M:

    beta P_tau - P_xx + gamma U_x,tau = 0      (5.1)   beta = s / (s + alpha^2)
    U_xx - alpha P_x = 0                       (5.2)   gamma = alpha / (s + alpha^2)
    x = 0:  P = 0  (drained),   U_x - alpha P = -1  (load p0 on top)
    x = 1:  P_x = 0  (no flow), U = 0  (rigid base)
    tau = 0:  P = gamma  (undrained response to the sudden load)

gamma is the loading efficiency of section 4.1 (p^u = gamma p0). U is the
downward displacement; U(0, tau) is the settlement of the top surface, the
quantity a settlement plate measures on site. It grows from 1 - alpha gamma
(instant, undrained) to 1 (fully drained, P = 0 everywhere).

Exact solution. Integrating Eq. (5.2) with the top condition gives
U_x = alpha P - 1 everywhere; substituting into (5.1) leaves P_tau = P_xx,
the section 4.1 problem scaled by gamma. So

    P(x, tau) = gamma * p_hat(x, tau)                  (p_hat: section 4.1 solution)
    U(x, tau) = (1 - x) - alpha gamma * integral_x^1 p_hat dx'

and the integral has a closed form in iterated error functions (see
integral_q). The PINN does NOT use this reduction: it is trained on the
coupled equations (5.1)-(5.2) as written, with two outputs U and P.

Two problems are built on it:
  forward   find U and P everywhere (data-driven solution, like 4.1 / 4.2),
  inverse   find the consolidation coefficient c from 20 noisy settlement
            readings of the top surface (data-driven discovery, like 4.3).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import erfc as np_erfc

from pinn.derivatives import grad
from problems import consolidation as cons

# Material constants (dimensionless). Generic values on purpose, so that a
# coefficient used in the wrong place shows up in the tests.
ALPHA = 0.9                                # Biot coefficient
S = 0.6                                    # K_v / M: drained stiffness over Biot modulus
BETA = S / (S + ALPHA**2)                  # storage coefficient in (5.1)
GAMMA = ALPHA / (S + ALPHA**2)             # loading efficiency, P at tau = 0+
T_MAX = 0.5                                # forward problem: tau in [0, 0.5], as in 4.1

N_COLLOCATION = 600
N_BOUNDARY = 150                           # per boundary condition
N_INITIAL = 150

# Inverse problem. Time is measured in field units t in [0, 1] (e.g. years),
# tau = c t, and c is unknown.
C_TRUE = 0.5                               # so tau covers [0, 0.5] again
C_START = 0.1                              # starting guess: 5x too small
N_OBS = 20                                 # settlement readings
OBS_NOISE = 0.01                           # std of reading noise, fraction of final settlement


# ------------------------------------------------------------ exact solution
def _ierfc(z, lib):
    """Integrated complementary error function: exp(-z^2)/sqrt(pi) - z erfc(z)."""
    erfc = torch.special.erfc if lib is torch else np_erfc
    return lib.exp(-z * z) / math.sqrt(math.pi) - z * erfc(z)


def integral_q(x, tau, n_images: int = 20):
    """integral_x^1 q(x', tau) dx', with q = 1 - p_hat the images sum of section 4.1.

    Each image pair erfc((2n + x)/s) + erfc((2n + 2 - x)/s), s = 2 sqrt(tau),
    integrates over [x, 1] to s [ierfc((2n + x)/s) - ierfc((2n + 2 - x)/s)]
    (the two terms at x' = 1 cancel). Exact near tau = 0, unlike the Fourier
    series. tau = 0 gives 0.
    """
    lib = torch if isinstance(tau, torch.Tensor) else np
    if lib is np:
        x, tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(tau, float))
    positive = tau > 0
    safe = lib.where(positive, tau, lib.ones_like(tau))
    s = 2.0 * lib.sqrt(safe)
    out = lib.zeros_like(x * safe)
    for n in range(n_images):
        out = out + (-1) ** n * s * (_ierfc((2 * n + x) / s, lib) - _ierfc((2 * n + 2 - x) / s, lib))
    return lib.where(positive, out, lib.zeros_like(out))


def exact_pressure(x, tau):
    return GAMMA * cons.exact(x, tau)


def exact_displacement(x, tau):
    """U = (1 - x) - alpha gamma (1 - x - integral_x^1 q)."""
    return (1.0 - x) * (1.0 - ALPHA * GAMMA) + ALPHA * GAMMA * integral_q(x, tau)


def exact_displacement_series(x, tau, n_terms: int = 4000):
    """Same quantity from the Fourier series of section 4.1, integrated term by term:
    integral_x^1 sin(m pi x'/2) dx' = (2/(m pi)) cos(m pi x/2) for odd m.
    numpy only; used to cross-check exact_displacement()."""
    x, tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(tau, float))
    integral_p = np.zeros_like(x)
    for j in range(n_terms):
        m = 2 * j + 1
        integral_p += 8.0 / (m * math.pi) ** 2 * np.cos(m * math.pi * x / 2) * np.exp(-(m * math.pi) ** 2 * tau / 4)
    return (1.0 - x) - ALPHA * GAMMA * integral_p


def settlement(tau):
    """Settlement of the top surface U(0, tau) (Terzaghi's settlement curve)."""
    lib = torch if isinstance(tau, torch.Tensor) else np
    return exact_displacement(lib.zeros_like(tau), tau)


# ------------------------------------------------------------ network outputs
# The network has two outputs, N = (N_u, N_p). Both transforms keep U = 0 at
# the base and P = 0 at the drained top exactly, as section 4.1 does with
# its "x * network" output. tau_of(t) converts the time input to tau: the
# identity in the forward problem, c * t in the inverse one.

def fields_paper(model, x, t, tau_scale=1.0):
    """Section 4.1 style: U = (1 - x) N_u, P = x N_p. Initial condition and
    the load on top are penalties in the loss."""
    n = model(x, t)
    return (1.0 - x) * n[:, 0:1], x * n[:, 1:2]


def fields_enriched(model, x, t, tau_scale=1.0):
    """Corner-enriched outputs (our fix from section 4.1, extended to U):

        P = gamma [ erf(x / 2 sqrt tau) + x t N_p ]
        U = (1 - x)(1 - alpha gamma) + alpha gamma g(x, tau) + (1 - x) t N_u

    g is the first image term of integral_q: the displacement that the
    half-space erf pressure produces. Both the jump at the corner and the
    initial state are built in exactly; the network learns smooth corrections.
    tau = tau_scale * t, so in the inverse problem the erf front moves at the
    trainable speed c.
    """
    n = model(x, t)
    tau = tau_scale * t
    s = 2.0 * torch.sqrt(tau.clamp_min(1e-30))
    g = torch.where(tau > 0, s * (_ierfc(x / s, torch) - _ierfc((2.0 - x) / s, torch)), torch.zeros_like(tau))
    P = GAMMA * (cons.half_space(x, tau) + x * t * n[:, 1:2])
    U = (1.0 - x) * (1.0 - ALPHA * GAMMA) + ALPHA * GAMMA * g + (1.0 - x) * t * n[:, 0:1]
    return U, P


# ------------------------------------------------------------ residuals
def residuals(fields, model, x, t, c=1.0):
    """Residuals of Eqs. (5.1) and (5.2) at (x, t), with tau = c t.

    In t, Eq. (5.1) reads beta P_t - c P_xx + gamma U_x,t = 0. c = 1 in the
    forward problem (t is already tau); in the inverse problem c is trainable.
    """
    U, P = fields(model, x, t, c)
    U_x = grad(U, x)
    P_x = grad(P, x)
    mass = BETA * grad(P, t) - c * grad(P_x, x) + GAMMA * grad(U_x, t)
    equilibrium = grad(U_x, x) - ALPHA * P_x
    return mass, equilibrium


def top_load_residual(fields, model, t, c=1.0):
    """U_x - alpha P + 1 at x = 0 (the load p0 on the top surface)."""
    x = torch.zeros_like(t).requires_grad_(True)
    U, P = fields(model, x, t, c)
    return grad(U, x) - ALPHA * P + 1.0


def base_flow_residual(fields, model, t, c=1.0):
    """P_x at x = 1 (impermeable base)."""
    x = torch.ones_like(t).requires_grad_(True)
    _, P = fields(model, x, t, c)
    return grad(P, x)


def initial_residual(fields, model, x, c=1.0):
    """P(x, 0) - gamma."""
    t = torch.zeros_like(x).requires_grad_(True)
    _, P = fields(model, x, t, c)
    return P - GAMMA


# ------------------------------------------------------------ point sets
def collocation_points(n: int = N_COLLOCATION, t_max: float = T_MAX, device=None):
    x = torch.rand(n, 1, device=device).requires_grad_(True)
    t = (t_max * torch.rand(n, 1, device=device)).requires_grad_(True)
    return x, t


def boundary_times(n: int = N_BOUNDARY, t_max: float = T_MAX, device=None):
    return (t_max * torch.rand(n, 1, device=device)).requires_grad_(True)


def initial_positions(n: int = N_INITIAL, device=None):
    return torch.rand(n, 1, device=device)


def observation_times(n: int = N_OBS, device=None):
    """Settlement readings, equally spaced on (0, 1] (none at t = 0)."""
    return torch.linspace(1.0 / n, 1.0, n, device=device).reshape(-1, 1)


def settlement_readings(noise: float, draw: int, n: int = N_OBS, device=None):
    """Noisy settlement readings U(0, c_true t_i) + noise * eps (Gaussian).
    The same draw uses the same standard-normal numbers at every noise level."""
    t = observation_times(n, device)
    gen = torch.Generator().manual_seed(20_000 + draw)
    eps = torch.randn(n, 1, generator=gen).to(device)
    return t, settlement(C_TRUE * t) + noise * eps


def test_grid(nx: int = 201, nt: int = 201, t_max: float = T_MAX, device=None):
    x = torch.linspace(0, 1, nx, device=device)
    t = torch.linspace(0, t_max, nt, device=device)
    T, X = torch.meshgrid(t, x, indexing="ij")
    return X.reshape(-1, 1), T.reshape(-1, 1), (nt, nx)


def classical_fit(t, u, c0: float = C_START) -> dict:
    """Classical baseline for the inverse problem: least-squares fit of
    Terzaghi's settlement curve U(0, c t) to the readings, over c only.
    It knows the closed-form settlement curve; the PINN does not."""
    from scipy.optimize import least_squares
    t = np.asarray(t, float).ravel()
    u = np.asarray(u, float).ravel()
    fit = least_squares(lambda p: settlement(p[0] * t) - u, x0=[c0], bounds=([1e-6], [np.inf]),
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    dof = max(len(t) - 1, 1)
    sigma2 = 2 * fit.cost / dof
    jtj = float((fit.jac.T @ fit.jac).item())
    se = float(np.sqrt(sigma2 / jtj)) if sigma2 > 0 else 0.0
    return {"c": float(fit.x[0]), "se_c": se}
