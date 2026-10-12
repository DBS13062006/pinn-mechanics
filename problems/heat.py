"""
Paper section 4.2: steady heat conduction with a uniform heat source.

    T_xx + T_yy + 1 = 0     on the square (-1, 1) x (-1, 1)
    T = 0                   on the boundary

Physical picture: a square plate is heated uniformly everywhere and its edges
are held at temperature 0, so it is hottest in the middle.

The paper says no analytical solution exists and compares against a
finite-element solution from deal.II. In fact the classic series solution
exists (this is also the Prandtl stress function of a square bar in torsion):

    T = (1 - x^2)/2 - (16/pi^3) sum_{k odd} (-1)^((k-1)/2) / k^3
                         * cosh(k pi y / 2) / cosh(k pi / 2) * cos(k pi x / 2)

The first term handles the source and the x-edges; the sum corrects the
y-edges. Two independent references are implemented and checked against each
other (tests/test_heat.py): this series, and our own finite-difference solver,
which plays the role of the paper's deal.II solution.

The paper's PINN uses no output transform, so T = 0 on the edges is only a
penalty in the loss, and it reports its largest error along the edges. It
suggests (but does not try) multiplying the network by a distance-like
function that vanishes on the boundary; pressure_hard() below does exactly
that with d = (1 - x^2)(1 - y^2).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from pinn.derivatives import grad

N_COLLOCATION = 1500
N_BOUNDARY = 500
T_CENTER = 0.29468541           # T(0, 0) of the exact solution (classic torsion constant)


# ------------------------------------------------------------ exact solution
def exact(x, y, n_terms: int = 1000):
    """Series solution above. numpy arrays or torch tensors."""
    is_torch = isinstance(x, torch.Tensor)
    lib = torch if is_torch else np
    if not is_torch:
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    out = (1.0 - x**2) / 2.0
    ay = lib.abs(y)
    for j in range(n_terms):
        k = 2 * j + 1
        a = k * math.pi / 2.0
        # cosh(a y) / cosh(a) without overflow
        ratio = lib.exp(a * (ay - 1.0)) * (1.0 + lib.exp(-2.0 * a * ay)) / (1.0 + math.exp(-2.0 * a))
        out = out - (16.0 / math.pi**3) * (-1) ** j / k**3 * ratio * lib.cos(a * x)
    return out


def finite_difference(n: int = 201) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent reference: 5-point finite differences on an n x n grid.

    Second-order accurate (error ~ h^2). Returns x, y (1D grids) and T[iy, ix].
    """
    x = np.linspace(-1, 1, n)
    h = x[1] - x[0]
    m = n - 2                                     # interior points per side
    main = sp.diags([-2 * np.ones(m), np.ones(m - 1), np.ones(m - 1)], [0, -1, 1])
    eye = sp.identity(m)
    lap = (sp.kron(eye, main) + sp.kron(main, eye)) / h**2
    interior = spla.spsolve(lap.tocsc(), -np.ones(m * m))
    T = np.zeros((n, n))
    T[1:-1, 1:-1] = interior.reshape(m, m)
    return x, x.copy(), T


# ------------------------------------------------------------ PINN pieces
def temperature_soft(model, x, y):
    """Paper's setup: the network output is T itself (no transform)."""
    return model(x, y)


def distance(x, y):
    """(1 - x^2)(1 - y^2): zero on every edge, positive inside, smooth."""
    return (1.0 - x**2) * (1.0 - y**2)


def temperature_hard(model, x, y):
    """The paper's suggested fix: T = d(x, y) * network, so T = 0 on the edges exactly."""
    return distance(x, y) * model(x, y)


def pde_residual(model, x, y, output=temperature_soft):
    """T_xx + T_yy + 1. Zero wherever the heat equation holds."""
    T = output(model, x, y)
    return grad(grad(T, x), x) + grad(grad(T, y), y) + 1.0


# ------------------------------------------------------------ point sets
def collocation_points(n: int = N_COLLOCATION, device=None):
    x = (2 * torch.rand(n, 1, device=device) - 1).requires_grad_(True)
    y = (2 * torch.rand(n, 1, device=device) - 1).requires_grad_(True)
    return x, y


def boundary_points(n: int = N_BOUNDARY, device=None):
    """n random points spread over the four edges."""
    s = 2 * torch.rand(n, 1, device=device) - 1
    edge = torch.randint(0, 4, (n, 1), device=device)
    one = torch.ones_like(s)
    # edges 0, 1: x = -1, +1 (y free);  edges 2, 3: y = -1, +1 (x free)
    x = torch.where(edge == 0, -one, torch.where(edge == 1, one, s))
    y = torch.where(edge == 2, -one, torch.where(edge == 3, one, s))
    return x, y


def test_grid(n: int = 201, device=None):
    g = torch.linspace(-1, 1, n, device=device)
    Y, X = torch.meshgrid(g, g, indexing="ij")    # rows = y, columns = x
    return X.reshape(-1, 1), Y.reshape(-1, 1), (n, n)


def distance_to_boundary(x, y):
    """Euclidean distance to the nearest edge of the square."""
    return np.minimum(1 - np.abs(x), 1 - np.abs(y))
