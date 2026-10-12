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
function that vanishes on the boundary; temperature_hard() below does exactly
that with d = (1 - x^2)(1 - y^2).

The corners (Step 5b). Along an edge T = 0, so T does not curve along it:
on the top edge T_xx = 0, and on the right edge T_yy = 0, so the PDE forces
T_xx = -1 there. At the corner where the two edges meet, T_xx would have to
be 0 and -1 at once. The exact solution resolves this by making T_xx depend on
the direction of approach: coming in at angle theta from the horizontal edge,
T_xx -> -2 theta / pi. A smooth function (any tanh network) has a single value
at each point, so no network can satisfy both the PDE and T = 0 at a corner:
  - with soft BC the network gives up a little of T = 0 near the corners, which
    is where its largest error is;
  - with hard BC, T = d * N gives T_xx = T_yy = 0 at a corner whatever N is, so
    the PDE residual there is exactly 1.
temperature_corner() adds the exact local solution at each corner
(corner_function) so that the network only has to learn a smooth remainder.
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


# ------------------------------------------------------------ corners (Step 5b)
CORNERS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
CORNER_RADIUS = 0.5             # width of the Gaussian fade-out around each corner term


def corner_function(x, y, cx, cy):
    """Exact solution at the corner (cx, cy), as if its two edges went on forever.

    With xi, eta = distances to the two edges meeting there and r, theta =
    polar coordinates from the corner (theta = 0 along the horizontal edge):

        u = r^2 [ -1/4 + cos(2 theta)/4 - (ln r sin(2 theta) + theta cos(2 theta)) / pi ]

    Derivation: try u = r^2 g(theta). The PDE gives g'' + 4 g = -1, so
    g = -1/4 + A cos(2 theta) + B sin(2 theta). u = 0 at theta = 0 needs
    A = 1/4, but then g(pi/2) = -1/2 and no B can fix it (sin(pi) = 0): this is
    the corner contradiction again. The way out is the harmonic function
    r^2 (ln r sin(2 theta) + theta cos(2 theta)) (the imaginary part of
    z^2 log z), which is 0 at theta = 0 and -pi r^2 / 2 at theta = pi/2; adding
    -1/pi of it makes u = 0 on both edges. Then u_xx = -2 theta / pi exactly.

    Uses only local information (a 90-degree corner with T = 0 on both sides),
    not the series solution. numpy arrays or torch tensors. At the corner point
    itself the value is 0 but the derivatives are undefined.
    """
    is_torch = isinstance(x, torch.Tensor)
    lib = torch if is_torch else np
    xi, eta = 1 - cx * x, 1 - cy * y
    r2 = xi**2 + eta**2
    r2 = r2.clamp_min(1e-300) if is_torch else np.maximum(r2, 1e-300)
    theta = torch.atan2(eta, xi) if is_torch else np.arctan2(eta, xi)
    return r2 * (-0.25 + 0.25 * lib.cos(2 * theta)
                 - (0.5 * lib.log(r2) * lib.sin(2 * theta) + theta * lib.cos(2 * theta)) / math.pi)


def corner_terms(x, y, radius: float = CORNER_RADIUS):
    """Sum of the four corner functions, each kept local.

    weight (1 + cx x)(1 + cy y)/4 is 1 at its own corner and 0 on the two far
    edges, so T = 0 still holds exactly on every edge. The Gaussian fade-out
    keeps each term near its corner; without it the four terms add up to more
    than the temperature itself and the network has to learn a large
    correction (tried in Step 5b: about 10x worse).
    """
    lib = torch if isinstance(x, torch.Tensor) else np
    out = 0.0
    for cx, cy in CORNERS:
        weight = (1 + cx * x) * (1 + cy * y) / 4
        fade = lib.exp(-((1 - cx * x) ** 2 + (1 - cy * y) ** 2) / radius**2)
        out = out + weight * fade * corner_function(x, y, cx, cy)
    return out


def temperature_corner(model, x, y):
    """Step 5b: T = d * network + corner terms. T = 0 on the edges exactly, and
    the corner behaviour no smooth network can produce is built in."""
    return temperature_hard(model, x, y) + corner_terms(x, y)


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


def distance_to_corner(x, y):
    """Euclidean distance to the nearest of the four corners."""
    return np.sqrt((1 - np.abs(x)) ** 2 + (1 - np.abs(y)) ** 2)


def corner_points(n: int, r_min: float = 1e-4, r_max: float = 1.0, generator=None, device=None):
    """n points spread over the four corners, log-uniform in the distance r to
    the corner (as many points between 1e-4 and 1e-3 as between 0.1 and 1).
    Returns x, y and r, for profiles of the residual vs distance to a corner."""
    u = torch.rand(n, 3, generator=generator)
    r = r_min * (r_max / r_min) ** u[:, :1]
    theta = (math.pi / 2) * u[:, 1:2]
    which = (4 * u[:, 2:3]).long().clamp_max(3)
    cx = torch.where(which % 2 == 0, 1.0, -1.0)
    cy = torch.where(which < 2, 1.0, -1.0)
    x = cx * (1 - r * torch.cos(theta))
    y = cy * (1 - r * torch.sin(theta))
    return x.to(device).requires_grad_(True), y.to(device).requires_grad_(True), r.to(device)
