"""
Checks of problems/heat.py that do not involve training.

The paper says no analytical solution exists; these tests check the series
solution three ways: against the known centre value, against our own
finite-difference solver (which must converge to it at second order), and by
plugging it into the PINN residual, which must return zero.
"""
import numpy as np
import pytest
import torch

from problems import heat


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_centre_value_matches_the_classic_constant():
    assert heat.exact(0.0, 0.0) == pytest.approx(heat.T_CENTER, abs=1e-8)


def test_series_and_finite_differences_agree_and_converge_at_second_order():
    errors = []
    for n in (101, 201):
        x, y, T = heat.finite_difference(n)
        X, Y = np.meshgrid(x, y)
        errors.append(np.abs(heat.exact(X, Y) - T).max())
    assert errors[1] < 1e-5
    assert 3.5 < errors[0] / errors[1] < 4.5        # halving h divides the error by ~4


def test_exact_solution_is_zero_on_the_boundary():
    s = np.linspace(-1, 1, 201)
    for x, y in [(s, -np.ones_like(s)), (s, np.ones_like(s)), (-np.ones_like(s), s), (np.ones_like(s), s)]:
        assert np.abs(heat.exact(x, y)).max() < 1e-8


def test_exact_solution_has_zero_pde_residual():
    model = lambda x, y: heat.exact(x, y)
    x = (1.8 * torch.rand(300, 1) - 0.9).requires_grad_(True)
    y = (1.8 * torch.rand(300, 1) - 0.9).requires_grad_(True)
    assert heat.pde_residual(model, x, y).abs().max().item() < 1e-8


def test_hard_output_is_zero_on_the_boundary_for_any_network():
    wild = lambda x, y: 3.0 + torch.sin(5 * x) * torch.cos(7 * y)
    x, y = heat.boundary_points(400)
    assert torch.all(heat.temperature_hard(wild, x, y) == 0)


def test_boundary_points_lie_on_the_edges_and_cover_all_four():
    x, y = heat.boundary_points(2000)
    assert torch.all((x.abs() == 1) | (y.abs() == 1))
    for mask in (x == -1, x == 1, y == -1, y == 1):
        assert mask.sum() > 300


# ---------------------------------------------------------------- corners
def test_exact_curvature_at_a_corner_depends_on_direction_not_distance():
    """Approaching corner (1, 1) at angle theta from the top edge, T_xx -> -2 theta / pi:
    0 along the top edge, -1 along the right edge. No smooth function can do this."""
    theta = np.array([0.1, np.pi / 4, np.pi / 2 - 0.1])
    for d in (0.1, 0.01, 0.001):
        x, y, h = 1 - d * np.cos(theta), 1 - d * np.sin(theta), d / 50
        T_xx = (heat.exact(x + h, y, 20000) - 2 * heat.exact(x, y, 20000) + heat.exact(x - h, y, 20000)) / h**2
        assert np.allclose(T_xx, -2 * theta / np.pi, atol=5e-3), (d, T_xx)


def test_any_hard_bc_network_has_pde_residual_exactly_one_at_the_corners():
    from pinn import MLP, set_seed
    set_seed(0)
    model = MLP(2, 1, 20, 3, lb=[-1, -1], ub=[1, 1])
    wild = lambda x, y: 3.0 + torch.sin(5 * x) * torch.cos(7 * y)
    x = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]], requires_grad=True)
    y = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]], requires_grad=True)
    for net in (model, wild):
        r = heat.pde_residual(net, x, y, heat.temperature_hard)
        assert torch.allclose(r, torch.ones_like(r), atol=1e-12)


def test_hard_bc_target_grows_like_a_logarithm_at_a_corner():
    """A hard-BC network must learn N = T / d, which grows by ln(10) / (2 pi) for
    every factor of 10 closer to a corner: no bounded network can follow it."""
    values = []
    for dist in (1e-2, 1e-3, 1e-4):
        x = np.array([1 - dist / np.sqrt(2)])                 # on the diagonal
        values.append(heat.exact(x, x, 20000)[0] / heat.distance(x, x)[0])
    steps = np.diff(values)
    assert np.all(steps > 0)
    assert steps[-1] == pytest.approx(np.log(10) / (2 * np.pi), rel=0.01)


def test_corner_function_solves_the_pde_and_is_zero_on_its_two_edges():
    from pinn import laplacian
    x = (2 * torch.rand(500, 1) - 1).requires_grad_(True)
    y = (2 * torch.rand(500, 1) - 1).requires_grad_(True)
    for cx, cy in heat.CORNERS:
        u = heat.corner_function(x, y, cx, cy)
        assert (laplacian(u, x, y) + 1).abs().max().item() < 1e-10
        s = torch.linspace(-1, 1, 101).reshape(-1, 1)
        assert heat.corner_function(torch.full_like(s, cx), s, cx, cy).abs().max() < 1e-14
        assert heat.corner_function(s, torch.full_like(s, cy), cx, cy).abs().max() < 1e-14


def test_corner_output_is_zero_on_the_boundary_for_any_network():
    wild = lambda x, y: 3.0 + torch.sin(5 * x) * torch.cos(7 * y)
    x, y = heat.boundary_points(400)
    assert heat.temperature_corner(wild, x, y).abs().max().item() < 1e-14


def test_corner_output_removes_the_corner_residual_for_any_network():
    wild = lambda x, y: 3.0 + torch.sin(5 * x) * torch.cos(7 * y)
    x, y, r = heat.corner_points(400, r_min=1e-5, r_max=1e-4)
    hard = heat.pde_residual(wild, x, y, heat.temperature_hard)
    corner = heat.pde_residual(wild, x, y, heat.temperature_corner)
    assert hard.abs().min().item() > 0.9          # ~1 near every corner
    assert corner.abs().max().item() < 0.02       # ~0 near every corner (shrinks in proportion to r)


def test_corner_points_lie_inside_at_the_requested_distance():
    x, y, r = heat.corner_points(2000)
    xn, yn = x.detach().numpy(), y.detach().numpy()
    assert np.all(np.abs(xn) <= 1) and np.all(np.abs(yn) <= 1)
    assert np.allclose(heat.distance_to_corner(xn, yn), r.numpy())
    assert r.min() >= 1e-4 and r.max() <= 1.0
