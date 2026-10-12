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
