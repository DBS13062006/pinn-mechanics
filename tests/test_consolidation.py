"""
Checks of problems/consolidation.py that do not involve training.

The two reference formulas are derived differently (Fourier series vs
method of images), so agreement between them is strong evidence both are
right. The exact solution is then plugged into the PINN residual functions,
which must return zero.
"""
import numpy as np
import pytest
import torch

from problems import consolidation as cons


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_images_and_fourier_series_agree():
    x = np.linspace(0, 1, 101)
    for t in [0.005, 0.01, 0.05, 0.2, 0.5]:
        a = cons.exact(x, np.full_like(x, t))
        b = cons.exact_series(x, np.full_like(x, t), n_terms=4000)
        assert np.max(np.abs(a - b)) < 1e-10, t


def test_boundary_and_initial_conditions_of_exact_solution():
    t = np.linspace(1e-4, cons.T_MAX, 50)
    assert np.max(np.abs(cons.exact(np.zeros_like(t), t))) < 1e-14          # p(0, t) = 0
    h = 1e-6
    dpdx = (cons.exact(np.full_like(t, 1 + h), t) - cons.exact(np.full_like(t, 1 - h), t)) / (2 * h)
    assert np.max(np.abs(dpdx)) < 1e-6                                     # p_x(1, t) = 0
    x = np.linspace(1e-3, 1, 50)
    assert np.all(cons.exact(x, np.zeros_like(x)) == 1.0)                  # p(x, 0) = 1


class ExactAsModel:
    """Stands in for a perfectly trained network: pressure() multiplies by x."""
    def __call__(self, x, t):
        return cons.exact(x, t) / x


def test_exact_solution_has_zero_pde_residual():
    x = (0.05 + 0.95 * torch.rand(500, 1)).requires_grad_(True)
    t = (0.01 + 0.49 * torch.rand(500, 1)).requires_grad_(True)
    r = cons.pde_residual(ExactAsModel(), x, t)
    assert r.abs().max().item() < 1e-8


def test_exact_solution_has_zero_neumann_and_initial_residuals():
    t = 0.01 + 0.49 * torch.rand(100, 1)
    assert cons.neumann_residual(ExactAsModel(), t).abs().max().item() < 1e-8
    x = 0.01 + 0.99 * torch.rand(100, 1)
    assert cons.initial_residual(ExactAsModel(), x).abs().max().item() < 1e-12


def test_output_transform_enforces_drained_top():
    model = lambda x, t: torch.full_like(x, 7.0)
    t = torch.rand(10, 1)
    assert torch.all(cons.pressure(model, torch.zeros_like(t), t) == 0)


def test_error_by_band_shares_sum_to_one():
    t = np.linspace(0, cons.T_MAX, 1000)
    bands = cons.error_by_band(np.ones_like(t), t)
    assert sum(b["share_sq_err"] for b in bands) == pytest.approx(1.0)


def test_half_space_term_solves_the_pde_by_itself():
    zero = lambda x, t: torch.zeros_like(x)
    x = (0.01 + 0.99 * torch.rand(500, 1)).requires_grad_(True)
    t = (0.001 + 0.499 * torch.rand(500, 1)).requires_grad_(True)
    r = cons.pde_residual(zero, x, t, output=cons.pressure_enriched)
    assert r.abs().max().item() < 1e-9


def test_enriched_output_makes_top_and_initial_conditions_exact_for_any_network():
    wild = lambda x, t: 5.0 + 3.0 * torch.sin(7 * x + 11 * t)
    t = torch.rand(50, 1)
    assert torch.all(cons.pressure_enriched(wild, torch.zeros_like(t), t) == 0)
    x = 0.001 + 0.999 * torch.rand(50, 1)
    assert torch.allclose(cons.pressure_enriched(wild, x, torch.zeros_like(x)), torch.ones_like(x))


class ExactCorrection:
    """The network a perfect enriched PINN would learn: (exact - erf term) / (x t)."""
    def __call__(self, x, t):
        return (cons.exact(x, t) - cons.half_space(x, t)) / (x * t)


def test_enriched_output_with_exact_correction_reproduces_exact_solution():
    x = (0.05 + 0.95 * torch.rand(300, 1)).requires_grad_(True)
    t = (0.01 + 0.49 * torch.rand(300, 1)).requires_grad_(True)
    p = cons.pressure_enriched(ExactCorrection(), x, t)
    assert torch.allclose(p, cons.exact(x, t), atol=1e-12)
    assert cons.pde_residual(ExactCorrection(), x, t, output=cons.pressure_enriched).abs().max() < 1e-7
    tb = 0.01 + 0.49 * torch.rand(100, 1)
    assert cons.neumann_residual(ExactCorrection(), tb, output=cons.pressure_enriched).abs().max() < 1e-8
