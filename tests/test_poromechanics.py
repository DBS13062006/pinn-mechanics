"""
Checks of problems/poromechanics.py that do not involve training.

The displacement has two independent formulas (iterated-erfc images and a
term-by-term integrated Fourier series); they must agree. The exact fields,
plugged into the PINN residuals of the COUPLED equations (5.1)-(5.2), must
give zero, and both output transforms must satisfy their exact constraints
for any network.
"""
import numpy as np
import pytest
import torch

from pinn import MLP
from problems import poromechanics as poro


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_displacement_images_and_series_agree():
    x = np.linspace(0, 1, 101)
    for tau in [0.01, 0.05, 0.2, 0.5]:
        a = poro.exact_displacement(x, np.full_like(x, tau))
        b = poro.exact_displacement_series(x, np.full_like(x, tau))
        assert np.max(np.abs(a - b)) < 1e-8, tau


def test_settlement_limits():
    # instantly: undrained settlement 1 - alpha gamma; long after: fully drained, 1
    assert poro.settlement(np.array([0.0]))[0] == pytest.approx(1 - poro.ALPHA * poro.GAMMA)
    assert poro.settlement(np.array([10.0]))[0] == pytest.approx(1.0, abs=1e-9)
    taus = np.linspace(0, 2, 200)
    assert np.all(np.diff(poro.settlement(taus)) >= 0)          # it only ever grows


def exact_fields(model, x, t, tau_scale=1.0):
    """Stands in for a perfectly trained network."""
    tau = tau_scale * t
    return poro.exact_displacement(x, tau), poro.exact_pressure(x, tau)


@pytest.mark.parametrize("c", [1.0, 0.5])
def test_exact_solution_satisfies_coupled_equations(c):
    torch.manual_seed(0)
    x = (0.02 + 0.96 * torch.rand(400, 1)).requires_grad_(True)
    t = (0.02 + 0.9 * torch.rand(400, 1)).requires_grad_(True)
    mass, eq = poro.residuals(exact_fields, None, x, t, c)
    assert mass.abs().max().item() < 1e-9
    assert eq.abs().max().item() < 1e-9
    tb = (0.02 + 0.9 * torch.rand(50, 1)).requires_grad_(True)
    assert poro.top_load_residual(exact_fields, None, tb, c).abs().max().item() < 1e-9
    assert poro.base_flow_residual(exact_fields, None, tb, c).abs().max().item() < 1e-9


@pytest.mark.parametrize("fields", [poro.fields_paper, poro.fields_enriched])
def test_output_transforms_hold_their_constraints(fields):
    torch.manual_seed(1)
    model = MLP(2, 2, width=16, depth=2)
    t = torch.rand(50, 1)
    U, _ = fields(model, torch.ones_like(t), t)
    _, P = fields(model, torch.zeros_like(t), t)
    assert U.abs().max().item() < 1e-14                          # U = 0 on the rigid base
    assert P.abs().max().item() < 1e-14                          # P = 0 on the drained top


def test_enriched_output_has_exact_initial_state():
    torch.manual_seed(2)
    model = MLP(2, 2, width=16, depth=2)
    x = torch.rand(100, 1) * 0.98 + 0.01
    U, P = poro.fields_enriched(model, x, torch.zeros_like(x))
    assert (P - poro.GAMMA).abs().max().item() < 1e-14
    assert (U - (1 - x) * (1 - poro.ALPHA * poro.GAMMA)).abs().max().item() < 1e-14


def test_classical_fit_recovers_c_from_clean_readings():
    t, u = poro.settlement_readings(noise=0.0, draw=0)
    fit = poro.classical_fit(t.numpy(), u.numpy())
    assert fit["c"] == pytest.approx(poro.C_TRUE, rel=1e-8)
