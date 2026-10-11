"""
Checks of problems/oscillator.py that do not involve training.

The key idea: plug the EXACT solution in where the network would go. If the
residual and initial-condition code are written correctly, they must return
(numerically) zero. A sign error or a missing factor in the ODE would show
up here immediately instead of as a mysterious wrong c or k after training.
"""
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from problems import oscillator as osc


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def exact_model(t):
    """Stands in for a perfectly trained network."""
    return osc.exact(t)


def test_closed_form_matches_numerical_ode_solver():
    t = np.linspace(0, osc.T_MAX, 1001)
    sol = solve_ivp(lambda _, y: [y[1], -(osc.C_TRUE * y[1] + osc.K_TRUE * y[0]) / osc.M],
                    [0, osc.T_MAX], [osc.U0, osc.V0], t_eval=t, rtol=1e-11, atol=1e-12)
    assert np.max(np.abs(osc.exact(t) - sol.y[0])) < 1e-9


def test_exact_solution_has_zero_residual_with_true_parameters():
    t = osc.test_times(2000).requires_grad_(True)
    r = osc.residual(exact_model, t, osc.C_TRUE, osc.K_TRUE)
    assert r.abs().max().item() < 1e-10


def test_wrong_parameters_give_nonzero_residual():
    t = osc.test_times(2000).requires_grad_(True)
    r = osc.residual(exact_model, t, osc.C_START, osc.K_START)
    assert r.abs().max().item() > 0.1


def test_exact_solution_satisfies_initial_conditions():
    assert osc.initial_condition_loss(exact_model, device=None).item() < 1e-20


def test_sampling_matches_the_paper():
    t_d = osc.observation_times(50)
    assert t_d.shape == (50, 1) and t_d[0].item() == 0.0 and t_d[-1].item() == osc.T_MAX
    t_f = osc.collocation_times(100)
    assert t_f.shape == (100, 1) and t_f.requires_grad
    assert 0.0 <= t_f.min().item() and t_f.max().item() <= osc.T_MAX


def test_implied_parameters_of_exact_curve_are_the_true_ones():
    c, k = osc.implied_parameters(exact_model, osc.test_times(2000))
    assert c == pytest.approx(osc.C_TRUE, rel=1e-8)
    assert k == pytest.approx(osc.K_TRUE, rel=1e-8)


def test_implied_parameters_follow_the_curve_not_the_truth():
    """A curve generated with other values must imply those other values."""
    other = lambda t: osc.exact(t, c=1.0, k=2.5)
    c, k = osc.implied_parameters(other, osc.test_times(2000))
    assert c == pytest.approx(1.0, rel=1e-8) and k == pytest.approx(2.5, rel=1e-8)


def test_amplitude_ratios():
    t = osc.test_times(3000)
    assert osc.amplitude_ratios(osc.exact(t), t) == pytest.approx([1.0, 1.0, 1.0])
    flat_tail = torch.where(t < 5.0, osc.exact(t), torch.zeros_like(t))
    ratios = osc.amplitude_ratios(flat_tail, t)
    assert ratios[0] == pytest.approx(1.0) and ratios[1] == 0.0 and ratios[2] == 0.0


def test_overdamped_parameters_are_rejected():
    with pytest.raises(ValueError):
        osc.exact(np.array([0.0]), c=10.0, k=4.0)
