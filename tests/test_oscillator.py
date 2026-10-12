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


# ---------------------------------------------------------------- Step 8: imperfect data
def test_noisy_observations_layout_and_reproducibility():
    for setting, (n, t_end) in osc.DATA_SETTINGS.items():
        t, u = osc.noisy_observations(setting, 0.0, draw=3)
        assert t.shape == (n, 1) and t[0, 0] == 0 and t[-1, 0] == pytest.approx(t_end)
        assert torch.allclose(u, osc.exact(t))                       # no noise: exact data
    t, u1 = osc.noisy_observations("full50", 0.01, draw=3)
    _, u3 = osc.noisy_observations("full50", 0.03, draw=3)
    _, other = osc.noisy_observations("full50", 0.01, draw=4)
    exact = osc.exact(t)
    assert torch.allclose(u3 - exact, 3 * (u1 - exact))              # same random numbers, scaled
    assert not torch.allclose(u1, other)                             # a different draw differs


def test_numerical_solver_matches_closed_form():
    t = np.linspace(0, osc.T_MAX, 200)
    assert np.abs(osc.solve_numerically(t, 0.4, 4.0) - osc.exact(t)).max() < 1e-9


def test_classical_fit_recovers_parameters_from_clean_data():
    for setting in osc.DATA_SETTINGS:
        t, u = osc.noisy_observations(setting, 0.0, draw=0)
        fit = osc.classical_fit(t.numpy(), u.numpy())
        assert fit["c"] == pytest.approx(osc.C_TRUE, rel=1e-6)
        assert fit["k"] == pytest.approx(osc.K_TRUE, rel=1e-6)


def test_cramer_rao_bound_scales_with_noise_and_matches_the_classical_spread():
    """The bound is linear in the noise, and the classical method (asymptotically
    the best possible) should scatter by about the bound: checked on 40 draws."""
    c1, k1 = osc.cramer_rao_std("early50", 0.01)
    c3, k3 = osc.cramer_rao_std("early50", 0.03)
    assert c3 == pytest.approx(3 * c1) and k3 == pytest.approx(3 * k1)
    assert c1 > k1                                   # damping is harder to pin down than stiffness
    fits = [osc.classical_fit(*(a.numpy() for a in osc.noisy_observations("early50", 0.03, d)))
            for d in range(40)]
    c_rel = np.array([f["c"] for f in fits]) / osc.C_TRUE - 1
    assert 0.7 < c_rel.std() / c3 < 1.3
    assert abs(c_rel.mean()) < 0.5 * c3              # no large bias
    se = np.median([f["se_c"] for f in fits]) / osc.C_TRUE
    assert 0.8 < se / c3 < 1.2                       # its own error bars agree with the bound
