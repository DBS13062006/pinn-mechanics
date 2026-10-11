"""
Fast checks of the PINN engine. Run from the repo root:

    python -m pytest -q

Each test targets one thing a later experiment silently depends on. If a
change to pinn/ breaks one of them, the experiments would give wrong numbers
without crashing, which is the worst kind of bug. Run this before committing
changes to pinn/.
"""
import math

import pytest
import torch

from pinn import (MLP, TrainConfig, Trainer, grad, laplacian, nth_grad,
                  relative_error, relative_l2, set_seed)

QUIET = dict(verbose=False)


@pytest.fixture(autouse=True)
def float64_and_seed():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    set_seed(0)
    yield
    torch.set_default_dtype(old)


# ---------------------------------------------------------------- derivatives
def test_grad_matches_known_derivatives():
    x = torch.linspace(-2, 2, 201).reshape(-1, 1).requires_grad_(True)
    u = x**3
    assert torch.allclose(grad(u, x), 3 * x**2)
    assert torch.allclose(nth_grad(u, x, 2), 6 * x)
    assert torch.allclose(nth_grad(u, x, 3), torch.full_like(x, 6.0))


def test_grad_of_unrelated_input_is_zero():
    x = torch.rand(10, 1, requires_grad=True)
    y = torch.rand(10, 1, requires_grad=True)
    u = y**2  # does not depend on x
    assert torch.equal(grad(u, x), torch.zeros_like(x))


def test_laplacian():
    x = torch.rand(50, 1, requires_grad=True)
    y = torch.rand(50, 1, requires_grad=True)
    u = x**2 + 3 * y**2  # u_xx + u_yy = 2 + 6
    assert torch.allclose(laplacian(u, x, y), torch.full_like(x, 8.0))


def test_network_derivatives_match_finite_differences():
    model = MLP(2, 1, width=30, depth=3, lb=[0, -1], ub=[1, 1])
    t = torch.rand(100, 1, requires_grad=True)
    x = (2 * torch.rand(100, 1) - 1).requires_grad_(True)
    u = model(t, x)
    u_t, u_x = grad(u, t), grad(u, x)
    u_xx = grad(u_x, x)

    h = 1e-4
    with torch.no_grad():
        fd_t = (model(t + h, x) - model(t - h, x)) / (2 * h)
        fd_xx = (model(t, x + h) - 2 * model(t, x) + model(t, x - h)) / h**2
    assert torch.allclose(u_t, fd_t, atol=1e-7)
    assert torch.allclose(u_xx, fd_xx, atol=1e-5)


# ----------------------------------------------------------------------- model
def test_inputs_are_scaled_to_minus_one_one():
    model = MLP(2, 1, lb=[0.0, -2.0], ub=[1.0, 2.0])
    corners = torch.tensor([[0.0, -2.0], [1.0, 2.0], [0.5, 0.0]])
    expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    assert torch.allclose(model.scale_inputs(corners), expected)


def test_both_call_styles_give_the_same_output():
    model = MLP(2, 3, width=10, depth=2)
    t, x = torch.rand(20, 1), torch.rand(20, 1)
    assert torch.equal(model(t, x), model(torch.cat([t, x], dim=1)))


def test_bad_bounds_are_rejected():
    with pytest.raises(ValueError):
        MLP(2, 1, lb=[0.0, 1.0], ub=[1.0, 1.0])


# --------------------------------------------------------------------- trainer
def test_weights_scale_the_loss_terms():
    p = torch.nn.Parameter(torch.tensor(0.0))
    trainer = Trainer([p], lambda: {"a": p * 0 + 1.0, "b": p * 0 + 2.0},
                      TrainConfig(weights={"b": 10.0}, **QUIET))
    total, terms = trainer.compute_loss()
    assert total.item() == pytest.approx(1.0 + 10.0 * 2.0)
    assert set(terms) == {"a", "b"}


def test_hooks_run_once_per_adam_step_only():
    p = torch.nn.Parameter(torch.tensor(1.0))
    calls = []
    cfg = TrainConfig(adam_iters=7, lbfgs_iters=50, **QUIET)
    Trainer([p], lambda: {"f": (p - 3) ** 2}, cfg, hooks=[lambda tr: calls.append(1)]).train()
    assert len(calls) == 7


def test_forward_problem_poisson_1d():
    """-u'' = pi^2 sin(pi x) on [0, 1], u(0) = u(1) = 0. Exact: u = sin(pi x)."""
    model = MLP(1, 1, width=20, depth=2)
    x_f = torch.linspace(0, 1, 64).reshape(-1, 1).requires_grad_(True)
    x_b = torch.tensor([[0.0], [1.0]])

    def loss_fn():
        u = model(x_f)
        residual = -nth_grad(u, x_f, 2) - math.pi**2 * torch.sin(math.pi * x_f)
        return {"pde": (residual**2).mean(), "bc": (model(x_b) ** 2).mean()}

    x_test = torch.linspace(0, 1, 257).reshape(-1, 1)
    eval_fn = lambda: {"rel_l2": relative_l2(model(x_test), torch.sin(math.pi * x_test))}
    cfg = TrainConfig(adam_iters=500, adam_lr=1e-2, lbfgs_iters=500,
                      log_every=50, eval_every=250, **QUIET)
    trainer = Trainer(model.parameters(), loss_fn, cfg, eval_fn=eval_fn).train()

    assert trainer.evals[-1]["rel_l2"] < 1e-3
    assert {"pde", "bc", "loss"} <= set(trainer.log[-1])
    assert trainer.log[-1]["loss"] < trainer.log[0]["loss"]


def test_inverse_problem_recovers_a_parameter():
    """u' = -a u with data from u = exp(-2x). The PINN must find a = 2."""
    model = MLP(1, 1, width=20, depth=2)
    a = torch.nn.Parameter(torch.tensor(0.5))  # deliberately 4x too small
    x_d = torch.linspace(0, 1, 20).reshape(-1, 1)
    u_d = torch.exp(-2.0 * x_d)
    x_f = torch.linspace(0, 1, 64).reshape(-1, 1).requires_grad_(True)

    def loss_fn():
        u = model(x_f)
        residual = grad(u, x_f) + a * u
        return {"data": ((model(x_d) - u_d) ** 2).mean(), "pde": (residual**2).mean()}

    cfg = TrainConfig(adam_iters=2000, adam_lr=1e-2, lbfgs_iters=500, **QUIET)
    Trainer(list(model.parameters()) + [a], loss_fn, cfg).train()
    assert relative_error(a.item(), 2.0) < 1e-3
