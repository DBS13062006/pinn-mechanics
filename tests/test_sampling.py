"""Checks of pinn/sampling.py (residual-based adaptive sampling)."""
import pytest
import torch

from pinn import TrainConfig, Trainer, set_seed
from pinn.sampling import RADHook, rad_sample


@pytest.fixture(autouse=True)
def float64_and_seed():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    set_seed(0)
    yield
    torch.set_default_dtype(old)


def uniform_1d(m):
    return (torch.rand(m, 1),)


def step_residual(x):
    """Residual 100 on x < 0.1, 0.01 elsewhere: a 'hard region' covering 10% of the domain."""
    return torch.where(x < 0.1, torch.full_like(x, 100.0), torch.full_like(x, 0.01))


def test_rad_concentrates_points_where_residual_is_large():
    # Pool 100x the sample, so drawing without replacement barely depletes the hard region.
    (x,) = rad_sample(step_residual, uniform_1d, n=1000, n_pool=100_000, k=1.0, c=1.0)
    # weights: ~11 inside (10% of pool), ~1 outside -> about 55% of samples inside
    frac = (x < 0.1).double().mean().item()
    assert 0.50 < frac < 0.60
    assert x.requires_grad


def test_large_c_gives_nearly_uniform_points():
    (x,) = rad_sample(step_residual, uniform_1d, n=1000, n_pool=100_000, k=1.0, c=1e6)
    assert abs((x < 0.1).double().mean().item() - 0.1) < 0.03


def test_hook_resamples_in_place_on_schedule():
    x = torch.rand(200, 1).requires_grad_(True)
    original_id = id(x)
    hook = RADHook([x], step_residual, uniform_1d, every=5, n_pool=2_000)
    p = torch.nn.Parameter(torch.tensor(1.0))

    def loss_fn():
        return {"f": ((x * p) ** 2).mean()}

    Trainer([p], loss_fn, TrainConfig(adam_iters=12, lbfgs_iters=0, verbose=False),
            hooks=[hook]).train()
    assert [it for it, _ in hook.history] == [5, 10]
    assert id(x) == original_id and x.shape == (200, 1)
    assert (x < 0.1).double().mean().item() > 0.4   # now concentrated in the hard region
