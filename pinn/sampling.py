"""
Where to put the collocation points.

Uniform random points spend most of their effort where the solution is easy.
Residual-based adaptive distribution (RAD; Wu, Zhu, Tan, Kartha & Lu, 2022,
"A comprehensive study of non-adaptive and residual-based adaptive sampling
for physics-informed neural networks") periodically re-draws the points so
that more of them land where the PDE is currently violated the most:

    density(x)  proportional to  |r(x)|^k / mean(|r|^k) + c

r is the PDE residual of the current network. k sharpens the focus (k = 1
is the paper's default) and c keeps a share of points spread uniformly
(c = 1: about half the points stay "uniform", so no region is abandoned).
This is the PINN version of adaptive mesh refinement, which our base paper
lists as future work.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch

PointSampler = Callable[[int], Sequence[torch.Tensor]]
ResidualFn = Callable[..., torch.Tensor]


def rad_sample(residual_fn: ResidualFn, sampler: PointSampler, n: int,
               n_pool: int = 20_000, k: float = 1.0, c: float = 1.0) -> tuple[torch.Tensor, ...]:
    """Draw n points from a pool of n_pool candidates, favouring large residuals.

    sampler(m) must return a tuple of (m, 1) coordinate tensors (e.g. x, t)
    spread uniformly over the domain; residual_fn(*coords) returns (m, 1).
    Returns new coordinate tensors with requires_grad=True.

    Points are drawn without replacement, so keep n_pool much larger than n
    (20x or more); otherwise the high-residual candidates run out and the
    sample ends up less concentrated than the formula above says.
    """
    pool = [p.detach().requires_grad_(True) for p in sampler(n_pool)]
    with torch.enable_grad():
        r = residual_fn(*pool).detach().abs().reshape(-1)
    rk = r**k
    weights = rk / rk.mean() + c
    idx = torch.multinomial(weights, n, replacement=False)
    return tuple(p.detach()[idx].clone().requires_grad_(True) for p in pool)


class RADHook:
    """Trainer hook: every `every` Adam steps, replace the collocation points
    IN PLACE with a fresh RAD sample of the same size.

    The loss function keeps referring to the same tensors, so it picks up the
    new points without any other change. Hooks only run during Adam: moving
    points during L-BFGS would invalidate its curvature history, so L-BFGS
    trains on the last set drawn.
    """

    def __init__(self, points: Sequence[torch.Tensor], residual_fn: ResidualFn,
                 sampler: PointSampler, every: int = 1_000, n_pool: int = 20_000,
                 k: float = 1.0, c: float = 1.0, keep_history: bool = True) -> None:
        self.points = list(points)
        self.residual_fn, self.sampler = residual_fn, sampler
        self.every, self.n_pool, self.k, self.c = every, n_pool, k, c
        self.keep_history = keep_history
        self.history: list[tuple[int, list[torch.Tensor]]] = []

    def __call__(self, trainer) -> None:
        if trainer.iteration % self.every != 0:
            return
        n = self.points[0].shape[0]
        new = rad_sample(self.residual_fn, self.sampler, n, self.n_pool, self.k, self.c)
        with torch.no_grad():
            for p, q in zip(self.points, new):
                p.copy_(q)
        if self.keep_history:
            self.history.append((trainer.iteration, [p.detach().cpu().clone() for p in self.points]))
