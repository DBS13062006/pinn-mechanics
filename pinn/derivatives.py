"""
Derivatives of the network output with respect to its INPUTS.

This is the heart of a PINN. Normal training only needs derivatives of the
loss with respect to the weights. A PINN also needs u_t, u_x, u_xx, ... with
respect to the coordinates, because those appear inside the loss itself
(the PDE residual). PyTorch's autograd gives them exactly, with no grid and
no finite-difference error.

Usage pattern (every problem module follows it):

    t = t.requires_grad_(True)       # mark coordinates we differentiate by
    x = x.requires_grad_(True)
    u = model(t, x)                  # keep t and x as separate tensors
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)              # differentiate the derivative again
    residual = u_t - kappa * u_xx    # e.g. the heat equation
"""
from __future__ import annotations

import torch


def grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Point-wise derivative du/dx.

    u: (N, 1) outputs, where u[i] depends only on point i.
    x: (N, 1) inputs with requires_grad=True.
    Returns an (N, 1) tensor with du[i]/dx[i].

    Why this works: autograd computes vector-Jacobian products. Passing
    grad_outputs = ones is the same as differentiating sum(u). Because u[i]
    depends only on x[i], d(sum u)/dx[i] = du[i]/dx[i]: one backward pass
    gives the derivative at every point at once. (This is also why a PINN
    must never use layers that mix points, such as BatchNorm.)

    create_graph=True records how the derivative itself was computed, so that
      - we can differentiate it again (u_x -> u_xx), and
      - the optimizer can backpropagate through it into the weights.
    """
    (du_dx,) = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        allow_unused=True,
    )
    # If u does not depend on x at all (e.g. a derivative of a constant),
    # autograd returns None. Mathematically the derivative is zero.
    if du_dx is None:
        return torch.zeros_like(x)
    return du_dx


def nth_grad(u: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    """n-th derivative d^n u / dx^n by repeated differentiation."""
    if n < 1:
        raise ValueError("n must be at least 1")
    for _ in range(n):
        u = grad(u, x)
    return u


def laplacian(u: torch.Tensor, *coords: torch.Tensor) -> torch.Tensor:
    """u_xx + u_yy + ... over the given coordinates (each (N, 1))."""
    return sum(grad(grad(u, c), c) for c in coords)
