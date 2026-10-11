"""
The neural network that stands in for the unknown solution, e.g. u(t, x).

A PINN does not need a special architecture: a plain fully connected network
with a smooth activation is enough. What makes it "physics-informed" is the
loss, built in pinn/train.py from derivatives computed in pinn/derivatives.py.

Three design choices matter here:

1. tanh activation. The loss contains 2nd derivatives (u_xx), so the
   activation must have a useful 2nd derivative. ReLU is piecewise linear:
   its 2nd derivative is zero almost everywhere, so a ReLU network literally
   cannot express u_xx. tanh is smooth to every order.

2. Input scaling to [-1, 1]. Physical coordinates can live on any range
   (t in [0, 1], x in [0, 2], ...). tanh saturates for large inputs, so we
   map every input coordinate to [-1, 1] using the domain bounds (lb, ub).
   This is what Raissi's original code does too.

3. Xavier (Glorot) initialization with zero biases. It keeps the size of the
   signal roughly constant from layer to layer at the start of training, so
   tanh units start in their responsive range instead of saturated.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected network: in_dim -> [width] * depth -> out_dim.

    Args:
        in_dim:  number of input coordinates, e.g. 2 for (t, x).
        out_dim: number of output fields, e.g. 1 for u(t, x).
        width:   neurons per hidden layer.
        depth:   number of hidden layers.
        lb, ub:  lower / upper bound of each input coordinate. Used to scale
                 inputs to [-1, 1]. Defaults to [0, 1] for every coordinate.
        activation: activation class (default nn.Tanh).
        init:    "xavier_normal" (Glorot normal, default) or "xavier_uniform"
                 (Glorot uniform). The paper uses normal for 4.1 and uniform
                 for 4.2 and 4.3.
        normalize: scale inputs to [-1, 1] (default True). False feeds raw
                 coordinates to the first layer; used only to test how much
                 the scaling matters.

    Call it either with one (N, in_dim) tensor, or with one (N, 1) tensor per
    coordinate:  model(t, x)  ==  model(torch.cat([t, x], dim=1)).
    The second form is what PINNs use, because autograd can then
    differentiate u with respect to t and x separately.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 50,
        depth: int = 4,
        lb: Sequence[float] | None = None,
        ub: Sequence[float] | None = None,
        activation: type[nn.Module] = nn.Tanh,
        init: str = "xavier_normal",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        inits = {"xavier_normal": nn.init.xavier_normal_,
                 "xavier_uniform": nn.init.xavier_uniform_}
        if init not in inits:
            raise ValueError(f"init must be one of {sorted(inits)}")
        self.normalize = normalize
        dtype = torch.get_default_dtype()
        lb_t = torch.zeros(in_dim) if lb is None else torch.as_tensor(lb, dtype=dtype)
        ub_t = torch.ones(in_dim) if ub is None else torch.as_tensor(ub, dtype=dtype)
        if lb_t.numel() != in_dim or ub_t.numel() != in_dim:
            raise ValueError(f"lb and ub need {in_dim} entries each")
        if torch.any(ub_t <= lb_t):
            raise ValueError("every ub must be larger than its lb")
        # Buffers are saved in state_dict and follow .to(device), but are not trained.
        self.register_buffer("lb", lb_t.to(dtype).reshape(1, -1))
        self.register_buffer("ub", ub_t.to(dtype).reshape(1, -1))

        sizes = [in_dim] + [width] * depth + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            inits[init](linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(sizes) - 2:  # no activation after the output layer
                layers.append(activation())
        self.layers = nn.Sequential(*layers)

    def scale_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Map each coordinate from [lb, ub] to [-1, 1]."""
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=1)
        if self.normalize:
            x = self.scale_inputs(x)
        return self.layers(x)
