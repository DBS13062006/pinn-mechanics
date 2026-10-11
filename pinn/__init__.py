"""
pinn: the reusable PINN engine for this project.

Filled in step by step:
  utils.py        reproducibility, device, timing, run bookkeeping   (Step 1)
  model.py        fully connected network + output transforms        (Step 2)
  derivatives.py  autograd helpers for u_x, u_xx, ...                (Step 2)
  train.py        Adam -> L-BFGS training loop with logging          (Step 2)
  sampling.py     uniform and residual-adaptive (RAD) point sampling (Step 7)
"""

__version__ = "0.1.0"
