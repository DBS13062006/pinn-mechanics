"""
pinn: the reusable PINN engine for this project.

  utils.py        reproducibility, device/precision, timing, bookkeeping
  model.py        fully connected network with input scaling
  derivatives.py  autograd helpers: grad, nth_grad, laplacian
  metrics.py      relative L2, max error, parameter error
  train.py        Adam -> L-BFGS trainer with per-term logging
  sampling.py     residual-based adaptive sampling (RAD) hook
"""
from pinn.derivatives import grad, laplacian, nth_grad
from pinn.metrics import max_abs_error, relative_error, relative_l2
from pinn.model import MLP
from pinn.train import TrainConfig, Trainer
from pinn.utils import (count_params, env_info, get_device, make_run_dir,
                        save_json, set_seed, setup_torch, timer)

__version__ = "0.2.0"

__all__ = [
    "MLP", "Trainer", "TrainConfig",
    "grad", "nth_grad", "laplacian",
    "relative_l2", "max_abs_error", "relative_error",
    "set_seed", "setup_torch", "get_device", "timer",
    "env_info", "make_run_dir", "save_json", "count_params",
]
