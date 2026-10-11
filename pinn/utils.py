"""
Shared utilities: reproducibility, device selection, timing, run bookkeeping.

Every experiment in this repo calls `set_seed` first and saves `env_info()`
next to its results, so any number in the report can be traced back to the
exact seed, library versions and hardware that produced it.
"""
from __future__ import annotations

import json
import os
import platform
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every random number generator the project uses.

    A PINN run has two sources of randomness:
      1. the random initial weights of the network, and
      2. the random collocation / boundary / data points.
    Both come from these generators, so seeding all of them makes a run
    repeatable.

    deterministic=True also forbids GPU kernels whose results can vary from
    run to run (slightly slower). Use it for the final runs in the report.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU and every CUDA device
    if deterministic:
        # cuBLAS needs this env var to give repeatable matrix products.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Device, precision and timing
# --------------------------------------------------------------------------
def get_device(prefer: str = "cuda") -> torch.device:
    """Return the GPU if one is usable, otherwise the CPU."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_torch(dtype: torch.dtype = torch.float64, prefer: str = "cuda") -> torch.device:
    """Set the project-wide number precision and return the device to use.

    Every experiment calls this first. After it, every new network and tensor
    is float64 unless stated otherwise.

    Why float64: the step 1 benchmark on the RTX 4050 measured 12.6 ms per
    training step in float32 and 12.2 ms in float64. With networks this small
    the GPU spends its time launching many tiny operations, not on arithmetic,
    so the slower float64 arithmetic is hidden. Extra precision is therefore
    free, and L-BFGS needs it (see pinn/train.py).
    """
    torch.set_default_dtype(dtype)
    return get_device(prefer)


@contextmanager
def timer(device: torch.device | str | None = None):
    """Wall-clock timer that is correct on the GPU.

    GPU work is queued asynchronously: Python returns before the GPU has
    finished. Without torch.cuda.synchronize() you would time the queueing,
    not the computation.

        with timer(device) as t:
            ...work...
        print(t["seconds"])
    """
    is_cuda = device is not None and torch.device(device).type == "cuda"
    result: dict = {}
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield result
    finally:
        if is_cuda:
            torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - start


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------
def env_info() -> dict:
    """Library versions and hardware, saved alongside every result."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_memory_gb=round(props.total_memory / 1024**3, 2),
            compute_capability=f"{props.major}.{props.minor}",
            cuda_runtime=torch.version.cuda,
        )
    return info


def make_run_dir(name: str, root: str | Path = "results/runs") -> Path:
    """Create a fresh, timestamped folder for one experiment run."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(obj, path: str | Path) -> None:
    """Write a dict to JSON (numbers, configs, env info)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def count_params(model: torch.nn.Module) -> int:
    """Number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
