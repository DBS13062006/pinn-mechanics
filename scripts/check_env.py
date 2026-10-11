"""
Step 1 environment check. Run from the repo root:

    python scripts/check_env.py

What it verifies (each one is something later steps rely on):
  [1] PyTorch is installed and can see your GPU.
  [2] Autograd gives correct 1st and 2nd derivatives, of a known function and
      of a neural network with respect to its INPUTS. This is exactly how a
      PINN evaluates u_t, u_x, u_xx inside its loss.
  [3] Seeding makes training runs exactly repeatable.
  [4] Speed of one PINN-style training step in float32 vs float64, on CPU and
      GPU. This decides where Adam (float32) and L-BFGS (float64) will run.
"""
from __future__ import annotations

import sys

import torch

from pinn.utils import env_info, save_json, set_seed, timer

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


# --------------------------------------------------------------------------
# Two tiny helpers. Proper versions live in pinn/ from Step 2 onwards.
# --------------------------------------------------------------------------
def d(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Derivative of a point-wise function with respect to its inputs.

    create_graph=True keeps the computation graph of the derivative itself,
    so we can differentiate it again (u_x -> u_xx) and later backpropagate
    through it into the network weights.

    Why outputs.sum()? autograd.grad needs a scalar to start from. Output i
    depends only on input i (every point is processed independently), so the
    gradient of the sum with respect to input i is exactly du_i/dx_i.
    This is also why a PINN must never use layers that mix points together,
    such as BatchNorm: the trick would silently give wrong derivatives.
    """
    return torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]


def mlp(n_in: int, width: int, depth: int, n_out: int = 1,
        dtype: torch.dtype = torch.float64) -> torch.nn.Sequential:
    """Fully connected tanh network: n_in -> [width] * depth -> n_out."""
    layers, prev = [], n_in
    for _ in range(depth):
        layers += [torch.nn.Linear(prev, width), torch.nn.Tanh()]
        prev = width
    layers.append(torch.nn.Linear(prev, n_out))
    return torch.nn.Sequential(*layers).to(dtype)


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# --------------------------------------------------------------------------
# [1] Environment
# --------------------------------------------------------------------------
def check_environment() -> tuple[dict, str]:
    header("[1] Environment")
    info = env_info()
    info["cpu_threads"] = torch.get_num_threads()
    for key, value in info.items():
        print(f"  {key:<20} {value}")
    if info["cuda_available"]:
        return info, PASS
    print("\n  No usable GPU found. Everything still runs on the CPU, just slower.")
    print("  On Windows the usual cause is a CPU-only PyTorch build: reinstall")
    print("  with the CUDA command from https://pytorch.org/get-started/locally/")
    return info, WARN


# --------------------------------------------------------------------------
# [2] Autograd correctness
# --------------------------------------------------------------------------
def check_autograd() -> tuple[dict, str]:
    header("[2] Autograd derivatives")

    # (a) A function whose derivatives we know: u = sin(x).
    x = torch.linspace(0, 2 * torch.pi, 1001, dtype=torch.float64, requires_grad=True)
    u = torch.sin(x)
    u_x = d(u, x)
    u_xx = d(u_x, x)
    err_ux = (u_x - torch.cos(x)).abs().max().item()
    err_uxx = (u_xx + torch.sin(x)).abs().max().item()
    print("  (a) u = sin(x):  should give u_x = cos(x), u_xx = -sin(x)")
    print(f"      max |u_x  - cos x| = {err_ux:.2e}")
    print(f"      max |u_xx + sin x| = {err_uxx:.2e}")

    # (b) A neural network u(t, x), checked against finite differences.
    set_seed(0)
    net = mlp(n_in=2, width=50, depth=3)
    tx = torch.rand(200, 2, dtype=torch.float64, requires_grad=True)  # columns: t, x
    u = net(tx)
    grad_u = d(u, tx)                     # shape (N, 2) = [u_t, u_x]
    u_t, u_x = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = d(u_x, tx)[:, 1:2]             # d(u_x)/d(t, x) = [u_xt, u_xx]; keep u_xx

    h = 1e-3
    base = tx.detach()
    e_t = torch.tensor([[h, 0.0]], dtype=torch.float64)
    e_x = torch.tensor([[0.0, h]], dtype=torch.float64)
    with torch.no_grad():
        fd_t = (net(base + e_t) - net(base - e_t)) / (2 * h)
        fd_xx = (net(base + e_x) - 2 * net(base) + net(base - e_x)) / h**2
    err_t = (u_t - fd_t).abs().max().item()
    err_xx = (u_xx - fd_xx).abs().max().item()
    print("\n  (b) tanh network u(t, x), autograd vs finite differences (h = 1e-3)")
    print(f"      max |u_t  autograd - FD| = {err_t:.2e}")
    print(f"      max |u_xx autograd - FD| = {err_xx:.2e}")
    print("      (small but nonzero: the FD formulas themselves have O(h^2) error)")

    ok = err_ux < 1e-12 and err_uxx < 1e-12 and err_t < 1e-5 and err_xx < 1e-5
    results = dict(sin_ux=err_ux, sin_uxx=err_uxx, net_ut_vs_fd=err_t, net_uxx_vs_fd=err_xx)
    return results, PASS if ok else FAIL


# --------------------------------------------------------------------------
# [3] Reproducibility
# --------------------------------------------------------------------------
def tiny_training_run(seed: int, device: torch.device) -> torch.Tensor:
    """30 Adam steps on a heat-equation residual u_t - u_xx. Returns weights."""
    set_seed(seed)
    net = mlp(2, 20, 2, dtype=torch.float32).to(device)
    pts = torch.rand(256, 2, device=device, requires_grad=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(30):
        opt.zero_grad()
        g = d(net(pts), pts)
        u_xx = d(g[:, 1:2], pts)[:, 1:2]
        loss = ((g[:, 0:1] - u_xx) ** 2).mean()
        loss.backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in net.parameters()]).cpu()


def check_reproducibility() -> tuple[dict, str]:
    header("[3] Reproducibility (same seed -> identical weights)")
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    results, status = {}, PASS
    for dev in devices:
        if dev.type == "cuda":
            # Deterministic mode so the GPU repeats itself bit for bit.
            set_seed(0, deterministic=True)
        a = tiny_training_run(0, dev)
        b = tiny_training_run(0, dev)
        c = tiny_training_run(1, dev)
        same = torch.equal(a, b)
        differs = not torch.equal(a, c)
        print(f"  {dev.type:<5} seed 0 twice -> identical: {same}   "
              f"seed 0 vs seed 1 -> different: {differs}")
        results[dev.type] = dict(identical=same, different_seed_differs=differs)
        if not (same and differs):
            status = FAIL
        if dev.type == "cuda":
            torch.use_deterministic_algorithms(False)
    return results, status


# --------------------------------------------------------------------------
# [4] Speed: float32 vs float64, CPU vs GPU
# --------------------------------------------------------------------------
def ms_per_pinn_step(device: torch.device, dtype: torch.dtype,
                     n_points: int = 2000, steps: int = 30, warmup: int = 5) -> float:
    """Time one PINN training step with the paper's consolidation network
    (5 hidden layers x 50 tanh units) on n_points collocation points."""
    set_seed(0)
    net = mlp(2, 50, 5, dtype=dtype).to(device)
    pts = torch.rand(n_points, 2, device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        g = d(net(pts), pts)
        u_xx = d(g[:, 1:2], pts)[:, 1:2]
        loss = ((g[:, 0:1] - u_xx) ** 2).mean()
        loss.backward()
        opt.step()

    for _ in range(warmup):  # first steps include one-off setup cost
        step()
    with timer(device) as t:
        for _ in range(steps):
            step()
    return 1000 * t["seconds"] / steps


def check_speed() -> tuple[dict, str]:
    header("[4] Speed of one PINN training step (2000 points, 5x50 tanh net)")
    combos = [("cpu", torch.float32), ("cpu", torch.float64)]
    if torch.cuda.is_available():
        combos += [("cuda", torch.float32), ("cuda", torch.float64)]

    results = {}
    print(f"  {'device':<7}{'dtype':<10}{'ms/step':>10}{'20k Adam steps':>18}")
    for dev_name, dtype in combos:
        ms = ms_per_pinn_step(torch.device(dev_name), dtype)
        name = str(dtype).replace("torch.", "")
        results[f"{dev_name}_{name}"] = ms
        print(f"  {dev_name:<7}{name:<10}{ms:>10.2f}{ms * 20000 / 60000:>14.1f} min")

    print()
    if torch.cuda.is_available():
        fastest32 = "cuda" if results["cuda_float32"] < results["cpu_float32"] else "cpu"
        fastest64 = "cuda" if results["cuda_float64"] < results["cpu_float64"] else "cpu"
        print(f"  -> Run Adam (float32) on: {fastest32}")
        print(f"  -> Run L-BFGS (float64) on: {fastest64}")
        print("     (laptop GPUs have weak float64 hardware, so the CPU can win there)")
    else:
        print("  -> CPU only for now. Fine for development; use Colab for big sweeps.")
    return results, PASS


# --------------------------------------------------------------------------
def main() -> int:
    env, s1 = check_environment()
    grads, s2 = check_autograd()
    repro, s3 = check_reproducibility()
    speed, s4 = check_speed()

    header("Summary")
    statuses = {"[1] environment": s1, "[2] autograd": s2,
                "[3] reproducibility": s3, "[4] speed": s4}
    for name, status in statuses.items():
        print(f"  {name:<22} {status}")

    save_json(dict(env=env, autograd=grads, reproducibility=repro,
                   ms_per_step=speed, status=statuses), "results/env_check.json")
    print("\n  Saved full report to results/env_check.json")
    return 1 if FAIL in statuses.values() else 0


if __name__ == "__main__":
    sys.exit(main())
