"""
Step 2: check the PINN engine on a problem with a known exact answer.

Problem: the 1D heat (diffusion) equation

    u_t = kappa * u_xx,          t in [0, 1], x in [0, 1]
    u(0, x) = sin(pi x)          initial condition
    u(t, 0) = u(t, 1) = 0        boundary conditions

with exact solution u = exp(-kappa pi^2 t) sin(pi x), kappa = 0.1.

Why this problem: Terzaghi consolidation (paper section 4.1) is the SAME
equation (p_t = c_v p_zz). Here the initial and boundary conditions agree
(both are 0 at x = 0), so the solution is smooth everywhere and a working
engine must solve it accurately. In Terzaghi's problem they disagree at the
corner (p = 1 initially, p = 0 at the drained boundary), which is exactly
what makes that problem harder. That contrast becomes Improvement 2 (Step 7).

Two parts, one per problem class from the paper:
  A. forward: kappa known, find u(t, x)        -> accuracy of the solution
  B. inverse: kappa unknown, find it from data -> accuracy of kappa

Run from the repo root:
    python experiments/step2_engine_check.py           # full run (GPU: a few minutes)
    python experiments/step2_engine_check.py --quick   # small run for a CPU
"""
from __future__ import annotations

import argparse
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from pinn import (MLP, TrainConfig, Trainer, env_info, grad, make_run_dir,
                  max_abs_error, relative_error, relative_l2, save_json,
                  set_seed, setup_torch)
from pinn.plotting import (INK_2, SERIES, apply_style, mark_lbfgs_start,
                           plot_loss_history)

KAPPA_TRUE = 0.1
KAPPA_START = 1.0  # inverse problem starts 10x away from the truth

SETTINGS = {
    "full":  dict(width=50, depth=4, n_f=2000, n_ic=200, n_bc=200, n_data=500,
                  adam_iters=5000, lbfgs_iters=3000),
    "quick": dict(width=20, depth=3, n_f=500, n_ic=50, n_bc=50, n_data=200,
                  adam_iters=2000, lbfgs_iters=1000),
}


def exact(t: torch.Tensor, x: torch.Tensor, kappa: float = KAPPA_TRUE) -> torch.Tensor:
    return torch.exp(-kappa * math.pi**2 * t) * torch.sin(math.pi * x)


def heat_residual(model, t, x, kappa):
    """u_t - kappa u_xx at the points (t, x). Zero wherever the PDE holds."""
    u = model(t, x)
    u_t = grad(u, t)
    u_xx = grad(grad(u, x), x)
    return u_t - kappa * u_xx


def test_grid(device, n=101):
    """Regular grid for measuring error. Never used for training."""
    t = torch.linspace(0, 1, n, device=device)
    x = torch.linspace(0, 1, n, device=device)
    T, X = torch.meshgrid(t, x, indexing="ij")
    return T.reshape(-1, 1), X.reshape(-1, 1), n


def rand(n, device, requires_grad=False):
    return torch.rand(n, 1, device=device).requires_grad_(requires_grad)


# ------------------------------------------------------------------- part A
def run_forward(s: dict, device, seed: int):
    set_seed(seed)
    model = MLP(2, 1, s["width"], s["depth"], lb=[0, 0], ub=[1, 1]).to(device)

    # Training points
    x_ic = rand(s["n_ic"], device)                       # t = 0, x random
    t_ic = torch.zeros_like(x_ic)
    u_ic = torch.sin(math.pi * x_ic)
    t_bc = rand(s["n_bc"], device)                       # x = 0 or 1, t random
    x_bc = torch.randint(0, 2, (s["n_bc"], 1), device=device).to(t_bc.dtype)
    t_f = rand(s["n_f"], device, requires_grad=True)     # collocation points
    x_f = rand(s["n_f"], device, requires_grad=True)

    def loss_fn():
        r = heat_residual(model, t_f, x_f, KAPPA_TRUE)
        return {
            "pde": (r**2).mean(),
            "ic": ((model(t_ic, x_ic) - u_ic) ** 2).mean(),
            "bc": (model(t_bc, x_bc) ** 2).mean(),
        }

    T, X, n = test_grid(device)
    U = exact(T, X)

    def eval_fn():
        pred = model(T, X)
        return {"rel_l2": relative_l2(pred, U), "max_err": max_abs_error(pred, U)}

    cfg = TrainConfig(adam_iters=s["adam_iters"], lbfgs_iters=s["lbfgs_iters"],
                      eval_every=500, print_every=1000)
    trainer = Trainer(model.parameters(), loss_fn, cfg, eval_fn=eval_fn).train()

    with torch.no_grad():
        pred = model(T, X).reshape(n, n).cpu().numpy()
    fields = dict(pred=pred, exact=U.reshape(n, n).cpu().numpy())
    return model, trainer, fields


# ------------------------------------------------------------------- part B
def run_inverse(s: dict, device, seed: int):
    set_seed(seed)
    model = MLP(2, 1, s["width"], s["depth"], lb=[0, 0], ub=[1, 1]).to(device)
    # The unknown physical constant is just one more trainable number.
    kappa = torch.nn.Parameter(torch.tensor(KAPPA_START, device=device))

    # Measurements of u at random points inside the domain (noise-free here;
    # noisy and sparse data is Improvement 3, Step 8).
    t_d, x_d = rand(s["n_data"], device), rand(s["n_data"], device)
    u_d = exact(t_d, x_d)
    t_f = rand(s["n_f"], device, requires_grad=True)
    x_f = rand(s["n_f"], device, requires_grad=True)

    def loss_fn():
        r = heat_residual(model, t_f, x_f, kappa)
        return {"data": ((model(t_d, x_d) - u_d) ** 2).mean(), "pde": (r**2).mean()}

    T, X, _ = test_grid(device)
    U = exact(T, X)

    def eval_fn():
        return {"kappa": kappa.item(),
                "kappa_rel_err": relative_error(kappa.item(), KAPPA_TRUE),
                "rel_l2": relative_l2(model(T, X), U)}

    cfg = TrainConfig(adam_iters=s["adam_iters"], lbfgs_iters=s["lbfgs_iters"],
                      eval_every=100, print_every=1000)
    params = list(model.parameters()) + [kappa]
    trainer = Trainer(params, loss_fn, cfg, eval_fn=eval_fn).train()
    return model, kappa, trainer


# ------------------------------------------------------------------- figure
def make_figure(fields, fwd: Trainer, inv: Trainer, path):
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    extent = [0, 1, 0, 1]  # x on the horizontal axis, t on the vertical axis

    ax = axes[0, 0]
    im = ax.imshow(fields["pred"], origin="lower", extent=extent, aspect="auto", cmap="seq_blue")
    fig.colorbar(im, ax=ax, label="u")
    ax.set(title="(a) Forward: PINN solution u(t, x)", xlabel="x", ylabel="t")
    ax.grid(False)

    ax = axes[0, 1]
    err = np.abs(fields["pred"] - fields["exact"])
    im = ax.imshow(np.maximum(err, 1e-8), origin="lower", extent=extent, aspect="auto",
                   cmap="seq_orange", norm=LogNorm())
    fig.colorbar(im, ax=ax, label="|PINN - exact|")
    rel = fwd.evals[-1]["rel_l2"]
    ax.set(title=f"(b) Forward: absolute error (rel. L2 = {rel:.1e})", xlabel="x", ylabel="t")
    ax.grid(False)

    plot_loss_history(axes[0, 2], fwd.log, ["pde", "ic", "bc"], "(c) Forward: loss terms")

    ax = axes[1, 0]
    it = [e["iter"] for e in inv.evals]
    ax.plot(it, [e["kappa"] for e in inv.evals], color=SERIES[0], label="PINN estimate")
    ax.axhline(KAPPA_TRUE, color=INK_2, lw=1.0, ls=":", label=f"true value {KAPPA_TRUE}")
    ax.set(title="(d) Inverse: estimate of kappa", xlabel="iteration", ylabel="kappa")
    ax.legend(loc="center left")

    ax = axes[1, 1]
    ax.plot(it, [e["kappa_rel_err"] for e in inv.evals], color=SERIES[0])
    ax.set_yscale("log")
    final = inv.evals[-1]["kappa_rel_err"]
    ax.set(title=f"(e) Inverse: relative error of kappa (final {final:.1e})",
           xlabel="iteration", ylabel="|kappa - true| / true")
    mark_lbfgs_start(ax, inv.log, label=False)
    mark_lbfgs_start(axes[1, 0], inv.log, label=False)

    plot_loss_history(axes[1, 2], inv.log, ["data", "pde"], "(f) Inverse: loss terms")

    fig.suptitle("Step 2 engine check: heat equation u_t = 0.1 u_xx", x=0.01, ha="left",
                 fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true", help="small settings for a CPU")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = setup_torch(torch.float64)
    mode = "quick" if args.quick else "full"
    s = SETTINGS[mode]
    run_dir = make_run_dir(f"step2_engine_check_{mode}")
    print(f"device {device}, mode {mode}, results -> {run_dir}")

    print("\n[A] forward problem: kappa known, find u")
    model_f, fwd, fields = run_forward(s, device, args.seed)
    print("\n[B] inverse problem: find kappa from measurements of u")
    model_i, kappa, inv = run_inverse(s, device, args.seed)

    fwd.save(run_dir / "forward")
    inv.save(run_dir / "inverse")
    torch.save(model_f.state_dict(), run_dir / "forward" / "model.pt")
    torch.save({"model": model_i.state_dict(), "kappa": kappa.detach().cpu()},
               run_dir / "inverse" / "model.pt")

    figure = ("results/figures/step2_engine_check.png" if mode == "full"
              else run_dir / "step2_engine_check_quick.png")
    make_figure(fields, fwd, inv, figure)

    f_m, i_m = fwd.evals[-1], inv.evals[-1]
    metrics = {
        "mode": mode, "seed": args.seed, "settings": s, "env": env_info(),
        "forward": {"rel_l2": f_m["rel_l2"], "max_err": f_m["max_err"], **fwd.timings},
        "inverse": {"kappa": i_m["kappa"], "kappa_rel_err": i_m["kappa_rel_err"],
                    "rel_l2": i_m["rel_l2"], **inv.timings},
    }
    save_json(metrics, run_dir / "metrics.json")

    # Pass/fail thresholds: generous, they only catch a broken engine.
    ok_f = f_m["rel_l2"] < 1e-2
    ok_i = i_m["kappa_rel_err"] < 1e-2
    print("\n" + "=" * 60)
    print(f"[A] forward  rel. L2 error   {f_m['rel_l2']:.2e}   max error {f_m['max_err']:.2e}"
          f"   {'PASS' if ok_f else 'FAIL'}")
    print(f"[B] inverse  kappa = {i_m['kappa']:.6f} (true {KAPPA_TRUE})"
          f"   rel. error {i_m['kappa_rel_err']:.2e}   {'PASS' if ok_i else 'FAIL'}")
    print(f"    time: forward {fwd.timings['total_seconds']:.0f}s, "
          f"inverse {inv.timings['total_seconds']:.0f}s")
    print(f"    figure: {figure}")
    print(f"    metrics: {run_dir / 'metrics.json'}")
    return 0 if (ok_f and ok_i) else 1


if __name__ == "__main__":
    sys.exit(main())
