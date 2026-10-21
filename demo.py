"""
Live demo: the paper's method vs our fix, on every problem, in a few minutes.

    python demo.py                 # all four problems, one after another
    python demo.py oscillator      # one problem: oscillator | consolidation | heat | poromechanics
    python demo.py all --no-show   # save the figures without opening windows

Each problem trains two PINNs from scratch with the SAME small budget, prints
progress as it goes, then opens a figure comparing them with the exact
solution (also saved to results/demo/). The budgets here are much smaller
than in experiments/ so that the demo runs live in front of an audience; the
full multi-seed numbers are in results/tables/.

Typical times on an Apple M5 (CPU): about 1 minute per problem.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from pinn import MLP, TrainConfig, Trainer, max_abs_error, relative_l2, set_seed, setup_torch
from problems import consolidation as cons
from problems import heat
from problems import oscillator as osc
from problems import poromechanics as poro

OUT = Path("results/demo")
DEVICE = torch.device("cpu")
SCALE = 1.0          # --quick halves every budget


def n(iters: int) -> int:
    return 0 if iters == 0 else max(1, int(iters * SCALE))


def banner(title: str, lines: list[str]) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    for line in lines:
        print(f"  {line}")
    print("=" * 78)


def train(name: str, params, loss_fn, adam: int, lbfgs: int, lr: float, weights=None, eval_fn=None):
    print(f"\n-> {name}")
    cfg = TrainConfig(adam_iters=n(adam), adam_lr=lr, lbfgs_iters=n(lbfgs), weights=weights or {},
                      print_every=max(n(adam) // 4, 1), eval_every=max(n(adam) // 4, 1),
                      lbfgs_chunk=50, verbose=True)
    return Trainer(params, loss_fn, cfg, eval_fn=eval_fn).train()


def verdict(metric: str, paper: float, ours: float) -> str:
    ratio = paper / ours if ours > 0 else float("inf")
    return f"{metric}: paper method {paper:.2e}  ->  our fix {ours:.2e}   ({ratio:,.0f}x better)"


def finish(fig, name: str, summary: list[str], show: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=130)
    print("\n" + "-" * 78)
    for line in summary:
        print("  " + line)
    print(f"  figure saved: {path}")
    print("-" * 78)
    if show:
        import matplotlib.pyplot as plt
        plt.show()


# ============================================================================ oscillator
def demo_oscillator(show: bool) -> None:
    import matplotlib.pyplot as plt
    banner("Paper 4.3  Inverse problem: find damping c and stiffness k of a vibrating mass",
           ["u'' + c u' + k u = 0, from 50 measurements of u(t).  Truth c = 0.4, k = 4; start 0.2, 2.",
            "Paper: train network, c and k together.  Ours: two-stage warm start."])
    t_d = osc.observation_times(50, DEVICE)
    u_d = osc.exact(t_d)
    W = {"ode": 0.25, "ic": 0.25, "data": 0.25}

    def setup(seed=0):
        set_seed(seed)
        model = MLP(1, 1, 32, 3, lb=[0.0], ub=[osc.T_MAX], init="xavier_uniform")
        c = torch.nn.Parameter(torch.tensor(osc.C_START))
        k = torch.nn.Parameter(torch.tensor(osc.K_START))
        t_f = osc.collocation_times(100, DEVICE)
        full = lambda: {"ode": (osc.residual(model, t_f, c, k) ** 2).mean(),
                        "ic": osc.initial_condition_loss(model, DEVICE),
                        "data": ((model(t_d) - u_d) ** 2).mean()}
        return model, c, k, full

    # paper: joint Adam
    m1, c1, k1, full1 = setup()
    t0 = time.perf_counter()
    train("paper method: Adam, network + c + k together", list(m1.parameters()) + [c1, k1], full1,
          adam=4_000, lbfgs=0, lr=5e-4, weights=W,
          eval_fn=lambda: {"c": c1.item(), "k": k1.item()})
    s1 = time.perf_counter() - t0

    # ours: two-stage
    m2, c2, k2, full2 = setup()
    t0 = time.perf_counter()
    data_only = lambda: {"ic": osc.initial_condition_loss(m2, DEVICE), "data": ((m2(t_d) - u_d) ** 2).mean()}
    train("our fix, stage 1: fit the data only (no physics yet)", list(m2.parameters()), data_only,
          adam=3_000, lbfgs=2_000, lr=5e-4, weights=W)
    cs, ks = osc.implied_parameters(m2, osc.test_times(2_000, DEVICE))
    print(f"\n-> stage 2: read c, k off the fitted curve by least squares:  c = {cs:.4f}, k = {ks:.4f}")
    with torch.no_grad():
        c2.fill_(cs)
        k2.fill_(ks)
    train("stage 3a: physics on, c and k frozen", list(m2.parameters()), full2, adam=0, lbfgs=1_000, lr=5e-4, weights=W)
    train("stage 3b: everything free", list(m2.parameters()) + [c2, k2], full2, adam=0, lbfgs=2_000, lr=5e-4,
          weights=W, eval_fn=lambda: {"c": c2.item(), "k": k2.item()})
    s2 = time.perf_counter() - t0

    err = lambda c, k: max(abs(c.item() - osc.C_TRUE) / osc.C_TRUE, abs(k.item() - osc.K_TRUE) / osc.K_TRUE)
    tt = osc.test_times(1_000, DEVICE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, m, c, k, title, sec in [(axes[0], m1, c1, k1, "paper method", s1),
                                    (axes[1], m2, c2, k2, "our fix (two-stage)", s2)]:
        with torch.no_grad():
            ax.plot(tt.numpy(), osc.exact(tt).numpy(), color="0.2", ls=":", lw=1.5, label="true motion")
            ax.plot(tt.numpy(), m(tt).numpy(), color="#2a78d6", label="PINN")
        ax.scatter(t_d.numpy(), u_d.numpy(), s=12, color="#eb6834", zorder=3, label="50 measurements")
        ax.set(title=f"{title}:  c = {c.item():.4f}, k = {k.item():.4f}  ({sec:.0f} s)",
               xlabel="time t", ylabel="displacement u")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Damped oscillator: discovering c and k (truth 0.4 and 4)", fontweight="bold")
    finish(fig, "oscillator", [
        f"truth: c = 0.4, k = 4",
        f"paper method ({s1:.0f} s): c = {c1.item():.4f}, k = {k1.item():.4f}   worst error {err(c1, k1):.1%}",
        f"our fix      ({s2:.0f} s): c = {c2.item():.4f}, k = {k2.item():.4f}   worst error {err(c2, k2):.2%}",
    ], show)


# ============================================================================ consolidation
def demo_consolidation(show: bool) -> None:
    import matplotlib.pyplot as plt
    banner("Paper 4.1  Forward problem: soil consolidation (pore pressure dissipating)",
           ["p_t = p_xx,  p(0,t) = 0 drained top,  p_x(1,t) = 0 rigid base,  p(x,0) = 1.",
            "Paper: p = x N(x,t).  Ours: p = erf(x / 2 sqrt t) + x t N(x,t), the corner jump built in."])
    X, T, shape = cons.test_grid(device=DEVICE)
    P = torch.as_tensor(cons.exact(X.numpy(), T.numpy()))
    W = {"pde": 0.25, "bc": 0.25 * 250 / 375, "ic": 0.25 * 125 / 375}
    results = {}
    for key, out_fn, title in [("paper", cons.pressure, "paper method: p = x N"),
                               ("ours", cons.pressure_enriched, "our fix: corner-enriched output")]:
        set_seed(0)
        model = MLP(2, 1, 50, 5, lb=[0, 0], ub=[1, cons.T_MAX], init="xavier_normal")
        x_f, t_f = cons.collocation_points()
        t_b, x_0 = cons.boundary_times(), cons.initial_positions()

        def loss_fn(model=model, out_fn=out_fn):
            terms = {"pde": (cons.pde_residual(model, x_f, t_f, out_fn) ** 2).mean(),
                     "bc": (cons.neumann_residual(model, t_b, out_fn) ** 2).mean()}
            if out_fn is cons.pressure:
                terms["ic"] = (cons.initial_residual(model, x_0, out_fn) ** 2).mean()
            return terms

        t0 = time.perf_counter()
        train(title, model.parameters(), loss_fn, adam=2_000, lbfgs=1_500, lr=1e-3, weights=W,
              eval_fn=lambda model=model, out_fn=out_fn: {"rel_l2": relative_l2(out_fn(model, X, T), P)})
        with torch.no_grad():
            pred = out_fn(model, X, T)
        results[key] = (pred.numpy().reshape(shape), relative_l2(pred, P), time.perf_counter() - t0, model, out_fn)

    ref = P.numpy().reshape(shape)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    xs = np.linspace(0, 1, 301)
    for ax, key, title in [(axes[0], "paper", "paper method"), (axes[1], "ours", "our fix")]:
        _, rel, sec, model, out_fn = results[key]
        for tp, col in zip((0.005, 0.02, 0.1, 0.4), ("#86b6ef", "#3987e5", "#184f95", "#0d366b")):
            xt = torch.as_tensor(xs).reshape(-1, 1)
            with torch.no_grad():
                ax.plot(xs, out_fn(model, xt, torch.full_like(xt, tp)).numpy().ravel(), color=col, label=f"t = {tp:g}")
            ax.plot(xs, cons.exact(xs, np.full_like(xs, tp)), color="0.2", ls=":", lw=1.0)
        ax.axhline(1.0, color="#eb6834", lw=0.8, ls="--")
        ax.set(title=f"{title}: rel. L2 error {rel:.1e}  ({sec:.0f} s)", xlabel="depth x",
               ylabel="pore pressure p (dotted = exact)", ylim=(-0.05, 1.15))
        ax.legend(loc="lower right", fontsize=8)
    ax = axes[2]
    tl = np.linspace(0, cons.T_MAX, shape[0])
    for key, col, lab in [("paper", "0.35", "paper method"), ("ours", "#2a78d6", "our fix")]:
        ax.semilogy(tl[1:], np.abs(results[key][0] - ref).max(axis=1)[1:], color=col, label=lab)
    ax.set(title="max error over depth vs time", xlabel="time t", ylabel="max |error|")
    ax.legend()
    fig.suptitle("1D consolidation: the error sits at the corner (x = 0, t = 0)", fontweight="bold")
    finish(fig, "consolidation", [
        verdict("relative L2 error", results["paper"][1], results["ours"][1]),
        f"highest pressure (physics: <= 1): paper {results['paper'][0].max():.3f}, ours {results['ours'][0].max():.3f}",
    ], show)


# ============================================================================ heat
def demo_heat(show: bool) -> None:
    import matplotlib.pyplot as plt
    banner("Paper 4.2  Forward problem: steady heat conduction in a square plate",
           ["T_xx + T_yy + 1 = 0 on [-1,1]^2,  T = 0 on the edges.",
            "Paper: T = N (edges as a penalty).  Ours: T = (1-x^2)(1-y^2) N + exact corner terms."])
    X, Y, shape = heat.test_grid(device=DEVICE)
    Tref = torch.as_tensor(heat.exact(X.numpy(), Y.numpy()))
    results = {}
    for key, out_fn, title in [("paper", heat.temperature_soft, "paper method: soft boundary condition"),
                               ("ours", heat.temperature_corner, "our fix: exact edges + corner terms")]:
        set_seed(0)
        model = MLP(2, 1, 50, 4, lb=[-1, -1], ub=[1, 1], init="xavier_uniform")
        x_f, y_f = heat.collocation_points()
        x_b, y_b = heat.boundary_points()

        def loss_fn(model=model, out_fn=out_fn):
            terms = {"pde": (heat.pde_residual(model, x_f, y_f, out_fn) ** 2).mean()}
            if out_fn is heat.temperature_soft:
                terms["bc"] = (out_fn(model, x_b, y_b) ** 2).mean()
            return terms

        t0 = time.perf_counter()
        train(title, model.parameters(), loss_fn, adam=1_500, lbfgs=1_500, lr=5e-4,
              weights={"pde": 0.5, "bc": 0.5},
              eval_fn=lambda model=model, out_fn=out_fn: {"max_err": max_abs_error(out_fn(model, X, Y), Tref)})
        with torch.no_grad():
            pred = out_fn(model, X, Y)
        results[key] = (pred.numpy().reshape(shape), max_abs_error(pred, Tref), time.perf_counter() - t0)

    ref = Tref.numpy().reshape(shape)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    im = axes[0].imshow(ref, extent=(-1, 1, -1, 1), origin="lower", cmap="inferno")
    axes[0].set(title="exact temperature", xlabel="x", ylabel="y")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    vmax = np.abs(results["paper"][0] - ref).max()
    for ax, key, title in [(axes[1], "paper", "paper method"), (axes[2], "ours", "our fix")]:
        pred, mx, sec = results[key]
        im = ax.imshow(np.abs(pred - ref), extent=(-1, 1, -1, 1), origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        ax.set(title=f"{title}: |error|, max {mx:.1e}  ({sec:.0f} s)", xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Steady heat conduction: same colour scale for both error maps", fontweight="bold")
    finish(fig, "heat", [verdict("max error", results["paper"][1], results["ours"][1])], show)


# ============================================================================ poromechanics
def demo_poromechanics(show: bool) -> None:
    import matplotlib.pyplot as plt
    banner("Paper 5 (future work)  Coupled poromechanics, Eqs. (5.1)-(5.2): discover c",
           ["One PINN for soil displacement U AND pore pressure P, coupled equations as written.",
            f"Find the consolidation coefficient c (true {poro.C_TRUE}, start {poro.C_START}) "
            f"from {poro.N_OBS} settlement readings with 1% noise.",
            "Paper recipe: U = (1-x) N_u, P = x N_p.  Ours: corner-enriched U and P."])
    t_obs, u_obs = poro.settlement_readings(poro.OBS_NOISE, draw=0, device=DEVICE)
    results = {}
    for key, fields, title in [("paper", poro.fields_paper, "paper recipe"),
                               ("ours", poro.fields_enriched, "our fix: corner-enriched outputs")]:
        set_seed(0)
        model = MLP(2, 2, 40, 4, lb=[0, 0], ub=[1, 1], init="xavier_normal")
        log_c = torch.tensor(np.log(poro.C_START), requires_grad=True)
        x_f, t_f = poro.collocation_points(t_max=1.0)
        t_b, x_0 = poro.boundary_times(t_max=1.0), poro.initial_positions()

        def loss_fn(model=model, fields=fields, log_c=log_c):
            c = torch.exp(log_c)
            mass, eq = poro.residuals(fields, model, x_f, t_f, c)
            terms = {"mass": (mass**2).mean(), "equil": (eq**2).mean(),
                     "load": (poro.top_load_residual(fields, model, t_b, c) ** 2).mean(),
                     "flow": (poro.base_flow_residual(fields, model, t_b, c) ** 2).mean(),
                     "data": ((fields(model, torch.zeros_like(t_obs), t_obs, c)[0] - u_obs) ** 2).mean()}
            if fields is poro.fields_paper:
                terms["ic"] = (poro.initial_residual(fields, model, x_0, c) ** 2).mean()
            return terms

        t0 = time.perf_counter()
        train(title, list(model.parameters()) + [log_c], loss_fn, adam=1_500, lbfgs=500, lr=1e-3,
              eval_fn=lambda log_c=log_c: {"c": float(torch.exp(log_c).detach())})
        results[key] = (model, fields, float(torch.exp(log_c).detach()), time.perf_counter() - t0)

    tt = torch.linspace(0, 1, 300).reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(tt.numpy(), poro.settlement(poro.C_TRUE * tt).numpy(), color="0.2", ls=":", lw=1.5, label="true settlement")
    ax.scatter(t_obs.numpy(), u_obs.numpy(), s=16, color="#eb6834", zorder=3, label="20 noisy readings")
    for key, col in [("paper", "0.45"), ("ours", "#2a78d6")]:
        model, fields, c, _ = results[key]
        with torch.no_grad():
            ax.plot(tt.numpy(), fields(model, torch.zeros_like(tt), tt, torch.tensor(c))[0].numpy(), color=col,
                    label=f"{'paper recipe' if key == 'paper' else 'our fix'}: c = {c:.3f}")
    ax.set(title="Surface settlement over time", xlabel="time t (field units)", ylabel="settlement (1 = final)")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    xs = torch.linspace(0, 1, 200).reshape(-1, 1)
    model, fields, c, _ = results["ours"]
    for tp, col in zip((0.02, 0.1, 0.4, 1.0), ("#86b6ef", "#3987e5", "#184f95", "#0d366b")):
        with torch.no_grad():
            _, Pq = fields(model, xs, torch.full_like(xs, tp), torch.tensor(c))
        ax.plot(xs.numpy(), Pq.numpy(), color=col, label=f"t = {tp:g}")
        ax.plot(xs.numpy(), poro.exact_pressure(xs, poro.C_TRUE * torch.full_like(xs, tp)).numpy(),
                color="0.2", ls=":", lw=1.0)
    ax.set(title="Pore pressure inside the soil, our fix (dotted = exact)", xlabel="depth x", ylabel="pore pressure P")
    ax.legend(fontsize=8)
    fig.suptitle("Coupled poromechanics: consolidation coefficient from settlement readings", fontweight="bold")
    fit = poro.classical_fit(t_obs.numpy(), u_obs.numpy())
    cp, co = results["paper"][2], results["ours"][2]
    finish(fig, "poromechanics", [
        f"true c = {poro.C_TRUE}",
        f"paper recipe ({results['paper'][3]:.0f} s): c = {cp:.4f}   error {abs(cp - poro.C_TRUE) / poro.C_TRUE:.1%}",
        f"our fix      ({results['ours'][3]:.0f} s): c = {co:.4f}   error {abs(co - poro.C_TRUE) / poro.C_TRUE:.2%}",
        f"classical curve fit (knows the closed-form answer): c = {fit['c']:.4f}",
    ], show)


DEMOS = {"oscillator": demo_oscillator, "consolidation": demo_consolidation,
         "heat": demo_heat, "poromechanics": demo_poromechanics}


def main() -> int:
    global SCALE
    parser = argparse.ArgumentParser(description="Live demo: the paper's PINN vs our fixes")
    parser.add_argument("problem", nargs="?", default="all", choices=["all", *DEMOS])
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    parser.add_argument("--quick", action="store_true", help="halve every training budget")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    if args.quick:
        SCALE = 0.5
    setup_torch(torch.float64, prefer="cpu")
    from pinn.plotting import apply_style
    apply_style()
    start = time.perf_counter()
    for name in (DEMOS if args.problem == "all" else [args.problem]):
        DEMOS[name](show=not args.no_show)
    print(f"\nDemo finished in {time.perf_counter() - start:.0f} s. Figures in {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
