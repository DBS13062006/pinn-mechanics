"""
Step 3b: fix the slow, unreliable discovery of c in the oscillator problem.

Diagnosis from Step 3: c and k simply follow the curve the network currently
draws. Early on, the network has only fitted the first oscillations and is
flat afterwards; a flat tail looks like heavy damping (c ~ 1), and since
u = 0 satisfies the ODE exactly while the late observations hold only 1% of
the data's energy, training can get stuck there. On top of that, c and k
START WRONG (0.2 and 2.0), so the ODE term pushes the network toward the
wrong solution while the data pull toward the right one.

Fix ("two-stage", a classic idea in ODE parameter estimation, here used as a
warm start for the PINN):
  Stage 1  fit the network to the observations and initial condition only
           (no ODE, so nothing rewards a flat tail and no wrong physics)
  Stage 2  read c and k off the fitted curve: least squares on its
           derivatives (problems/oscillator.py: implied_parameters)
  Stage 3a full PINN loss with c, k FROZEN at those values: the now roughly
           correct physics pulls the whole curve, tail included, into shape
  Stage 3b full PINN loss, c and k free: final joint polish

Configurations (10 seeds each; the paper's Adam-100k baseline is read from
Step 3's committed table instead of being rerun):
  adam20k_lbfgs        best method from Step 3, for comparison
  two_stage            the full fix
  two_stage_no_init    ablation: skip stage 2 (and 3a): c, k start at 0.2, 2.0
  two_stage_no_freeze  ablation: skip stage 3a

Usage (from the repo root):
  python experiments/step3b_oscillator_fix.py --device cpu --jobs 5
  python experiments/step3b_oscillator_fix.py --quick        # smoke test
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import multiprocessing as mp
import pickle
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn import (MLP, TrainConfig, Trainer, env_info, make_run_dir,
                  relative_l2, save_json, set_seed, setup_torch)
from pinn.plotting import BLUES, INK_2, MUTED, SERIES, apply_style
from problems import oscillator as osc

# Paper section 4.3 settings (same as Step 3)
WIDTH, DEPTH, INIT, LR = 32, 3, "xavier_uniform", 5e-4
WEIGHTS = {"ode": 0.25, "ic": 0.25, "data": 0.25}
N_DATA, N_F = 50, 100
STAGE1_ADAM = 10_000
LBFGS_MAX = 50_000
SUCCESS_TOL = 0.01

CONFIGS = ["adam20k_lbfgs", "two_stage", "two_stage_no_init", "two_stage_no_freeze"]
LABELS = {
    "paper": "paper\n(Adam 100k)",
    "adam20k_lbfgs": "Adam 20k\n+ L-BFGS",
    "two_stage_no_init": "two-stage\nwithout stage 2",
    "two_stage_no_freeze": "two-stage\nwithout stage 3a",
    "two_stage": "two-stage\n(full fix)",
}
PLOT_ORDER = ["paper", "adam20k_lbfgs", "two_stage_no_init", "two_stage_no_freeze", "two_stage"]
STEP3_TABLE = Path("results/tables/step3_oscillator_runs.csv")


def run_one(config: str, seed: int, device_name: str, quick: bool = False) -> dict:
    """One configuration, one seed. Runs in its own process."""
    device = setup_torch(torch.float64, prefer=device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    set_seed(seed)
    model = MLP(1, 1, WIDTH, DEPTH, lb=[0.0], ub=[osc.T_MAX], init=INIT).to(device)
    c = torch.nn.Parameter(torch.tensor(osc.C_START, device=device))
    k = torch.nn.Parameter(torch.tensor(osc.K_START, device=device))
    t_d = osc.observation_times(N_DATA, device)
    u_d = osc.exact(t_d)
    t_f = osc.collocation_times(N_F, device)
    t_test = osc.test_times(10_000, device)
    u_test = osc.exact(t_test)
    t_diag = osc.test_times(2_000, device)
    net = list(model.parameters())
    lbfgs_max = 500 if quick else LBFGS_MAX

    def loss_data_only():
        return {"ic": osc.initial_condition_loss(model, device),
                "data": ((model(t_d) - u_d) ** 2).mean()}

    def loss_full():
        r = osc.residual(model, t_f, c, k)
        return {"ode": (r**2).mean(),
                "ic": osc.initial_condition_loss(model, device),
                "data": ((model(t_d) - u_d) ** 2).mean()}

    def cfg(adam, lbfgs):
        return TrainConfig(adam_iters=adam, adam_lr=LR, lbfgs_iters=lbfgs,
                           weights=WEIGHTS, verbose=False)

    stages: dict[str, float] = {}
    iters = 0
    t0 = time.perf_counter()
    c_star = k_star = None

    if config == "adam20k_lbfgs":
        tr = Trainer(net + [c, k], loss_full, cfg(1_000 if quick else 20_000, lbfgs_max)).train()
        iters += tr.iteration
        stages["joint"] = time.perf_counter() - t0
    else:
        # Stage 1: data + initial condition only
        tr = Trainer(net, loss_data_only, cfg(1_000 if quick else STAGE1_ADAM, lbfgs_max)).train()
        iters += tr.iteration
        stages["stage1"] = time.perf_counter() - t0
        # Stage 2: c and k implied by the fitted curve
        c_star, k_star = osc.implied_parameters(model, t_diag)
        if config != "two_stage_no_init":
            with torch.no_grad():
                c.fill_(c_star)
                k.fill_(k_star)
        # Stage 3a: physics on, c and k frozen
        if config == "two_stage":
            t1 = time.perf_counter()
            tr = Trainer(net, loss_full, cfg(0, lbfgs_max)).train()
            iters += tr.iteration
            stages["stage3a"] = time.perf_counter() - t1
        # Stage 3b: everything free
        t1 = time.perf_counter()
        tr = Trainer(net + [c, k], loss_full, cfg(0, lbfgs_max)).train()
        iters += tr.iteration
        stages["stage3b"] = time.perf_counter() - t1

    seconds = time.perf_counter() - t0
    with torch.no_grad():
        u_rel = relative_l2(model(t_test), u_test)
    t_g = t_test.clone().requires_grad_(True)
    ode_test = (osc.residual(model, t_g, c, k) ** 2).mean().item()
    c_err = abs(c.item() - osc.C_TRUE) / osc.C_TRUE
    k_err = abs(k.item() - osc.K_TRUE) / osc.K_TRUE
    out = {
        "config": config, "seed": seed, "device": str(device),
        "c": c.item(), "k": k.item(), "c_err": c_err, "k_err": k_err,
        "success": c_err < SUCCESS_TOL and k_err < SUCCESS_TOL,
        "u_rel_l2": u_rel, "ode_test": ode_test, "iters_total": iters, "seconds": seconds,
        "c_stage2": c_star, "k_stage2": k_star,
        "c_err_stage2": None if c_star is None else abs(c_star - osc.C_TRUE) / osc.C_TRUE,
        "k_err_stage2": None if k_star is None else abs(k_star - osc.K_TRUE) / osc.K_TRUE,
    }
    out.update({f"seconds_{name}": v for name, v in stages.items()})
    return out


# ---------------------------------------------------------------- outputs
def load_paper_baseline() -> list[dict]:
    """The paper-setting runs from Step 3 (5 seeds), so we don't rerun 100k-iteration jobs."""
    if not STEP3_TABLE.exists():
        return []
    rows = []
    with open(STEP3_TABLE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["config"] == "paper":
                rows.append({"config": "paper", "seed": int(r["seed"]),
                             "c_err": float(r["c_err"]), "k_err": float(r["k_err"]),
                             "success": r["success"] == "True", "seconds": float(r["seconds"])})
    return rows


def write_tables(runs: list[dict], paper: list[dict], tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    cols = ["config", "seed", "c", "k", "c_err", "k_err", "success", "u_rel_l2", "ode_test",
            "c_stage2", "k_stage2", "c_err_stage2", "k_err_stage2", "iters_total", "seconds",
            "seconds_stage1", "seconds_stage3a", "seconds_stage3b", "seconds_joint", "device"]
    with open(tables / "step3b_oscillator_fix_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({c: r.get(c, "") for c in cols})

    lines = ["# Step 3b: fixing slow discovery of c (oscillator, paper section 4.3)", "",
             "Truth: c = 0.4, k = 4. Start: c = 0.2, k = 2. A run succeeds if both end within 1%.",
             "The paper row is Step 3's 5 runs with the paper's exact settings.", "",
             "| method | succeeded | c error (median / worst) | k error (median / worst) | time per run (mean) |",
             "|---|---|---|---|---|"]
    for config in PLOT_ORDER:
        rs = paper if config == "paper" else [r for r in runs if r["config"] == config]
        if not rs:
            continue
        lines.append(
            f"| {LABELS[config].replace(chr(10), ' ')} | {sum(r['success'] for r in rs)}/{len(rs)} "
            f"| {statistics.median(r['c_err'] for r in rs):.1e} / {max(r['c_err'] for r in rs):.1e} "
            f"| {statistics.median(r['k_err'] for r in rs):.1e} / {max(r['k_err'] for r in rs):.1e} "
            f"| {statistics.mean(r['seconds'] for r in rs):.0f} s |")
    two = [r for r in runs if r["config"] == "two_stage"]
    if two:
        lines += ["", "Two-stage fix, after stage 2 only (c and k read off the data-only fit, "
                  "before any physics):",
                  f"c error median {statistics.median(r['c_err_stage2'] for r in two):.1e}, "
                  f"worst {max(r['c_err_stage2'] for r in two):.1e}; "
                  f"k error median {statistics.median(r['k_err_stage2'] for r in two):.1e}, "
                  f"worst {max(r['k_err_stage2'] for r in two):.1e}."]
    (tables / "step3b_oscillator_fix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def role_color(config: str) -> str:
    """Color by role, not by method: the x-axis label already names each method."""
    if config == "two_stage":
        return SERIES[0]                       # the fix: strong blue
    if config.startswith("two_stage"):
        return BLUES(0.25)                     # ablations: light blue
    return INK_2                               # baselines: gray


def strip(ax, groups: list[tuple[str, list[dict]]], key: str, log: bool = True) -> None:
    """One column of dots per method; hollow dots are failed runs."""
    for i, (config, rs) in enumerate(groups):
        if not rs:
            continue
        color = role_color(config)
        xs = i + np.linspace(-0.15, 0.15, len(rs))
        ok = np.array([r["success"] for r in rs])
        vals = np.array([r[key] for r in rs])
        ax.scatter(xs[ok], vals[ok], s=34, color=color, zorder=3)
        ax.scatter(xs[~ok], vals[~ok], s=34, facecolors="none", edgecolors=color,
                   linewidths=1.5, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(groups)), [LABELS[c] for c, _ in groups], fontsize=8)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.grid(axis="x", visible=False)


def make_figure(runs: list[dict], paper: list[dict], path: Path) -> None:
    apply_style()
    groups = [(c, paper if c == "paper" else [r for r in runs if r["config"] == c])
              for c in PLOT_ORDER]
    groups = [(c, rs) for c, rs in groups if rs]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    for ax, key, name, letter in [(axes[0, 0], "c_err", "c", "a"), (axes[0, 1], "k_err", "k", "b")]:
        strip(ax, groups, key)
        ax.axhline(SUCCESS_TOL, color=MUTED, lw=0.8, ls="--")
        ax.annotate("1% (success threshold)", xy=(1, SUCCESS_TOL), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=8)
        n_ok = ", ".join(f"{sum(r['success'] for r in rs)}/{len(rs)}" for _, rs in groups)
        ax.set(title=f"({letter}) Final error of {name}, every seed (hollow = failed)",
               ylabel=f"|{name} - true| / true")
        if name == "c":
            ax.text(0.01, 0.02, f"succeeded: {n_ok}", transform=ax.transAxes, color=INK_2, fontsize=8)

    ax = axes[1, 0]
    strip(ax, groups, "seconds", log=True)
    ax.set(title="(c) Wall-clock time per run", ylabel="seconds (log scale)")

    ax = axes[1, 1]
    two = [r for r in runs if r["config"] == "two_stage"]
    for r in two:
        color = SERIES[0] if r["success"] else MUTED
        ax.plot([0, 1], [r["c_err_stage2"], r["c_err"]], color=color, lw=1.0, alpha=0.8)
        ax.scatter([0, 1], [r["c_err_stage2"], r["c_err"]], s=26, color=color, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["after stage 2\n(data-only fit, no physics)", "after stage 3\n(full PINN)"])
    ax.set_xlim(-0.4, 1.4)
    ax.grid(axis="x", visible=False)
    ax.axhline(SUCCESS_TOL, color=MUTED, lw=0.8, ls="--")
    ax.set(title="(d) Two-stage fix: error of c before and after the physics",
           ylabel="|c - true| / true")

    fig.suptitle("Step 3b: fixing the slow, unreliable discovery of c", x=0.01, ha="left",
                 fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3b: oscillator fix")
    parser.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = args.seeds[:2]

    run_dir = make_run_dir("step3b_oscillator_fix" + ("_quick" if args.quick else ""))
    tasks = [(c, s) for c in args.configs for s in args.seeds]
    print(f"{len(tasks)} runs on {args.device} with {args.jobs} in parallel -> {run_dir}")

    runs, t0 = [], time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, c, s, args.device, args.quick) for c, s in tasks]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            runs.append(r)
            print(f"  [{n:>2}/{len(tasks)}] {r['config']:<20} seed {r['seed']}  c {r['c']:.5f}  "
                  f"k {r['k']:.5f}  {'ok    ' if r['success'] else 'FAILED'}  {r['seconds']:5.0f}s  "
                  f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    runs.sort(key=lambda r: (CONFIGS.index(r["config"]), r["seed"]))
    with open(run_dir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)
    paper = load_paper_baseline()
    if args.quick:
        tables, figure = run_dir / "tables", run_dir / "step3b_oscillator_fix_quick.png"
    else:
        tables, figure = Path("results/tables"), Path("results/figures/step3b_oscillator_fix.png")
    write_tables(runs, paper, tables)
    make_figure(runs, paper, figure)
    save_json({"args": vars(args), "env": env_info()}, run_dir / "metrics.json")

    print("\n" + (tables / "step3b_oscillator_fix_summary.md").read_text(encoding="utf-8"))
    print(f"figure: {figure}\ntables: {tables}\nraw runs: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
