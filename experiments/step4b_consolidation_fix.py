"""
Step 4b: fix the consolidation PINN's error near the corner (x = 0, t = 0).

Step 4 found that the error sits where the 500 random collocation points are
not (first 2% of the time window: ~11 points, most of the error), that the
training loss there says nothing about the real accuracy, and that every run
predicts a physically impossible pressure above 1.

Two fixes, same budget as Step 4's cheaper setting (Adam 10k, then L-BFGS):
  rad           Residual-based adaptive sampling: every 1000 Adam steps the
                500 collocation points are re-drawn where the PDE residual is
                large (pinn/sampling.py). General: knows nothing about corners.
  enriched      Corner-enriched output p = erf(x / 2 sqrt t) + x t N(x, t)
                (problems/consolidation.py). The erf term is the exact solution
                for a layer with no base, which carries the jump; the network
                only learns the smooth effect of the base. Uses one piece of
                physics knowledge, so it is problem-specific.
  enriched_rad  both together.
Baselines are read from Step 4's committed table (same seeds, same machine):
  paper (Adam 50k + L-BFGS) and Adam 10k + L-BFGS.

Usage (from the repo root):
  python experiments/step4b_consolidation_fix.py --device cpu --jobs 5
  python experiments/step4b_consolidation_fix.py --quick
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
from matplotlib.colors import LogNorm

# Shared settings and grids from Step 4 (same folder, so a plain import works).
import step4_consolidation as s4
from pinn import (MLP, TrainConfig, Trainer, env_info, make_run_dir,
                  relative_l2, save_json, set_seed, setup_torch)
from pinn.plotting import BLUES, INK, INK_2, MUTED, SERIES, apply_style
from pinn.sampling import RADHook
from problems import consolidation as cons

METHODS = ["rad", "enriched", "enriched_rad"]
LABELS = {"paper": "paper\n(Adam 50k)", "adam10k": "baseline\n(Adam 10k)", "rad": "adaptive\nsampling (RAD)",
          "enriched": "corner-enriched\noutput", "enriched_rad": "enriched\n+ RAD"}
ORDER = ["paper", "adam10k", "rad", "enriched", "enriched_rad"]
ADAM_ITERS = 10_000
RAD_EVERY, RAD_POOL = 1_000, 20_000
STEP4_TABLE = Path("results/tables/step4_consolidation_runs.csv")
METRICS = ["rel_l2", "max_err_t_ge_0.01", "max_err_t_ge_0.05", "err_share_t_lt_0.01",
           "coll_points_t_lt_0.01", "train_pde_loss", "test_pde_loss", "p_min", "p_max", "seconds"]


def run_one(method: str, seed: int, device_name: str, quick: bool = False) -> dict:
    device = setup_torch(torch.float64, prefer=device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, x_f, t_f, t_b, x_0 = s4.build(device, seed)
    enriched = method.startswith("enriched")
    out_fn = cons.pressure_enriched if enriched else cons.pressure

    def loss_fn():
        terms = {"pde": (cons.pde_residual(model, x_f, t_f, out_fn) ** 2).mean(),
                 "bc": (cons.neumann_residual(model, t_b, out_fn) ** 2).mean()}
        if not enriched:  # with enrichment p = 1 at t = 0 holds exactly
            terms["ic"] = (cons.initial_residual(model, x_0, out_fn) ** 2).mean()
        return terms

    hooks = []
    if method.endswith("rad"):
        hooks.append(RADHook(
            [x_f, t_f], lambda x, t: cons.pde_residual(model, x, t, out_fn),
            lambda m: cons.collocation_points(m, device=device),
            every=200 if quick else RAD_EVERY, n_pool=RAD_POOL))

    X, T, shape = cons.test_grid(device=device)
    P = torch.as_tensor(cons.exact(X.cpu().numpy(), T.cpu().numpy()), device=device)

    def eval_fn():
        return {"rel_l2": relative_l2(out_fn(model, X, T), P)}

    cfg = TrainConfig(adam_iters=1_000 if quick else ADAM_ITERS, adam_lr=s4.LR,
                      lbfgs_iters=500 if quick else s4.LBFGS_MAX, eval_every=1_000,
                      log_every=100, weights=s4.WEIGHTS, verbose=False)
    trainer = Trainer(model.parameters(), loss_fn, cfg, eval_fn=eval_fn, hooks=hooks).train()

    # --- diagnostics (same as Step 4) ---------------------------------------
    with torch.no_grad():
        p_uni = out_fn(model, X, T).cpu().numpy().reshape(shape)
    x_uni, t_uni = X.cpu().numpy().reshape(shape), T.cpu().numpy().reshape(shape)
    err_uni = np.abs(p_uni - cons.exact(x_uni, t_uni))
    XL, TL, shape_l = s4.log_grid(device)
    with torch.no_grad():
        p_log = out_fn(model, XL, TL).cpu().numpy().reshape(shape_l)
    err_log = np.abs(p_log - cons.exact(XL.cpu().numpy().reshape(shape_l), TL.cpu().numpy().reshape(shape_l)))
    xr, tr_ = cons.collocation_points(20_000, device=device)
    res_fresh = cons.pde_residual(model, xr, tr_, out_fn).detach()
    t_coll = t_f.detach().cpu().numpy().ravel()
    x_coll = x_f.detach().cpu().numpy().ravel()
    bands = cons.error_by_band(err_uni[1:], t_uni[1:])
    profiles = {}
    for tp in s4.PROFILE_TIMES:
        xs = torch.linspace(0, 1, 401, device=device).reshape(-1, 1)
        with torch.no_grad():
            profiles[tp] = out_fn(model, xs, torch.full_like(xs, tp)).cpu().numpy().ravel()

    final = trainer.log[-1]
    return {
        "config": method, "seed": seed, "device": str(device),
        "rel_l2": trainer.evals[-1]["rel_l2"],
        "max_err_t_ge_0.01": float(err_uni[t_uni >= 0.01].max()),
        "max_err_t_ge_0.05": float(err_uni[t_uni >= 0.05].max()),
        "err_share_t_lt_0.01": bands[0]["share_sq_err"],
        "coll_points_t_lt_0.01": int((t_coll < 0.01).sum()),
        "train_pde_loss": final["pde"], "test_pde_loss": float((res_fresh**2).mean()),
        "p_min": float(p_uni.min()), "p_max": float(p_uni.max()),
        "iters_total": trainer.iteration, "seconds": trainer.timings["total_seconds"],
        "max_err_vs_t": err_log.max(axis=1),
        "profiles": profiles, "x_coll": x_coll, "t_coll": t_coll,
    }


def load_step4() -> list[dict]:
    if not STEP4_TABLE.exists():
        return []
    rows = []
    with open(STEP4_TABLE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {"config": r["config"], "seed": int(r["seed"])}
            row.update({m: float(r[m]) for m in METRICS})
            rows.append(row)
    return rows


def write_tables(runs: list[dict], base: list[dict], tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    cols = ["config", "seed"] + METRICS + ["iters_total", "device"]
    with open(tables / "step4b_consolidation_fix_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    med = lambda rs, k: statistics.median(r[k] for r in rs)
    worst = lambda rs, k: max(r[k] for r in rs)
    lines = ["# Step 4b: fixing the consolidation error near the corner (paper section 4.1)", "",
             "Median over 5 seeds (worst in brackets). Paper and baseline rows are Step 4's runs.", "",
             "| method | rel. L2 error | max error, t >= 0.01 | max error, t >= 0.05 "
             "| highest predicted pressure (must be <= 1) | collocation points in t < 0.01 "
             "| PDE loss on fresh points | time per run |",
             "|---|---|---|---|---|---|---|---|"]
    for config in ORDER:
        rs = [r for r in (base if config in ("paper", "adam10k") else runs) if r["config"] == config]
        if not rs:
            continue
        lines.append(
            f"| {LABELS[config].replace(chr(10), ' ')} "
            f"| {med(rs, 'rel_l2'):.1e} ({worst(rs, 'rel_l2'):.1e}) "
            f"| {med(rs, 'max_err_t_ge_0.01'):.1e} ({worst(rs, 'max_err_t_ge_0.01'):.1e}) "
            f"| {med(rs, 'max_err_t_ge_0.05'):.1e} ({worst(rs, 'max_err_t_ge_0.05'):.1e}) "
            f"| {med(rs, 'p_max'):.3f} ({worst(rs, 'p_max'):.3f}) "
            f"| {med(rs, 'coll_points_t_lt_0.01'):.0f} of 500 "
            f"| {med(rs, 'test_pde_loss'):.1e} "
            f"| {statistics.mean(r['seconds'] for r in rs):.0f} s |")
    (tables / "step4b_consolidation_fix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip(ax, groups, key, log=True):
    for i, (config, rs) in enumerate(groups):
        color = SERIES[0] if config == "enriched" else (BLUES(0.3) if config in METHODS else INK_2)
        xs = i + np.linspace(-0.15, 0.15, len(rs))
        ax.scatter(xs, [r[key] for r in rs], s=34, color=color, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(groups)), [LABELS[c] for c, _ in groups], fontsize=8)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.grid(axis="x", visible=False)


def make_figure(runs: list[dict], base: list[dict], path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    groups = [(c, [r for r in (base if c in ("paper", "adam10k") else runs) if r["config"] == c]) for c in ORDER]
    groups = [(c, rs) for c, rs in groups if rs]

    # (a) max error vs time, median over seeds, per new method (baseline curve not stored in Step 4's CSV)
    ax = axes[0, 0]
    for i, method in enumerate(METHODS):
        rs = [r for r in runs if r["config"] == method]
        if rs:
            ax.plot(s4.T_LOG, np.median([r["max_err_vs_t"] for r in rs], axis=0), color=SERIES[i],
                    label=LABELS[method].replace("\n", " "))
    ax.axhline(0.045, color=INK_2, lw=0.9, ls=":")
    ax.annotate("paper's reported max error (0.045)", xy=(1, 0.045), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="(a) Max error over depth vs time (median of seeds)", xlabel="time t (log scale)",
           ylabel="max over x of |error|")
    ax.legend(loc="lower left", fontsize=8)

    # (b) profiles near the corner, enriched vs RAD, seed 0
    ax = axes[0, 1]
    xs = np.linspace(0, 1, 401)
    for tp, shade in zip(s4.PROFILE_TIMES[:2], [0.55, 1.0]):
        ax.plot(xs, cons.exact(xs, np.full_like(xs, tp)), color=INK, lw=1.0, ls=":")
        for i, method in enumerate(["rad", "enriched"]):
            r = next((r for r in runs if r["config"] == method and r["seed"] == 0), None)
            if r is not None:
                ax.plot(xs, r["profiles"][tp], color=SERIES[METHODS.index(method)], alpha=shade,
                        label=f"{LABELS[method].replace(chr(10), ' ')}, t = {tp:g}")
    ax.set(title="(b) Profiles near the corner (seed 0; dotted = exact)", xlabel="depth x",
           ylabel="pore pressure p", xlim=(0, 0.5), ylim=(-0.05, 1.15))
    ax.legend(loc="lower right", fontsize=7)

    # (c) where RAD put the points (seed 0)
    ax = axes[0, 2]
    r = next((r for r in runs if r["config"] == "rad" and r["seed"] == 0), None)
    if r is not None:
        keep = r["t_coll"] >= s4.T_LOG[0]
        ax.scatter(r["x_coll"][keep], r["t_coll"][keep], s=6, color=SERIES[0], alpha=0.7, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(s4.T_LOG[0], cons.T_MAX)
        n_small = int((r["t_coll"] < 0.01).sum())
        ax.set(title=f"(c) RAD's final 500 points (seed 0): {n_small} in t < 0.01",
               xlabel="depth x", ylabel="time t (log scale)")

    for ax, key, title, log in [
        (axes[1, 0], "rel_l2", "(d) Relative L2 error, every seed", True),
        (axes[1, 1], "max_err_t_ge_0.01", "(e) Max error for t >= 0.01, every seed", True),
        (axes[1, 2], "p_max", "(f) Highest predicted pressure (physics: <= 1)", False),
    ]:
        strip(ax, groups, key, log)
        ax.set_title(title)
    axes[1, 1].axhline(0.045, color=INK_2, lw=0.9, ls=":")
    axes[1, 2].axhline(1.0, color=INK_2, lw=0.9, ls=":")
    fig.suptitle("Step 4b: fixing the consolidation error near the corner", x=0.01, ha="left",
                 fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4b: consolidation fixes")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = args.seeds[:2]
    run_dir = make_run_dir("step4b_consolidation_fix" + ("_quick" if args.quick else ""))
    tasks = [(m, s) for m in args.methods for s in args.seeds]
    print(f"{len(tasks)} runs on {args.device} with {args.jobs} in parallel -> {run_dir}")
    runs, t0 = [], time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, m, s, args.device, args.quick) for m, s in tasks]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            runs.append(r)
            print(f"  [{n:>2}/{len(tasks)}] {r['config']:<13} seed {r['seed']}  rel L2 {r['rel_l2']:.2e}  "
                  f"max err (t>=0.01) {r['max_err_t_ge_0.01']:.1e}  p_max {r['p_max']:.3f}  "
                  f"{r['seconds']:5.0f}s  (elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    runs.sort(key=lambda r: (METHODS.index(r["config"]), r["seed"]))
    with open(run_dir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)
    base = load_step4()
    if args.quick:
        tables, figure = run_dir / "tables", run_dir / "step4b_consolidation_fix_quick.png"
    else:
        tables, figure = Path("results/tables"), Path("results/figures/step4b_consolidation_fix.png")
    write_tables(runs, base, tables)
    make_figure(runs, base, figure)
    save_json({"args": vars(args), "env": env_info()}, run_dir / "metrics.json")
    print("\n" + (tables / "step4b_consolidation_fix_summary.md").read_text(encoding="utf-8"))
    print(f"figure: {figure}\ntables: {tables}\nraw runs: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
