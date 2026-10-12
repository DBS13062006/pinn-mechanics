"""
Step 5: reproduce paper section 4.2 (steady heat conduction with a uniform
source on a square) with 5 seeds, and test the fix the paper suggests but
never implements: enforce T = 0 on the edges exactly by multiplying the
network by a function that vanishes there.

Configurations (the paper's network, points, loss weights and learning rate):
  paper    no output transform (T = 0 on the edges is only a loss penalty),
           Adam 50k then L-BFGS, as in the paper
  soft5k   same, but Adam 5k then L-BFGS
  hard5k   T = (1 - x^2)(1 - y^2) * network, Adam 5k then L-BFGS; the loss
           is just the PDE

Reference: the exact series solution (problems/heat.py), checked against our
own finite-difference solver. The paper says no analytical solution exists.

Usage (from the repo root):
  python experiments/step5_heat.py --bench
  python experiments/step5_heat.py --device cpu --jobs 5
  python experiments/step5_heat.py --device cuda
  python experiments/step5_heat.py --quick
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

from pinn import (MLP, TrainConfig, Trainer, env_info, make_run_dir,
                  relative_l2, save_json, set_seed, setup_torch)
from pinn.plotting import BLUES, INK_2, MUTED, SERIES, apply_style
from problems import heat

# Paper section 4.2 settings
WIDTH, DEPTH, INIT, LR = 50, 4, "xavier_uniform", 5e-4
WEIGHTS = {"pde": 0.5, "bc": 0.5}
CONFIGS = {
    "paper":  dict(adam_iters=50_000, hard=False),
    "soft5k": dict(adam_iters=5_000, hard=False),
    "hard5k": dict(adam_iters=5_000, hard=True),
}
LABELS = {"paper": "paper: soft BC\n(Adam 50k)", "soft5k": "soft BC\n(Adam 5k)",
          "hard5k": "hard BC\n(Adam 5k)"}
LBFGS_MAX = 50_000
PAPER_MAX_ERR = 5.2e-3          # colour-bar maximum of the paper's Fig. 3
# Bins of distance to the nearest edge, matched to the 0.01 test-grid spacing. The first bin is the edge itself.
DIST_BINS = np.array([0.0, 1e-9, 0.015, 0.035, 0.075, 0.15, 0.3, 1.0001])
DIST_X = np.array([0.004, 0.01, 0.025, 0.055, 0.11, 0.22, 0.6])     # where each bin is drawn
EDGE_FLOOR = 1e-12                                                  # guard for exact zeros on the log axis


def build(device, seed: int):
    set_seed(seed)
    model = MLP(2, 1, WIDTH, DEPTH, lb=[-1, -1], ub=[1, 1], init=INIT).to(device)
    x_f, y_f = heat.collocation_points(device=device)
    x_b, y_b = heat.boundary_points(device=device)
    return model, x_f, y_f, x_b, y_b


def make_loss(model, out_fn, hard, x_f, y_f, x_b, y_b):
    def loss_fn():
        terms = {"pde": (heat.pde_residual(model, x_f, y_f, out_fn) ** 2).mean()}
        if not hard:
            terms["bc"] = (out_fn(model, x_b, y_b) ** 2).mean()
        return terms
    return loss_fn


def run_one(config: str, seed: int, device_name: str, quick: bool = False) -> dict:
    device = setup_torch(torch.float64, prefer=device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    spec = CONFIGS[config]
    out_fn = heat.temperature_hard if spec["hard"] else heat.temperature_soft
    model, x_f, y_f, x_b, y_b = build(device, seed)
    loss_fn = make_loss(model, out_fn, spec["hard"], x_f, y_f, x_b, y_b)

    X, Y, shape = heat.test_grid(device=device)
    T_ex = torch.as_tensor(heat.exact(X.cpu().numpy(), Y.cpu().numpy()), device=device)

    def eval_fn():
        return {"rel_l2": relative_l2(out_fn(model, X, Y), T_ex)}

    cfg = TrainConfig(adam_iters=500 if quick else spec["adam_iters"], adam_lr=LR,
                      lbfgs_iters=300 if quick else LBFGS_MAX, eval_every=1_000,
                      log_every=100, weights=WEIGHTS, verbose=False)
    trainer = Trainer(model.parameters(), loss_fn, cfg, eval_fn=eval_fn).train()

    # --- diagnostics ----------------------------------------------------------
    with torch.no_grad():
        T = out_fn(model, X, Y).cpu().numpy().reshape(shape)
    xg, yg = X.cpu().numpy().reshape(shape), Y.cpu().numpy().reshape(shape)
    err = np.abs(T - heat.exact(xg, yg))
    d = heat.distance_to_boundary(xg, yg)
    edge, band, inner = d < 1e-12, (d > 1e-12) & (d < 0.05), d >= 0.2

    xr, yr = heat.collocation_points(20_000, device=device)
    r2 = (heat.pde_residual(model, xr, yr, out_fn).detach() ** 2).cpu().numpy().ravel()
    xr_np, yr_np = xr.detach().cpu().numpy().ravel(), yr.detach().cpu().numpy().ravel()
    corner = (1 - np.abs(xr_np) < 0.1) & (1 - np.abs(yr_np) < 0.1)

    idx = np.digitize(d.ravel(), DIST_BINS) - 1
    err_vs_dist = np.array([err.ravel()[idx == i].max() if np.any(idx == i) else np.nan
                            for i in range(len(DIST_BINS) - 1)])
    return {
        "config": config, "seed": seed, "device": str(device),
        "rel_l2": trainer.evals[-1]["rel_l2"], "max_err": float(err.max()),
        "max_err_edge": float(err[edge].max()), "max_err_band": float(err[band].max()),
        "max_err_interior": float(err[inner].max()),
        "share_sq_err_band": float((err[d < 0.05] ** 2).sum() / (err**2).sum()),
        "T_min": float(T.min()), "T_center_err": float(abs(T[shape[0] // 2, shape[1] // 2] - heat.T_CENTER)),
        "train_pde_loss": trainer.log[-1]["pde"], "test_pde_loss": float(r2.mean()),
        "test_pde_loss_corners": float(r2[corner].mean()), "test_pde_loss_rest": float(r2[~corner].mean()),
        "iters_total": trainer.iteration, "seconds": trainer.timings["total_seconds"],
        "err_map": err.astype(np.float32), "err_vs_dist": err_vs_dist, "log": trainer.log,
    }


# ---------------------------------------------------------------- benchmark
def bench(steps: int = 60) -> None:
    print(f"Timing {steps} Adam steps of the paper's heat setup (soft BC)...")
    results = {}
    for name in ["cpu", "cuda"]:
        if name == "cuda" and not torch.cuda.is_available():
            continue
        device = setup_torch(torch.float64, prefer=name)
        if name == "cpu":
            torch.set_num_threads(1)
        model, x_f, y_f, x_b, y_b = build(device, 0)
        loss_fn = make_loss(model, heat.temperature_soft, False, x_f, y_f, x_b, y_b)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        def step():
            opt.zero_grad()
            sum(WEIGHTS[k] * v for k, v in loss_fn().items()).backward()
            opt.step()

        for _ in range(10):
            step()
        if name == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        if name == "cuda":
            torch.cuda.synchronize()
        results[name] = 1000 * (time.perf_counter() - t0) / steps
        print(f"  {name:<5} {results[name]:6.2f} ms/step")
    total = 5 * sum(c["adam_iters"] for c in CONFIGS.values())
    print(f"\nAll 15 runs: {total:,} Adam steps, plus L-BFGS (assume +40%).")
    for name, ms in results.items():
        print(f"  {name:<5} one run at a time: ~{total * ms / 60000 * 1.4:5.0f} min")
    if "cpu" in results:
        longest = CONFIGS["paper"]["adam_iters"] + 2 * CONFIGS["soft5k"]["adam_iters"]
        print(f"  cpu   5 runs in parallel (--jobs 5): ~{longest * results['cpu'] / 60000 * 1.4:5.0f} min"
              " (needs 5 free cores; slower if the laptop throttles)")
    print(f"\nCPU cores on this machine: {mp.cpu_count()}")


# ---------------------------------------------------------------- outputs
COLS = ["config", "seed", "rel_l2", "max_err", "max_err_edge", "max_err_band", "max_err_interior",
        "share_sq_err_band", "T_min", "T_center_err", "train_pde_loss", "test_pde_loss",
        "test_pde_loss_corners", "test_pde_loss_rest", "iters_total", "seconds", "device"]


def write_tables(runs: list[dict], tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "step5_heat_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    med = lambda rs, k: statistics.median(r[k] for r in rs)
    worst = lambda rs, k: max(r[k] for r in rs)
    lines = ["# Step 5: steady heat conduction (paper section 4.2)", "",
             "Errors against the exact series solution. Median over 5 seeds (worst in brackets).",
             f"The paper's Fig. 3 shows a maximum error of about {PAPER_MAX_ERR:.1e}, on the edges.", "",
             "| configuration | rel. L2 error | max error | max error on the edges | max error, interior "
             "(>= 0.2 from an edge) | share of squared error within 0.05 of an edge (10% of the area) "
             "| lowest temperature (physics: >= 0) | PDE loss: training / fresh points | time per run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for config in CONFIGS:
        rs = [r for r in runs if r["config"] == config]
        if not rs:
            continue
        lines.append(
            f"| {LABELS[config].replace(chr(10), ' ')} "
            f"| {med(rs, 'rel_l2'):.1e} ({worst(rs, 'rel_l2'):.1e}) "
            f"| {med(rs, 'max_err'):.1e} ({worst(rs, 'max_err'):.1e}) "
            f"| {med(rs, 'max_err_edge'):.1e} ({worst(rs, 'max_err_edge'):.1e}) "
            f"| {med(rs, 'max_err_interior'):.1e} ({worst(rs, 'max_err_interior'):.1e}) "
            f"| {med(rs, 'share_sq_err_band'):.0%} "
            f"| {med(rs, 'T_min'):.1e} ({min(r['T_min'] for r in rs):.1e}) "
            f"| {med(rs, 'train_pde_loss'):.1e} / {med(rs, 'test_pde_loss'):.1e} "
            f"| {statistics.mean(r['seconds'] for r in rs):.0f} s |")
    hard = [r for r in runs if r["config"] == "hard5k"]
    if hard:
        lines += ["", "Hard BC, PDE loss on fresh points near the four corners (within 0.1 of both edges) vs "
                  f"elsewhere: {med(hard, 'test_pde_loss_corners'):.1e} vs {med(hard, 'test_pde_loss_rest'):.1e} "
                  "(median)."]
    (tables / "step5_heat_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip(ax, groups, key, log=True):
    for i, (config, rs) in enumerate(groups):
        color = SERIES[0] if config == "hard5k" else INK_2
        xs = i + np.linspace(-0.15, 0.15, len(rs))
        ax.scatter(xs, [r[key] for r in rs], s=34, color=color, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(groups)), [LABELS[c] for c, _ in groups], fontsize=8)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.grid(axis="x", visible=False)


def make_figure(runs: list[dict], path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    g = np.linspace(-1, 1, 201)
    Xg, Yg = np.meshgrid(g, g)

    ax = axes[0, 0]
    im = ax.pcolormesh(g, g, heat.exact(Xg, Yg), cmap="seq_blue", shading="auto")
    fig.colorbar(im, ax=ax, label="T")
    ax.contour(g, g, heat.exact(Xg, Yg), levels=6, colors="white", linewidths=0.6)
    ax.set(title="(a) Exact series solution (reference)", xlabel="x", ylabel="y", aspect="equal")
    ax.grid(False)

    norm = LogNorm(vmin=1e-6, vmax=1e-2)
    for ax, config, letter in [(axes[0, 1], "paper", "b"), (axes[0, 2], "hard5k", "c")]:
        r = next((r for r in runs if r["config"] == config and r["seed"] == 0), None)
        if r is None:
            ax.axis("off")
            continue
        im = ax.pcolormesh(g, g, np.maximum(r["err_map"], 1e-7), cmap="seq_orange", norm=norm, shading="auto")
        fig.colorbar(im, ax=ax, label="|PINN - exact|")
        ax.set(title=f"({letter}) Error, {LABELS[config].replace(chr(10), ' ')}, seed 0",
               xlabel="x", ylabel="y", aspect="equal")
        ax.grid(False)

    ax = axes[1, 0]
    for i, config in enumerate(CONFIGS):
        rs = [r for r in runs if r["config"] == config]
        if rs:
            color = SERIES[0] if config == "hard5k" else (INK_2 if config == "paper" else MUTED)
            curve = np.maximum(np.nanmedian([r["err_vs_dist"] for r in rs], axis=0), EDGE_FLOOR)
            ax.plot(DIST_X, curve, "o-", ms=4, color=color, label=LABELS[config].replace("\n", " "))
    ax.axvline(0.007, color=MUTED, lw=0.6)
    ax.annotate("left of the line: on the edge itself\n(hard BC ~1e-9: zero up to the\naccuracy of the exact series)",
                xy=(0.004, 2e-9), xytext=(10, 4), textcoords="offset points", color=INK_2, fontsize=7)
    ax.axhline(PAPER_MAX_ERR, color=INK_2, lw=0.9, ls=":")
    ax.annotate("paper's max error", xy=(1, PAPER_MAX_ERR), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="(d) Max error vs distance to the nearest edge", xlabel="distance to edge (log scale)",
           ylabel="max |error| (median of seeds)")
    ax.legend(loc="center right", fontsize=8)

    groups = [(c, [r for r in runs if r["config"] == c]) for c in CONFIGS]
    groups = [(c, rs) for c, rs in groups if rs]
    strip(axes[1, 1], groups, "rel_l2")
    axes[1, 1].set_title("(e) Relative L2 error, every seed")
    strip(axes[1, 2], groups, "T_min", log=False)
    axes[1, 2].axhline(0.0, color=INK_2, lw=0.9, ls=":")
    axes[1, 2].set_title("(f) Lowest predicted temperature (physics: >= 0)")

    fig.suptitle("Step 5: steady heat conduction (paper section 4.2)", x=0.01, ha="left", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5: steady heat conduction")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--resume", metavar="RUN_DIR",
                        help="continue an interrupted run: reuse the trainings already saved in RUN_DIR")
    args = parser.parse_args()
    if args.bench:
        bench()
        return 0
    if args.quick:
        args.seeds = args.seeds[:2]
    run_dir = Path(args.resume) if args.resume else make_run_dir("step5_heat" + ("_quick" if args.quick else ""))
    # Every finished training is saved here immediately, so an interrupted run loses
    # only the trainings still in progress. --resume RUN_DIR picks up from these files.
    partial = run_dir / "partial"
    partial.mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted(partial.glob("*.pkl")):
        with open(f, "rb") as fh:
            runs.append(pickle.load(fh))
    done = {(r["config"], r["seed"]) for r in runs}
    tasks = sorted([(c, s) for c in args.configs for s in args.seeds if (c, s) not in done],
                   key=lambda cs: -CONFIGS[cs[0]]["adam_iters"])     # longest first
    if done:
        print(f"resuming: {len(done)} trainings already saved in {partial}, {len(tasks)} to go")
    print(f"{len(tasks)} runs on {args.device} with {args.jobs} in parallel -> {run_dir}")
    print(f"(if this gets interrupted, rerun the same command with --resume {run_dir})")
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, c, s, args.device, args.quick) for c, s in tasks]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            runs.append(r)
            with open(partial / f"{r['config']}_seed{r['seed']}.pkl", "wb") as fh:
                pickle.dump(r, fh)
            print(f"  [{n:>2}/{len(tasks)}] {r['config']:<7} seed {r['seed']}  rel L2 {r['rel_l2']:.2e}  "
                  f"max err {r['max_err']:.1e}  T_min {r['T_min']:+.1e}  {r['seconds']:6.0f}s  "
                  f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    runs.sort(key=lambda r: (list(CONFIGS).index(r["config"]), r["seed"]))
    with open(run_dir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)
    if args.quick:
        tables, figure = run_dir / "tables", run_dir / "step5_heat_quick.png"
    else:
        tables, figure = Path("results/tables"), Path("results/figures/step5_heat.png")
    write_tables(runs, tables)
    make_figure(runs, figure)
    save_json({"args": vars(args), "env": env_info()}, run_dir / "metrics.json")
    print("\n" + (tables / "step5_heat_summary.md").read_text(encoding="utf-8"))
    print(f"figure: {figure}\ntables: {tables}\nraw runs: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
