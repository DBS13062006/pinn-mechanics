"""
Step 5b: why both heat PINNs go wrong at the corners of the square, and a fix.

Step 5 found that the soft-BC PINN (the paper's) has its largest error on the
edges, and that the hard-BC PINN's PDE loss on fresh points is ~100x its
training value, almost all of it near the four corners.

The cause is the square itself (details in problems/heat.py). Along the top
edge T = 0, so T_xx = 0 there; along the right edge T = 0, so T_yy = 0 and the
PDE forces T_xx = -1. At the corner T_xx would have to be both. The exact
solution gets around this because its T_xx depends on the direction of
approach (-2 theta / pi); a tanh network is smooth and cannot. So every
network must break either the boundary condition (soft BC) or the PDE (hard
BC: the residual is exactly 1 at each corner, whatever the network).

Configurations (Step 5's network, points, loss weights; Adam 5k then L-BFGS):
  soft5k    Step 5's soft BC, rerun with corner diagnostics
  hard5k    Step 5's hard BC, rerun with corner diagnostics
  corner5k  hard BC + the exact local solution at each corner
            (problems/heat.py: temperature_corner); the network only learns
            the smooth remainder. Same idea as the erf term in Step 4b.
soft5k and hard5k train exactly as in Step 5, so on the same machine their
errors must match Step 5's table (checked and reported).

Usage (from the repo root):
  python experiments/step5b_heat_corners.py --device cpu --jobs 3
  python experiments/step5b_heat_corners.py --quick
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import math
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

# Shared settings from Step 5 (same folder, so a plain import works).
import step5_heat as s5
from pinn import (TrainConfig, Trainer, env_info, make_run_dir, relative_l2,
                  save_json, setup_torch)
from pinn.plotting import INK, INK_2, MUTED, SERIES, apply_style
from problems import heat

ADAM_ITERS = 5_000
CONFIGS = {"soft5k": heat.temperature_soft, "hard5k": heat.temperature_hard,
           "corner5k": heat.temperature_corner}
LABELS = {"soft5k": "soft BC", "hard5k": "hard BC", "corner5k": "hard BC +\ncorner terms"}
COLORS = {"soft5k": MUTED, "hard5k": SERIES[0], "corner5k": SERIES[2]}
STEP5_TABLE = Path("results/tables/step5_heat_runs.csv")
NEAR_CORNER = 0.15                      # "near a corner": 4 quarter discs = 1.8% of the area
# Max error vs distance to the nearest corner, on the 0.01 test grid. The first bin is the corner point itself.
CORNER_BINS = np.array([0.0, 1e-9, 0.015, 0.035, 0.075, 0.15, 0.3, 0.6, 1.5])
CORNER_X = np.array([0.004, 0.012, 0.025, 0.05, 0.11, 0.21, 0.42, 0.9])
# PDE residual vs distance to the nearest corner, on points log-uniform in that distance.
N_PROFILE = 8_000
PROFILE_BINS = np.logspace(-4, 0, 9)
PROFILE_X = np.sqrt(PROFILE_BINS[:-1] * PROFILE_BINS[1:])
EVAL_SEED = 1234                        # same fresh points for every run


def binned(values, where, bins, reduce):
    idx = np.digitize(where, bins) - 1
    return np.array([reduce(values[idx == i]) if np.any(idx == i) else np.nan for i in range(len(bins) - 1)])


def run_one(config: str, seed: int, device_name: str, quick: bool = False) -> dict:
    device = setup_torch(torch.float64, prefer=device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    out_fn = CONFIGS[config]
    soft = config == "soft5k"
    # Training is identical to Step 5's run_one, so soft5k / hard5k reproduce Step 5 exactly.
    model, x_f, y_f, x_b, y_b = s5.build(device, seed)
    loss_fn = s5.make_loss(model, out_fn, not soft, x_f, y_f, x_b, y_b)

    X, Y, shape = heat.test_grid(device=device)
    T_ex = torch.as_tensor(heat.exact(X.cpu().numpy(), Y.cpu().numpy()), device=device)

    def eval_fn():
        return {"rel_l2": relative_l2(out_fn(model, X, Y), T_ex)}

    cfg = TrainConfig(adam_iters=500 if quick else ADAM_ITERS, adam_lr=s5.LR,
                      lbfgs_iters=300 if quick else s5.LBFGS_MAX, eval_every=1_000,
                      log_every=100, weights=s5.WEIGHTS, verbose=False)
    trainer = Trainer(model.parameters(), loss_fn, cfg, eval_fn=eval_fn).train()

    # --- error on the test grid --------------------------------------------------
    with torch.no_grad():
        T = out_fn(model, X, Y).cpu().numpy().reshape(shape)
    xg, yg = X.cpu().numpy().reshape(shape), Y.cpu().numpy().reshape(shape)
    err = np.abs(T - heat.exact(xg, yg))
    dc = heat.distance_to_corner(xg, yg)
    i_max = np.unravel_index(err.argmax(), shape)
    near = dc < NEAR_CORNER

    # --- PDE residual on fresh points (the same points for every run) -----------
    gen = torch.Generator().manual_seed(EVAL_SEED)
    xu = (2 * torch.rand(20_000, 1, generator=gen) - 1).to(device).requires_grad_(True)
    yu = (2 * torch.rand(20_000, 1, generator=gen) - 1).to(device).requires_grad_(True)
    r2 = (heat.pde_residual(model, xu, yu, out_fn).detach() ** 2).cpu().numpy().ravel()
    xn, yn = xu.detach().cpu().numpy().ravel(), yu.detach().cpu().numpy().ravel()
    corner_box = (1 - np.abs(xn) < 0.1) & (1 - np.abs(yn) < 0.1)       # Step 5's definition
    xc, yc, rc = heat.corner_points(N_PROFILE, generator=gen, device=device)
    rc2 = (heat.pde_residual(model, xc, yc, out_fn).detach() ** 2).cpu().numpy().ravel()

    return {
        "config": config, "seed": seed, "device": str(device),
        "rel_l2": trainer.evals[-1]["rel_l2"], "max_err": float(err.max()),
        "max_err_x": float(xg[i_max]), "max_err_y": float(yg[i_max]),
        "max_err_dist_corner": float(dc[i_max]),
        "max_err_near_corners": float(err[near].max()), "max_err_rest": float(err[~near].max()),
        "share_sq_err_near_corners": float((err[near] ** 2).sum() / (err**2).sum()),
        "err_at_corner_points": float(err[dc < 1e-9].max()),
        "T_min": float(T.min()),
        "train_pde_loss": trainer.log[-1]["pde"], "test_pde_loss": float(r2.mean()),
        "test_pde_loss_corners": float(r2[corner_box].mean()), "test_pde_loss_rest": float(r2[~corner_box].mean()),
        "iters_total": trainer.iteration, "lbfgs_iterations": trainer.timings.get("lbfgs_iterations"),
        "seconds": trainer.timings["total_seconds"],
        "err_map": err.astype(np.float32),
        "err_vs_corner": binned(err.ravel(), dc.ravel(), CORNER_BINS, np.max),
        "res_vs_corner": binned(rc2, rc.cpu().numpy().ravel(), PROFILE_BINS, lambda v: float(np.sqrt(v.mean()))),
        "log": trainer.log,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


# ---------------------------------------------------------------- outputs
COLS = ["config", "seed", "rel_l2", "max_err", "max_err_x", "max_err_y", "max_err_dist_corner",
        "max_err_near_corners", "max_err_rest", "share_sq_err_near_corners", "err_at_corner_points",
        "T_min", "train_pde_loss", "test_pde_loss", "test_pde_loss_corners", "test_pde_loss_rest",
        "iters_total", "lbfgs_iterations", "seconds", "device"]


def compare_with_step5(runs: list[dict]) -> str:
    """soft5k / hard5k train exactly as in Step 5: their errors should match Step 5's table."""
    if not STEP5_TABLE.exists():
        return "Step 5 table not found: reproducibility check skipped."
    with open(STEP5_TABLE, newline="", encoding="utf-8") as f:
        base = {(r["config"], int(r["seed"])): r for r in csv.DictReader(f)}
    diffs = []
    for r in runs:
        b = base.get((r["config"], r["seed"]))
        if b is not None and r["config"] in ("soft5k", "hard5k"):
            for k in ("rel_l2", "max_err"):
                diffs.append(abs(r[k] - float(b[k])) / float(b[k]))
    if not diffs:
        return "No matching Step 5 runs: reproducibility check skipped."
    return (f"Reproducibility: soft5k and hard5k reran Step 5's training; over {len(diffs) // 2} runs their "
            f"rel. L2 and max errors differ from Step 5's table by at most {max(diffs):.1e} (relative). "
            "Exact zeros mean the same machine produced bit-identical training.")


def write_tables(runs: list[dict], tables: Path, quick: bool = False) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "step5b_heat_corners_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    med = lambda rs, k: statistics.median(r[k] for r in rs)
    worst = lambda rs, k: max(r[k] for r in rs)
    n_seeds = max(sum(r["config"] == c for r in runs) for c in CONFIGS)
    lines = ["# Step 5b: the corners of the heat problem", "",
             f"Errors against the exact series solution. Median over {n_seeds} seeds (worst in brackets). "
             + ("QUICK TEST RUN (Adam 500, L-BFGS 300): not results." if quick else
                "Adam 5k then L-BFGS for every configuration."), "",
             "| configuration | rel. L2 error | max error | max error: distance to the nearest corner "
             f"| share of squared error within {NEAR_CORNER} of a corner (1.8% of the area) "
             "| PDE loss on fresh points near the corners | PDE loss: fresh / training points "
             "| lowest temperature (physics: >= 0) | time per run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for config in CONFIGS:
        rs = [r for r in runs if r["config"] == config]
        if not rs:
            continue
        at_corner = sum(r["max_err_dist_corner"] < 1e-9 for r in rs)
        lines.append(
            f"| {LABELS[config].replace(chr(10), ' ')} "
            f"| {med(rs, 'rel_l2'):.1e} ({worst(rs, 'rel_l2'):.1e}) "
            f"| {med(rs, 'max_err'):.1e} ({worst(rs, 'max_err'):.1e}) "
            f"| {med(rs, 'max_err_dist_corner'):.2f} (at a corner point in {at_corner}/{len(rs)} seeds) "
            f"| {med(rs, 'share_sq_err_near_corners'):.0%} "
            f"| {med(rs, 'test_pde_loss_corners'):.1e} "
            f"| {statistics.median(r['test_pde_loss'] / r['train_pde_loss'] for r in rs):.1f}x "
            f"| {med(rs, 'T_min'):.1e} ({min(r['T_min'] for r in rs):.1e}) "
            f"| {statistics.mean(r['seconds'] for r in rs):.0f} s |")
    lines += ["", "\"Near the corners\" for the PDE loss means within 0.1 of both edges, as in Step 5.", "",
              "Facts about the problem itself (no training involved; checked in tests/test_heat.py):",
              "- Exact solution: approaching a corner at angle theta from the horizontal edge, T_xx -> -2 theta / pi "
              "(0 along that edge, -1 along the other), at every distance. A smooth network has one value there.",
              "- Any hard-BC network T = (1 - x^2)(1 - y^2) N has PDE residual exactly 1 at the four corners.",
              "- The function a hard-BC network must learn, N = T / ((1 - x^2)(1 - y^2)), grows by "
              f"ln(10) / (2 pi) = {math.log(10) / (2 * math.pi):.3f} for every factor of 10 closer to a corner.",
              "", "" if quick else compare_with_step5(runs)]
    (tables / "step5b_heat_corners_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def exact_curvature_by_angle(theta: np.ndarray, dist: float) -> np.ndarray:
    """T_xx of the exact solution at distance dist from corner (1, 1), angle theta from the top edge."""
    x, y, h = 1 - dist * np.cos(theta), 1 - dist * np.sin(theta), dist / 50
    f = lambda xx: heat.exact(xx, y, 20000)
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def strip(ax, groups, key):
    for i, (config, rs) in enumerate(groups):
        xs = i + np.linspace(-0.15, 0.15, len(rs))
        ax.scatter(xs, [r[key] for r in rs], s=34, color=COLORS[config], zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)), [LABELS[c] for c, _ in groups], fontsize=8)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.grid(axis="x", visible=False)


def make_figure(runs: list[dict], path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    groups = [(c, [r for r in runs if r["config"] == c]) for c in CONFIGS]
    groups = [(c, rs) for c, rs in groups if rs]

    # (a) the contradiction, measured on the exact solution
    ax = axes[0, 0]
    theta = np.linspace(0.02, math.pi / 2 - 0.02, 23)
    for dist, marker, color in [(0.1, "o", INK), (0.01, "s", INK_2), (0.001, "^", MUTED)]:
        ax.plot(np.degrees(theta), exact_curvature_by_angle(theta, dist), marker, ms=4.5, mfc="none",
                color=color, label=f"exact solution, {dist:g} from the corner")
    ax.plot([0, 90], [0, -1], color=SERIES[1], lw=1.2, ls="--", label="-2 theta / pi (corner analysis)", zorder=0)
    ax.annotate("along the horizontal edge,\nT = 0 forces T_xx = 0", xy=(0, 0), xytext=(14, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.annotate("along the vertical edge,\nthe PDE forces T_xx = -1", xy=(90, -1), xytext=(-14, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8, color=INK_2)
    ax.set(title="(a) Exact T_xx near a corner depends on direction",
           xlabel="direction of approach (degrees from the horizontal edge)", ylabel="T_xx", xlim=(-3, 93))
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=7.5)   # below the top annotation

    # (b) PDE residual vs distance to the nearest corner
    ax = axes[0, 1]
    for config, rs in groups:
        curve = np.nanmedian([r["res_vs_corner"] for r in rs], axis=0)
        ax.plot(PROFILE_X, curve, "o-", ms=4, color=COLORS[config], label=LABELS[config].replace("\n", " "))
    ax.axhline(1.0, color=INK_2, lw=0.9, ls=":")
    ax.annotate("hard BC: exactly 1 at the corner, for any network", xy=(1, 1.0),
                xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="(b) PDE residual vs distance to a corner",
           xlabel="distance to corner (log scale)", ylabel="RMS PDE residual on fresh points (median of seeds)")
    ax.legend(loc="lower left", fontsize=8)

    # (c) max error vs distance to the nearest corner
    ax = axes[0, 2]
    for config, rs in groups:
        curve = np.maximum(np.nanmedian([r["err_vs_corner"] for r in rs], axis=0), 1e-12)
        ax.plot(CORNER_X, curve, "o-", ms=4, color=COLORS[config], label=LABELS[config].replace("\n", " "))
    ax.axvline(0.007, color=MUTED, lw=0.6)
    ax.annotate("left of the line:\nthe corner point itself", xy=(0.0075, 0.3), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="(c) Max error vs distance to a corner",
           xlabel="distance to corner (log scale)", ylabel="max |error| (median of seeds)")
    ax.legend(loc="lower right", fontsize=8)

    # (d), (e) error maps around one corner: the one where hard BC seed 0 is worst
    hard0 = next((r for r in runs if r["config"] == "hard5k" and r["seed"] == 0), None)
    corner0 = next((r for r in runs if r["config"] == "corner5k" and r["seed"] == 0), None)
    n, k = 201, 51                                   # grid points per side; zoom = last 0.5 of each axis
    g = np.linspace(-1, 1, n)
    if hard0 is not None:
        e = hard0["err_map"]
        cy, cx = max([(1, 1), (1, -1), (-1, 1), (-1, -1)],
                     key=lambda c: e[slice(-k, None) if c[0] > 0 else slice(0, k),
                                     slice(-k, None) if c[1] > 0 else slice(0, k)].max())
    else:
        cx = cy = 1
    rows = slice(-k, None) if cy > 0 else slice(0, k)
    cols = slice(-k, None) if cx > 0 else slice(0, k)
    norm = LogNorm(vmin=1e-7, vmax=1e-3)
    for ax, r, letter in [(axes[1, 0], hard0, "d"), (axes[1, 1], corner0, "e")]:
        if r is None:
            ax.axis("off")
            continue
        im = ax.pcolormesh(g[cols], g[rows], np.maximum(r["err_map"][rows, cols], 1e-8),
                           cmap="seq_orange", norm=norm, shading="auto")
        fig.colorbar(im, ax=ax, label="|PINN - exact|")
        ax.plot([cx], [cy], "o", ms=5, color=INK, clip_on=False)
        what = "hard BC" if letter == "d" else "hard BC + corner terms"
        ax.set(title=f"({letter}) {what}, seed 0: error near corner ({cx:+d}, {cy:+d})",
               xlabel="x", ylabel="y", aspect="equal")
        ax.grid(False)

    strip(axes[1, 2], groups, "max_err")
    axes[1, 2].set_title("(f) Max error over the square, every seed")

    fig.suptitle("Step 5b: the corners of the heat problem", x=0.01, ha="left", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5b: the corners of the heat problem")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", metavar="RUN_DIR",
                        help="continue an interrupted run: reuse the trainings already saved in RUN_DIR")
    args = parser.parse_args()
    if args.quick:
        args.seeds = args.seeds[:2]
    run_dir = Path(args.resume) if args.resume else make_run_dir("step5b_heat_corners" + ("_quick" if args.quick else ""))
    # Every finished training is saved here immediately (see Step 5).
    partial = run_dir / "partial"
    partial.mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted(partial.glob("*.pkl")):
        with open(f, "rb") as fh:
            runs.append(pickle.load(fh))
    done = {(r["config"], r["seed"]) for r in runs}
    order = {"corner5k": 0, "hard5k": 1, "soft5k": 2}                    # slowest first
    tasks = sorted([(c, s) for c in args.configs for s in args.seeds if (c, s) not in done],
                   key=lambda cs: order[cs[0]])
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
            print(f"  [{n:>2}/{len(tasks)}] {r['config']:<8} seed {r['seed']}  rel L2 {r['rel_l2']:.2e}  "
                  f"max err {r['max_err']:.1e}  corner PDE loss {r['test_pde_loss_corners']:.1e}  "
                  f"{r['seconds']:6.0f}s  (elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    runs.sort(key=lambda r: (list(CONFIGS).index(r["config"]), r["seed"]))
    with open(run_dir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)
    if args.quick:
        tables, figure = run_dir / "tables", run_dir / "step5b_heat_corners_quick.png"
    else:
        tables, figure = Path("results/tables"), Path("results/figures/step5b_heat_corners.png")
    write_tables(runs, tables, args.quick)
    make_figure(runs, figure)
    save_json({"args": vars(args), "env": env_info()}, run_dir / "metrics.json")
    print("\n" + (tables / "step5b_heat_corners_summary.md").read_text(encoding="utf-8"))
    print(f"figure: {figure}\ntables: {tables}\nraw runs: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
