"""
Step 4: reproduce paper section 4.1 (Terzaghi consolidation, a forward
problem) with 5 seeds, and measure WHERE the error is.

Configurations (the paper's network, points and loss weights in both):
  paper       the paper exactly: Adam 50k (lr 1e-3), then L-BFGS to convergence
  adam10k     Adam 10k, then L-BFGS. The paper's loss curve is flat after ~5k
              iterations; this tests whether the other 40k do anything.

Diagnostics, all against the exact (method-of-images) solution:
  * error as a function of time on a grid that zooms toward t = 0 (log scale),
    because the true solution jumps at the corner (x = 0, t = 0) and a
    "max error" on a uniform grid only says how close that grid gets to it;
  * how the squared error and the 500 collocation points are spread over
    time bands;
  * training PDE loss vs the same loss on 20,000 fresh random points.

Usage (from the repo root):
  python experiments/step4_consolidation.py --bench
  python experiments/step4_consolidation.py --device cpu --jobs 5
  python experiments/step4_consolidation.py --device cuda
  python experiments/step4_consolidation.py --quick
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
from pinn.plotting import (BLUES, INK, INK_2, MUTED, SERIES, apply_style,
                           mark_lbfgs_start)
from problems import consolidation as cons

# Paper section 4.1 settings
WIDTH, DEPTH, INIT, LR = 50, 5, "xavier_normal", 1e-3
N_B = cons.N_BOUNDARY_X + cons.N_INITIAL
# One boundary loss averaged over all 375 boundary points, weight 1/4 (paper);
# logged as two terms, so each gets its share of that weight.
WEIGHTS = {"pde": 0.25, "bc": 0.25 * cons.N_BOUNDARY_X / N_B, "ic": 0.25 * cons.N_INITIAL / N_B}
CONFIGS = {"paper": dict(adam_iters=50_000), "adam10k": dict(adam_iters=10_000)}
LABELS = {"paper": "paper (Adam 50k + L-BFGS)", "adam10k": "Adam 10k + L-BFGS"}
LBFGS_MAX = 50_000
T_LOG = np.logspace(-4, np.log10(cons.T_MAX), 120)     # time axis that zooms toward t = 0
PROFILE_TIMES = (0.0025, 0.01, 0.05, 0.2)


def build(device, seed: int):
    set_seed(seed)
    model = MLP(2, 1, WIDTH, DEPTH, lb=[0, 0], ub=[1, cons.T_MAX], init=INIT).to(device)
    x_f, t_f = cons.collocation_points(device=device)
    t_b = cons.boundary_times(device=device)
    x_0 = cons.initial_positions(device=device)
    return model, x_f, t_f, t_b, x_0


def make_loss(model, x_f, t_f, t_b, x_0):
    def loss_fn():
        return {"pde": (cons.pde_residual(model, x_f, t_f) ** 2).mean(),
                "bc": (cons.neumann_residual(model, t_b) ** 2).mean(),
                "ic": (cons.initial_residual(model, x_0) ** 2).mean()}
    return loss_fn


def log_grid(device, nx: int = 201):
    """x uniform on [0, 1], t log-spaced on [1e-4, 0.5]."""
    x = torch.linspace(0, 1, nx, device=device)
    t = torch.as_tensor(T_LOG, device=device, dtype=x.dtype)
    T, X = torch.meshgrid(t, x, indexing="ij")
    return X.reshape(-1, 1), T.reshape(-1, 1), (len(T_LOG), nx)


def run_one(config: str, seed: int, device_name: str, quick: bool = False) -> dict:
    device = setup_torch(torch.float64, prefer=device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, x_f, t_f, t_b, x_0 = build(device, seed)
    loss_fn = make_loss(model, x_f, t_f, t_b, x_0)

    X, T, shape = cons.test_grid(device=device)                     # uniform 201 x 201, includes t = 0
    P = torch.as_tensor(cons.exact(X.cpu().numpy(), T.cpu().numpy()), device=device)

    def eval_fn():
        return {"rel_l2": relative_l2(cons.pressure(model, X, T), P)}

    adam = 1_000 if quick else CONFIGS[config]["adam_iters"]
    cfg = TrainConfig(adam_iters=adam, adam_lr=LR, lbfgs_iters=500 if quick else LBFGS_MAX,
                      eval_every=1_000, log_every=100, weights=WEIGHTS, verbose=False)
    trainer = Trainer(model.parameters(), loss_fn, cfg, eval_fn=eval_fn).train()

    # --- diagnostics -------------------------------------------------------
    with torch.no_grad():
        p_uni = cons.pressure(model, X, T).cpu().numpy().reshape(shape)
    x_uni, t_uni = X.cpu().numpy().reshape(shape), T.cpu().numpy().reshape(shape)
    err_uni = np.abs(p_uni - cons.exact(x_uni, t_uni))

    XL, TL, shape_l = log_grid(device)
    with torch.no_grad():
        p_log = cons.pressure(model, XL, TL).cpu().numpy().reshape(shape_l)
    x_log, t_log = XL.cpu().numpy().reshape(shape_l), TL.cpu().numpy().reshape(shape_l)
    err_log = np.abs(p_log - cons.exact(x_log, t_log))
    XLg, TLg = XL.clone().requires_grad_(True), TL.clone().requires_grad_(True)
    res_log = np.abs(cons.pde_residual(model, XLg, TLg).detach().cpu().numpy().reshape(shape_l))

    # PDE loss on fresh random points (the paper's "PDE test loss")
    xr, tr_ = cons.collocation_points(20_000, device=device)
    res_fresh = cons.pde_residual(model, xr, tr_).detach()
    t_fresh = tr_.detach().cpu().numpy().ravel()
    res_fresh_np = res_fresh.cpu().numpy().ravel()

    t_coll = t_f.detach().cpu().numpy().ravel()
    x_coll = x_f.detach().cpu().numpy().ravel()
    bands = cons.error_by_band(err_uni[1:], t_uni[1:])              # t > 0 rows
    band_info = []
    for b, (lo, hi) in zip(bands, cons.BANDS):
        m = (t_fresh >= lo) & (t_fresh < hi)
        band_info.append({**b, "collocation_points": int(((t_coll >= lo) & (t_coll < hi)).sum()),
                          "mean_abs_residual": float(np.abs(res_fresh_np[m]).mean())})

    profiles = {}
    for tp in PROFILE_TIMES:
        xs = torch.linspace(0, 1, 401, device=device).reshape(-1, 1)
        with torch.no_grad():
            profiles[tp] = cons.pressure(model, xs, torch.full_like(xs, tp)).cpu().numpy().ravel()

    final = trainer.log[-1]
    return {
        "config": config, "seed": seed, "device": str(device),
        "rel_l2": trainer.evals[-1]["rel_l2"],
        "max_err_t_ge_0.01": float(err_uni[t_uni >= 0.01].max()),
        "max_err_t_ge_0.05": float(err_uni[t_uni >= 0.05].max()),
        "err_share_t_lt_0.01": band_info[0]["share_sq_err"],
        "err_share_t_lt_0.05": band_info[0]["share_sq_err"] + band_info[1]["share_sq_err"],
        "coll_points_t_lt_0.01": band_info[0]["collocation_points"],
        "coll_points_corner": int(((x_coll < 0.1) & (t_coll < 0.01)).sum()),
        "train_pde_loss": final["pde"], "train_ic_loss": final["ic"], "train_bc_loss": final["bc"],
        "test_pde_loss": float((res_fresh**2).mean()),
        "p_min": float(p_uni.min()), "p_max": float(p_uni.max()),
        "iters_total": trainer.iteration,
        "lbfgs_iters": int(trainer.timings.get("lbfgs_iterations", 0)),
        "seconds": trainer.timings["total_seconds"],
        # arrays for the figure
        "bands": band_info, "max_err_vs_t": err_log.max(axis=1), "err_log": err_log,
        "res_log": res_log, "x_log": x_log[0], "profiles": profiles,
        "x_coll": x_coll, "t_coll": t_coll, "log": trainer.log, "evals": trainer.evals,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }


# ---------------------------------------------------------------- benchmark
def bench(steps: int = 100) -> None:
    print(f"Timing {steps} Adam steps of the paper's consolidation setup...")
    results = {}
    for name in ["cpu", "cuda"]:
        if name == "cuda" and not torch.cuda.is_available():
            continue
        device = setup_torch(torch.float64, prefer=name)
        if name == "cpu":
            torch.set_num_threads(1)
        model, x_f, t_f, t_b, x_0 = build(device, 0)
        loss_fn = make_loss(model, x_f, t_f, t_b, x_0)
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
    print(f"\nAll 10 runs: {total:,} Adam steps, plus L-BFGS (roughly +10-20%).")
    for name, ms in results.items():
        print(f"  {name:<5} one run at a time:  ~{total * ms / 60000 * 1.15:5.0f} min")
    if "cpu" in results:
        longest = CONFIGS["paper"]["adam_iters"] + CONFIGS["adam10k"]["adam_iters"]
        print(f"  cpu   5 runs in parallel (--jobs 5): ~{longest * results['cpu'] / 60000 * 1.15:5.0f} min"
              " (needs 5 free cores; slower if the laptop throttles)")
    print(f"\nCPU cores on this machine: {mp.cpu_count()}")


# ---------------------------------------------------------------- outputs
def write_tables(runs: list[dict], tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    cols = ["config", "seed", "rel_l2", "max_err_t_ge_0.01", "max_err_t_ge_0.05",
            "err_share_t_lt_0.01", "err_share_t_lt_0.05", "coll_points_t_lt_0.01",
            "coll_points_corner", "train_pde_loss", "test_pde_loss", "train_ic_loss",
            "train_bc_loss", "p_min", "p_max", "iters_total", "lbfgs_iters", "seconds", "device"]
    with open(tables / "step4_consolidation_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)

    med = lambda rs, k: statistics.median(r[k] for r in rs)
    lines = ["# Step 4: 1D consolidation (paper section 4.1)", "",
             "Errors against the exact solution. Median over seeds (worst in brackets).",
             "The paper reports a 'perfect match' with max absolute error about 0.045 (its Fig. 2).", "",
             "| configuration | runs | rel. L2 error | max error, t >= 0.01 | max error, t >= 0.05 "
             "| share of squared error in t < 0.01 (2% of domain) | collocation points in t < 0.01 "
             "| PDE loss: training / fresh points | time per run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for config in CONFIGS:
        rs = [r for r in runs if r["config"] == config]
        if not rs:
            continue
        lines.append(
            f"| {LABELS[config]} | {len(rs)} "
            f"| {med(rs, 'rel_l2'):.1e} ({max(r['rel_l2'] for r in rs):.1e}) "
            f"| {med(rs, 'max_err_t_ge_0.01'):.3f} ({max(r['max_err_t_ge_0.01'] for r in rs):.3f}) "
            f"| {med(rs, 'max_err_t_ge_0.05'):.3f} ({max(r['max_err_t_ge_0.05'] for r in rs):.3f}) "
            f"| {med(rs, 'err_share_t_lt_0.01'):.0%} "
            f"| {med(rs, 'coll_points_t_lt_0.01'):.0f} of 500 "
            f"| {med(rs, 'train_pde_loss'):.1e} / {med(rs, 'test_pde_loss'):.1e} "
            f"| {statistics.mean(r['seconds'] for r in rs):.0f} s |")
    (tables / "step4_consolidation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(runs: list[dict], path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    paper = sorted((r for r in runs if r["config"] == "paper"), key=lambda r: r["seed"])
    ref = paper[0] if paper else runs[0]
    x_log = ref["x_log"]

    # (a) pressure profiles at several times
    ax = axes[0, 0]
    xs = np.linspace(0, 1, 401)
    shades = [BLUES(v) for v in np.linspace(0.45, 1.0, len(PROFILE_TIMES))]
    for tp, color in zip(PROFILE_TIMES, shades):
        ax.plot(xs, cons.exact(xs, np.full_like(xs, tp)), color=color, lw=1.0, ls=":")
        ax.plot(xs, ref["profiles"][tp], color=color, label=f"t = {tp:g}")
    ax.set(title="(a) Pressure profiles: PINN (solid) vs exact (dotted)", xlabel="depth x",
           ylabel="pore pressure p", ylim=(-0.05, 1.12))
    ax.legend(loc="lower right")

    # (b) error map with time on a log axis
    ax = axes[0, 1]
    im = ax.pcolormesh(x_log, T_LOG, np.maximum(ref["err_log"], 1e-6), cmap="seq_orange",
                       norm=LogNorm(vmin=1e-4, vmax=1), shading="auto")
    fig.colorbar(im, ax=ax, label="|PINN - exact|")
    ax.set_yscale("log")
    ax.set(title="(b) Error map (time axis zooms toward t = 0)", xlabel="depth x",
           ylabel="time t (log scale)")
    ax.grid(False)

    # (c) max error over depth vs time, every run
    ax = axes[0, 2]
    for i, config in enumerate(CONFIGS):
        for r in (r for r in runs if r["config"] == config):
            ax.plot(T_LOG, r["max_err_vs_t"], color=SERIES[i], lw=1.0, alpha=0.7,
                    label=LABELS[config] if r["seed"] == min(q["seed"] for q in runs if q["config"] == config) else None)
    ax.axhline(0.045, color=INK_2, lw=0.9, ls=":")
    ax.annotate("paper's reported max error (0.045)", xy=(1, 0.045), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="(c) Max error over depth vs time, all seeds", xlabel="time t (log scale)",
           ylabel="max over x of |error|")
    ax.legend(loc="lower left", fontsize=8)

    # (d) PDE residual map + where the collocation points are
    ax = axes[1, 0]
    im = ax.pcolormesh(x_log, T_LOG, np.maximum(ref["res_log"], 1e-6), cmap="seq_blue",
                       norm=LogNorm(vmin=1e-4, vmax=1e3), shading="auto")
    fig.colorbar(im, ax=ax, label="|p_t - p_xx| (PDE residual)")
    keep = ref["t_coll"] >= T_LOG[0]
    ax.scatter(ref["x_coll"][keep], ref["t_coll"][keep], s=5, color=INK, alpha=0.6, lw=0)
    ax.set_yscale("log")
    ax.set(title="(d) PDE residual (color) and the 500 training points (dots)", xlabel="depth x",
           ylabel="time t (log scale)")
    ax.grid(False)

    # (e) training loss vs fresh-point loss
    ax = axes[1, 1]
    for i, config in enumerate(CONFIGS):
        rs = [r for r in runs if r["config"] == config]
        for r in rs:
            ax.plot([0, 1], [r["train_pde_loss"], r["test_pde_loss"]], color=SERIES[i], lw=1.0, alpha=0.7)
            ax.scatter([0, 1], [r["train_pde_loss"], r["test_pde_loss"]], color=SERIES[i], s=24, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["at the 500 training points\n(what the optimizer sees)",
                           "at 20,000 fresh points\n(the real accuracy)"])
    ax.set_xlim(-0.4, 1.4)
    ax.grid(axis="x", visible=False)
    ax.set(title="(e) PDE loss: training points vs fresh points", ylabel="mean squared PDE residual")

    # (f) error share vs domain share, per time band
    ax = axes[1, 2]
    labels = [b["band"] for b in ref["bands"]]
    width = 0.27
    pos = np.arange(len(labels))
    dom = [b["share_of_domain"] for b in ref["bands"]]
    pts = [statistics.median(r["bands"][i]["collocation_points"] for r in runs) / cons.N_COLLOCATION
           for i in range(len(labels))]
    err = [statistics.median(r["bands"][i]["share_sq_err"] for r in runs) for i in range(len(labels))]
    ax.bar(pos - width, dom, width, color=MUTED, label="share of the domain")
    ax.bar(pos, pts, width, color=SERIES[0], label="share of collocation points")
    ax.bar(pos + width, err, width, color=SERIES[1], label="share of squared error")
    ax.set_xticks(pos, [f"t in {l}" for l in labels])
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set(title="(f) The error sits where the points are not", ylabel="fraction (median over runs)")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Step 4: 1D consolidation (paper section 4.1)", x=0.01, ha="left", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4: 1D consolidation")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if args.bench:
        bench()
        return 0
    if args.quick:
        args.seeds = args.seeds[:2]

    run_dir = make_run_dir("step4_consolidation" + ("_quick" if args.quick else ""))
    # Longest runs first, so parallel workers finish at about the same time.
    tasks = sorted([(c, s) for c in args.configs for s in args.seeds],
                   key=lambda cs: -CONFIGS[cs[0]]["adam_iters"])
    print(f"{len(tasks)} runs on {args.device} with {args.jobs} in parallel -> {run_dir}")
    runs, t0 = [], time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, c, s, args.device, args.quick) for c, s in tasks]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            runs.append(r)
            print(f"  [{n:>2}/{len(tasks)}] {r['config']:<8} seed {r['seed']}  rel L2 {r['rel_l2']:.2e}  "
                  f"max err (t>=0.01) {r['max_err_t_ge_0.01']:.3f}  {r['seconds']:6.0f}s  "
                  f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    runs.sort(key=lambda r: (list(CONFIGS).index(r["config"]), r["seed"]))
    with open(run_dir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)
    if args.quick:
        tables, figure = run_dir / "tables", run_dir / "step4_consolidation_quick.png"
    else:
        tables, figure = Path("results/tables"), Path("results/figures/step4_consolidation.png")
    write_tables(runs, tables)
    make_figure(runs, figure)
    save_json({"args": vars(args), "env": env_info()}, run_dir / "metrics.json")
    print("\n" + (tables / "step4_consolidation_summary.md").read_text(encoding="utf-8"))
    print(f"figure: {figure}\ntables: {tables}\nraw runs (incl. trained weights): {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
