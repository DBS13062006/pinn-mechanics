"""
Step 3: reproduce paper section 4.3 (find damping c and stiffness k), with
5 random seeds instead of the paper's single run, and find out WHY c
converges so slowly.

Configurations (all share the paper's network, data and starting guesses):
  paper          the paper exactly: Adam only, 100k iterations, lr 5e-4
  adam5k_lbfgs   Adam for 5k iterations, then L-BFGS until it converges
  adam20k_lbfgs  Adam for 20k iterations, then L-BFGS until it converges

Usage (from the repo root):
  python experiments/step3_oscillator.py --bench          # time CPU vs GPU, then choose
  python experiments/step3_oscillator.py --device cuda    # all configs, seeds 0-4, one at a time
  python experiments/step3_oscillator.py --device cpu --jobs 5   # 5 runs in parallel on CPU cores
  python experiments/step3_oscillator.py --quick          # 1-minute smoke test

Outputs:
  results/figures/step3_oscillator.png        report figure            (committed)
  results/tables/step3_oscillator_runs.csv    one row per run          (committed)
  results/tables/step3_oscillator_summary.md  one row per config       (committed)
  results/runs/<timestamp>_step3_oscillator/  full trajectories        (not committed)
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
from pinn.plotting import (BLUES, INK, INK_2, MUTED, SERIES, apply_style,
                           mark_lbfgs_start)
from problems import oscillator as osc

# Paper section 4.3 settings
PAPER = dict(width=32, depth=3, init="xavier_uniform", lr=5e-4, n_data=50, n_f=100,
             weights={"ode": 0.25, "ic": 0.25, "data": 0.25})

CONFIGS = {
    "paper":         dict(adam_iters=100_000, lbfgs_iters=0,      eval_every=500),
    "adam5k_lbfgs":  dict(adam_iters=5_000,   lbfgs_iters=50_000, eval_every=100),
    "adam20k_lbfgs": dict(adam_iters=20_000,  lbfgs_iters=50_000, eval_every=100),
}
QUICK_ADAM = {"paper": 3_000, "adam5k_lbfgs": 500, "adam20k_lbfgs": 1_000}
SNAPSHOT_ITERS = (1_000, 4_000, 10_000, 50_000)
LABELS = {"paper": "paper (Adam 100k)", "adam5k_lbfgs": "Adam 5k + L-BFGS",
          "adam20k_lbfgs": "Adam 20k + L-BFGS"}
SUCCESS_TOL = 0.01  # a run "succeeds" if both c and k end within 1% of the truth


# ---------------------------------------------------------------- one run
def build(device, seed: int):
    """Network, unknown parameters and point sets, exactly as in the paper."""
    set_seed(seed)
    model = MLP(1, 1, PAPER["width"], PAPER["depth"], lb=[0.0], ub=[osc.T_MAX],
                init=PAPER["init"]).to(device)
    c = torch.nn.Parameter(torch.tensor(osc.C_START, device=device))
    k = torch.nn.Parameter(torch.tensor(osc.K_START, device=device))
    t_d = osc.observation_times(PAPER["n_data"], device)
    u_d = osc.exact(t_d)
    t_f = osc.collocation_times(PAPER["n_f"], device)
    return model, c, k, t_d, u_d, t_f


def run_one(config: str, seed: int, device_name: str, quick: bool = False) -> dict:
    """Train one configuration with one seed. Runs in its own process."""
    device = setup_torch(torch.float64, prefer=device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)  # parallel runs must not fight over cores
    spec = dict(CONFIGS[config])
    if quick:
        spec["adam_iters"] = QUICK_ADAM[config]
        spec["lbfgs_iters"] = min(spec["lbfgs_iters"], 500)

    model, c, k, t_d, u_d, t_f = build(device, seed)
    t_test = osc.test_times(10_000, device)
    u_test = osc.exact(t_test)
    t_diag = osc.test_times(2_000, device)

    def loss_fn():
        r = osc.residual(model, t_f, c, k)
        return {"ode": (r**2).mean(),
                "ic": osc.initial_condition_loss(model, device),
                "data": ((model(t_d) - u_d) ** 2).mean()}

    def eval_fn():
        u = model(t_test)
        with torch.enable_grad():
            c_star, k_star = osc.implied_parameters(model, t_diag)
        a0, a1, a2 = osc.amplitude_ratios(u, t_test)
        return {"c": c.item(), "k": k.item(), "c_implied": c_star, "k_implied": k_star,
                "u_rel_l2": relative_l2(u, u_test), "amp_0_5": a0, "amp_5_10": a1, "amp_10_15": a2}

    snapshots = {}
    t_plot = osc.test_times(1_000, device)

    def snapshot_hook(trainer):
        if trainer.iteration in SNAPSHOT_ITERS:
            with torch.no_grad():
                snapshots[trainer.iteration] = model(t_plot).cpu().numpy().ravel()

    cfg = TrainConfig(adam_iters=spec["adam_iters"], adam_lr=PAPER["lr"],
                      lbfgs_iters=spec["lbfgs_iters"], eval_every=spec["eval_every"],
                      log_every=100, weights=PAPER["weights"], verbose=False)
    trainer = Trainer(list(model.parameters()) + [c, k], loss_fn, cfg,
                      eval_fn=eval_fn, hooks=[snapshot_hook]).train()

    # Paper's "ODE test loss": mean squared residual on the 10000 test points
    t_g = t_test.clone().requires_grad_(True)
    ode_test = (osc.residual(model, t_g, c, k) ** 2).mean().item()
    with torch.no_grad():
        u_final = model(t_plot).cpu().numpy().ravel()

    ev = trainer.evals
    c_err = abs(c.item() - osc.C_TRUE) / osc.C_TRUE
    k_err = abs(k.item() - osc.K_TRUE) / osc.K_TRUE
    return {
        "config": config, "seed": seed, "device": str(device),
        "c": c.item(), "k": k.item(), "c_err": c_err, "k_err": k_err,
        "success": c_err < SUCCESS_TOL and k_err < SUCCESS_TOL,
        "u_rel_l2": ev[-1]["u_rel_l2"], "ode_test": ode_test,
        "data_loss": trainer.log[-1]["data"],
        "iters_total": trainer.iteration,
        "iters_to_1pct": iterations_to_stay_within(ev, SUCCESS_TOL),
        "lbfgs_iters": int(trainer.timings.get("lbfgs_iterations", 0)),
        "seconds": trainer.timings["total_seconds"],
        # full histories, for the figure and the run folder
        "evals": ev, "log": trainer.log, "snapshots": snapshots, "u_final": u_final,
        "t_plot": t_plot.cpu().numpy().ravel(),
    }


def iterations_to_stay_within(evals: list[dict], tol: float):
    """First evaluated iteration from which BOTH c and k stay within tol until the end."""
    ok = [abs(e["c"] - osc.C_TRUE) / osc.C_TRUE < tol and abs(e["k"] - osc.K_TRUE) / osc.K_TRUE < tol
          for e in evals]
    if not ok or not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return evals[i]["iter"]


# ---------------------------------------------------------------- benchmark
def bench(steps: int = 300) -> None:
    """Time the paper's training step on one CPU core and on the GPU."""
    print(f"Timing {steps} Adam steps of the paper's oscillator setup...")
    results = {}
    for name in ["cpu", "cuda"]:
        if name == "cuda" and not torch.cuda.is_available():
            continue
        device = setup_torch(torch.float64, prefer=name)
        if name == "cpu":
            torch.set_num_threads(1)
        model, c, k, t_d, u_d, t_f = build(device, 0)
        opt = torch.optim.Adam(list(model.parameters()) + [c, k], lr=5e-4)

        def step():
            opt.zero_grad()
            r = osc.residual(model, t_f, c, k)
            loss = (r**2).mean() + osc.initial_condition_loss(model, device) \
                + ((model(t_d) - u_d) ** 2).mean()
            loss.backward()
            opt.step()

        for _ in range(20):
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

    paper_steps = 5 * CONFIGS["paper"]["adam_iters"]
    print(f"\nThe 5 'paper' runs are {paper_steps:,} Adam steps in total (the other configs are much shorter).")
    for name, ms in results.items():
        print(f"  {name:<5} one run at a time: ~{paper_steps * ms / 60000:5.0f} min")
    if "cpu" in results:
        print(f"  cpu   5 runs in parallel (--jobs 5): ~{CONFIGS['paper']['adam_iters'] * results['cpu'] / 60000:5.0f} min"
              " (if you have at least 5 free cores)")
    print(f"\nCPU cores on this machine: {mp.cpu_count()}")


# ---------------------------------------------------------------- outputs
def write_tables(runs: list[dict], tables: Path) -> list[dict]:
    tables.mkdir(parents=True, exist_ok=True)
    cols = ["config", "seed", "c", "k", "c_err", "k_err", "success", "u_rel_l2", "ode_test",
            "data_loss", "iters_total", "iters_to_1pct", "lbfgs_iters", "seconds", "device"]
    with open(tables / "step3_oscillator_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow({c: r[c] for c in cols})

    summary = []
    for config in CONFIGS:
        rs = [r for r in runs if r["config"] == config]
        if not rs:
            continue
        conv = [r["iters_to_1pct"] for r in rs if r["iters_to_1pct"] is not None]
        summary.append({
            "config": LABELS[config], "runs": len(rs),
            "succeeded": sum(r["success"] for r in rs),
            "c_err_median": statistics.median(r["c_err"] for r in rs),
            "c_err_worst": max(r["c_err"] for r in rs),
            "k_err_median": statistics.median(r["k_err"] for r in rs),
            "k_err_worst": max(r["k_err"] for r in rs),
            "iters_to_1pct_median": statistics.median(conv) if conv else None,
            "seconds_mean": statistics.mean(r["seconds"] for r in rs),
        })

    lines = ["# Step 3: damped oscillator (paper section 4.3)", "",
             "Truth: c = 0.4, k = 4. Start: c = 0.2, k = 2. A run succeeds if both end within 1%.",
             "Paper's single run: c = 0.40125 (0.31% off), k = 3.9933 (0.17% off).", "",
             "| configuration | succeeded | c error (median / worst) | k error (median / worst) "
             "| iterations until both stay within 1% (median) | time per run |",
             "|---|---|---|---|---|---|"]
    for s in summary:
        it = f"{s['iters_to_1pct_median']:,.0f}" if s["iters_to_1pct_median"] is not None else "never"
        lines.append(f"| {s['config']} | {s['succeeded']}/{s['runs']} "
                     f"| {s['c_err_median']:.1e} / {s['c_err_worst']:.1e} "
                     f"| {s['k_err_median']:.1e} / {s['k_err_worst']:.1e} | {it} "
                     f"| {s['seconds_mean']:.0f} s |")
    (tables / "step3_oscillator_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def make_figure(runs: list[dict], path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    paper_runs = sorted((r for r in runs if r["config"] == "paper"), key=lambda r: r["seed"])
    ref = paper_runs[0]
    t = ref["t_plot"]
    u_true = osc.exact(t)

    # (a) the network's curve at several points in training
    ax = axes[0, 0]
    ax.plot(t, u_true, color=INK, lw=1.0, ls=":", label="true solution")
    shades = [BLUES(x) for x in np.linspace(0.35, 1.0, len(ref["snapshots"]))]
    for (it, u), color in zip(sorted(ref["snapshots"].items()), shades):
        ax.plot(t, u, color=color, label=f"network at {it:,} it.")
    t_d = osc.observation_times(PAPER["n_data"]).numpy().ravel()
    ax.plot(t_d, osc.exact(t_d), "o", ms=3.5, color=INK_2, label="50 observations", zorder=5)
    for a, _ in osc.WINDOWS[1:]:
        ax.axvline(a, color=MUTED, lw=0.6)
    ax.set(title="(a) The network learns the signal left to right", xlabel="t", ylabel="u(t)")
    ax.legend(loc="upper right", fontsize=7)

    # (b) amplitude recovered in each time window
    ax = axes[0, 1]
    it = [e["iter"] for e in ref["evals"]]
    for i, (key, (a, b)) in enumerate(zip(["amp_0_5", "amp_5_10", "amp_10_15"], osc.WINDOWS)):
        ax.plot(it, [e[key] for e in ref["evals"]], color=SERIES[i], label=f"t in [{a:.0f}, {b:.0f}]")
    ax.set_xscale("log")
    ax.set(title="(b) Fraction of true amplitude captured, per window",
           xlabel="iteration (log scale)", ylabel="network amplitude / true amplitude")
    ax.legend(loc="center right")

    # (c) c and the c implied by the current curve
    ax = axes[0, 2]
    for r in paper_runs[1:]:
        ax.plot([e["iter"] for e in r["evals"]], [e["c"] for e in r["evals"]],
                color=SERIES[0], lw=0.8, alpha=0.3)
    ax.plot(it, [e["c"] for e in ref["evals"]], color=SERIES[0], label="trained c (seed 0; others faint)")
    ax.plot(it, [e["c_implied"] for e in ref["evals"]], color=SERIES[1], lw=1.2, ls="--",
            label="c* implied by the network's curve")
    ax.axhline(osc.C_TRUE, color=INK_2, lw=1.0, ls=":", label="true c = 0.4")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.6)
    ax.set(title="(c) c simply follows the network's curve", xlabel="iteration (log scale)", ylabel="c")
    ax.legend(loc="upper right", fontsize=7)

    # (d) k trajectories for the paper setting
    ax = axes[1, 0]
    for r in paper_runs:
        lw, alpha = (1.6, 1.0) if r is ref else (0.8, 0.3)
        ax.plot([e["iter"] for e in r["evals"]], [e["k"] for e in r["evals"]],
                color=SERIES[0], lw=lw, alpha=alpha)
    ax.axhline(osc.K_TRUE, color=INK_2, lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set(title="(d) k settles early (paper setting, all seeds)", xlabel="iteration (log scale)", ylabel="k")

    # (e), (f) final errors of every run, per configuration
    configs = [c for c in CONFIGS if any(r["config"] == c for r in runs)]
    for ax, key, name in [(axes[1, 1], "c_err", "c"), (axes[1, 2], "k_err", "k")]:
        for i, config in enumerate(configs):
            rs = [r for r in runs if r["config"] == config]
            xs = i + np.linspace(-0.12, 0.12, len(rs))
            ok = [r["success"] for r in rs]
            ax.scatter(xs[ok], [r[key] for r, o in zip(rs, ok) if o], s=36, color=SERIES[i],
                       zorder=3, label=None)
            ax.scatter(xs[~np.array(ok)], [r[key] for r, o in zip(rs, ok) if not o], s=36,
                       facecolors="none", edgecolors=SERIES[i], linewidths=1.5, zorder=3)
        ax.axhline(SUCCESS_TOL, color=MUTED, lw=0.8, ls="--")
        ax.annotate("1% (success threshold)", xy=(1, SUCCESS_TOL), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=8)
        paper_err = {"c": abs(0.40125215 - 0.4) / 0.4, "k": abs(3.9933197 - 4) / 4}[name]
        ax.axhline(paper_err, color=INK_2, lw=0.8, ls=":")
        ax.annotate("paper's single run", xy=(0, paper_err), xycoords=("axes fraction", "data"),
                    xytext=(4, 4), textcoords="offset points", color=INK_2, fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(configs)), [LABELS[c] for c in configs])
        ax.set_xlim(-0.5, len(configs) - 0.5)
        ax.grid(axis="x", visible=False)
        ax.set(title=f"({'e' if name == 'c' else 'f'}) Final error of {name}, every seed "
                     "(hollow = failed)", ylabel=f"|{name} - true| / true")

    fig.suptitle("Step 3: damped oscillator, find c and k (paper section 4.3)",
                 x=0.01, ha="left", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3: damped oscillator inverse problem")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--jobs", type=int, default=1, help="runs to execute in parallel")
    parser.add_argument("--quick", action="store_true", help="tiny smoke test")
    parser.add_argument("--bench", action="store_true", help="time CPU vs GPU and exit")
    args = parser.parse_args()

    if args.bench:
        bench()
        return 0
    if args.quick:
        args.seeds = args.seeds[:2]

    run_dir = make_run_dir("step3_oscillator" + ("_quick" if args.quick else ""))
    tasks = [(c, s) for c in args.configs for s in args.seeds]
    print(f"{len(tasks)} runs on {args.device} with {args.jobs} in parallel -> {run_dir}")

    runs, t0 = [], time.perf_counter()
    ctx = mp.get_context("spawn")  # works the same on Windows and Linux
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
        futures = {pool.submit(run_one, c, s, args.device, args.quick): (c, s) for c, s in tasks}
        for n, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            runs.append(r)
            print(f"  [{n:>2}/{len(tasks)}] {r['config']:<14} seed {r['seed']}  "
                  f"c {r['c']:.5f}  k {r['k']:.5f}  "
                  f"{'ok    ' if r['success'] else 'FAILED'}  {r['seconds']:6.0f}s  "
                  f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    runs.sort(key=lambda r: (list(CONFIGS).index(r["config"]), r["seed"]))
    with open(run_dir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)

    if args.quick:
        tables, figure = run_dir / "tables", run_dir / "step3_oscillator_quick.png"
    else:
        tables, figure = Path("results/tables"), Path("results/figures/step3_oscillator.png")
    summary = write_tables(runs, tables)
    if any(r["config"] == "paper" for r in runs):
        make_figure(runs, figure)
    save_json({"args": vars(args), "env": env_info(), "summary": summary}, run_dir / "metrics.json")

    print("\n" + (tables / "step3_oscillator_summary.md").read_text(encoding="utf-8"))
    print(f"figure: {figure}\ntables: {tables}\nraw runs: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
