"""
How well can c and k be found from imperfect data, and how does the
PINN compare with simpler methods? The paper only tried clean data at 50
points and compared against nothing.

Data (problems/oscillator.py: noisy_observations):
  noise   0%, 1%, 3%, 10% of the initial displacement (Gaussian)
  full50  50 points over [0, 15] (the paper's setting)
  full15  15 points over [0, 15] (fewer than 3 per oscillation)
  early50 50 points over [0, 5] only; the method must work out the rest
10 random draws per (setting, noise level). Every method sees the same data.

Methods (all start from c = 0.2, k = 2 and know the ODE, m, u(0), u'(0)):
  classical      ODE-constrained least squares (scipy): adjust c, k until the
                 numerical ODE solution fits the data. The textbook method.
  plain NN       fit a network to the data alone, then read c and k off its
                 derivatives (stages 1-2 of the two-stage PINN, recorded from
                 the same runs, so no extra training)
  standard PINN  Adam 20k + L-BFGS on the full PINN loss (best of the reproduction)
  two-stage PINN the fix from oscillator_two_stage.py

Reference: the Cramer-Rao bound, the smallest spread ANY unbiased method can
achieve with this much noise (problems/oscillator.py: cramer_rao_std). With
noise, "within 1%" stops being a fair target: at 3% noise even an ideal method
misses c by about 1.3% half the time. So errors are also reported as a
multiple of what an ideal method would typically get.

Usage (from the repo root):
  python experiments/oscillator_noise.py --device cpu --jobs 3
  python experiments/oscillator_noise.py --quick
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

# Shared settings from the two-stage script (same folder, so a plain import works).
import oscillator_two_stage as s3b
from pinn import (MLP, TrainConfig, Trainer, env_info, make_run_dir,
                  relative_l2, save_json, set_seed, setup_torch)
from pinn.plotting import INK_2, MUTED, SERIES, apply_style
from problems import oscillator as osc

TRAINED = ["standard_pinn", "two_stage", "classical"]          # what actually runs (slowest first)
METHODS = ["classical", "plain_nn", "standard_pinn", "two_stage"]  # what is reported
LABELS = {"classical": "classical (ODE least squares)", "plain_nn": "plain NN + regression",
          "standard_pinn": "standard PINN (Adam 20k + L-BFGS)", "two_stage": "two-stage PINN"}
SHORT = {"classical": "classical", "plain_nn": "plain NN", "standard_pinn": "standard PINN",
         "two_stage": "two-stage PINN"}
COLORS = {"classical": INK_2, "plain_nn": SERIES[2], "standard_pinn": SERIES[1], "two_stage": SERIES[0]}
SETTING_TITLES = {"full50": "50 points over [0, 15] (paper)", "full15": "15 points over [0, 15]",
                  "early50": "50 points over [0, 5] only"}
TYPICAL = 0.6745            # median |error| of a normal estimator = 0.6745 x its std
ERR_FLOOR = 1e-6            # errors below this are drawn at this value
N_TEST = 10_000


def curve_error(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    return float(np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true))


def run_one(method: str, setting: str, noise: float, draw: int, device_name: str,
            quick: bool = False) -> list[dict]:
    """One method on one dataset. Returns one row per reported method
    (the two-stage run also yields the plain-NN result)."""
    device = setup_torch(torch.float64, prefer=device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    t_d, u_d = osc.noisy_observations(setting, noise, draw, device)
    t_end = osc.DATA_SETTINGS[setting][1]
    t_test = osc.test_times(N_TEST, device)
    u_test = osc.exact(t_test)
    base = {"setting": setting, "noise": noise, "draw": draw, "device": str(device)}

    def row(name, c, k, seconds, u_rel, **extra):
        c_err, k_err = abs(c - osc.C_TRUE) / osc.C_TRUE, abs(k - osc.K_TRUE) / osc.K_TRUE
        return {**base, "method": name, "c": c, "k": k, "c_err": c_err, "k_err": k_err,
                "within_1pct": c_err < 0.01 and k_err < 0.01, "u_rel_l2": u_rel,
                "seconds": seconds, **extra}

    if method == "classical":
        t0 = time.perf_counter()
        fit = osc.classical_fit(t_d.cpu().numpy(), u_d.cpu().numpy())
        seconds = time.perf_counter() - t0
        u_fit = osc.solve_numerically(t_test.cpu().numpy().ravel(), fit["c"], fit["k"])
        return [row("classical", fit["c"], fit["k"], seconds,
                    curve_error(u_fit, u_test.cpu().numpy().ravel()),
                    se_c=fit["se_c"], se_k=fit["se_k"], converged=fit["converged"])]

    set_seed(draw)
    model = MLP(1, 1, s3b.WIDTH, s3b.DEPTH, lb=[0.0], ub=[osc.T_MAX], init=s3b.INIT).to(device)
    c = torch.nn.Parameter(torch.tensor(osc.C_START, device=device))
    k = torch.nn.Parameter(torch.tensor(osc.K_START, device=device))
    t_f = osc.collocation_times(s3b.N_F, device)          # the ODE is enforced on all of [0, 15]
    t_diag = torch.linspace(0.0, t_end, 2_000, device=device).reshape(-1, 1)   # measured window only
    net = list(model.parameters())
    lbfgs_max = 500 if quick else s3b.LBFGS_MAX

    def loss_data_only():
        return {"ic": osc.initial_condition_loss(model, device), "data": ((model(t_d) - u_d) ** 2).mean()}

    def loss_full():
        r = osc.residual(model, t_f, c, k)
        return {"ode": (r**2).mean(), "ic": osc.initial_condition_loss(model, device),
                "data": ((model(t_d) - u_d) ** 2).mean()}

    def cfg(adam, lbfgs):
        return TrainConfig(adam_iters=adam, adam_lr=s3b.LR, lbfgs_iters=lbfgs,
                           weights=s3b.WEIGHTS, verbose=False)

    def u_error():
        with torch.no_grad():
            return relative_l2(model(t_test), u_test)

    t0 = time.perf_counter()
    if method == "standard_pinn":
        Trainer(net + [c, k], loss_full, cfg(1_000 if quick else 20_000, lbfgs_max)).train()
        return [row("standard_pinn", c.item(), k.item(), time.perf_counter() - t0, u_error())]

    # two-stage, stage 1: data + initial condition only
    Trainer(net, loss_data_only, cfg(1_000 if quick else s3b.STAGE1_ADAM, lbfgs_max)).train()
    seconds_nn = time.perf_counter() - t0
    # stage 2: c and k implied by the fitted curve, over the measured window only
    c_nn, k_nn = osc.implied_parameters(model, t_diag)
    nn_row = row("plain_nn", c_nn, k_nn, seconds_nn, u_error())
    # A noisy fit can imply physically impossible values (negative damping or
    # stiffness). Then keep the starting guess rather than start the PINN there.
    fallback = not (c_nn >= 0 and k_nn > 0)
    if not fallback:
        with torch.no_grad():
            c.fill_(c_nn)
            k.fill_(k_nn)
    # stage 3a: physics on, c and k frozen; stage 3b: everything free
    Trainer(net, loss_full, cfg(0, lbfgs_max)).train()
    Trainer(net + [c, k], loss_full, cfg(0, lbfgs_max)).train()
    two_row = row("two_stage", c.item(), k.item(), time.perf_counter() - t0, u_error(),
                  stage2_fallback=fallback)
    return [nn_row, two_row]


# ---------------------------------------------------------------- outputs
COLS = ["method", "setting", "noise", "draw", "c", "k", "c_err", "k_err", "within_1pct", "u_rel_l2",
        "seconds", "se_c", "se_k", "converged", "stage2_fallback", "device"]


def groups(runs):
    out = {}
    for r in runs:
        out.setdefault((r["method"], r["setting"], r["noise"]), []).append(r)
    return out


def write_tables(runs: list[dict], tables: Path, quick: bool) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "oscillator_noise_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({c: r.get(c, "") for c in COLS})
    g = groups(runs)
    n_draws = max(len(v) for v in g.values())
    med = lambda rs, key: statistics.median(r[key] for r in rs)
    lines = ["# Damped oscillator: finding c and k from noisy and sparse data", "",
             f"Truth c = 0.4, k = 4; every method starts from c = 0.2, k = 2. Median relative error over "
             f"{n_draws} noise draws, worst in brackets." + (" QUICK TEST RUN: not results." if quick else ""),
             "Noise: standard deviation as a fraction of the initial displacement.",
             "'ideal' = typical (median) error of the best possible unbiased method at this noise level "
             "(Cramer-Rao bound); with no noise it is 0.", ""]
    for name, key in [("c (damping)", "c_err"), ("k (stiffness)", "k_err")]:
        lines += [f"## Error in {name}", "",
                  "| data | noise | ideal | " + " | ".join(SHORT[m] for m in METHODS) + " |",
                  "|---|---|---|" + "---|" * len(METHODS)]
        for setting in osc.DATA_SETTINGS:
            for noise in osc.NOISE_LEVELS:
                cr = osc.cramer_rao_std(setting, noise)[0 if key == "c_err" else 1] * TYPICAL
                cells = []
                for m in METHODS:
                    rs = g.get((m, setting, noise))
                    cells.append(f"{med(rs, key):.1e} ({max(r[key] for r in rs):.1e})" if rs else "-")
                lines.append(f"| {setting} | {noise:.0%} | {cr:.1e} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Error of the recovered curve u(t) over [0, 15] (relative L2, median)", "",
              "| data | noise | " + " | ".join(SHORT[m] for m in METHODS) + " |",
              "|---|---|" + "---|" * len(METHODS)]
    for setting in osc.DATA_SETTINGS:
        for noise in osc.NOISE_LEVELS:
            cells = [f"{med(g[(m, setting, noise)], 'u_rel_l2'):.1e}" if (m, setting, noise) in g else "-"
                     for m in METHODS]
            lines.append(f"| {setting} | {noise:.0%} | " + " | ".join(cells) + " |")
    lines += ["", "## Cost", "",
              "| method | time per dataset (mean) | runs within 1% for both c and k, no noise |", "|---|---|---|"]
    for m in METHODS:
        rs = [r for r in runs if r["method"] == m]
        clean = [r for r in rs if r["noise"] == 0]
        if rs:
            lines.append(f"| {LABELS[m]} | {statistics.mean(r['seconds'] for r in rs):.1f} s "
                         f"| {sum(r['within_1pct'] for r in clean)}/{len(clean)} |")
    fb = [r for r in runs if r["method"] == "two_stage" and r.get("stage2_fallback")]
    lines += ["", f"Two-stage runs whose data-only fit implied impossible values (c < 0 or k <= 0), so stage 3 "
              f"started from the initial guess instead: {len(fb)} of "
              f"{sum(r['method'] == 'two_stage' for r in runs)}.",
              "The plain NN's time is its share of the two-stage run (stage 1 only)."]
    (tables / "oscillator_noise_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(runs: list[dict], path: Path) -> None:
    apply_style()
    g = groups(runs)
    settings = list(osc.DATA_SETTINGS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True, sharey="row")
    offsets = dict(zip(METHODS, np.linspace(-0.27, 0.27, len(METHODS))))
    xs = np.arange(len(osc.NOISE_LEVELS))
    for row, (key, name) in enumerate([("c_err", "c (damping)"), ("k_err", "k (stiffness)")]):
        for col, setting in enumerate(settings):
            ax = axes[row, col]
            # ideal: typical error of the best possible unbiased method
            ideal = [osc.cramer_rao_std(setting, n)[row] * TYPICAL for n in osc.NOISE_LEVELS]
            ax.plot(xs[1:], ideal[1:], color=MUTED, lw=1.4, ls="--", zorder=1,
                    label="ideal (Cramer-Rao)" if col == 0 else None)
            for m in METHODS:
                meds = []
                for i, noise in enumerate(osc.NOISE_LEVELS):
                    rs = g.get((m, setting, noise), [])
                    vals = np.maximum([r[key] for r in rs], ERR_FLOOR)
                    jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
                    ax.scatter(i + offsets[m] + jitter, vals, s=9, color=COLORS[m], alpha=0.45,
                               linewidths=0, zorder=2)
                    meds.append(np.median(vals) if len(vals) else np.nan)
                ax.plot(xs + offsets[m], meds, "o-", ms=5, lw=1.6, color=COLORS[m], zorder=3,
                        label=LABELS[m] if col == 0 else None)
            ax.set_yscale("log")
            ax.set_xticks(xs, [f"{n:.0%}" for n in osc.NOISE_LEVELS])
            ax.set_xlim(-0.5, len(xs) - 0.5)
            ax.grid(axis="x", visible=False)
            ax.set_xlabel("noise (fraction of initial displacement)")
            letter = "abcdef"[row * 3 + col]
            ax.set_title(f"({letter}) {name}: {SETTING_TITLES[setting]}")
            if col == 0:
                ax.set_ylabel(f"|{name.split()[0]} - true| / true")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8.5, frameon=False)
    axes[1, 0].annotate(f"dots = draws, line = median; dashed = typical error of an ideal method; "
                        f"errors below {ERR_FLOOR:.0e} drawn at {ERR_FLOOR:.0e}",
                        xy=(0.01, 0.01), xycoords="axes fraction", fontsize=7, color=INK_2)
    fig.suptitle("Damped oscillator: finding c and k from noisy and sparse data", x=0.01, ha="left", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Damped oscillator: noisy and sparse data")
    parser.add_argument("--methods", nargs="+", default=TRAINED, choices=TRAINED)
    parser.add_argument("--settings", nargs="+", default=list(osc.DATA_SETTINGS), choices=list(osc.DATA_SETTINGS))
    parser.add_argument("--noise", nargs="+", type=float, default=list(osc.NOISE_LEVELS))
    parser.add_argument("--draws", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", metavar="RUN_DIR",
                        help="continue an interrupted run: reuse the trainings already saved in RUN_DIR")
    args = parser.parse_args()
    if args.quick:
        args.draws = args.draws[:2]
    run_dir = Path(args.resume) if args.resume else make_run_dir("oscillator_noise" + ("_quick" if args.quick else ""))
    partial = run_dir / "partial"            # every finished task is saved at once (see heat_reproduce.py)
    partial.mkdir(parents=True, exist_ok=True)

    def fname(m, s, n, d):
        return partial / f"{m}_{s}_noise{n:g}_draw{d}.pkl"

    runs, tasks = [], []
    for m in args.methods:
        for s in args.settings:
            for n in args.noise:
                for d in args.draws:
                    f = fname(m, s, n, d)
                    if f.exists():
                        with open(f, "rb") as fh:
                            runs += pickle.load(fh)
                    else:
                        tasks.append((m, s, n, d))
    if runs:
        print(f"resuming: {len(runs)} results already saved in {partial}")
    print(f"{len(tasks)} runs on {args.device} with {args.jobs} in parallel -> {run_dir}")
    print(f"(if this gets interrupted, rerun the same command with --resume {run_dir})")
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, *task, args.device, args.quick): task for task in tasks}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            rows = fut.result()
            with open(fname(*futures[fut]), "wb") as fh:
                pickle.dump(rows, fh)
            runs += rows
            m, s, n, d = futures[fut]
            r = rows[-1]
            print(f"  [{i:>3}/{len(tasks)}] {m:<13} {s:<7} noise {n:<4g} draw {d}  c {r['c']:.4f}  k {r['k']:.4f}  "
                  f"{r['seconds']:5.0f}s  (elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    order = {m: i for i, m in enumerate(METHODS)}
    runs.sort(key=lambda r: (list(osc.DATA_SETTINGS).index(r["setting"]), r["noise"], order[r["method"]], r["draw"]))
    with open(run_dir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)
    if args.quick:
        tables, figure = run_dir / "tables", run_dir / "oscillator_noise_quick.png"
    else:
        tables, figure = Path("results/tables"), Path("results/figures/oscillator_noise.png")
    write_tables(runs, tables, args.quick)
    make_figure(runs, figure)
    save_json({"args": vars(args), "env": env_info()}, run_dir / "metrics.json")
    print("\n" + (tables / "oscillator_noise_summary.md").read_text(encoding="utf-8"))
    print(f"figure: {figure}\ntables: {tables}\nraw runs: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
