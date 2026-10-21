"""
The paper's future work (section 5): the COUPLED poromechanics equations
(5.1)-(5.2), solved with one PINN for both the soil displacement U and the
pore pressure P. See problems/poromechanics.py for the equations and the
exact solution.

Two problems, mirroring the paper's two classes:

  forward  (data-driven solution)  find U(x, t) and P(x, t).
      paper     section 4.1's recipe applied to the coupled system:
                U = (1 - x) N_u, P = x N_p; initial condition and top load
                are penalties in the loss.
      enriched  our section 4.1 fix carried over: the erf corner term in P
                and the matching displacement term in U are built in, so the
                initial state is exact and the network learns smooth parts.

  inverse  (data-driven discovery)  find the consolidation coefficient c
           (which sets how fast the soil settles; c = k K / mu in physical
           terms, so it gives the permeability k) from 20 settlement
           readings of the top surface, as a geotechnical engineer would
           from a settlement plate. True c = 0.5, start c = 0.1 (5x off).
      paper / enriched outputs, clean readings and readings with 1% noise.
      Baseline: least-squares fit of Terzaghi's closed-form settlement curve.

Every configuration runs over 5 seeds. Network: 4 x 40 tanh, Glorot normal;
Adam 5k (lr 1e-3) then L-BFGS (up to 5k), all loss weights 1.

Usage (from the repo root):
  python experiments/poromechanics.py --jobs 5            # everything, 25 runs
  python experiments/poromechanics.py --quick             # 2 seeds, short budget
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

from pinn import (MLP, TrainConfig, Trainer, env_info, make_run_dir, relative_l2,
                  save_json, set_seed, setup_torch)
from pinn.plotting import BLUES, INK, INK_2, MUTED, SERIES, apply_style
from problems import poromechanics as poro

WIDTH, DEPTH, LR = 40, 4, 1e-3
ADAM, LBFGS = 5_000, 5_000
FIELDS = {"paper": poro.fields_paper, "enriched": poro.fields_enriched}
TASKS = {  # name -> (problem, output, noise)
    "fwd_paper": ("forward", "paper", 0.0),
    "fwd_enriched": ("forward", "enriched", 0.0),
    "inv_paper": ("inverse", "paper", 0.0),
    "inv_enriched": ("inverse", "enriched", 0.0),
    "inv_enriched_noisy": ("inverse", "enriched", poro.OBS_NOISE),
}
LABELS = {"fwd_paper": "paper recipe", "fwd_enriched": "corner-enriched",
          "inv_paper": "paper recipe,\nclean", "inv_enriched": "enriched,\nclean",
          "inv_enriched_noisy": "enriched,\n1% noise"}
PROFILE_TIMES = (0.01, 0.05, 0.2, 0.5)
TAU_CURVE = np.linspace(0, poro.T_MAX, 201)
SUCCESS_TOL = 0.01


def run_one(task: str, seed: int, device_name: str, quick: bool = False) -> dict:
    device = setup_torch(torch.float64, prefer=device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    problem, output, noise = TASKS[task]
    fields = FIELDS[output]
    inverse = problem == "inverse"
    t_max = 1.0 if inverse else poro.T_MAX

    set_seed(seed)
    model = MLP(2, 2, WIDTH, DEPTH, lb=[0, 0], ub=[1, t_max], init="xavier_normal").to(device)
    x_f, t_f = poro.collocation_points(t_max=t_max, device=device)
    t_b = poro.boundary_times(t_max=t_max, device=device)
    x_0 = poro.initial_positions(device=device)
    params = list(model.parameters())
    if inverse:
        log_c = torch.tensor(np.log(poro.C_START), device=device, requires_grad=True)
        params.append(log_c)
        t_obs, u_obs = poro.settlement_readings(noise, draw=seed, device=device)
        c_of = lambda: torch.exp(log_c)
    else:
        c_of = lambda: 1.0

    def loss_fn():
        c = c_of()
        mass, eq = poro.residuals(fields, model, x_f, t_f, c)
        terms = {"mass": (mass**2).mean(), "equilibrium": (eq**2).mean(),
                 "top_load": (poro.top_load_residual(fields, model, t_b, c) ** 2).mean(),
                 "base_flow": (poro.base_flow_residual(fields, model, t_b, c) ** 2).mean()}
        if output == "paper":  # the enriched output has the initial state built in
            terms["initial"] = (poro.initial_residual(fields, model, x_0, c) ** 2).mean()
        if inverse:
            x_top = torch.zeros_like(t_obs)
            U, _ = fields(model, x_top, t_obs, c)
            terms["data"] = ((U - u_obs) ** 2).mean()
        return terms

    X, T, shape = poro.test_grid(t_max=t_max, device=device)
    tau_grid = (poro.C_TRUE * T) if inverse else T
    U_ref = poro.exact_displacement(X, tau_grid)
    P_ref = poro.exact_pressure(X, tau_grid)

    def eval_fn():
        U, P = fields(model, X, T, c_of())
        out = {"rel_l2_U": relative_l2(U, U_ref), "rel_l2_P": relative_l2(P, P_ref)}
        if inverse:
            out["c"] = float(c_of().detach())
        return out

    cfg = TrainConfig(adam_iters=800 if quick else ADAM, adam_lr=LR,
                      lbfgs_iters=300 if quick else LBFGS, eval_every=100,
                      log_every=100, verbose=False)
    trainer = Trainer(params, loss_fn, cfg, eval_fn=eval_fn).train()

    with torch.no_grad():
        U, P = fields(model, X, T, c_of())
    U, P = U.cpu().numpy().reshape(shape), P.cpu().numpy().reshape(shape)
    Ur, Pr = U_ref.cpu().numpy().reshape(shape), P_ref.cpu().numpy().reshape(shape)
    t_np = T.cpu().numpy().reshape(shape)
    late = (t_np * (poro.C_TRUE if inverse else 1.0)) >= 0.01
    final = trainer.evals[-1]
    row = {
        "task": task, "seed": seed, "device": str(device),
        "rel_l2_U": final["rel_l2_U"], "rel_l2_P": final["rel_l2_P"],
        "max_err_U": float(np.abs(U - Ur).max()),
        "max_err_P_tau_ge_0.01": float(np.abs(P - Pr)[late].max()),
        "P_max": float(P.max()), "settlement_err": float(np.abs(U[:, 0] - Ur[:, 0]).max()),
        "iters_total": trainer.iteration, "seconds": trainer.timings["total_seconds"],
        "final_loss": trainer.log[-1]["loss"],
    }
    # curves for the figure (in tau, the physical consolidation time)
    t_curve = torch.as_tensor(TAU_CURVE / (poro.C_TRUE if inverse else 1.0), device=device).reshape(-1, 1)
    xs = torch.linspace(0, 1, 201, device=device).reshape(-1, 1)
    with torch.no_grad():
        row["settlement_curve"] = fields(model, torch.zeros_like(t_curve), t_curve, c_of())[0].cpu().numpy().ravel()
        prof = {}
        for tau in PROFILE_TIMES:
            tt = torch.full_like(xs, tau / (poro.C_TRUE if inverse else 1.0))
            Uq, Pq = fields(model, xs, tt, c_of())
            prof[tau] = (Uq.cpu().numpy().ravel(), Pq.cpu().numpy().ravel())
        row["profiles"] = prof
    if inverse:
        c_hat = float(c_of().detach())
        row.update(c=c_hat, c_rel_err=abs(c_hat - poro.C_TRUE) / poro.C_TRUE,
                   c_history=[(e["iter"], e["c"]) for e in trainer.evals])
        fit = poro.classical_fit(t_obs.cpu().numpy(), u_obs.cpu().numpy())
        row.update(c_classical=fit["c"], c_classical_rel_err=abs(fit["c"] - poro.C_TRUE) / poro.C_TRUE)
    return row


# ---------------------------------------------------------------------------- tables
def write_tables(runs: list[dict], tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    cols = ["task", "seed", "rel_l2_U", "rel_l2_P", "max_err_U", "max_err_P_tau_ge_0.01", "P_max",
            "settlement_err", "c", "c_rel_err", "c_classical", "c_classical_rel_err",
            "iters_total", "seconds", "device"]
    with open(tables / "poromechanics_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    med = lambda rs, k: statistics.median(r[k] for r in rs)
    worst = lambda rs, k: max(r[k] for r in rs)
    group = lambda t: [r for r in runs if r["task"] == t]
    n = len({r["seed"] for r in runs})
    lines = ["# Coupled poromechanics, Eqs. (5.1)-(5.2) (paper section 5, future work)", "",
             f"Median over {n} seeds (worst in brackets). Errors against the exact solution.", "",
             "## Forward problem: displacement U and pore pressure P", "",
             "| output | rel. L2 error, U | rel. L2 error, P | max error U | max error P, tau >= 0.01 "
             "| highest P (physics: <= gamma = %.3f) | time per run |" % poro.GAMMA,
             "|---|---|---|---|---|---|---|"]
    for t in ("fwd_paper", "fwd_enriched"):
        rs = group(t)
        if rs:
            lines.append(
                f"| {LABELS[t]} | {med(rs, 'rel_l2_U'):.1e} ({worst(rs, 'rel_l2_U'):.1e}) "
                f"| {med(rs, 'rel_l2_P'):.1e} ({worst(rs, 'rel_l2_P'):.1e}) "
                f"| {med(rs, 'max_err_U'):.1e} | {med(rs, 'max_err_P_tau_ge_0.01'):.1e} "
                f"| {med(rs, 'P_max'):.3f} ({worst(rs, 'P_max'):.3f}) | {statistics.mean(r['seconds'] for r in rs):.0f} s |")
    lines += ["", f"## Inverse problem: consolidation coefficient c from {poro.N_OBS} settlement readings", "",
              f"True c = {poro.C_TRUE}, start c = {poro.C_START}. A run succeeds if c ends within 1%.", "",
              "| method | succeeded | c error, median (worst) | classical curve fit, same readings | rel. L2 error, P | time per run |",
              "|---|---|---|---|---|---|"]
    for t in ("inv_paper", "inv_enriched", "inv_enriched_noisy"):
        rs = group(t)
        if rs:
            ok = sum(r["c_rel_err"] < SUCCESS_TOL for r in rs)
            lines.append(
                f"| {LABELS[t].replace(chr(10), ' ')} | {ok}/{len(rs)} "
                f"| {med(rs, 'c_rel_err'):.1e} ({worst(rs, 'c_rel_err'):.1e}) "
                f"| {med(rs, 'c_classical_rel_err'):.1e} ({worst(rs, 'c_classical_rel_err'):.1e}) "
                f"| {med(rs, 'rel_l2_P'):.1e} | {statistics.mean(r['seconds'] for r in rs):.0f} s |")
    (tables / "poromechanics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- figure
def make_figure(runs: list[dict], path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    pick = lambda t: next((r for r in runs if r["task"] == t and r["seed"] == 0), None)
    colors = {"fwd_paper": INK_2, "fwd_enriched": SERIES[0], "inv_paper": INK_2,
              "inv_enriched": SERIES[0], "inv_enriched_noisy": SERIES[1]}
    xs = np.linspace(0, 1, 201)
    shades = [BLUES(v) for v in np.linspace(0.45, 1.0, len(PROFILE_TIMES))]

    # (a) settlement curve
    ax = axes[0, 0]
    ax.plot(TAU_CURVE, poro.settlement(TAU_CURVE), color=INK, ls=":", lw=1.2, label="exact")
    for t in ("fwd_paper", "fwd_enriched"):
        r = pick(t)
        if r is not None:
            ax.plot(TAU_CURVE, r["settlement_curve"], color=colors[t], label=LABELS[t])
    ax.set(title="(a) Surface settlement U(0, t) (seed 0)", xlabel="consolidation time t",
           ylabel="settlement (1 = final)")
    ax.legend(loc="lower right")

    # (b), (c) pressure and displacement profiles, enriched vs exact
    for ax, k, name in [(axes[0, 1], 1, "pore pressure P"), (axes[0, 2], 0, "displacement U")]:
        r = pick("fwd_enriched")
        rp = pick("fwd_paper")
        for tau, col in zip(PROFILE_TIMES, shades):
            ref = (poro.exact_pressure if k else poro.exact_displacement)(xs, np.full_like(xs, tau))
            ax.plot(xs, ref, color=col, ls=":", lw=1.0)
            if r is not None:
                ax.plot(xs, r["profiles"][tau][k], color=col, label=f"t = {tau:g}")
            if rp is not None and tau == PROFILE_TIMES[0]:
                ax.plot(xs, rp["profiles"][tau][k], color=SERIES[1], lw=1.0, ls="--",
                        label=f"paper recipe, t = {tau:g}")
        ax.set(title=f"({'b' if k else 'c'}) {name}: enriched (solid), exact (dotted)",
               xlabel="depth x", ylabel=name)
        ax.legend(loc="lower right" if k else "upper right", fontsize=7)

    # (d) forward errors, every seed
    ax = axes[1, 0]
    groups = [("fwd_paper", "rel_l2_U", "U"), ("fwd_paper", "rel_l2_P", "P"),
              ("fwd_enriched", "rel_l2_U", "U"), ("fwd_enriched", "rel_l2_P", "P")]
    for i, (t, key, f) in enumerate(groups):
        rs = [r for r in runs if r["task"] == t]
        ax.scatter(i + np.linspace(-0.15, 0.15, max(len(rs), 1))[:len(rs)], [r[key] for r in rs],
                   s=30, color=colors[t], zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(4), [f"paper\n{f}" if "paper" in t else f"enriched\n{f}" for t, _, f in groups], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_title("(d) Forward: relative L2 error, every seed")

    # (e) inverse: c over training
    ax = axes[1, 1]
    for t in ("inv_paper", "inv_enriched", "inv_enriched_noisy"):
        for j, r in enumerate(rr for rr in runs if rr["task"] == t):
            it, c = zip(*r["c_history"])
            ax.plot(np.maximum(it, 1), c, color=colors[t], lw=1.0, alpha=0.8,
                    label=LABELS[t].replace("\n", " ") if j == 0 else None)
    ax.axhline(poro.C_TRUE, color=INK, ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set(title="(e) Inverse: c during training (true 0.5, start 0.1)", xlabel="iteration (log scale)",
           ylabel="consolidation coefficient c")
    ax.legend(loc="lower right", fontsize=7)

    # (f) inverse: final error vs classical fit
    ax = axes[1, 2]
    tasks = [t for t in ("inv_paper", "inv_enriched", "inv_enriched_noisy") if any(r["task"] == t for r in runs)]
    for i, t in enumerate(tasks):
        rs = [r for r in runs if r["task"] == t]
        off = np.linspace(-0.12, 0.12, len(rs))
        ax.scatter(i - 0.18 + off, [max(r["c_rel_err"], 1e-7) for r in rs], s=30, color=colors[t], zorder=3)
        ax.scatter(i + 0.18 + off, [max(r["c_classical_rel_err"], 1e-7) for r in rs], s=30,
                   facecolors="none", edgecolors=MUTED, zorder=3)
    ax.axhline(SUCCESS_TOL, color=INK_2, lw=0.9, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(range(len(tasks)), [LABELS[t] for t in tasks], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set(title="(f) Inverse: error in c (filled: PINN, open: curve fit)", ylabel="relative error in c")

    fig.suptitle("Coupled poromechanics (paper section 5): one PINN for displacement and pore pressure",
                 x=0.01, ha="left", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Coupled poromechanics PINN (paper section 5)")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = args.seeds[:2]
    run_dir = make_run_dir("poromechanics" + ("_quick" if args.quick else ""))
    jobs = [(t, s) for t in args.tasks for s in args.seeds]
    print(f"{len(jobs)} runs on {args.device} with {args.jobs} in parallel -> {run_dir}")
    runs, t0 = [], time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, t, s, args.device, args.quick) for t, s in jobs]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            runs.append(r)
            extra = f"  c {r['c']:.4f} (classical {r['c_classical']:.4f})" if "c" in r else ""
            print(f"  [{n:>2}/{len(jobs)}] {r['task']:<19} seed {r['seed']}  rel L2 U {r['rel_l2_U']:.1e}  "
                  f"P {r['rel_l2_P']:.1e}{extra}  {r['seconds']:5.0f}s  "
                  f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    runs.sort(key=lambda r: (list(TASKS).index(r["task"]), r["seed"]))
    with open(run_dir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)
    if args.quick:
        tables, figure = run_dir / "tables", run_dir / "poromechanics_quick.png"
    else:
        tables, figure = Path("results/tables"), Path("results/figures/poromechanics.png")
    write_tables(runs, tables)
    make_figure(runs, figure)
    save_json({"args": vars(args), "env": env_info()}, run_dir / "metrics.json")
    print("\n" + (tables / "poromechanics_summary.md").read_text(encoding="utf-8"))
    print(f"figure: {figure}\ntables: {tables}\nraw runs: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
