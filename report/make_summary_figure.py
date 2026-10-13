"""
Draw the one-glance summary figure for the README and report from the
committed results tables (no training, runs in a second).

    python report/make_summary_figure.py

Writes results/figures/summary.png. Each panel compares the paper's method,
the strongest alternative we tried, and our fix, on the metric that the
problem's diagnosis singled out.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinn.plotting import AXIS, INK, INK_2, MUTED, SERIES, apply_style

TABLES = Path("results/tables")
OUT = Path("results/figures/summary.png")
OURS = SERIES[0]
OTHERS = ["#c9c8c1", "#a3a29a"]   # paper, alternative: quiet neutrals so the fix stands out


def rows(table: str, config: str) -> list[dict]:
    with open(TABLES / f"{table}_runs.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["config"] == config]


def values(table: str, config: str, key: str) -> list[float]:
    return [float(r[key]) for r in rows(table, config)]


def error_panel(ax, groups, ylabel, title):
    """Median bar per method, every seed as a dot, log scale."""
    rng = np.random.default_rng(0)
    for i, (label, vals, color) in enumerate(groups):
        med = statistics.median(vals)
        ax.bar(i, med, width=0.62, color=color, zorder=2)
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(vals)), vals, s=14, color=INK_2,
                   alpha=0.55, linewidths=0, zorder=3)
        ax.annotate(f"{med:.1e}", (i, max(vals)), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 8)   # headroom for the labels and the headline
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set(ylabel=ylabel, title=title)
    ax.grid(axis="x", visible=False)
    paper, ours = statistics.median(groups[0][1]), statistics.median(groups[-1][1])
    ax.text(0.98, 0.97, f"{paper / ours:,.0f}x lower", transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color=OURS)


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"wspace": 0.32})

    # (a) oscillator: how often c and k are found, and at what cost
    ax = axes[0]
    osc = [("paper\n(Adam 100k)", "oscillator_reproduce", "paper", OTHERS[0]),
           ("Adam 20k\n+ L-BFGS", "oscillator_two_stage", "adam20k_lbfgs", OTHERS[1]),
           ("two-stage\n(ours)", "oscillator_two_stage", "two_stage", OURS)]
    times = []
    for i, (label, table, config, color) in enumerate(osc):
        ok = [r["success"] in ("True", "1", "true") for r in rows(table, config)]
        secs = statistics.mean(values(table, config, "seconds"))
        times.append(secs)
        share = 100 * sum(ok) / len(ok)
        ax.bar(i, share, width=0.62, color=color, zorder=2)
        ax.annotate(f"{sum(ok)}/{len(ok)}", (i, share), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
        ax.text(i, 4, f"{secs:.0f} s\nper run", ha="center", va="bottom", fontsize=8,
                color="white" if color == OURS else INK_2, zorder=3)
    ax.set_xticks(range(3), [o[0] for o in osc])
    ax.set_yticks(range(0, 101, 20))
    ax.set(ylim=(0, 132), ylabel="seeds finding c and k within 1% (%)",
           title="(a) Damped oscillator: finding c and k")
    ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.97, f"{times[0] / times[-1]:.0f}x faster", transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color=OURS)

    # (b) consolidation: overall accuracy
    error_panel(axes[1], [
        ("paper\n(Adam 50k)", values("consolidation_reproduce", "paper", "rel_l2"), OTHERS[0]),
        ("adaptive\nsampling (RAD)", values("consolidation_enrichment", "rad", "rel_l2"), OTHERS[1]),
        ("corner-enriched\noutput (ours)", values("consolidation_enrichment", "enriched", "rel_l2"), OURS),
    ], "relative L2 error", "(b) Consolidation: error near the corner")

    # (c) heat: worst-case error, which sits at the edges and then the corners
    error_panel(axes[2], [
        ("paper\n(soft BC)", values("heat_reproduce", "paper", "max_err"), OTHERS[0]),
        ("exact BC", values("heat_corners", "hard5k", "max_err"), OTHERS[1]),
        ("exact BC +\ncorner terms (ours)", values("heat_corners", "corner5k", "max_err"), OURS),
    ], "max error over the plate", "(c) Heat conduction: worst-case error")

    for ax in axes:
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_color(AXIS)
    fig.text(0.01, 0.02, "Bars: median over seeds (5 or 10). Dots: individual seeds. "
             "All errors against exact solutions.", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"figure: {OUT}")


if __name__ == "__main__":
    main()
