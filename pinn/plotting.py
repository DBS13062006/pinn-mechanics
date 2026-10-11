"""
One consistent look for every figure in the report.

Colors come from a colorblind-checked palette: series colors are always used
in the same fixed order (blue, orange, aqua, ...), magnitudes use a single
hue from light to dark, and text stays in neutral ink rather than series
colors. Using the helpers below keeps all figures consistent without
thinking about it.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical series colors, in fixed order. Never cycle or reorder them.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Sequential (magnitude) maps: one hue, light -> dark.
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fde3d6", "#f6b08f", "#eb6834", "#c24e1f", "#8f3914", "#5e250c"])


def apply_style() -> None:
    """Call once before making figures."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "image.cmap": "seq_blue",
    })
    for cmap in (BLUES, ORANGES):
        if cmap.name not in plt.colormaps():
            mpl.colormaps.register(cmap)


def lbfgs_start(log: list[dict]) -> int | None:
    """Iteration where Adam handed over to L-BFGS (None if L-BFGS never ran)."""
    if not any(row["phase"] == "lbfgs" for row in log):
        return None
    return max((row["iter"] for row in log if row["phase"] == "adam"), default=0)


def mark_lbfgs_start(ax, log: list[dict], label: bool = True) -> None:
    """Dashed vertical line where L-BFGS takes over."""
    switch = lbfgs_start(log)
    if switch is None:
        return
    ax.axvline(switch, color=MUTED, lw=0.8, ls="--")
    if label:
        ax.annotate("Adam | L-BFGS", xy=(switch, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -10), textcoords="offset points", ha="right",
                    color=INK_2, fontsize=8)


def plot_loss_history(ax, log: list[dict], terms: list[str], title: str = "",
                      legend_loc: str = "lower left") -> None:
    """Each loss term vs iteration on a log scale, with the Adam -> L-BFGS switch marked."""
    iters = [row["iter"] for row in log]
    for i, name in enumerate(terms):
        ax.plot(iters, [row[name] for row in log], color=SERIES[i], label=name)
    mark_lbfgs_start(ax, log)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss term")
    ax.legend(loc=legend_loc)
    if title:
        ax.set_title(title)
