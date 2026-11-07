"""One plotting style for every figure (report + app). Palette = validated categorical order."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

PALETTE = ["#1A8FB0", "#FF2E63", "#7469B6", "#2E9E6E", "#B04A98", "#E07B39", "#3E6FD0", "#B88A10"]  # same as ds/viz.py
STAYS, LEAVES = PALETTE[0], PALETTE[1]
GRAY = "#6B6F7B"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#C9BFD6",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#ECE7F0",
    "grid.linewidth": 0.8, "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "lines.linewidth": 2,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelcolor": "#252A34",
    "xtick.color": "#252A34", "ytick.color": "#252A34", "axes.titlecolor": "#142850", "legend.frameon": False, "figure.dpi": 110,
})


def fig(w=7, h=4, n=1, m=1):
    f, ax = plt.subplots(m, n, figsize=(w, h))
    return f, ax


def save(f, name: str):
    f.tight_layout()
    f.savefig(config.FIGURES / f"{name}.png", dpi=130, bbox_inches="tight")
    return f


from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

DIV = LinearSegmentedColormap.from_list("churn_div", ["#1A8FB0", "#F2F0F4", "#FF2E63"])  # two poles + neutral midpoint
