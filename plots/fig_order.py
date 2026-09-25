"""Section-3 figure: greedy vs cheapest-first deployment order, relative to the best fixed order (in-sample).

    python plots/fig_order.py        # reads plots/fig_order_data/*.json -> plots/fig_order.pdf (+ .png)
                                     # and copies the PDF to figures/fig_order.pdf

One panel per benchmark; x = solve rate of the arbitrageur's allocation c*(lambda), y = its expected spend
under a fixed population order relative to the BEST fixed order of the funded models (the optimum of the
deployable class; 0% = that baseline): the Section-3 greedy (eq. pop_order_greedy) and the models sorted by
average cost per attempt. IN-SAMPLE by design (SOURCE = "in"): allocation, both heuristic orders and the best
order are computed on all problems, so the baseline is a floor and the figure is an optimizer diagnostic
(an optimizer diagnostic). SOURCE = "cv" draws the cross-validated series instead, where the baseline
is selected on the training fold and a data-independent order can beat it on the held-out problems (the
leave-one-out penalty of a discrete choice decided by a few problems). The per-instance oracle order is in
the records but not drawn (not deployable). Records: `experiments/cv_order.py`; style: `plots/fig1.py`.
"""
import json
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

import fig1 as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "fig_order_data")
PANEL_FILES = ["livecodebench_priced.json", "terminal_bench2_priced.json"]        # main text (fig_order)
APPENDIX_FILES = ["terminal_bench4_priced.json", "deepswe_priced.json"]           # appendix (fig_order_app)
OUT = os.path.join(HERE, "fig_order")
PAPER_COPY = os.path.join(HERE, "..", "figures", "fig_order.pdf")

FIGSIZE = (F.PS.width(0.8), 1.95)       # drawn at final size for width=0.8\textwidth in order.tex
SOURCE = "in"                           # "in" = in-sample series (the paper), "cv" = cross-validated series
BASELINE = "best"                       # 0% line: best fixed order of the funded models
LINES = {                               # order -> (label, colour, linestyle, linewidth, alpha, zorder)
    "greedy":   ("Greedy (ours)", F.ACCENT, "-", F.ACCENT_LW, F.PROFIT_ALPHA, 5),
    "cheapest": ("Cheapest first", F.HAZE, (0, (4, 2)), F.ACCENT_LW, 0.9, 4),
}
YLABEL = "Cost relative to\nbest fixed order"
XLABEL = "Solve rate"
ENVELOPE_ONLY = True                    # cv only: draw the lambdas whose pooled greedy point is on the reported (Pareto) curve


def load(path):
    d = json.load(open(path))
    for k in list(d):
        if k.startswith("spend_") or k.startswith("solve_"):
            d[k] = np.array(d[k], dtype=float)
    return d


def envelope(u, R):
    """Indices of the lower-left envelope of (solve rate, spend) points (as cv.pareto)."""
    o = np.argsort(u)
    keep, best = np.zeros(u.size, dtype=bool), np.inf
    for i in o[::-1]:
        if R[i] < best:
            keep[i], best = True, R[i]
    return keep


def draw_panel(ax, d):
    x = d[f"solve_{SOURCE}"] * 100.0
    base = d[f"spend_{BASELINE}_{SOURCE}"]
    sel = np.isfinite(x) & (base > 0)
    if ENVELOPE_ONLY and SOURCE == "cv":
        sel &= envelope(d["solve_cv"], d["spend_greedy_cv"])
    order = np.argsort(x[sel])
    ylo, yhi = 0.0, 0.0
    for name, (label, color, ls, lw, alpha, z) in LINES.items():
        y = (d[f"spend_{name}_{SOURCE}"] / base - 1.0) * 100.0
        ax.plot(x[sel][order], y[sel][order], color=color, ls=ls, lw=lw * F.SC, alpha=alpha, zorder=z,
                solid_capstyle="round", dash_capstyle="round", label=label)
        ylo, yhi = min(ylo, float(np.nanmin(y[sel]))), max(yhi, float(np.nanmax(y[sel])))
    ax.axhline(0.0, color=F.AXIS, lw=0.6, zorder=1)
    ax.set_title(d["title"], fontsize=F.FS["title"] * F.SC, pad=6 * F.SC)
    ax.set_xlabel(XLABEL, fontsize=F.FS["label"] * F.SC)
    F.style(ax, ylabel=YLABEL, log_y=False)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    multi = sel & (np.asarray(d["n_funded"]) > 1)                      # x starts where the cascade has >1 model
    x0, x1 = (float(np.nanmin(x[multi])) if multi.any() else 0.0), float(np.nanmax(x[sel]))
    ax.set_xlim(x0, x1)
    if x1 - x0 <= 50:
        ax.xaxis.set_major_locator(MultipleLocator(10))
    pad = 0.06 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad)
    handles = [Line2D([0], [0], color=c, ls=ls, lw=lw, alpha=a, label=lab) for lab, c, ls, lw, a, _ in LINES.values()]
    leg = ax.legend(handles=handles, fontsize=F.FS["legend"] * F.SC, loc="upper left", handlelength=2.2,
                    handletextpad=0.6, borderaxespad=0.3)
    for t in leg.get_texts():
        t.set_color(F.INK)


def draw(items, out=OUT, figsize=FIGSIZE, ncols=None):
    n = len(items); ncols = ncols or n; nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, d in zip(axs.ravel(), items):
        draw_panel(ax, d)
        lo, hi = ax.get_ylim()                       # axis is in percent
        if hi - lo < 0.5:                            # both orders optimal everywhere (DeepSWE): keep a readable axis
            ax.set_ylim(-0.05, 1.0)
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout(w_pad=F.W_PAD * F.SC)
    fig.savefig(out + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out + ".pdf")
    return fig


def _mirror(out):
    copy = os.path.join(os.path.dirname(PAPER_COPY), os.path.basename(out) + ".pdf")
    if os.path.isdir(os.path.dirname(copy)):
        shutil.copy(out + ".pdf", copy); print("copied to", os.path.normpath(copy))


def main(keys=None):
    """No arguments: the paper's two-panel figure (LiveCodeBench, Terminal-Bench 2.0 -> fig_order) and the
    appendix two-panel figure (Terminal-Bench 4.0, DeepSWE -> fig_order_app), both mirrored into
    figures/. With dataset keys: one panel per key, three per row at full text width, written
    to fig_order_all (mirrored as well); the paper's figures are left untouched."""
    if not keys:
        for files, out in ((PANEL_FILES, OUT), (APPENDIX_FILES, OUT + "_app")):
            draw([load(os.path.join(DATA_DIR, f)) for f in files], out=out)
            _mirror(out)
        return
    items = [load(os.path.join(DATA_DIR, f"{k}.json")) for k in keys]
    n = len(items); ncols = min(3, n); nrows = -(-n // ncols)
    out = OUT + "_all"
    draw(items, out=out, figsize=(F.PS.width(1.0), 1.95 * nrows), ncols=ncols)
    _mirror(out)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
