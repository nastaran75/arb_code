"""Figure 1 of the paper: cost vs solve rate per benchmark (individual models + arbitrageur) and the
arbitrageur's profit margin.

Reproduce (from the repo root, any Python with numpy + matplotlib):

    python plots/fig1.py                # reads plots/fig1_data/*.json -> plots/fig1.pdf (+ .png)
                                        # and copies the PDF to figures/fig1.pdf;
                                        # same for the appendix pair -> fig1_appendix.pdf

The JSON files hold only the curves drawn here (a few hundred KB); they are produced from the raw
per-problem data by `experiments/cv.py --paper` (see that file for the model and cost conventions).
`experiments/make_panel.py` imports the drawing code below, so there is a single implementation.

Data record (one JSON per panel, in PANEL_FILES order):
  title, legend (benchmark name without version), xlabel, ylabel, n_problems, models[M],
  solve_rate[G]   : x grid in percent
  model_cost[M][G]: expected $ over the benchmark for each individual model to reach solve_rate (null = unreachable)
  arb_cost[G]     : same for the arbitrageur's optimal allocation (in-sample)
  market_cost[G]  : cheapest single model at each solve rate (in-sample)
  arb_cost_cv[G], market_cost_cv[G] : cross-validated versions (experiments/cv.py); when present THESE are
                    drawn (arbitrageur line and profit margin) -- the paper's evaluation. The individual model
                    curves are fixed policies indexed by their budget, so they need no cross-validation.
  floor           : $ at the smallest budget considered; the cost panels clip curves below it
"""
import json
import os
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import paperstyle as PS   # applies the paper rcParams on import
from matplotlib.lines import Line2D
from matplotlib.ticker import (FuncFormatter, LogLocator, MaxNLocator, MultipleLocator, NullLocator,
                               PercentFormatter)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "fig1_data")
# Main-text Figure 2: LiveCodeBench + Terminal-Bench 2.0 (as in the submitted version).
# The other two benchmarks (Terminal-Bench 4.0, DeepSWE) go to the appendix companion,
# each figure with its own gain panel over the benchmarks it shows.
PANEL_FILES = ["livecodebench_priced.json", "terminal_bench2_priced.json"]
APPENDIX_FILES = ["terminal_bench4_priced.json", "deepswe_priced.json"]
OUT = os.path.join(HERE, "fig1")                                    # -> fig1.pdf / fig1.png
OUT_APP = os.path.join(HERE, "fig1_appendix")
# Mirrored into <release>/figures/ (the paper reads its figures from there).
PAPER_COPY = os.path.join(HERE, "..", "figures", "fig1.pdf")
PAPER_COPY_APP = os.path.join(HERE, "..", "figures", "fig1_appendix.pdf")

# ---- style ----------------------------------------------------------------------------------------------------
HAZE, HAZE_ALPHA = "#2a78d6", 0.38       # individual models: one blue, translucent
# The cheapest-first arbitrageur is NOT drawn here: on this data it costs the same as greedy
# except above ~80% solve rate, where the gap peaks at 1.10x -- invisible against a four-decade
# log cost axis. `cv.py` still records it as `arb_cheap_cost_cv`; plots/fig_order.py shows it
# properly, as a premium over the best fixed order on a 0-10% axis.
ACCENT, ACCENT_ALPHA = "#eb6834", 0.65   # arbitrageur: orange, translucent so a tracked model line stays visible inside it
INK, GRID, AXIS = PS.INK, PS.GRID, PS.AXIS
HAZE_LW, ACCENT_LW = 1.0, 2.0
PROFIT_ALPHA, PROFIT_LW = 0.80, None     # profit panel: PROFIT_LW=None -> same as ACCENT_LW
PROFIT_CUT_JUMP_PP = 5.0                 # cut each profit line before its last >5pp jump at a cheapest-model switch
CV_INSAMPLE_ALPHA, CV_INSAMPLE_LS = 0.28, (0, (2, 2))   # in-sample reference curves when a record carries CV curves
PER_QUERY = False                        # divide every cost curve by n_problems, so the x axis is
                                         # cost PER QUERY -- the same unit as the bar figures
                                         # (make_acc_bars.py). False = total cost over the benchmark.
# Left edge of the cost axis per benchmark (keyed by the record's `legend`). Below these the
# curves are all bunched near 0% and carry no information; cutting there spends the panel width
# on the region where the arbitrageur separates from the models. None/absent = start at the
# cheapest finite cost in the record.
XMIN = {"LiveCodeBench": 0.01, "Terminal-Bench": 1.0}

# What the third panel plots. "ratio" = how many times fewer problems the arbitrageur leaves
# unsolved at the same cost, (1-p_market)/(1-p_arb); "pp" = the raw difference in solve rate.
# The raw difference understates gains at the top of the range -- 85%->95% is far harder than
# 10%->20% but scores the same 10 points -- whereas the ratio is a difference of log failure
# probabilities, the scale the model itself is built on (failure = exp(-alpha c), see `alpha`).
GAIN_MODE = "pp"

SHOW_INSAMPLE = False                    # records with CV curves: also draw the in-sample arbitrageur/margin as a faint reference
BENCH_LINESTYLES = ["-", "--", "-.", ":"]
SHORT_TITLE = {"LiveCodeBench": "LCB", "Terminal-Bench 2.0": "TB 2.0",
               "Terminal-Bench 4.0": "TB 4.0", "DeepSWE": "DeepSWE"}   # gain-panel legend at 4-benchmark width

# Type, ink and page geometry come from paperstyle (importing it applies the rcParams).
# Re-exported here because fig_order.py / make_panel.py
# read them off this module as F.FS, F.SC, F.AXIS, F.W_PAD, F.FIGSIZE.
FS = PS.FS                               # TRUE points on the page -- see paperstyle
SC = PS.SC                               # font/stroke scale (make_panel.py --figwidth sets it)
W_PAD = PS.W_PAD
FIGSIZE = (PS.width(1.0), 1.95)          # inches; drawn at final size for width=\textwidth


# ---- data -----------------------------------------------------------------------------------------------------
def load(path):
    d = json.load(open(path))
    for k in ("solve_rate", "model_cost", "arb_cost", "market_cost"):
        d[k] = np.array(d[k], dtype=float)                          # null -> nan
    return d


def dump(d, path):
    """Write a data record (numpy arrays -> lists, nan -> null, 6 significant digits)."""
    def conv(v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, float):
            return None if not np.isfinite(v) else float(f"{v:.6g}")
        return v
    with open(path, "w") as f:
        json.dump({k: conv(v) for k, v in d.items()}, f, separators=(",", ":"))


# ---- drawing --------------------------------------------------------------------------------------------------
def _dollars(v, _pos=None):
    """Dollar tick label. Below a tenth of a cent, %g gives '1e-06'; use a mathtext power of
    ten instead, which is both shorter and readable (LiveCodeBench per-query costs are ~1e-6)."""
    if v >= 1000:
        return f"${v / 1000:g}k"        # "$1k" fits where "$1,000" collides with its neighbour
    if v >= 1:
        return f"${v:,.0f}"
    if v >= 1e-3:
        return f"${v:g}"
    return rf"$\${{10}}^{{{int(round(np.log10(v)))}}}$"


def style(ax, ylabel=None, log_y=True, swap=False):
    """swap=True puts cost on x (log, dollar ticks) and solve rate on y (percent) -- the
    accuracy-per-cost orientation of the cost panels. Otherwise solve rate is on x."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, ls="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5 * SC, width=0.5, labelsize=FS["tick"] * SC, colors=AXIS, labelcolor=INK)
    if swap:
        ax.set_xscale("log")
        # numticks=5, not 12: on x these labels are ~0.4in wide and LiveCodeBench spans five
        # decades, so a decade-per-tick grid overprints itself. LogLocator decimates the stride.
        # At the 4-benchmark width (SC < 1) the panels are ~1.1in, so decimate harder still.
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5 if SC >= 1.0 else 3))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FS["label"] * SC)
        return
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(NullLocator())
        if ylabel and "$" in ylabel:
            ax.yaxis.set_major_formatter(FuncFormatter(_dollars))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS["label"] * SC)


def draw_cost(ax, d):
    """Cost-vs-solve-rate panel: haze of individual models drawn OVER the arbitrageur frontier."""
    x, floor = d["solve_rate"], d["floor"]
    # The records store cost summed over the benchmark; per query divides by the problem count
    # (88 for Terminal-Bench, 713 for LiveCodeBench). The floor scales with it so the clip is
    # unchanged. This is the unit the bar figures use, so $0.17/query means the same thing in both.
    scale = float(d["n_problems"]) if PER_QUERY else 1.0
    model_cost = np.where(d["model_cost"] >= floor, d["model_cost"], np.nan) / scale   # clip at the budget floor:
    arb_cost = np.where(d["arb_cost"] >= floor, d["arb_cost"], np.nan) / scale         # every line enters from the bottom edge
    floor = floor / scale
    for row in model_cost:
        ax.plot(row, x, color=HAZE, alpha=HAZE_ALPHA, lw=HAZE_LW, zorder=5, solid_capstyle="round")
    cv = d.get("arb_cost_cv")                                                  # optional cross-validated arbitrageur
    if cv is not None:
        cv = np.asarray(cv, dtype=float) / scale
        cv = np.where(cv >= floor, cv, np.nan)
        if SHOW_INSAMPLE:
            ax.plot(arb_cost, x, color=ACCENT, alpha=CV_INSAMPLE_ALPHA, lw=ACCENT_LW, ls=CV_INSAMPLE_LS, zorder=2)
        ax.plot(cv, x, color=ACCENT, alpha=ACCENT_ALPHA, lw=ACCENT_LW, zorder=3, solid_capstyle="round")
    else:
        ax.plot(arb_cost, x, color=ACCENT, alpha=ACCENT_ALPHA, lw=ACCENT_LW, zorder=3, solid_capstyle="round")
    # Benchmark named by a panel title (the shared legend sits above these); the y label is
    # then just the quantity, so the dataset name is not repeated on every axis.
    ax.set_title(d["title"], fontsize=FS["title"] * SC, pad=4 * SC)
    ax.set_xlabel("Cost per query ($)" if PER_QUERY else "Total cost ($)", fontsize=FS["label"] * SC)
    style(ax, ylabel="Solve rate", swap=True)
    cs = np.concatenate([model_cost.ravel(), arb_cost]); cs = cs[np.isfinite(cs)]
    x0 = XMIN.get(d.get("legend"))
    ax.set_xlim(float(cs.min()) if x0 is None else float(x0) / scale, float(cs.max()))   # cost on x (log)
    ax.set_ylim(0.0, float(x.max()))                # solve rate on y
    handles = [Line2D([0], [0], color=HAZE, alpha=min(1.0, HAZE_ALPHA * 2.2), lw=1.4, label="Models"),
               Line2D([0], [0], color=ACCENT, alpha=ACCENT_ALPHA, lw=ACCENT_LW, label="Arbitrageur (ours)")]
    for ov in d.get("overlays", []):                                           # extra curves (e.g. brute-force search)
        oy = np.asarray(ov["y"], dtype=float) / scale; oy = np.where(oy >= floor, oy, np.nan)
        kw = dict(color=ov.get("color", INK), alpha=ov.get("alpha", 1.0), zorder=ov.get("zorder", 6))
        if ov.get("kind", "line") == "dots":
            mk = dict(marker=ov.get("marker", "o"), mew=ov.get("mew", 0.0))
            if ov.get("hollow"):
                mk.update(mfc="none", mec=kw["color"], mew=ov.get("mew", 0.7 * SC))
            ax.plot(oy, ov["x"], ls="none", ms=ov.get("ms", 2.6) * SC, **mk, **kw)   # (cost, solve rate)
            handles.append(Line2D([0], [0], ls="none", ms=ov.get("ms", 2.6) * SC * 1.3, label=ov["label"], **mk, **kw))
        else:
            ax.plot(oy, ov["x"], ls=ov.get("ls", "--"), lw=ov.get("lw", 1.2) * SC, solid_capstyle="round",
                    dash_capstyle="round", **kw)
            handles.append(Line2D([0], [0], ls=ov.get("ls", "--"), lw=ov.get("lw", 1.2) * SC, label=ov["label"], **kw))
    return handles                                 # drawn once, above the panels, by draw_fig1


def margin(d, cut_jump_pp=None, cv=False):
    """Profit margin (%) vs solve rate: (cheapest single model - arbitrageur) / cheapest single model, where a
    single model reaches that solve rate. cut_jump_pp: truncate before the last upward jump (> cut_jump_pp per
    grid step) that coincides with the cheapest model switching identity (the terminal spike).
    cv=True uses the cross-validated arbitrageur and market curves (`arb_cost_cv`, `market_cost_cv`)."""
    model_cost = d["model_cost"]
    if cv:
        arb_cost, market = np.asarray(d["arb_cost_cv"], dtype=float), np.asarray(d["market_cost_cv"], dtype=float)
    else:
        arb_cost, market = d["arb_cost"], d["market_cost"]
    finite = np.isfinite(market) & np.isfinite(arb_cost)
    with np.errstate(all="ignore"):
        m = np.where(finite, np.clip((market - arb_cost) / market, 0.0, 1.0) * 100.0, np.nan)
    if cut_jump_pp is not None:
        who = np.argmin(np.where(np.isnan(model_cost), np.inf, model_cost), axis=0)
        jump = np.zeros_like(m, dtype=bool)
        jump[1:] = finite[1:] & finite[:-1] & (np.diff(m) > cut_jump_pp) & (np.diff(who) != 0)
        if jump.any():
            m[jump.nonzero()[0].max():] = np.nan
    return d["solve_rate"], m


def gain(d, cv=False, mode=None):
    """Accuracy gain of the arbitrageur over the best single model, as a function of total cost:
    at each cost C, invert both cost-to-reach-solve-rate curves and take the difference in solve
    rate (percentage points). This is the dual of `margin`, which reads the same two curves
    horizontally (cost saved at a fixed solve rate) instead of vertically."""
    g = np.asarray(d["solve_rate"], dtype=float)
    if cv:
        arb, mkt = np.asarray(d["arb_cost_cv"], float), np.asarray(d["market_cost_cv"], float)
    else:
        arb, mkt = np.asarray(d["arb_cost"], float), np.asarray(d["market_cost"], float)
    out = []
    for c in (arb, mkt):
        ok = np.isfinite(c)
        o = np.argsort(c[ok])
        out.append((c[ok][o], g[ok][o]))
    (ac, ag), (mc, mg) = out
    lo, hi = max(ac.min(), mc.min()), min(ac.max(), mc.max())
    if not (hi > lo):
        return np.array([]), np.array([])
    C = np.logspace(np.log10(lo), np.log10(hi), 400)
    pa, pm = np.interp(C, ac, ag) / 100.0, np.interp(C, mc, mg) / 100.0
    if (mode or GAIN_MODE) == "pp":
        return C, 100.0 * (pa - pm)
    fa, fm = np.clip(1 - pa, 1e-4, 1.0), np.clip(1 - pm, 1e-4, 1.0)   # clip: p=1 -> infinite ratio
    return C, fm / fa


def draw_profit(ax, items):
    """Profit-margin line per benchmark, all in the arbitrageur colour, told apart by line style."""
    lw = ACCENT_LW if PROFIT_LW is None else PROFIT_LW
    xhi, yhi = -np.inf, 0.0
    for d, ls in zip(items, BENCH_LINESTYLES):
        has_cv = d.get("arb_cost_cv") is not None
        if has_cv and SHOW_INSAMPLE:                                           # in-sample margin as faint reference
            x, m_in = margin(d, cut_jump_pp=PROFIT_CUT_JUMP_PP)
            ax.plot(x, m_in, color=ACCENT, ls=ls, alpha=CV_INSAMPLE_ALPHA, lw=lw, zorder=4)
        x, m = gain(d, cv=has_cv)                                          # accuracy gain vs cost
        if x.size == 0:
            continue
        x = x / (float(d["n_problems"]) if PER_QUERY else 1.0)
        ok = np.isfinite(m)
        # label by `title`, not `legend`: with both Terminal-Bench versions drawn, the
        # version-stripped legend field ("Terminal-Bench") is ambiguous between them. At the
        # 4-benchmark width the panel is too narrow for full names, so abbreviate.
        lab = SHORT_TITLE.get(d["title"], d["title"]) if len(items) > 2 else d["title"]
        ax.plot(x, m, color=ACCENT, ls=ls, alpha=PROFIT_ALPHA, lw=lw, zorder=5, solid_capstyle="round",
                dash_capstyle="round", label=lab)
        xhi, yhi = max(xhi, x[ok].max()), max(yhi, m[ok].max())
    ax.set_title("Gain over the best single model", fontsize=FS["title"] * SC, pad=6 * SC)
    # Cost on x (log, dollars) like the other panels; y is a difference in solve rate, in
    # percentage points, so it gets a plain numeric formatter rather than style()'s percent.
    ax.set_xlabel("Cost per query ($)" if PER_QUERY else "Total cost ($)", fontsize=FS["label"] * SC)
    if GAIN_MODE == "pp":
        style(ax, ylabel="Accuracy gain (pp)", swap=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))     # swap=True's percent locator is
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))   # wrong at this scale
        ax.set_ylim(0.0, float(yhi) * 1.08)
    else:
        style(ax, ylabel="Failure rate reduction", swap=True)
        ax.set_yscale("log")                                 # a ratio is multiplicative
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0), numticks=12))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}$\\times$"))
        ax.set_ylim(0.97, float(yhi) * 1.12)                 # 1x = no gain; below it, arb is worse
    # Left edge: the cheapest XMIN of the benchmarks drawn here, so this panel starts where
    # the cost panels do rather than in the flat region far below either of them.
    x0 = [XMIN[d["legend"]] for d in items if d.get("legend") in XMIN]
    if x0:
        ax.set_xlim(min(x0) / (float(items[0]["n_problems"]) if PER_QUERY else 1.0), float(xhi))
    return ax.get_legend_handles_labels()[0]       # drawn above the panel by draw_fig1


def draw_fig1(items, out=OUT, figsize=FIGSIZE, with_profit=True):
    """items: list of data records (one cost panel each) [+ one profit panel]. Writes out.pdf and out.png."""
    n_ax = len(items) + (1 if with_profit else 0)
    fig, axs = plt.subplots(1, n_ax, figsize=figsize)
    axs = np.atleast_1d(axs)
    handles = []
    for ax, d in zip(axs, items):
        handles = draw_cost(ax, d) or handles      # identical across cost panels; keep the last
    for ax in axs[1:len(items)]:                   # cost panels share one y-title (leftmost only);
        ax.set_ylabel("")                          # cleared before tight_layout so the space is reclaimed
    profit_handles = draw_profit(axs[-1], items) if with_profit else None
    fig.tight_layout(w_pad=W_PAD * SC)
    # Shared legend above the cost panels, spanning them. Anchored to the axes (not the figure)
    # so tight_layout's margin is not left as a gap; the profit panel keeps its own legend.
    cost_axs = axs[:len(items)]
    x0 = cost_axs[0].get_position().x0, cost_axs[-1].get_position().x1
    fig.legend(handles=handles, loc="lower center",
               # clears the panel titles, which now sit between the axes and the legend
               bbox_to_anchor=(sum(x0) / 2, cost_axs[0].get_position().y1 + 0.09),
               ncol=len(handles), fontsize=FS["legend"] * SC, frameon=False,
               handlelength=2.2, columnspacing=1.6, labelcolor=INK)
    if profit_handles:
        # Same treatment for the gain panel: one row, matching the shared legend on the left.
        # Two entries side by side are wider than this panel, so it is anchored to the panel's
        # RIGHT edge and grows leftwards -- centred it would run past \textwidth and LaTeX
        # would scale the whole figure down. Spacing is tightened for the same reason.
        pos = axs[-1].get_position()
        leg = fig.legend(handles=profit_handles, loc="lower center",
                         bbox_to_anchor=((pos.x0 + pos.x1) / 2, pos.y1 + 0.09),
                         ncol=2, fontsize=FS["legend"] * SC, frameon=False,
                         handlelength=1.6, columnspacing=1.0, handletextpad=0.5, labelcolor=INK)
        # Two entries side by side are wider than this panel, so centring them on it leaves the
        # legend hanging past the right edge of the figure. Measure the drawn legend and slide it
        # left until its right edge lines up with the panel's.
        fig.canvas.draw()
        lb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        if lb.x1 > pos.x1:
            leg.set_bbox_to_anchor(((pos.x0 + pos.x1) / 2 - (lb.x1 - pos.x1), pos.y1 + 0.09))
    fig.savefig(out + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})   # no timestamp -> byte-reproducible
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out + ".pdf")
    return fig


def main():
    # This script owns the paper's frontier figures: main Figure 2 (fig1.pdf, LiveCodeBench +
    # Terminal-Bench 2.0) and its appendix twin (fig1_appendix.pdf, Terminal-Bench 4.0 + DeepSWE).
    # Their revenue-heatmap companions (fig_rev_main.pdf, fig_rev_appendix.pdf) are rendered and
    # mirrored by experiments/fig_appendix.py.
    for files, out, copy in ((PANEL_FILES, OUT, PAPER_COPY), (APPENDIX_FILES, OUT_APP, PAPER_COPY_APP)):
        items = [load(os.path.join(DATA_DIR, f)) for f in files]
        draw_fig1(items, out=out)
        if copy and os.path.isdir(os.path.dirname(copy)):
            shutil.copy(out + ".pdf", copy)
            print("copied to", os.path.normpath(copy))


if __name__ == "__main__":
    main()
