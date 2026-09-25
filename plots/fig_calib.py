"""Section-4 figure: how much calibration do you have to buy?

x = attempts sampled per (problem, model) cell to estimate the pass@k curves -- the UPFRONT cost,
before a single query is served. y = solve rate the resulting allocation then reaches at a fixed
deployment budget, scored on the true (10,000-sample) curves. The oracle is the n = 10,000 truth,
drawn as the ceiling; it is not a deployable method, it is what the estimators are trying to reach.

    python plots/fig_calib.py     # reads plots/fig_passk_data/<key>_n<n>.json -> plots/fig_calib.pdf

The paper draws n = {5, 50}. `--ns 1,2,3,4,5` draws any other set of records instead, writing to
its own file stem and NOT mirroring into the manuscript -- that is the sweep this figure was always
meant to be, and `draw_vs_n()` is its natural reading: solve rate reached against the calibration
size n, one line per estimator, at each of the record's fixed oracle targets.

Both estimators extrapolate past the n observed attempts: the plug-in reports 1 - (1 - c/n)^k at
any k (validate_estimator.GEOM_EXTRAPOLATE), the beta-binomial by construction. Freezing the
plug-in at k = n instead would cap its frontier at the solve rate n attempts per model can reach,
which is a property of the cap and not of the estimator.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import (FixedLocator, LogLocator, MaxNLocator, MultipleLocator,
                               NullLocator, PercentFormatter, FuncFormatter)

import fig1 as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "fig_passk_data")
OUT = os.path.join(HERE, "fig_calib")
# The paper uses the FIGURE, not the table: table() still writes plots/calib_table.tex if you want
# the same numbers in tabular form, but nothing includes it (pass mirror=True to copy it in).
PAPER_TABLE = os.path.join(HERE, "..", "figures", "calib_table.tex")
# Mirrored into <release>/figures/ (the paper reads its figures from there).
PAPER_FIGS = [os.path.join(HERE, "..", "figures", "fig_calib_gap.pdf")]
PANELS = ["monkey_math", "monkey_codecontests", "resmat2_aime2024", "resmat2_aime2025"]
PANELS_APP = ("resmat2_gmmlu", "resmat2_mmlupro")      # appendix companion (fig_calib_gap_app)
NS = [2, 4, 8, 16]                       # the n values drawn (paper); --ns overrides
NS_PAPER = (2, 4, 8, 16)                 # only this set is mirrored into the manuscript
N_ORACLE = 10000                         # attempts per cell behind the true curves
# Left edge of the budget axis: the records' `x_extrap` for the smallest n -- the solve rate from
# which the oracle allocation spends more than n attempts of some model, so an n-sample estimate
# must extrapolate. Below it nothing is extrapolated and every method coincides exactly, which is
# a third of the panel spent on curves that are identical by construction.
CUT_AT_EXTRAP = True
BUDGET_FRAC = 0.95                       # deployment budget: where the oracle reaches this
                                         # fraction of its reachable solve rate. Lower fractions
                                         # put every method in the region needing no extrapolation,
                                         # where all of them tie and the panel says nothing.
# Estimators drawn in the calibration panels and listed in their legends.
LINES = {
    "geom": ("Arbitrageur using plug-in", F.HAZE, "o"),
    "zibb": ("Arbitrageur using EB pass@$k$ (ours)", F.ACCENT, "s"),
}
LEGEND_MODES = {"geom", "zibb"}
ORACLE_LS = (0, (4, 2))                  # oracle drawn dashed, so it reads as a reference
# Legend is split: colour identifies the estimator, line style identifies the calibration size.
# Folding n into every label instead would make five entries too wide for the figure.
NLS = ("-", (0, (3, 1.6)))


def n_legend_alpha(al, n_of_them):
    """Opacity for a legend swatch. Only the >2-n ramp is carried into the legend: with two n values
    the line style already tells them apart, and fading those swatches would change the paper figure
    for no gain."""
    return al if n_of_them > len(NLS) else None


def n_style(i, n_of_them):
    """(linestyle, alpha) for the i-th smallest n. Two n values keep the paper's solid/dashed pair;
    more than two switch to one solid line per n on a lightness ramp -- five dash patterns are not
    tellable apart at this size, five opacities are, and the ramp reads in the right direction
    (faint = little calibration data)."""
    if n_of_them <= len(NLS):
        return NLS[i], F.PROFIT_ALPHA
    return "-", 0.30 + 0.70 * (i / (n_of_them - 1))


class StackedNHandle:
    """Legend proxy for an orange estimator line stacked above its blue counterpart."""

    def __init__(self, linestyle, alpha):
        self.linestyle = linestyle
        self.alpha = alpha


class HandlerStackedN(HandlerBase):
    def create_artists(self, legend, handle, xdescent, ydescent, width, height, fontsize, trans):
        x = [xdescent, xdescent + width]
        lines = [
            Line2D(x, [ydescent + 0.85 * height] * 2, color=F.ACCENT,
                   ls=handle.linestyle, lw=F.ACCENT_LW, alpha=handle.alpha),
            Line2D(x, [ydescent + 0.15 * height] * 2, color=F.HAZE,
                   ls=handle.linestyle, lw=F.ACCENT_LW, alpha=handle.alpha),
        ]
        for line in lines:
            line.set_transform(trans)
        return lines


FIGSIZE = (F.PS.width(0.78), 1.95)


def achieved(cost, grid, C):
    """Solve rate a cost-to-reach-solve-rate curve delivers at deployment budget C."""
    ok = np.isfinite(cost) & (cost > 0)
    c, g = cost[ok], grid[ok]
    o = np.argsort(c)
    return float(np.interp(C, c[o], g[o]))


def budget_ref(d0, g, oc):
    """Deployment-budget anchor: the oracle's cost at BUDGET_FRAC of the terminal cut, or of the
    oracle's reach when the record has no terminal spike (x_cut null, e.g. the IRSL benchmarks)."""
    xc = d0["x_cut"] if d0.get("x_cut") is not None else float(g[np.isfinite(oc)].max())
    return float(oc[int(np.argmin(np.abs(g - BUDGET_FRAC * xc)))])


def rows():
    """(dataset -> {(mode, n): solve rate}, oracle solve rate, deployment budget) for the table."""
    out = {}
    for key in PANELS:
        recs = {n: json.load(open(os.path.join(DATA_DIR, f"{key}_n{n}.json"))) for n in NS}
        d0 = recs[NS[0]]
        g = np.array(d0["solve_rate"], float)
        oc = np.array(d0["oracle_cost"], float)
        C = budget_ref(d0, g, oc)
        vals = {(m, n): achieved(np.array(recs[n]["oracle_cost"], float)
                                 * np.array(recs[n]["est"][m]["realized_ratio_med"], float),
                                 np.array(recs[n]["solve_rate"], float), C)
                for m in LINES for n in NS}
        out[key] = dict(legend=d0["legend"], vals=vals, oracle=achieved(oc, g, C), budget=C)
    return out


def table(path=os.path.join(HERE, "calib_table.tex"), mirror=False):
    """LaTeX table: solve rate each policy REACHES at the budget the oracle needs for a target.
    Rows are the round targets in each record's `targets` block, so the operating point is stated
    rather than chosen by a constant; the target column doubles as the oracle's own solve rate.
    Where a policy cannot spend that much (its own frontier tops out earlier) the entry is its
    ceiling: extra budget buys it nothing, so that is what it delivers there."""
    nn = len(NS)
    span = " & ".join(rf"\multicolumn{{{nn}}}{{c}}{{{lab}}}" for lab, _, _ in LINES.values())
    rule = "".join(rf"\cmidrule(lr){{{3 + i * nn}-{2 + (i + 1) * nn}}}" for i in range(len(LINES)))
    cols = " & ".join(rf"$n={n}$" for _ in LINES for n in NS)
    L = [r"% generated by plots/fig_calib.py -- do not edit by hand",
         rf"\begin{{tabular}}{{ll{'c' * (nn * len(LINES))}}}", r"\toprule",
         rf" & & {span} \\", rule,
         rf"Benchmark & Oracle & {cols} \\"]
    for key in PANELS:
        recs = {n: json.load(open(os.path.join(DATA_DIR, f"{key}_n{n}.json"))) for n in NS}
        d0 = recs[NS[0]]
        tg = d0["targets"]
        L.append(r"\midrule")
        for i_, (sr, C) in enumerate(zip(tg["solve_rate"], tg["oracle_cost"])):
            cells = []
            for mode in LINES:
                for n in NS:
                    d = recs[n]
                    cost = (np.array(d["oracle_cost"], float)
                            * np.array(d["est"][mode]["realized_ratio_med"], float))
                    cells.append(f"${achieved(cost, np.array(d['solve_rate'], float), C):.1f}\\%$")
            name = d0["legend"] if i_ == 0 else ""
            L.append(f"{name} & ${sr:.0f}\\%$ & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)
    if mirror and os.path.isdir(os.path.dirname(PAPER_TABLE)):
        import shutil
        shutil.copy(path, PAPER_TABLE)
        print("mirrored to", os.path.normpath(PAPER_TABLE))
    return "\n".join(L)


def rmse_vs_n(key, ns, k=1.0, reps=10, B=2000):
    """Per-problem pass@k RMSE vs n for the plug-in, ours, and Kazdan's population estimate.
    Same draws as the calibration records (seeded identically); bars are bootstrap SE over
    problems. Computed here from the counts -- the fig_passk_data records carry frontiers,
    not estimation error."""
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
    import warnings
    warnings.filterwarnings("ignore")
    from scipy.special import betaln
    import validate_estimator as v
    import estimator as est
    v.configure(key)
    cfg, models, q, c, n = v.load(key)[:5]
    M = len(models)
    K = np.array([float(k)])
    y_true = np.column_stack([v.chen_curve(c[:, j], n[:, j], K)[:, 0] for j in range(M)])
    P = y_true.shape[0]
    IDX = np.random.default_rng(0).integers(0, P, size=(B, P))
    # Per-model RMSE, matching what the method actually does: each model gets its own (a, b) fit,
    # so its error is a quantity of its own. The headline line is the MEAN over the per-model
    # RMSEs -- not the RMSE of the pooled squared errors, which averages MSEs and therefore
    # weights a model quadratically in its error (a model 3x worse counts 9x).
    out = {mode: {"per": [], "mean": [], "se": []} for mode in ("geom", "zibb", "kaz")}
    for n_sub in ns:
        errs = {m_: [[] for _ in range(M)] for m_ in out}            # per model: list of (P,) mse
        for rep in range(reps):
            rng = np.random.default_rng(v.SEED + 1000 * n_sub + rep)
            c_s, n_s, _, _ = v.subsample(c, n, n_sub, rng, fixed=cfg.get("fixed"))
            for j in range(M):
                a_, b_ = est.fit_bb(c_s[:, j], n_s[:, j], prior_pow=0.0)
                e = {"geom": est.geom_curve(c_s[:, j], n_s[:, j], K, extrapolate=True)[:, 0],
                     "zibb": est.zibb_curve(c_s[:, j], n_s[:, j], K, a_, b_, 1e-12)[:, 0],
                     "kaz": np.full(P, 1.0 - np.exp(betaln(a_, b_ + K[0]) - betaln(a_, b_)))}
                for m_, u in e.items():
                    errs[m_][j].append((u - y_true[:, j]) ** 2)
        for m_ in out:
            mse_j = np.array([np.mean(errs[m_][j], axis=0) for j in range(M)])   # (M, P)
            per = np.sqrt(mse_j.mean(1))                                          # (M,)
            out[m_]["per"].append(per)
            out[m_]["mean"].append(float(per.mean()))
            boot = np.mean([np.sqrt(mse_j[j][IDX].mean(1)) for j in range(M)], axis=0)
            out[m_]["se"].append(float(boot.std(ddof=1)))
    for m_ in out:
        out[m_]["per"] = np.array(out[m_]["per"]).T                               # (M, len(ns))
    return out


# the paper's set and ORDER (matches the gap figure, fig_gap_dist); only this set is mirrored
PANELS_PAPER = ("monkey_math", "monkey_codecontests", "resmat2_aime2024", "resmat2_aime2025")


def draw_shortfall(out=os.path.join(HERE, "fig_calib_gap"), panels=None, legend=True,
                   figw=1.0, mirror="auto"):
    """Achieved solve rate against the oracle's, at the budget the oracle needs for each target.
    The oracle is then the diagonal y = x -- a visible reference line rather than a flat one --
    and a policy that falls short of it drops below the diagonal. One DECISION-level panel per
    benchmark, in the gap figure's order; the ESTIMATION level lives in fig_gap_dist.
    `mirror`: "auto" copies into the paper only for the default NS/panels; a path copies there;
    None never copies. `legend=False` for the appendix companion (conventions in the caption)."""
    panels = list(panels or PANELS)
    fig, axs = plt.subplots(1, len(panels), figsize=(F.PS.width(figw), 1.75))
    axs = np.atleast_1d(axs)
    for ax, key in zip(axs, panels):
        recs = {n: json.load(open(os.path.join(DATA_DIR, f"{key}_n{n}.json"))) for n in NS}
        d0 = recs[NS[0]]
        tg = d0["targets"]
        T = np.array(tg["solve_rate"], float)
        pad = 0.06 * (T.max() - T.min())
        lo, hi = T.min() - pad, T.max() + pad
        ax.plot([lo, hi], [lo, hi], color=F.INK, lw=0.9, ls=ORACLE_LS, zorder=3)   # the oracle
        for mode, (label, colr, mk) in LINES.items():
            for i, n in enumerate(NS):
                ls, al = n_style(i, len(NS))
                d = recs[n]
                sr = np.array(d["solve_rate"], float)
                oc = np.array(d["oracle_cost"], float)
                # med/lo/hi are the draw quartiles of realized cost; a HIGHER cost ratio reaches a
                # LOWER solve rate at the same budget, so the hi-ratio curve is the band's bottom.
                ys = {}
                for tag in ("med", "lo", "hi"):
                    r_ = np.array([np.nan if x is None else x
                                   for x in d["est"][mode][f"realized_ratio_{tag}"]], float)
                    ys[tag] = np.array([achieved(oc * r_, sr, C) for C in tg["oracle_cost"]])
                ax.errorbar(T, ys["med"], yerr=[ys["med"] - ys["hi"], ys["lo"] - ys["med"]],
                            fmt="none", ecolor=colr, elinewidth=0.7, capsize=1.2, capthick=0.7,
                            alpha=al * 0.9, zorder=4)
                ax.plot(T, ys["med"], color=colr, ls=ls, marker=mk, ms=3.0, lw=F.ACCENT_LW,
                        alpha=al, zorder=5, solid_capstyle="round")
        ax.set_title(d0["legend"], fontsize=F.FS["annot"] * F.SC, pad=3 * F.SC)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)          # equal ranges: the oracle is the diagonal
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
        ax.yaxis.set_major_locator(MultipleLocator(10))     # solve rate in 10pp steps
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=F.GRID, lw=0.5, ls="-", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5 * F.SC, width=0.5, labelsize=F.FS["tick"] * F.SC,
                       colors=F.AXIS, labelcolor=F.INK)
    axs[0].set_ylabel("Solve rate", fontsize=F.FS["label"] * F.SC)
    # Method and n are separate encodings (colour+marker vs opacity+linestyle), so list each once
    # in two compact rows instead of showing the full method-by-n cross product.
    method_handles = [Line2D([], [], color=F.INK, lw=0.9, ls=ORACLE_LS,
                             label="Arbitrageur using oracle")]
    for mode, (label, colr, mk) in LINES.items():
        if mode in LEGEND_MODES:
            method_handles.append(Line2D([], [], color=colr, marker=mk, ms=3.0,
                                         lw=F.ACCENT_LW, label=label))
    n_handles, n_labels = [], []
    for i, n in enumerate(NS):
        ls, al = n_style(i, len(NS))
        n_handles.append(StackedNHandle(ls, n_legend_alpha(al, len(NS))))
        n_labels.append(f"$n={n}$")
    fig.tight_layout(w_pad=F.W_PAD * F.SC)
    p0, p1 = axs[0].get_position(), axs[-1].get_position()
    # repeated x-labels collide at this panel width: one centred title instead
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.16, "Oracle solve rate",
             ha="center", va="top", fontsize=F.FS["label"] * F.SC, color=F.INK)
    if legend:
        center = (p0.x0 + p1.x1) / 2
        fig.legend(handles=method_handles, loc="lower center",
                   bbox_to_anchor=(center, p0.y1 + 0.15), ncol=len(method_handles),
                   fontsize=F.FS["legend"] * F.SC, frameon=False, handlelength=1.6,
                   columnspacing=0.8, handletextpad=0.5, labelcolor=F.INK)
        fig.legend(handles=n_handles, labels=n_labels, loc="lower center",
                   bbox_to_anchor=(center, p0.y1 + 0.05), ncol=len(n_handles),
                   fontsize=F.FS["legend"] * F.SC, frameon=False, handlelength=1.6,
                   columnspacing=1.8, handletextpad=0.5, labelcolor=F.INK,
                   markerfirst=True, handler_map={StackedNHandle: HandlerStackedN()})
    fig.savefig(out + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out + ".pdf")
    import shutil
    if mirror not in (None, "auto") and os.path.isdir(os.path.dirname(mirror)):
        shutil.copy(out + ".pdf", mirror)
        print("mirrored to", os.path.normpath(mirror))
        return
    if mirror != "auto" or tuple(NS) != NS_PAPER or tuple(panels) != PANELS_PAPER:
        return                       # exploratory sweeps and non-paper benchmark sets: no mirror
    for dst in PAPER_FIGS:                            # this is the paper's Section-4 figure
        if os.path.isdir(os.path.dirname(dst)):
            shutil.copy(out + ".pdf", dst)
            print("mirrored to", os.path.normpath(dst))


def draw_vs_n(out=None, targets=(-3, -1)):
    """THE sweep reading: x = calibration size n (the upfront cost, before a single query is served),
    y = solve rate the resulting allocation reaches at the deployment budget the oracle needs for a
    fixed target. One panel per (dataset, target); the oracle target is the dashed ceiling. A curve
    approaching the ceiling from below says the estimator has bought enough calibration.

    `targets` indexes into each record's `targets` block. The default is the two HARDEST targets:
    at an easy target the oracle allocation never spends more than n attempts of any model, so
    nothing is extrapolated, every estimator is exact and the panel is a flat tie (the same reason
    BUDGET_FRAC is high and CUT_AT_EXTRAP trims the other views)."""
    out = out or os.path.join(HERE, "fig_calib_vs_n")
    cols = [(key, ti) for key in PANELS for ti in targets]
    fig, axs = plt.subplots(1, len(cols), figsize=(F.PS.width(1.0), 2.05))
    x = np.array(NS, float)
    for ax, (key, ti) in zip(np.atleast_1d(axs), cols):
        recs = {n: json.load(open(os.path.join(DATA_DIR, f"{key}_n{n}.json"))) for n in NS}
        d0 = recs[NS[0]]
        tg = d0["targets"]
        T = float(tg["solve_rate"][ti])
        C = float(tg["oracle_cost"][ti])
        ax.axhline(T, color=F.INK, lw=0.9, ls=ORACLE_LS, zorder=3)           # the oracle
        ys = []
        for mode, (label, colr, mk) in LINES.items():
            y = []
            for n in NS:
                d = recs[n]
                cost = (np.array(d["oracle_cost"], float)
                        * np.array(d["est"][mode]["realized_ratio_med"], float))
                y.append(achieved(cost, np.array(d["solve_rate"], float), C))
            ys += y
            ax.plot(x, y, color=colr, marker=mk, ms=3.0, lw=F.ACCENT_LW,
                    alpha=F.PROFIT_ALPHA, zorder=5, solid_capstyle="round")
        ax.set_title(f"{d0['legend']}, {T:.0f}%", fontsize=F.FS["title"] * F.SC, pad=4 * F.SC)
        ax.set_xlabel("Calibration attempts $n$", fontsize=F.FS["label"] * F.SC)
        ax.set_xlim(x.min() - 0.25, x.max() + 0.25)
        ax.xaxis.set_major_locator(FixedLocator(NS))
        lo = min(ys + [T]); hi = max(ys + [T]); pad = 0.10 * (hi - lo) + 1e-9
        ax.set_ylim(lo - pad, hi + pad)
        # A panel can span a fraction of a point (an easy target where everyone ties); 0 decimals
        # would then print the same label on every tick.
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0 if hi - lo > 4 else 1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=F.GRID, lw=0.5, ls="-", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5 * F.SC, width=0.5, labelsize=F.FS["tick"] * F.SC,
                       colors=F.AXIS, labelcolor=F.INK)
    a = np.atleast_1d(axs)
    a[0].set_ylabel("Solve rate reached", fontsize=F.FS["label"] * F.SC)
    handles = [Line2D([], [], color=F.INK, lw=0.9, ls=ORACLE_LS, label="Oracle ($n=10{,}000$)")]
    handles += [Line2D([], [], color=colr, marker=mk, ms=3.0, lw=F.ACCENT_LW, label=label)
                for label, colr, mk in LINES.values()]
    fig.tight_layout(w_pad=F.W_PAD * F.SC)
    p0, p1 = a[0].get_position(), a[-1].get_position()
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=((p0.x0 + p1.x1) / 2, p0.y1 + 0.09), ncol=3,
               fontsize=F.FS["legend"] * F.SC, frameon=False, handlelength=1.8,
               columnspacing=1.0, labelcolor=F.INK)
    fig.savefig(out + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out + ".pdf")


def main():
    """Solve rate against deployment budget C, with calibration size n as the line style.
    fig_passk draws one panel per (dataset, n); here n is folded into a single panel per dataset
    so the effect of buying more calibration is visible at every C rather than across columns."""
    fig, axs = plt.subplots(1, len(PANELS), figsize=FIGSIZE)
    for ax, key in zip(np.atleast_1d(axs), PANELS):
        recs = {n: json.load(open(os.path.join(DATA_DIR, f"{key}_n{n}.json"))) for n in NS}
        d0 = recs[NS[0]]
        g = np.array(d0["solve_rate"], float)
        oc = np.array(d0["oracle_cost"], float)
        C = budget_ref(d0, g, oc)
        ax.axvline(C, color=F.AXIS, lw=0.7, ls=(0, (1, 2)), zorder=1)   # the table's operating point
        ax.annotate("Table 1", xy=(C, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(-2, 0), textcoords="offset points", ha="right", va="bottom",
                    fontsize=F.FS["dense"] * F.SC, color=F.AXIS, rotation=90)
        cs = oc[np.isfinite(oc) & (oc > 0)]
        ax.plot(oc, g, color=F.INK, lw=F.ACCENT_LW, zorder=8, solid_capstyle="round")
        for mode, (label, colr, _mk) in LINES.items():
            for i, n in enumerate(NS):
                ls, al = n_style(i, len(NS))
                d = recs[n]
                cost = np.array(d["oracle_cost"], float) * np.array(d["est"][mode]["realized_ratio_med"], float)
                sr = np.array(d["solve_rate"], float)
                ax.plot(cost, sr, color=colr, ls=ls, lw=F.ACCENT_LW, alpha=al,
                        zorder=5, solid_capstyle="round", dash_capstyle="round")
                # Where a policy's own frontier tops out (the plug-in believes the problems it never
                # saw solved are unsolvable, so its claimed curve stops), extra budget buys it nothing
                # and the solve rate is flat from there on. Drawing that
                # continuation is what makes the gap at a given budget visible -- without it the
                # curve simply stops and the reader cannot see how far behind it has fallen.
                ok = np.isfinite(cost) & np.isfinite(sr)
                if ok.any():
                    j = int(np.nanargmax(np.where(ok, cost, -np.inf)))
                    if cost[j] < float(cs.max()):
                        ax.plot([cost[j], float(cs.max())], [sr[j], sr[j]], color=colr,
                                ls=(0, (1, 2)), lw=F.ACCENT_LW * 0.8, alpha=0.55 * al / F.PROFIT_ALPHA,
                                zorder=4)
                        ax.plot([cost[j]], [sr[j]], marker="o", ms=2.6, color=colr,
                                alpha=al, ls="none", zorder=6)
        ax.set_title(d0["legend"], fontsize=F.FS["title"] * F.SC, pad=4 * F.SC)
        ax.set_xscale("log")
        lo = float(cs.min())
        if CUT_AT_EXTRAP and d0.get("x_extrap") is not None:
            lo = float(oc[int(np.argmin(np.abs(g - d0["x_extrap"])))])
        ax.set_xlim(lo, float(cs.max()))
        # NOT x_cut: that cut exists to trim the terminal spike out of the MARGIN panel, and
        # using it here crops the highest budgets -- exactly where the estimators separate.
        ybot = 0.0 if not (CUT_AT_EXTRAP and d0.get("x_extrap") is not None) else 0.96 * d0["x_extrap"]
        ax.set_ylim(ybot, float(np.nanmax(g[np.isfinite(oc)])))
        ax.xaxis.set_major_locator(LogLocator(base=100, numticks=6))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=F.GRID, lw=0.5, ls="-", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5 * F.SC, width=0.5, labelsize=F.FS["tick"] * F.SC,
                       colors=F.AXIS, labelcolor=F.INK)
        ax.set_xlabel("Deployment budget", fontsize=F.FS["label"] * F.SC)
    a = np.atleast_1d(axs)
    a[0].set_ylabel("Solve rate reached", fontsize=F.FS["label"] * F.SC)
    handles = [Line2D([], [], color=F.INK, lw=F.ACCENT_LW, label="Oracle ($n=10{,}000$)")]
    handles += [Line2D([], [], ls="none", label="")] * (len(NS) - 1)      # pad to one column
    for label, colr, _mk in LINES.values():
        for i, n in enumerate(NS):
            ls, al = n_style(i, len(NS))
            handles.append(Line2D([], [], color=colr, ls=ls, lw=F.ACCENT_LW,
                                  alpha=n_legend_alpha(al, len(NS)), label=f"{label}, $n={n}$"))
    fig.tight_layout(w_pad=F.W_PAD * F.SC)
    p0, p1 = a[0].get_position(), a[-1].get_position()
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=((p0.x0 + p1.x1) / 2, p0.y1 + 0.11), ncol=1 + len(LINES),
               fontsize=F.FS["legend"] * F.SC, frameon=False, handlelength=1.8,
               columnspacing=1.0, labelcolor=F.INK)
    fig.savefig(OUT + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", OUT + ".pdf")
    if tuple(NS) == NS_PAPER and tuple(PANELS) == PANELS_PAPER:
        table()                                       # the table's layout assumes exactly two n
        draw_shortfall()                              # fig_calib_gap: the figure the paper includes
        draw_shortfall(out=os.path.join(HERE, "fig_calib_gap_app"), panels=PANELS_APP,
                       legend=True, figw=0.95,       # appendix companion uses the same legend
                       mirror=os.path.join(HERE, "..", "figures",
                                           "fig_calib_gap_app.pdf"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", help="comma-separated calibration sizes to draw (default 5,50 = the paper). "
                                 "Any other set writes to its own file stem and is not mirrored.")
    ap.add_argument("--panels", help="comma-separated dataset keys instead of the Monkey Business pair "
                                     "(e.g. resmat2_aime,resmat2_mmlu). Writes to its own file stem; "
                                     "never mirrored.")
    A = ap.parse_args()
    ptag = ""
    if A.panels:
        PANELS = A.panels.split(",")
        ptag = "_" + "-".join(PANELS)
    if A.ns or A.panels:
        if A.ns:
            NS = [int(x) for x in A.ns.split(",")]
        missing = [f"{k}_n{n}" for k in PANELS for n in NS
                   if not os.path.exists(os.path.join(DATA_DIR, f"{k}_n{n}.json"))]
        if missing:
            raise SystemExit("no records for: " + ", ".join(missing) + "\n"
                             "build them with:  cd ../experiments && "
                             f"python validate_calibration.py {' '.join(PANELS)}"
                             f" && ../.venv/bin/python make_fig_passk_data.py {' '.join(PANELS)}")
        tag = ptag + ("_n" + "-".join(map(str, NS)) if A.ns else "")
        OUT = os.path.join(HERE, "fig_calib" + tag)
        main()
        draw_shortfall(os.path.join(HERE, "fig_calib_gap" + tag))
        draw_vs_n(os.path.join(HERE, "fig_calib_vs_n" + tag))
    else:
        main()
