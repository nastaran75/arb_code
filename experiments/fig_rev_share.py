"""Revenue share along the frontier: which systems the arbitrageur's spend reaches, against the
solve rate it achieves. Cross-validated (cv_replay: allocations fit per fold, money attributed
under the greedy cascade on held-out problems), so a share is what a system would actually invoice.

    ../.venv/bin/python fig_rev_share.py deepswe_priced
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plots"))
import fig1 as F                            # noqa: E402  paper style
import cv_replay as CR                      # noqa: E402
from make_cost_bars import short, short_deepswe, family     # noqa: E402
from make_figure import DATASETS            # noqa: E402


def base_model(name):
    """Systems sharing a base model pool into one series: Agent__Model keeps the Model,
    a DeepSWE config drops its reasoning-effort suffix, an LCB system is already a model.
    Case-insensitive: Terminal-Bench agents spell the same model both 'gpt-5.3-codex' and
    'GPT-5.3-Codex', which otherwise splits one base model into two series."""
    return (name.split("__", 1)[1] if "__" in name else family(name)).lower()


def base_label(name):
    if "__" in name:
        lab = short(name)                           # "Agent · Model" -> keep the model half
        return lab.split(" · ", 1)[1] if " · " in lab else lab
    return short_deepswe(family(name)) if name == name.lower() else short(name)


def fmt_cost(c):
    if c >= 0.095:
        return f"${c:,.2f}"
    return "$" + np.format_float_positional(c, precision=2, fractional=False)  # no sci notation


def attempt_cost(key, model):
    """Mean per-attempt cost of a system, averaged over tasks (the jsonl's per-task mean_cost)."""
    import json as _json
    path = os.path.join(DATASETS[key]["base_dir"], f"{model}.jsonl")
    costs = [_json.loads(l)["mean_cost"] for l in open(path)]
    return float(np.mean(costs))

PEAK_MIN = 0.02                             # name a system only if its share peaks above this
TOP_CAP = 8                                 # ...and at most this many (palette + legend budget);
                                            # the rest pool into Other only if they carry real money
OTHER_C = "0.72"
# fig1 palette first, then distinct steps from the validated theme
COLORS = [F.ACCENT, F.HAZE, "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300",
          "#e34948", "#6d4c2f", "#00857a"]              # 10 distinct; series beyond that recycle


def main(key, k=None, xaxis="solve"):
    r = CR.replay(key, k=k)
    models, solve, spend, rev = r["models"], r["solve"], r["spend"], r["rev"]
    x = solve.mean(1) * 100.0                              # pooled CV solve rate per lambda
    tot = rev.sum(1)                                       # (L, M) money reaching each system
    sp = spend.mean(1)                                     # expected $ per task at each lambda
    keep = CR.envelope(x / 100.0, sp)                      # monotone frontier, sorted by spend
    x, tot, rev_k, sp = x[keep], tot[keep], rev[keep], sp[keep]
    # Terminal cut (the make_fig_passk_data.terminal_cut shape, on shares): in the terminal
    # region the frontier is near-vertical and the last dollars flip wholesale between near-tied
    # portfolios fold-to-fold -- on this data shares move 70-76pp PER STEP there, against <=1.1pp
    # anywhere in the body. Cut at the first step, within TERM_WINDOW_PP of the ceiling, whose
    # total share movement exceeds TERM_MOVE_PP; both thresholds sit an order of magnitude clear
    # of the data on either side.
    TERM_WINDOW_PP, TERM_MOVE_PP = 3.0, 20.0
    share_ = tot / np.clip(tot.sum(1, keepdims=True), 1e-12, None)
    move = np.abs(np.diff(share_, axis=0)).sum(1) * 100.0
    bad = np.nonzero((x[1:] >= x.max() - TERM_WINDOW_PP) & (move > TERM_MOVE_PP))[0]
    if bad.size:
        x, tot, rev_k, sp = x[:bad[0] + 1], tot[:bad[0] + 1], rev_k[:bad[0] + 1], sp[:bad[0] + 1]
    groups = {}
    for j, m in enumerate(models):
        groups.setdefault(base_model(m), []).append(j)
    gnames = list(groups)
    rev_k = np.stack([rev_k[:, :, idx].sum(2) for idx in groups.values()], axis=2)
    tot = np.stack([tot[:, idx].sum(1) for idx in groups.values()], axis=1)
    print(f"aggregated {len(models)} systems -> {len(gnames)} base models")

    def group_cost(gi):
        """Revenue-weighted mean per-attempt cost over the group's member systems."""
        idx = list(groups.values())[gi]
        w = np.array([max(r["rev"][:, :, j].sum(), 1e-12) for j in idx])
        c = np.array([attempt_cost(key, models[j]) for j in idx])
        return float((w * c).sum() / w.sum())


    # Bin the frontier into solve-rate bins of about twice the one-task quantum (100/P), and
    # report the MONEY-WEIGHTED mean within each bin: the axis then carries only the resolution
    # the task count supports. Adjacent lambda points are near-tied portfolios whose money lands
    # on different systems; averaging them money-weighted inside a bin is exactly "the share of
    # spend for targets in this band", which is what the y-axis claims.
    BIN_PP = {"terminal_bench2_priced": 2.0, "deepswe_priced": 1.5,
              "terminal_bench4_priced": 3.0,               # 66 tasks: 1 task = 1.5pp
              "livecodebench_priced": 0.5}.get(key, 2.0)
    edges = np.arange(np.floor(x.min()), x.max() + BIN_PP, BIN_PP)
    which = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    xb, cb, rb = [], [], []
    for b_ in range(len(edges) - 1):
        m_ = which == b_
        if not m_.any():
            continue
        w = tot[m_].sum(1)                                 # money at each lambda in the bin
        xb.append(float((x[m_] * w).sum() / w.sum()))
        cb.append(float((sp[m_] * w).sum() / w.sum()))     # money-weighted budget per task
        rb.append(rev_k[m_].sum(0))                        # (P, M): pooled money in the bin
    x = np.array(xb)
    xc = np.array(cb)
    rev_k = np.stack(rb)                                   # (Lb, P, M)
    tot = rev_k.sum(1)
    if xaxis == "budget":
        # Same bins and cuts as the solve-rate view (solve and C are monotone along the
        # frontier), only the plotted coordinate changes: the deployment budget per task.
        x = xc

    # 95% bootstrap band over TASKS (the convention of the revenue-bar figure): resample the
    # tasks, recompute each share as a ratio of sums; one index set shared across lambdas and
    # systems, so the bands are paired and move together.
    B_BOOT = 1000
    P = rev_k.shape[1]
    IDX = np.random.default_rng(0).integers(0, P, size=(B_BOOT, P))
    den = rev_k.sum(2)                                     # (L, P) spend per task
    den_b = den[:, IDX].sum(2)                             # (L, B)
    share = tot / np.clip(tot.sum(1, keepdims=True), 1e-12, None)
    top = [int(j) for j in np.argsort(share.max(0))[::-1] if share[:, j].max() > PEAK_MIN][:TOP_CAP]
    rest = [j for j in range(len(gnames)) if j not in top]
    def band(j):
        num_b = rev_k[:, :, j][:, IDX].sum(2)              # (L, B)
        with np.errstate(invalid="ignore", divide="ignore"):
            sh = np.where(den_b > 0, num_b / np.where(den_b > 0, den_b, 1.0), np.nan)
        return np.nanpercentile(sh, 2.5, axis=1) * 100, np.nanpercentile(sh, 97.5, axis=1) * 100

    top.sort(key=group_cost)                               # legend (and colours) in price order
    series = [(f"{base_label(models[groups[gnames[j]][0]])} — {fmt_cost(group_cost(j))}/attempt",
               share[:, j], COLORS[i % len(COLORS)], band(j)) for i, j in enumerate(top)]
    # Other appears only when the pooled remainder itself carries visible money: a pool of
    # zero-earners (TB4: four systems that never receive a dollar) would just add a flat line.
    if rest and share[:, rest].sum(1).max() > PEAK_MIN:
        series.append((f"Other ({len(rest)} base models)", share[:, rest].sum(1), OTHER_C, None))
    o = np.argsort(x)
    fig, ax = plt.subplots(figsize=(F.PS.width(0.62), 2.3))
    for l_, s_, c_, b_ in series:
        if b_ is not None:
            lo_, hi_ = b_
            ax.fill_between(x[o], lo_[o], hi_[o], color=c_, alpha=0.15, lw=0, zorder=2)
        ax.plot(x[o], s_[o] * 100, color=c_, lw=F.ACCENT_LW, alpha=0.9, label=l_,
                solid_capstyle="round", zorder=4)
    if xaxis == "budget":
        ax.set_xscale("log")
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, 100)
        ax.set_xlabel("Deployment budget per task ($, CV)", fontsize=F.PS.FS["label"])
    else:
        # Left edge: a long one-system monopoly (flat 100%/0% lines) says one sentence and spends
        # most of the axis on it. Default: start where a second system first invoices anything,
        # rounded down. Terminal-Bench's second system enters almost immediately but stays
        # marginal, so the rule degenerates there; X_LO pins it past the Minimax monopoly instead.
        X_LO = {"terminal_bench2_priced": 40.0}
        if key in X_LO:
            lo = X_LO[key]
        else:
            entry = [x[np.argmax(s_ > 0.01)] for _, s_, _, _ in series[1:] if (s_ > 0.01).any()]
            lo = 5.0 * np.floor(min(entry) / 5.0) if entry else x.min()
        ax.set_xlim(lo, x.max()); ax.set_ylim(0, 100)
        ax.set_xlabel("Solve rate reached (CV)", fontsize=F.PS.FS["label"])
    ax.set_ylabel("Share of arbitrageur spend", fontsize=F.PS.FS["label"])
    from matplotlib.ticker import PercentFormatter
    if xaxis != "budget":
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(labelsize=F.PS.FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=F.PS.FS["legend"],
                    frameon=False, handlelength=1.2, labelspacing=0.4)
    out = os.path.join(HERE, f"fig_rev_share_{key}" + ("" if k is None else f"_k{k}")
                       + ("_C" if xaxis == "budget" else ""))
    fig.savefig(out + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out + ".pdf")


if __name__ == "__main__":
    kv = None
    av = sys.argv[1:]
    if "--k" in av:
        i = av.index("--k"); kv = int(av[i + 1]); del av[i:i + 2]
    xa = "budget" if "--x-budget" in av else "solve"
    av = [a_ for a_ in av if a_ != "--x-budget"]
    main(av[0] if av else "deepswe_priced", k=kv, xaxis=xa)
