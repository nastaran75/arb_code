"""Revenue spectrum: the revenue-share data as a heatmap instead of crossing lines.

Rows = base models sorted by per-attempt price (cheapest at the top, so reading order = entry
order along the frontier); x = deployment budget per task, log; cell = share of the
arbitrageur's spend in that budget bin. Composition data with 6-10 series occludes itself as
lines; here nothing overlaps, colour carries only magnitude (identity is the row label), and
the paper's claim -- revenue succession follows price -- is the diagonal ridge.

    ../.venv/bin/python fig_rev_heat.py                # all four priced datasets
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plots"))
import fig1 as F                            # noqa: E402
import cv_replay as CR                      # noqa: E402
from fig_rev_share import base_model, base_label, fmt_cost, attempt_cost    # noqa: E402
from make_cost_bars import short as SHORT                                  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("rev", ["#ffffff", "#fbd0bd", F.ACCENT, "#7a2d10"])
ROW_MIN = 0.01                              # a row must peak above this share to appear


def binned(key, group=True):
    """(labels_with_price, cost_per_attempt, budget_bin_centers, share (rows, bins)) --
    the exact prep of fig_rev_share.main (envelope, terminal cut, money-weighted bins,
    base-model grouping), returned as a matrix instead of drawn.

    `group=False` keeps one row per SYSTEM (Agent x Model) instead of pooling the systems
    that share a base model. The allocation itself is always solved per system; grouping is
    a display choice, and ungrouped is what shows which scaffold the money actually reached."""
    r = CR.replay(key, verbose=False)
    models, solve, spend, rev = r["models"], r["solve"], r["spend"], r["rev"]
    x = solve.mean(1) * 100.0
    sp = spend.mean(1)
    tot = rev.sum(1)
    keep = CR.envelope(x / 100.0, sp)
    x, tot, rev_k, sp = x[keep], tot[keep], rev[keep], sp[keep]
    share_ = tot / np.clip(tot.sum(1, keepdims=True), 1e-12, None)
    move = np.abs(np.diff(share_, axis=0)).sum(1) * 100.0
    bad = np.nonzero((x[1:] >= x.max() - 3.0) & (move > 20.0))[0]
    if bad.size:
        x, tot, rev_k, sp = (a[:bad[0] + 1] for a in (x, tot, rev_k, sp))
    BIN_PP = {"terminal_bench2_priced": 2.0, "deepswe_priced": 1.5,
              "terminal_bench4_priced": 3.0, "livecodebench_priced": 0.5}.get(key, 2.0)
    edges = np.arange(np.floor(x.min()), x.max() + BIN_PP, BIN_PP)
    which = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    cb, rb = [], []
    for b_ in range(len(edges) - 1):
        m_ = which == b_
        if not m_.any():
            continue
        w = tot[m_].sum(1)
        cb.append(float((sp[m_] * w).sum() / w.sum()))
        rb.append(rev_k[m_].sum(0))
    rev_b = np.stack([r_.sum(0) for r_ in rb])             # (bins, systems)
    groups = {}
    for j, m in enumerate(models):
        groups.setdefault(base_model(m) if group else m, []).append(j)
    share = np.stack([rev_b[:, idx].sum(1) for idx in groups.values()], 1)
    share = share / np.clip(share.sum(1, keepdims=True), 1e-12, None)
    costs, labels = [], []
    for g, idx in groups.items():
        w = np.array([max(rev_b[:, j].sum(), 1e-12) for j in idx])
        c = np.array([attempt_cost(key, models[j]) for j in idx])
        costs.append(float((w * c).sum() / w.sum()))
        labels.append(base_label(models[idx[0]]) if group else SHORT(models[idx[0]]))
    return labels, np.array(costs), np.array(cb), share


MONO_SHARE = 0.95          # a bin is 'monopoly' when one row holds more than this share of spend
LEFT_MARGIN_DECADES = 0.15 # how far before the first non-monopoly bin the axis starts (log10 units)
X_LO = {}                  # per-dataset override of the left edge, $ per task


def draw(key, ax, cbar=False, prices=True, group=True):
    """`prices=False` drops the per-attempt cost from the row labels (rows stay sorted by it).
    `group=False`: one row per system instead of per base model (see `binned`)."""
    labels, costs, cbins, share = binned(key, group=group)
    keep = np.nonzero(share.max(0) > ROW_MIN)[0]
    order = keep[np.argsort(costs[keep])]                  # cheapest at the top
    S = share[:, order].T * 100                            # (rows, bins)
    edges = np.sqrt(cbins[1:] * cbins[:-1])
    edges = np.concatenate([[cbins[0] ** 2 / edges[0]], edges, [cbins[-1] ** 2 / edges[-1]]])
    Y = np.arange(len(order) + 1)
    pc = ax.pcolormesh(edges, Y, S[::-1], cmap=CMAP, vmin=0, vmax=100,
                       edgecolors="white", linewidth=0.4)
    ax.set_xscale("log")
    # Left edge: drop the ultra-low budgets where one system holds essentially all the spend --
    # on a log axis that monopoly stretches over 1.5-2 decades and squeezes the region where
    # revenue actually changes hands into the right third. Start half a decade before the first
    # bin in which the dominant row falls below MONO_SHARE, so the hand-over itself is visible.
    # X_LO overrides per dataset (budget in $); the right edge is the frontier's end as before.
    mono = S.max(0)                                        # dominant row's share per bin, %
    i0 = int(np.argmax(mono < 100 * MONO_SHARE)) if (mono < 100 * MONO_SHARE).any() else 0
    x_lo = X_LO.get(key, edges[i0] / 10 ** LEFT_MARGIN_DECADES)
    ax.set_xlim(max(x_lo, edges[0]), edges[-1])
    names = [f"{labels[j]}  ({fmt_cost(costs[j])})" if prices else labels[j] for j in order]
    ax.set_yticks(Y[:-1] + 0.5)
    ax.set_yticklabels(names[::-1], fontsize=F.PS.FS["dense"])
    ax.set_xlabel("Deployment budget per task ($, CV)", fontsize=F.PS.FS["label"])
    ax.tick_params(axis="x", labelsize=F.PS.FS["tick"], length=2.5)
    ax.tick_params(axis="y", length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    from make_figure import DATASETS
    ax.set_title(DATASETS[key]["title"], fontsize=F.PS.FS["title"], pad=4)
    return pc


def main(keys, group=True):
    n = len(keys)
    fig, axs = plt.subplots(n, 1, figsize=(F.PS.width(0.9), 1.05 + 1.5 * n), squeeze=False)
    for ax, key in zip(axs[:, 0], keys):
        pc = draw(key, ax, group=group)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label("Share of arbitrageur spend (%)", fontsize=F.PS.FS["label"])
    cb.ax.tick_params(labelsize=F.PS.FS["tick"])
    fig.tight_layout(rect=(0, 0, 0.9, 1), h_pad=1.4)
    out = os.path.join(HERE, "fig_rev_heat_" + "-".join(k.split("_")[0] for k in keys)
                       + ("" if group else "_bysystem"))
    fig.savefig(out + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out + ".pdf")


if __name__ == "__main__":
    argv = sys.argv[1:]
    group = "--by-system" not in argv                      # opt-in: do NOT pool by base model
    ks = [a for a in argv if not a.startswith("--")] or [
        "deepswe_priced", "terminal_bench4_priced",
        "terminal_bench2_priced", "livecodebench_priced"]
    main(ks, group=group)
