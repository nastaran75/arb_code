"""1xN panels of Section-2 arbitrage results (publication style).

  python make_panel.py                       -> cost panel: individual models (haze) + arbitrage (accent)
  python make_panel.py --profit              -> profit-margin panel: (frontier - arbitrage)/frontier, one axis per dataset
  python make_panel.py --with-profit         -> cost panels + ONE extra panel with a profit-margin line per dataset
  python make_panel.py KEY [KEY ...]         -> dataset keys override the default set
  --out NAME        output basename (default fig_panel / fig_profit_panel)
  --palette NAME    colour scheme from PALETTES (default "blue-orange")
  --figwidth W      physical figure width in inches (e.g. 5.5 = \textwidth); fonts/strokes rescale so the
                    PDF can be included at width=\textwidth without shrinking (default: 3.15 in per panel)
  --figsize W,H     whole-figure size in inches with fonts/strokes left at the reference sizes (e.g. 8.2,2.3)
  --extrapolate     plain geometric extrapolation (default: none past the observed attempts, see make_figure.py)
  --estimator E     'chen' (default: unbiased pass@k, tabulated path, greedy order) | 'geom' (closed form)

No market-frontier line is drawn in the cost panel; the frontier is computed internally
for the profit-margin panel. Each axis carries its own y-label; titles are the bare dataset names.

NOTE: the curves here are IN-SAMPLE. The paper's Figure 2 is the cross-validated version,
produced by `cv.py --paper` (same `compute()` for the individual models, CV arbitrageur and
market); use this script for exploration and for the in-sample diagnostics of Appendix C.
"""
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

import arbitrage as arb
from make_figure import DATASETS

# Drawing code lives in plots/fig1.py (single implementation shared with the paper's reproduction script).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots"))
import fig1 as F  # noqa: E402

PALETTES = {                      # haze colour, haze alpha, accent colour
    "blue-orange": (F.HAZE, F.HAZE_ALPHA, F.ACCENT),
    "aqua-violet": ("#1baf7a", 0.32, "#4a3aa7"),
    "blue-navy":   ("#2a78d6", 0.26, "#0d366b"),
}
PROFIT = "#4a3aa7"                     # colour of the per-dataset profit panels (--profit)

PANEL = ["livecodebench_priced", "terminal_bench2_priced"]

PANEL_W, PANEL_H = 3.15, 2.75          # inches per panel at the reference size


def set_figwidth(width_in, n_ax):
    """Render at a physical width (inches, e.g. \\textwidth = 5.5) so the PDF is included 1:1 in the paper:
    fonts and strokes in plots/fig1.py are scaled down (floor 78% / 75%)."""
    global PANEL_W, PANEL_H
    ref = 3.15 * n_ax
    F.SC = max(0.78, width_in / ref)
    PANEL_W = width_in / n_ax
    PANEL_H = 2.75 * max(0.62, width_in / ref)
    k = max(0.75, width_in / ref)
    F.HAZE_LW, F.ACCENT_LW = F.HAZE_LW * k, F.ACCENT_LW * k


ESTIMATOR = "chen"                     # pass@k estimator for fitting AND evaluation (see estimator.py)
PTS_PER_DECADE = 50                    # budget-grid resolution of the tabulated path


def compute(cfg, n_budget=70, extrapolate=False, ext_decades=4, estimator=None):
    """Per-dataset curves. Returns (grid, model_cost, arb_cost, market_cost, floor), costs in $ over the
    benchmark (P problems). Single-model curves are evaluated on a budget grid extended `ext_decades`
    below bmin so every model is defined down to ~0% solve rate (otherwise the market frontier at low
    solve rates is spuriously an expensive model); `floor` = bmin*P is where the cost panels clip.

    `estimator` (default ESTIMATOR): 'chen' = unbiased pass@k of Chen et al. with a randomized last
    attempt, fit and evaluated through the tabulated path (arbitrage_tab; non-concave, grid-argmax
    coordinate step); 'geom' = closed-form geometric plug-in (arbitrage.py; concave, certified).
    Allocations are always deployed in the greedy population order of Section 3."""
    estimator = estimator or ESTIMATOR
    base_dir, models = cfg["base_dir"], cfg["models"]
    if models is None:
        models = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(base_dir, "*.jsonl")))
    results = arb.load_results(base_dir)
    alpha, zcap, q, ids = arb.build_alpha(results, models, extrapolate=extrapolate)
    P = len(ids)
    if estimator == "geom":
        budgets = np.logspace(np.log10(cfg["bmin"]) - ext_decades, np.log10(cfg["bmax"]), n_budget + 10 * ext_decades)
        model_perf = np.array([arb.single_model_perf(alpha[:, j], zcap[:, j], budgets) for j in range(len(models))])
        model_spend = np.array([arb.single_model_spend(alpha[:, j], zcap[:, j], budgets) for j in range(len(models))])
        arb_perf, arb_spend, allocs = arb.arbitrage_frontier(alpha, zcap, n_budget)
    else:
        import estimator as est
        import arbitrage_tab as tab
        lo, hi = np.log10(cfg["bmin"]) - ext_decades, np.log10(cfg["bmax"])
        budgets = np.logspace(lo, hi, int((hi - lo) * PTS_PER_DECADE))
        U, _ = est.pass_curves(results, models, ids, q, budgets, estimator, extrapolate=extrapolate)   # per-problem costs
        _, n_rec = est.counts(results, models, ids)                 # no money past the recorded attempts
        tab.SPEND_CAP = n_rec * np.asarray(q, dtype=float) if estimator in ("chen", "chen_lin", "chen_int", "geom") else None
        model_perf, model_spend = tab.single_model(U, budgets)
        arb_perf, arb_spend, allocs = tab.arbitrage_frontier(U, budgets, n=n_budget, robust=(estimator != "geom"), n_head=16)
    # The frontier runs continuously down to zero budget (dense lambda head in
    # arbitrage_frontier); the cost panel truncates it at the models' cost floor.
    pos = arb_spend > 0
    arb_perf, arb_spend = arb_perf[pos], arb_spend[pos]

    grid = np.linspace(max(cfg["perf_lo"], min(model_perf.min(), arb_perf.min())), float(arb_perf.max()), 300)
    model_cost = np.array([arb.invert_to_cost(model_perf[j], model_spend[j], grid) for j in range(len(models))])
    with np.errstate(all="ignore"):
        market_cost = np.nanmin(model_cost, axis=0)          # cheapest single model at each solve rate (not drawn)
    arb_cost = arb.invert_to_cost(arb_perf, arb_spend, grid)
    # Every single-model allocation is a feasible arbitrage allocation, so the arbitrageur's cost is
    # at most the cheapest single model's. This defines the curve below the sweep's first allocation
    # (below one attempt the Chen curves are linear in budget, so the sweep starts at one full attempt)
    # and removes the ~1% cascade overhead where the sweep's allocation is marginally worse than a
    # single model (margin exactly 0 there instead of slightly negative).
    arb_cost = np.fmin(arb_cost, market_cost)
    return grid, model_cost * P, arb_cost * P, market_cost * P, cfg["bmin"] * P


def _title(cfg):
    return re.sub(r"\s*\(\d+ systems\)", "", cfg["title"])   # bare dataset name


def _title(cfg):
    return re.sub(r"\s*\(\d+ systems\)", "", cfg["title"])   # bare dataset name


def as_record(cfg, data):
    """Package compute() output as the data record plots/fig1.py draws (see fig1.py docstring)."""
    grid, model_cost, arb_cost, market_cost, floor = data
    title = _title(cfg)
    return dict(title=title, legend=re.sub(r"\s+\d+(\.\d+)?$", "", title), xlabel=cfg["xlabel"],
                ylabel=cfg.get("ylabel", "Cost ($)"), floor=float(floor), solve_rate=grid * 100.0,
                model_cost=model_cost, arb_cost=arb_cost, market_cost=market_cost)


def cost_panel(keys, extrapolate=False, name="fig_panel", palette="blue-orange", with_profit=False):
    F.HAZE, F.HAZE_ALPHA, F.ACCENT = PALETTES[palette]
    n_ax = len(keys) + (1 if with_profit else 0)
    items = [as_record(DATASETS[k], compute(DATASETS[k], extrapolate=extrapolate)) for k in keys]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    F.draw_fig1(items, out=out, figsize=(PANEL_W * n_ax, PANEL_H), with_profit=with_profit)


def profit_panel(keys, extrapolate=False, name="fig_profit_panel", palette="blue-orange"):
    fig, axs = plt.subplots(1, len(keys), figsize=(F.PS.width(1.0), 2.2))   # final size for width=\textwidth
    axs = np.atleast_1d(axs)
    for ax, key in zip(axs, keys):
        cfg = DATASETS[key]
        x, margin = F.margin(as_record(cfg, compute(cfg, extrapolate=extrapolate)))
        ax.plot(x, margin, color=PROFIT, lw=F.ACCENT_LW, zorder=5, solid_capstyle="round")
        ax.set_title(_title(cfg), fontsize=F.FS["title"], pad=6)
        ax.set_xlabel(cfg["xlabel"], fontsize=F.FS["label"])
        F.style(ax, ylabel="Arbitrage profit margin (%)", log_y=False)
        ax.set_ylim(0, None)
    fig.tight_layout(w_pad=2.0)
    _save(fig, name)


def _save(fig, name):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    argv = sys.argv[1:]

    def _opt(flag, default):
        if flag in argv:
            i = argv.index(flag); v = argv[i + 1]; del argv[i:i + 2]; return v
        return default
    name = _opt("--out", None)
    palette = _opt("--palette", "blue-orange")
    F.ACCENT_ALPHA = float(_opt("--accent-alpha", F.ACCENT_ALPHA))
    F.ACCENT_LW = float(_opt("--accent-lw", F.ACCENT_LW))
    _ha = _opt("--haze-alpha", None)
    if _ha is not None:
        PALETTES[palette] = (PALETTES[palette][0], float(_ha), PALETTES[palette][2])
    ESTIMATOR = _opt("--estimator", ESTIMATOR)               # 'chen' (default) | 'geom'
    _fw = _opt("--figwidth", None)                          # physical width in inches (e.g. 5.5 = \textwidth)
    _fs = _opt("--figsize", None)                           # "W,H" in inches for the whole figure; fonts unchanged
    keys = [a for a in argv if not a.startswith("--")] or PANEL
    extrapolate = "--extrapolate" in argv
    n_ax = len(keys) + (1 if "--with-profit" in argv else 0)
    if _fw is not None:
        set_figwidth(float(_fw), n_ax)
    if _fs is not None:
        W, H = (float(v) for v in _fs.split(","))
        PANEL_W, PANEL_H = W / n_ax, H
    if "--profit" in argv:
        profit_panel(keys, extrapolate=extrapolate, name=name or "fig_profit_panel", palette=palette)
    else:
        cost_panel(keys, extrapolate=extrapolate, name=name or "fig_panel", palette=palette,
                   with_profit="--with-profit" in argv)
