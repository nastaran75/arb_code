"""Paired estimation-error gaps (Figure 9): histograms over random calibration draws of

    Delta_ours = RMSE(plug-in) - RMSE(ours),

where a draw's RMSE is the per-model pass@k RMSE against the Chen truth on the full attempts,
averaged over models AND over the k grid KS (the deployable range; the k=1 slice alone scores
the least decision-relevant point of the curve). Every estimator is fitted on the SAME
n-attempt subsample of each draw; mass right of the zero line = that method beats the plug-in
on that draw. Per-draw predictions are cached in gap_cache/.

    ../.venv/bin/python fig_gap_dist.py                    # n = 8, all six benchmarks
    ../.venv/bin/python fig_gap_dist.py --n 50 --min-p1 1e-3
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fig1 as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

# The six benchmarks of the paper, in the figure's order (KEYS then KEYS_APP).
KEYS = [("monkey_math", "MATH"), ("monkey_codecontests", "CodeContests"),
        ("resmat2_aime2024", "AIME 2024"), ("resmat2_aime2025", "AIME 2025")]
KEYS_APP = [("resmat2_gmmlu", "Global-MMLU-Lite"), ("resmat2_mmlupro", "MMLU-Pro")]
N_SUB = 8
DRAWS = 100
KS = [1, 2, 5, 10, 20, 50, 100, 200, 500]   # the curve is scored on this grid, then averaged
BINS = 16
# each series measures improvement over the plug-in: right of zero = that method beats it
SERIES = {"ours": ("Plug-in $-$ ours", F.ACCENT)}
CACHE = os.path.join(HERE, "gap_cache")
MIN_P1 = None                  # Truong et al.'s no-power filter; --min-p1 sets it


def load_estimates(key, n_sub, draws):
    """(y_true (P,M,K), {arm: (draws,P,M,K)}) on the seeded draws, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{key}_n{n_sub}_r{draws}_zibb_kavg.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["y_true"], {k[4:]: z[k] for k in z.files if k.startswith("est_")}
    import warnings
    warnings.filterwarnings("ignore")
    import validate_estimator as v
    import estimator as est
    v.configure(key)
    cfg, models, q, c, n = v.load(key)[:5]
    M = len(models)
    K = np.array(KS, float)
    y_true = np.stack([v.chen_curve(c[:, j], n[:, j], K) for j in range(M)], axis=1)  # (P, M, K)
    e = {a: [] for a in ("plug", "ours")}
    for rep in range(draws):
        rng = np.random.default_rng(v.SEED + 1000 * n_sub + rep)
        c_s, n_s, _, _ = v.subsample(c, n, n_sub, rng, fixed=cfg.get("fixed"))
        P_, O_ = np.empty_like(y_true), np.empty_like(y_true)
        for j in range(M):
            # the paper's estimator, exactly as in the calibration figure (validate_estimator "zibb"):
            # zero-inflated Beta-Binomial prior with the (a+b)^-2.5 hyperprior, per-query posterior
            a_, b_, pi0 = est.fit_zibb(c_s[:, j], n_s[:, j])
            P_[:, j] = est.geom_curve(c_s[:, j], n_s[:, j], K, extrapolate=True)
            O_[:, j] = est.zibb_curve(c_s[:, j], n_s[:, j], K, a_, b_, pi0)
        e["plug"].append(P_); e["ours"].append(O_)
        if rep % 20 == 19:
            print(f"[{key}] draw {rep + 1}/{draws}", flush=True)
    e = {a: np.array(x) for a, x in e.items()}
    np.savez_compressed(path, y_true=y_true, **{"est_" + a: X for a, X in e.items()})
    return y_true, e


def gaps(key, n_sub, draws):
    """{method: (draws,) paired k-averaged RMSE gap, plug-in minus that method}."""
    y_true, e = load_estimates(key, n_sub, draws)
    if MIN_P1 is not None:
        keep = y_true[:, :, 0].max(1) >= MIN_P1
        print(f"[{key}] min-p1 filter {MIN_P1:g}: keeping {keep.sum()}/{keep.size} problems", flush=True)
        y_true = y_true[keep]
        e = {a: X[:, keep] for a, X in e.items()}
    M = y_true.shape[1]

    def rmse_per_draw(X):                    # (draws, P, M, K) -> (draws,): mean over models, k
        err2 = (X - y_true[None]) ** 2
        return np.sqrt(err2.mean(1)).mean((1, 2))          # rmse over P, then mean over (M, K)

    base = rmse_per_draw(e["plug"])
    return {m: base - rmse_per_draw(e[m]) for m in SERIES}


def panel(ax, gp, name):
    from matplotlib.ticker import MaxNLocator
    allg = np.concatenate(list(gp.values()))
    lo, hi = allg.min(), allg.max()
    pad = 0.10 * (hi - lo) if hi > lo else 1e-3
    # shared bin edges anchored at zero so no bar straddles the reference line
    w = (hi - lo + 2 * pad) / BINS
    edges = np.arange(np.floor((lo - pad) / w), np.ceil((hi + pad) / w) + 1) * w
    # ".. % > 0" annotation: just right of the zero line when it hugs the panel's left edge,
    # just LEFT of it when the line sits in the interior (a visible negative tail, e.g.
    # CodeContests) -- either way it never overprints the line or the bars' mode.
    # the drawn x range includes BOTH the bars and the zero line (axvline extends the limits
    # when all bars are on one side), so the fraction must be computed over that union
    x0, x1 = min(edges[0], 0.0), max(edges[-1], 0.0)
    zf = np.clip((0.0 - x0) / (x1 - x0), 0.0, 1.0)     # zero line in axes fraction
    ann_x, ann_ha = (zf - 0.03, "right") if zf > 0.25 else (zf + 0.06, "left")
    for i, (c, (label, colr)) in enumerate(SERIES.items()):
        g = gp[c]
        ax.hist(g, bins=edges, color=colr, alpha=0.55, edgecolor="white", linewidth=0.3, zorder=3)
        ax.text(ann_x, 0.96 - 0.13 * i, f"{np.mean(g > 0) * 100:.0f}% > 0",
                transform=ax.transAxes, ha=ann_ha, va="top", fontsize=F.FS["annot"], color=colr)
    ax.axvline(0.0, color="#b02a2e", lw=0.9, ls=(0, (3, 2)), zorder=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=False))
    ax.set_title(name, fontsize=F.FS["annot"], pad=3)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.5, labelsize=F.FS["tick"], colors=F.AXIS, labelcolor=F.INK)


def main(keys, out, paper=None, figw=1.0, ncols=None):
    """One panel per benchmark, `ncols` per row (default: all in one row); the shared x label
    sits under the bottom row and each row's first panel carries the y label."""
    n = len(keys); ncols = ncols or n; nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(F.PS.width(figw), 1.55 * nrows), squeeze=False)
    for ax, (key, name) in zip(axs.ravel(), keys):
        gp = gaps(key, N_SUB, DRAWS)
        panel(ax, gp, name)
        print(f"[{key}] " + "  ".join(f"{c}: {np.mean(g > 0) * 100:3.0f}%>0 med {np.median(g):+.4f}"
                                      for c, g in gp.items()), flush=True)
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    for row in axs:
        row[0].set_ylabel("Density", fontsize=F.FS["label"])
    fig.tight_layout(w_pad=F.W_PAD, h_pad=1.6)
    q0, q1 = axs[-1][0].get_position(), axs[-1][-1].get_position()
    fig.text((q0.x0 + q1.x1) / 2, q0.y0 - 0.175 / nrows,
             r"Plug-in RMSE $-$ EB pass@$k$ (ours) RMSE",
             ha="center", va="top", fontsize=F.FS["label"], color=F.INK)
    fig.savefig(out + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out + ".pdf")
    if paper and os.path.isdir(os.path.dirname(paper)):
        import shutil
        shutil.copy(out + ".pdf", paper)
        print("mirrored to", os.path.normpath(paper))
    plt.close(fig)


OUT = OUT_DEFAULT = os.path.join(HERE, "fig_gap_dist")
PAPER_DIR = os.path.join(HERE, "..", "figures")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-p1", type=float, help="drop problems whose best model's oracle pass@1 "
                                                 "is below this (Truong et al.'s no-power filter)")
    ap.add_argument("--n", type=int, help=f"calibration size (default {N_SUB}; non-default "
                                          "values write to their own file stem)")
    A = ap.parse_args()
    if A.n is not None and A.n != N_SUB:
        N_SUB = A.n
        OUT = OUT + f"_n{A.n}"
    if A.min_p1 is not None:
        MIN_P1 = A.min_p1
        OUT = OUT + f"_minp{A.min_p1:g}"
    if OUT == OUT_DEFAULT:                         # default run: all six benchmarks in one 2x3 (the paper's Figure 9)
        main(KEYS + KEYS_APP, OUT + "_all",
             paper=os.path.join(PAPER_DIR, "fig_gap_dist_all.pdf"), figw=0.9, ncols=3)
    else:                                          # --n / --min-p1 variants: main set, no mirror
        main(KEYS, OUT)
