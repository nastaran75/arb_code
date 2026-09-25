"""Data records for the pass@k-section figure (plots/fig_passk.py): what the arbitrageur achieves and what it
believes when its pass@k curves are estimated from n samples per (problem, model).

Reads the validate_calibration.py cache of a dataset (R draws x N_SUBS; estimators: geometric plug-in
1 - (1 - c/n)^k extrapolated to any k, and the zero-inflated, hyperprior-regularized
Beta-Binomial extension used by the released Figure 5/10 records) and the
oracle (Chen unbiased pass@k on all attempts, k <= KCAP), and writes plots/fig_passk_data/<key>_n<n>.json:

  solve_rate[G]      x grid (percent)
  oracle_cost[G]     arbitrage frontier fitted on the true curves (total expected spend over the problems)
  market_cost[G]     cheapest single model on the true curves
  oracle_margin[G]   (market - oracle) / market, percent
  x_lo               first solve rate (percent) at which the oracle frontier is no more expensive than the market
                     (below it the budget grid's floor, not the data, sets the curves), or null
  x_extrap           solve rate (percent) from which the oracle allocation spends more than n attempts of some
                     model (an n-sample estimate must extrapolate pass@k beyond it), or null
  x_cut              solve rate (percent) of the last grid point before the terminal spike of the oracle margin
                     (the cheapest model's cost exploding at the end of its reach; see terminal_cut), or null
  est[mode]          label, reach[G] / claim_reach[G] (fraction of draws whose realized / claimed frontier reaches s)
                     and, over ALL draws (median, 25th, 75th percentile; a draw that does not reach s counts as an
                     infinite cost, so a statistic is null where fewer than half of the draws reach s):
                       realized_margin_*  (market - realized) / market, percent   -> profitability
                       claimed_ratio_*    claimed / oracle cost                    -> frontier approximation
                       realized_ratio_*   realized / oracle cost                   -> regret
  k_curve[K]         the k grid of the pass@k curves below. NOT the budget grid: mapping budget -> k
                     leaves only a handful of points below k = 10 and none of them at k = 1, so the
                     curves are evaluated directly on this grid instead (every integer to 10, then
                     log-spaced to KCAP).
  model_pass_oracle  [M][K] aggregate pass@k per model on the TRUE curves (Chen unbiased on all
                     10,000 attempts): mean over problems of u_m(x, k) -- what fig_passk_curves.py draws
  model_pass[mode]   [M][K] the same aggregate under each estimator, median over the draws
  targets            the same three quantities at the fixed solve-rate targets of validate_estimator.TARGETS
  + n_sub, n_draws, n_problems, n_models, models, kcap, title, legend, xlabel

Usage (from experiments/):
    VAL_CACHE=val_cache ../.venv/bin/python make_fig_passk_data.py monkey_math monkey_codecontests --n-subs 5,50
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np

import estimator as est
import validate_estimator as v

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "plots", "fig_passk_data")
MODES = {"geom": "Plug-in", "zibb": "Beta-binomial"}
LEGEND = {"monkey_math": "MATH", "monkey_codecontests": "CodeContests",
          "resmat2_aime2024": "AIME 2024", "resmat2_aime2025": "AIME 2025",
          "resmat2_gmmlu": "Global-MMLU-Lite", "resmat2_mmlupro": "MMLU-Pro"}
BAND = (25, 75)            # percentile band over the draws
G = 300                    # grid points
CUT_JUMP_PP = 5.0          # terminal-spike rule of plots/fig1.py
# k grid of the per-model pass@k curves: every integer out to 20 (where the small-k behaviour is,
# and where an n = 5 estimate starts extrapolating), then log-spaced to the KCAP tail.
K_CURVE = np.unique(np.concatenate([np.arange(1.0, 21.0), np.geomspace(20.0, 5000.0, 35)]))


# Our estimator: the zero-inflated beta-binomial fit (spike pi0 + the (a+b)^-2.5 hyperprior of
# Gelman et al., BDA3 section 5.3), predicting through the per-problem posterior `zibb_curve`.
#   fitter,          prior_pow
BB_VARIANTS = {
    "zibb":     (est.fit_zibb, est.PRIOR_POW),
}
PI0_OFF = 1.0 / (1.0 + np.exp(-est.LOGIT_PI0_OFF))


def curve_estimates(cfg, c, n, n_sub, reps):
    """{mode: (M, K) median-over-draws aggregate pass@k} of the plug-in and of our estimator on the
    k grid K_CURVE, from the same seeded draws as validate_calibration.run_one."""
    M = c.shape[1]
    modes = ["geom"] + list(BB_VARIANTS)
    per_draw = {m: [] for m in modes}
    for rep in reps:
        rng = np.random.default_rng(v.SEED + 1000 * n_sub + rep)             # as in validate_calibration.run_one
        c_s, n_s, _, _ = v.subsample(c, n, n_sub, rng, fixed=cfg.get("fixed"))
        cols = {m: [] for m in modes}
        for j in range(M):
            cj, nj = c_s[:, j], n_s[:, j]
            cols["geom"].append(est.geom_curve(cj, nj, K_CURVE,
                                               extrapolate=v.GEOM_EXTRAPOLATE).mean(0))
            for m, (fit, pw) in BB_VARIANTS.items():
                pars = fit(cj, nj, prior_pow=pw)
                a_, b_ = pars[0], pars[1]
                pi0 = pars[2] if len(pars) == 3 else PI0_OFF
                cols[m].append(est.zibb_curve(cj, nj, K_CURVE, a_, b_, pi0).mean(0))
        for m in modes:
            per_draw[m].append(np.array(cols[m]))
    return {m: np.median(per_draw[m], 0) for m in modes}


def find_cache(key, M, ntag):
    cands = [p for p in glob.glob(os.path.join(v.CACHE, f"val_calib_{key}_M{M}_*.pkl")) if ntag in os.path.basename(p)]
    if not cands:
        raise SystemExit(f"no validate_calibration cache for {key} (M={M}, tag '{ntag}') in {v.CACHE}; "
                         f"run validate_calibration.py {key} first")
    return max(cands, key=os.path.getmtime)


def envelope_cost(perf, spend, grid):
    """Cheapest cost with perf >= s for each s in grid (Pareto lower envelope), interpolated in LOG cost --
    the curves live on a log axis, and the claimed frontiers of n = 5 can be sparse in the tail. NaN (unreached)
    beyond the last envelope point. Interpolated in log cost."""
    keep = spend > 0                                  # the zero-spend point of a frontier is not a price
    perf, spend = np.asarray(perf)[keep], np.asarray(spend)[keep]
    order = np.argsort(spend)
    p, c = perf[order], spend[order]
    p_env = np.maximum.accumulate(p)
    keep = np.concatenate([[True], np.diff(p_env) > 0])
    p_env, c_env = p_env[keep], c[keep]
    out = np.exp(np.interp(grid, p_env, np.log(c_env), left=np.nan, right=np.nan))
    out[grid > p_env[-1]] = np.nan
    return out


def band_stats(A, fill):
    """Median / 25th / 75th percentile over the draws (axis 0). A draw that does not reach a solve rate counts as
    `fill` (+inf for a cost or a cost ratio, -inf for a margin) rather than being dropped, so the statistics are
    over ALL draws; a non-finite statistic (fewer than half of the draws reach) is reported as NaN."""
    A = np.where(np.isfinite(A), A, fill)
    med = np.median(A, 0)
    lo, hi = np.percentile(A, BAND, axis=0)
    return [np.where(np.isfinite(x), x, np.nan) for x in (med, lo, hi)]


def summarize(rs, mode, grid, cost_o, market_o, P):
    """Per-estimator statistics on `grid` from the replicates `rs` (all with the same n_sub)."""
    Cr = np.array([envelope_cost(*r[mode]["realized"], grid) * P for r in rs])   # (R, G) true cost of the fitted policies
    Cc = np.array([envelope_cost(*r[mode]["claimed"], grid) * P for r in rs])    # (R, G) cost the estimator claims
    e = dict(label=MODES[mode], reach=np.isfinite(Cr).mean(0), claim_reach=np.isfinite(Cc).mean(0))
    with np.errstate(all="ignore"):
        for name, A, fill in (("realized_margin", (market_o[None, :] - Cr) / market_o[None, :] * 100.0, -np.inf),
                              ("claimed_ratio", Cc / cost_o[None, :], np.inf),
                              ("realized_ratio", Cr / cost_o[None, :], np.inf)):
            e[name + "_med"], e[name + "_lo"], e[name + "_hi"] = band_stats(A, fill)
    return e


def terminal_cut(grid, margin_o):
    """Solve rate (percent) of the last grid point BEFORE the terminal spike of the oracle margin, or None.
    The spike: at the end of the cheapest model's reach its cost explodes, so the margin jumps by > CUT_JUMP_PP
    per grid step and never comes back down (plots/fig1.py cuts the same artefact). Only the last 15% of the grid
    is eligible, so a jump followed by a plateau in the body of the curve is never cut."""
    finite = np.isfinite(margin_o)
    with np.errstate(all="ignore"):
        d = np.diff(margin_o)
    post = np.nonzero(finite[1:] & finite[:-1] & (d > CUT_JUMP_PP))[0] + 1        # indices of the post-jump points
    term = [i for i in post if i >= 0.85 * grid.size and np.nanmin(margin_o[i:]) >= margin_o[i] - 1e-9]
    return float(grid[min(term) - 1] * 100.0) if term else None


def extrapolation_start(ap_o, kmax, n_sub):
    """Solve rate (percent) from which the ORACLE allocation spends more than n_sub attempts of some model
    (kmax = max_i c_i / q_i at each frontier point): beyond it an n_sub-sample estimate has to extrapolate
    pass@k past the observed attempts. First crossing along the frontier; None if never."""
    o = np.argsort(ap_o)
    ext = np.asarray(kmax)[o] > n_sub
    if not ext.any():
        return None
    i = int(np.argmax(ext))
    if not ext[i:].all():
        print(f"  [warn] n={n_sub}: oracle attempts are not monotone along the frontier; shading from the first crossing")
    return float(ap_o[o][i] * 100.0)


def to_json(o):
    if isinstance(o, dict):
        return {k: to_json(x) for k, x in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [to_json(x) for x in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(f"{float(o):.6g}")
    if isinstance(o, (int, np.integer)):
        return int(o)
    return o


def main(argv):
    ns = tuple(int(x) for x in argv[argv.index("--n-subs") + 1].split(",")) if "--n-subs" in argv else None
    keys = [a for i, a in enumerate(argv) if not a.startswith("--") and not (i > 0 and argv[i - 1].startswith("--"))] \
        or ["monkey_math", "monkey_codecontests"]
    ntag = "" if ns is None else "_n" + "-".join(map(str, ns))
    os.makedirs(OUT_DIR, exist_ok=True)
    for key in keys:
        v.configure(key)
        if ns is not None:
            v.N_SUBS = ns
        orc = v.oracle(key)
        cfg, q, models = orc["cfg"], orc["q"], orc["models"]
        P, M = orc["U_true"].shape[0], len(models)
        path = find_cache(key, M, ntag)
        runs = pickle.load(open(path, "rb")); print(f"[{key}] {len(runs)} replicates from {os.path.basename(path)}")
        ap_o, asp_o = orc["arb"]
        grid = np.linspace(max(cfg.get("perf_lo", 0.0), 0.02), float(ap_o.max()), G)
        cost_o = envelope_cost(ap_o, asp_o, grid) * P
        model_cost = np.array([envelope_cost(orc["model_perf"][m], orc["model_spend"][m], grid) for m in range(M)]) * P
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            market_o = np.nanmin(model_cost, axis=0)
        margin_o = (market_o - cost_o) / market_o * 100.0
        x_cut = terminal_cut(grid, margin_o)
        dom = np.isfinite(margin_o) & (margin_o >= 0.0)               # below the budget floor the frontier is not computed
        x_lo = float(grid[np.argmax(dom)] * 100.0) if dom.any() else None
        # Per-model aggregate pass@k vs k (Test A), for plots/fig_passk_curves.py. The cached
        # `model_perf_est` lives on the budget grid, which is log-spaced over four decades of spend
        # and so lands only ~4 points in 1 <= k <= 10 (none at k = 1) -- too coarse to draw the
        # small-k curves. So the curves are re-evaluated here on K_CURVE from the counts. The draws
        # are the SAME ones validate_calibration scored: subsample() is a pure function of the seed, and the
        # seed is fixed by (n_sub, rep) exactly as in validate_calibration.run_one.
        c_all, n_all = v.load(key)[3], v.load(key)[4]
        oracle_curve = np.array([v.chen_curve(c_all[:, j], n_all[:, j], K_CURVE).mean(0)
                                 for j in range(M)])
        T = np.array(v.TARGETS[key], float)
        cost_oT = envelope_cost(ap_o, asp_o, T) * P
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            market_oT = np.nanmin([envelope_cost(orc["model_perf"][m], orc["model_spend"][m], T) for m in range(M)], axis=0) * P
        for n_sub in sorted(set(r["n_sub"] for r in runs)):
            rs = [r for r in runs if r["n_sub"] == n_sub]
            x_extrap = extrapolation_start(ap_o, orc["kmax"], n_sub)
            rec = dict(key=key, title=cfg["title"], legend=LEGEND.get(key, key), xlabel=cfg["xlabel"],
                       n_sub=n_sub, n_draws=len(rs), n_problems=P, n_models=M, models=list(models), kcap=v.KCAP,
                       solve_rate=grid * 100.0, oracle_cost=cost_o, market_cost=market_o, oracle_margin=margin_o, x_lo=x_lo, x_cut=x_cut,
                       x_extrap=x_extrap,
                       k_curve=K_CURVE, model_pass_oracle=oracle_curve,
                       model_pass=curve_estimates(cfg, c_all, n_all, n_sub, [r["rep"] for r in rs]),
                       est={}, targets=dict(solve_rate=T * 100.0, oracle_cost=cost_oT, market_cost=market_oT,
                                            oracle_margin=(market_oT - cost_oT) / market_oT * 100.0, est={}))
            for mode in MODES:
                if mode not in rs[0]:
                    continue
                rec["est"][mode] = summarize(rs, mode, grid, cost_o, market_o, P)
                rec["targets"]["est"][mode] = summarize(rs, mode, T, cost_oT, market_oT, P)
            out = os.path.join(OUT_DIR, f"{key}_n{n_sub}.json")
            with open(out, "w") as f:
                json.dump(to_json(rec), f, separators=(",", ":"))
            print("wrote", os.path.normpath(out))
            # summary for the text: realized/oracle (regret), claimed/oracle (honesty), reach, at the fixed targets
            print(f"  n={n_sub}: target | " + " | ".join(f"{MODES[m]:>34s}" for m in rec['targets']['est']))
            print("         oracle margin | " + " | ".join(f"{'real/orc':>10s} {'claim/orc':>10s} {'reach':>6s} {'margin':>5s}" for _ in rec['targets']['est']))
            for ti, t in enumerate(T):
                cells = []
                for m, e in rec["targets"]["est"].items():
                    cells.append(f"{e['realized_ratio_med'][ti]:10.2f} {e['claimed_ratio_med'][ti]:10.2f} "
                                 f"{e['reach'][ti]*100:5.0f}% {e['realized_margin_med'][ti]:5.0f}")
                print(f"  {t*100:5.0f}%   {rec['targets']['oracle_margin'][ti]:5.0f}%      | " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
