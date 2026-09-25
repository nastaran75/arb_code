"""Shared protocol of the estimator experiments (Section 5.2 and Appendix D): data loading, the
budget -> attempts map, the pass@k curve tables of each estimator, exact subsampling of the recorded
attempts, the held-out oracle, and the scoring of an allocation under another curve table.

Reference ("oracle"): Chen et al. (2021) unbiased pass@k computed on the HELD-OUT attempts (all recorded
attempts minus the n_sub used for estimation), with every model capped at k <= KCAP attempts
(k <= n/2 keeps the estimator low-variance). The cap is a per-model box constraint in the optimizer,
applied identically to oracle and estimated curves. Subsample n_sub attempts per (problem, model) cell
(hypergeometric, i.e. exact subsampling of the recorded attempts), re-estimate the curves, run the
arbitrage on the estimate and score the chosen allocations under the held-out curves:
        oracle   = arbitrage on the held-out curves
        claimed  = arbitrage on estimated curves, self-scored
        realized = estimated allocations scored on the held-out curves (what a user gets)

Attempt costs `q` are per (problem, model), (P, M): the Monkey Business builders write a per-problem
cost (2 * params * that problem's mean generation length); the IRSL sets carry one cost per model
(active parameters), stored per row all the same. A (M,) vector is accepted everywhere and broadcast.

Drivers: validate_calibration.py (replicates -> val_cache/), make_fig_passk_data.py (records for
plots/fig_calib.py), plots/fig_gap_dist.py (estimation error).
"""
from __future__ import annotations

import os
# one BLAS thread per process: the Pool already uses all cores and the arrays are tiny, so
# multi-threaded BLAS only oversubscribes the machine (must be set before numpy is imported)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob

import numpy as np
from scipy.special import gammaln

import arbitrage as arb
import arbitrage_tab as at
import estimator as est
from make_figure import DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
N_SUBS = (2, 4, 8, 16)       # calibration sizes (attempts per cell) drawn in the paper
R = 10                       # replicates per n_sub
N_FRONTIER = 70              # lambda points per frontier (arbitrage_tab.arbitrage_frontier default n)
TARGETS = {"monkey_math": (0.5, 0.7, 0.8, 0.9, 0.95),
           "monkey_codecontests": (0.1, 0.15, 0.2, 0.25, 0.3, 0.35),
           # 30-problem IRSL sets (union-solvable ceilings 93.3% / 90.0% / 100% / 100%)
           "resmat2_aime2024": (0.4, 0.5, 0.6, 0.7, 0.8),
           "resmat2_aime2025": (0.4, 0.5, 0.6, 0.7, 0.8),
           "resmat2_gmmlu": (0.6, 0.7, 0.8, 0.9, 0.95),
           "resmat2_mmlupro": (0.6, 0.7, 0.8, 0.9, 0.95)}
CACHE = os.environ.get("VAL_CACHE", os.path.join(HERE, "val_cache"))
N_BUDGET = 64
SEED = 0
KCAP = 5000.0                # max attempts per model (train and eval)
# Per-dataset protocol. The Monkey Business sets (10,000 attempts per cell) use the defaults above; the
# IRSL sets (2,048-2,560 attempts per cell) hold out ~2,500 attempts for the Chen reference, so the
# extrapolation cap scales down with it (same kcap <= n_rest / 2 rule).
VAL_SETTINGS = {
    "monkey_math": dict(n_subs=(2, 4, 8, 16), kcap=5000.0),
    "monkey_codecontests": dict(n_subs=(2, 4, 8, 16), kcap=5000.0),
    "resmat2_aime2024": dict(n_subs=(2, 4, 8, 16), kcap=1000.0),
    "resmat2_aime2025": dict(n_subs=(2, 4, 8, 16), kcap=1000.0),
    "resmat2_gmmlu": dict(n_subs=(2, 4, 8, 16), kcap=1000.0),
    "resmat2_mmlupro": dict(n_subs=(2, 4, 8, 16), kcap=1000.0),
}
# Fractional attempts (k < 1) as a randomized purchase, u(k) = k * u(1). Off in the paper: every attempt is cheap
# relative to the budgets of interest.
LINEAR_BELOW_ONE = False
# The plug-in baseline of Section 5.2 is 1 - (1 - c/n)^k, EXTRAPOLATED past the n observed attempts: that is
# what an arbitrageur holding n samples would report at any k. (`estimator.geom_curve` defaults to the other
# convention, frozen at k = n, which is the right one for the Chen curves of the market experiments.)
GEOM_EXTRAPOLATE = True


def q_rows(q, R):
    """(R, M) attempt costs: a (M,) vector is broadcast over the R rows, a (R, M) matrix is used as is."""
    q = np.asarray(q, float)
    return np.broadcast_to(q[None, :], (R, q.size)) if q.ndim == 1 else q


def model_cap(kcap, q):
    """(M,) budget beyond which every problem's curve is flat: kcap attempts at the model's highest price."""
    return np.asarray(kcap, float) * np.max(q_rows(q, 1), axis=0)


def kmax_of(allocs, q):
    """(F,) most attempts any (problem, model) buys under each allocation row of `allocs` (F, M)."""
    return (np.asarray(allocs)[:, None, :] / q_rows(q, 1)[None]).max((1, 2))


def cap_cells(U, budget, q, ncap):
    """Flatten each cell's curve beyond its own attempt cap: U[i, j, b] = U_ij(min(b, ncap_ij * q_ij)) (in place).
    ncap (P, M); +inf = no per-cell cap. Used for the fixed models (cap = observed n_ij, as in the paper).
    q: (M,) or (P, M), see q_rows."""
    P, M, B = U.shape
    qq = q_rows(q, P)
    for j in range(M):
        cap_b = ncap[:, j] * qq[:, j]
        if not np.any(np.isfinite(cap_b)):
            continue
        idx = np.clip(np.searchsorted(budget, cap_b), 1, B - 1)
        b0, b1 = budget[idx - 1], budget[idx]
        w = np.clip((cap_b - b0) / (b1 - b0), 0.0, 1.0)
        rows = np.arange(P)
        Ucap = (1 - w) * U[rows, j, idx - 1] + w * U[rows, j, idx]
        beyond = budget[None, :] > cap_b[:, None]
        U[:, j, :] = np.where(beyond, Ucap[:, None], U[:, j, :])
    return U


def linearize_below_one(U, budget, q):
    """U[r, m, b] <- (k/1) * U(k = 1) wherever k = budget / q[r, m] < 1 (in place; returns U).
    q: (M,) or (R, M), see q_rows; rows whose one-attempt price is off the grid are left alone."""
    R, M, B = U.shape
    qq = q_rows(q, R)
    for j in range(M):
        qj = qq[:, j]
        k = budget[None, :] / qj[:, None]                                        # (R, B)
        rows = np.flatnonzero((qj <= budget[-1]) & (k < 1).any(1))
        if rows.size == 0:
            continue
        i = np.clip(np.searchsorted(budget, qj[rows]), 1, B - 1); b0, b1 = budget[i - 1], budget[i]
        w = (qj[rows] - b0) / (b1 - b0)
        U1 = (1 - w) * U[rows, j, i - 1] + w * U[rows, j, i]                    # per-row u at exactly one attempt
        sub = k[rows] < 1
        U[rows, j, :] = np.where(sub, k[rows] * U1[:, None], U[rows, j, :])
    return U


def kcap_models(cfg, M):
    """Per-model attempt cap (M,): KCAP, or kcap_fixed for the fixed (all-attempts) models."""
    return np.asarray(cfg.get("kcap_m", np.full(M, KCAP)), float)


def configure(key):
    """Apply the per-dataset protocol (N_SUBS, KCAP) for `key`; call before load()/oracle()/run_replicates()."""
    global N_SUBS, KCAP, LINEAR_BELOW_ONE
    s = VAL_SETTINGS.get(key, {})
    N_SUBS = tuple(s.get("n_subs", (5, 10, 50)))
    KCAP = float(s.get("kcap", 5000.0))
    LINEAR_BELOW_ONE = bool(s.get("linear_below_one", False))
    return s


def cache_tag():
    """Suffix identifying the protocol conventions in cache file names."""
    return "_lin" if LINEAR_BELOW_ONE else ""



_DATA = {}


def load(key):
    """Load + count once per process; the parent calls this before forking the Pool so workers inherit it."""
    if key not in _DATA:
        _DATA[key] = _load(key)
    return _DATA[key]


def _load(key):
    cfg = DATASETS[key]
    s = VAL_SETTINGS.get(key, {})
    base = cfg["base_dir"]
    models = s.get("models") or cfg["models"] or sorted(os.path.basename(p)[:-6] for p in glob.glob(base + "/*.jsonl"))
    res = arb.load_results(base)
    by_id = {m: {r["id"]: r for r in res[m]} for m in models}
    ids = sorted(set.intersection(*[set(by_id[m]) for m in models]))
    fixed = np.array([m in s.get("fixed", ()) for m in models])
    if "min_n" in s:                                  # problems where every reference model has enough attempts to hold out
        ids = [i for i in ids if all(len(by_id[m][i]["attempts"]) >= s["min_n"] for m, f in zip(models, fixed) if not f)]
    q = np.array([[by_id[m][i]["mean_cost"] for m in models] for i in ids])            # (P, M) price per attempt
    c, n = est.counts(res, models, ids)
    ncap = np.where(fixed[None, :], n, np.inf)                     # fixed cells: flat beyond their observed attempts
    cfg = dict(cfg, fixed=fixed, kcap_m=np.where(fixed, n.max(0), KCAP), ncap=ncap)
    hi = s["bmax"] if "bmax" in s else cfg["bmax"] * 20
    budget = np.logspace(np.log10(cfg["bmin"] * 0.5), np.log10(hi), N_BUDGET)
    return cfg, models, q, c, n, budget


def chen_curve(c_col, n_col, k):
    """Chen et al. unbiased pass@k, extended to real k via gammaln. (P,B)."""
    c_col = c_col[:, None]; n_col = n_col[:, None]; kk = k[None, :] if k.ndim == 1 else k   # (B,) or (P, B)
    with np.errstate(invalid="ignore"):
        logr = (gammaln(n_col - c_col + 1) - gammaln(np.maximum(n_col - c_col - kk + 1, 1e-300))
                - gammaln(n_col + 1) + gammaln(np.maximum(n_col - kk + 1, 1e-300)))
    r = np.where(kk <= n_col - c_col, np.exp(logr), 0.0)
    return 1.0 - r


def curves_from_counts(c, n, q, budget, mode, kcap=None, ncap=None, extrapolate=None):
    """U[p, m, b] from count arrays, flat beyond kcap attempts (default: the current KCAP) and, if given, beyond the
    per-cell caps ncap (P, M). mode in {'geom', 'zibb', 'chen'}.
    `extrapolate` (default: GEOM_EXTRAPOLATE) only affects 'geom': True = 1 - (1 - c/n)^k at any k."""
    P, M = c.shape
    kc = np.broadcast_to(np.asarray(KCAP if kcap is None else kcap, float), (M,))
    ex = GEOM_EXTRAPOLATE if extrapolate is None else bool(extrapolate)
    U = np.empty((P, M, budget.size))
    qq = q_rows(q, P)
    for j in range(M):
        k = np.minimum(budget[None, :] / qq[:, j, None], kc[j])                  # (P, B) attempts affordable
        if mode == "geom":
            U[:, j, :] = est.geom_curve(c[:, j], n[:, j], k, extrapolate=ex)
        elif mode == "zibb":
            a, b, pi0 = est.fit_zibb(c[:, j], n[:, j])
            U[:, j, :] = est.zibb_curve(c[:, j], n[:, j], k, a, b, pi0)
        elif mode == "chen":
            U[:, j, :] = chen_curve(c[:, j], n[:, j], k)
        elif mode == "kaz":
            # Kazdan et al. (2025): population beta-binomial by plain type-II MLE. One curve per
            # model, assigned to EVERY query -- s_ij never enters the prediction.
            from scipy.special import betaln as _bl
            a_, b_ = est.fit_bb(c[:, j], n[:, j], prior_pow=0.0)
            U[:, j, :] = 1.0 - np.exp(_bl(a_, b_ + k) - _bl(a_, b_))              # (P, B): k is per problem
        else:
            raise ValueError(mode)
    if LINEAR_BELOW_ONE:
        U = linearize_below_one(U, budget, q)
    return cap_cells(U, budget, q, ncap) if ncap is not None else U



def _frontier(U, budget, cap):
    """arbitrage_frontier (numpy); always returns numpy arrays."""
    return at.arbitrage_frontier(U, budget, n=N_FRONTIER, cap=cap)


def subsample(c, n, n_sub, rng, fixed=None):
    """Draw n_sub attempts without replacement from each (problem, model) cell.
    Returns (c_sub, n_sub_arr, c_rest, n_rest): the draw and the held-out remainder.
    Columns flagged in `fixed` are not subsampled: all attempts are used on both sides."""
    ns = np.minimum(n, n_sub).astype(int)                       # fixed columns may have n < n_sub (overwritten below)
    c_sub = rng.hypergeometric(c.astype(int), (n - c).astype(int), ns).astype(float)
    n_sub_arr = ns.astype(float)
    c_rest, n_rest = c - c_sub, n - n_sub_arr
    if fixed is not None and np.any(fixed):
        f = np.asarray(fixed, bool)
        c_sub[:, f] = c[:, f]; n_sub_arr[:, f] = n[:, f]; c_rest[:, f] = c[:, f]; n_rest[:, f] = n[:, f]
    return c_sub, n_sub_arr, c_rest, n_rest


def score(U_ref, cumsp_ref, budget, allocs):
    perf = np.array([at.pool_perf(U_ref, budget, a) for a in allocs])
    spend = np.array([at.expected_spend(U_ref, cumsp_ref, budget, a) for a in allocs])
    return perf, spend


def oracle(key):
    cfg, models, q, c, n, budget = load(key)
    kcap = kcap_models(cfg, len(models))
    ncap = cfg.get("ncap")
    U_true = curves_from_counts(c, n, q, budget, "chen", kcap, ncap)       # Chen unbiased, k <= KCAP (fixed cells: n_ij)
    U_geom = curves_from_counts(c, n, q, budget, "geom", kcap, ncap)
    ap, asp, allocs = _frontier(U_true, budget, model_cap(kcap, q))
    perf_t, spend_t = at.single_model(U_true, budget)
    perf_ch, _ = at.single_model(U_geom, budget)
    kmax = kmax_of(allocs, q)
    return dict(cfg=cfg, models=models, q=q, budget=budget, U_true=U_true, arb=(ap, asp), allocs=allocs,
                kmax=kmax, model_perf=perf_t, model_spend=spend_t, model_perf_chen=perf_ch)

