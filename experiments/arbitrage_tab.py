"""Arbitrage on *tabulated* per-(problem,model) pass curves U[p,m,b].

Same objective and lambda-sweep solver as `arbitrage.py`, but on arbitrary tabulated
curves (from `estimator.pass_curves`, per-problem attempt costs allowed) rather than the
closed-form exponential model. THIS IS THE PATH THE PAPER'S FIGURES USE, with the
Chen unbiased pass@k curves ('chen'); the geometric curves reproduce `arbitrage.py`
to <1%.

Conventions (see Appendix C of the paper):
  - spend: `expected_spend(..., order="greedy")` (DEFAULT) = expected spend under the
    fixed Section-3 greedy population order (`greedy_order`, `spend_fixed`);
    `order="oracle"` = per-instance index-rule order (`oracle_spend`, needs p_i(x) of the
    query; reference only). `best_fixed_order` / `local_search_order` find the best fixed
    order for diagnostics.
  - solver: `arbitrage_frontier(robust=True)` takes, in each coordinate step, the exact
    argmax over the budget grid instead of assuming a monotone marginal; required for the
    Chen curves, whose failure probability is log-concave in k.
  - `lambda_grid`, `lams=`, `c_inits=`: a shared shadow-price grid and warm starts, used by
    `cv.py` to refit each cross-validation fold from the full-data solution.
"""
from __future__ import annotations

import numpy as np


def _interp_scalar(curve_PB: np.ndarray, budget: np.ndarray, x: float) -> np.ndarray:
    """Interpolate each row's curve (shared grid `budget`) at scalar `x`."""
    if x <= budget[0]:
        f = x / budget[0] if budget[0] > 0 else 0.0
        return curve_PB[:, 0] * f
    if x >= budget[-1]:
        return curve_PB[:, -1]
    hi = int(np.searchsorted(budget, x))
    lo = hi - 1
    f = (x - budget[lo]) / (budget[hi] - budget[lo])
    return curve_PB[:, lo] + f * (curve_PB[:, hi] - curve_PB[:, lo])


# Per-cell spend cap, (P, M) dollars, or None. Set by the curve builders (cv.curves, make_panel.compute)
# to n*q -- the price of the n RECORDED attempts -- whenever the utility curves are frozen at n
# (estimators chen / chen_lin / chen_int / geom). Beyond the cap the frozen curve is flat, so money
# spent there buys nothing; with the cap no money is spent past the n-th attempt, and every reported
# cost is the bill of the implementable policy "at most n attempts per model per query". Left None
# for extrapolated curves (chen_bb, bb, zibb), where attempts past n are meaningful.
SPEND_CAP = None


def cum_spend(U: np.ndarray, budget: np.ndarray, cap: np.ndarray = None) -> np.ndarray:
    """Expected spend to commit budget b on a model: int_0^b (1-U) dt. (P,B)->(P,B).
    `cap` (P,): stop spending at that budget per row (see SPEND_CAP)."""
    fbar = 1.0 - U
    b0 = np.concatenate([[0.0], budget])
    f0 = np.concatenate([np.ones((U.shape[0], 1)), fbar], axis=1)
    dt = np.diff(b0)
    seg = 0.5 * (f0[:, 1:] + f0[:, :-1]) * dt[None, :]
    cs = np.cumsum(seg, axis=1)
    if cap is None:
        return cs
    bb = np.minimum(budget[None, :], np.asarray(cap, dtype=float)[:, None])          # (P,B) clamped budget
    return np.stack([np.interp(bb[p], budget, cs[p]) for p in range(U.shape[0])])


def _cap(m):
    return None if SPEND_CAP is None else SPEND_CAP[:, m]


def single_model(U: np.ndarray, budget: np.ndarray):
    """Per-model aggregate (perf(budget), spend(budget)). U: (P,M,B)."""
    perf = U.mean(0)                                   # (M,B)
    spend = np.array([cum_spend(U[:, m, :], budget, _cap(m)).mean(0) for m in range(U.shape[1])])
    return perf, spend                                 # (M,B), (M,B)


def pool_perf(U, budget, c):
    fbar = np.stack([1.0 - _interp_scalar(U[:, m, :], budget, c[m]) for m in range(U.shape[1])], 1)
    return float(np.mean(1.0 - np.prod(fbar, axis=1)))


def _cells(U, cumsp, budget, c):
    """Per-cell (u, R) at allocation c: solve prob and expected spend given reached. (P,M) each."""
    M = U.shape[1]
    u = np.stack([_interp_scalar(U[:, m, :], budget, c[m]) for m in range(M)], 1)
    R = np.stack([_interp_scalar(cumsp[m], budget, c[m]) for m in range(M)], 1)
    return u, R


def spend_fixed(U, cumsp, budget, c, order):
    """Expected spend under ONE population order (deployable convention)."""
    u, R = _cells(U, cumsp, budget, c)
    order = np.asarray(order)
    logf = np.log(np.clip(1.0 - u, 1e-12, 1.0))[:, order]
    R = R[:, order]
    reach = np.exp(np.cumsum(logf, axis=1) - logf)
    return float((reach * R).sum(1).mean())


def greedy_order(U, cumsp, budget, c):
    """Section-3 greedy population order on tabulated curves (see arbitrage.greedy_order)."""
    u, R = _cells(U, cumsp, budget, c)
    M = U.shape[1]
    funded = [i for i in range(M) if c[i] > 0]
    rho = np.ones(U.shape[0])
    order = []
    while funded:
        best, best_val = None, -np.inf
        for i in funded:
            den = float((rho * R[:, i]).mean())
            val = float((rho * u[:, i]).mean()) / den if den > 0 else -np.inf
            if val > best_val:
                best, best_val = i, val
        if best is None:                     # no candidate has positive expected spend: order is immaterial
            order += funded
            break
        order.append(best)
        funded.remove(best)
        rho = rho * (1.0 - u[:, best])
    return np.array(order + [i for i in range(M) if c[i] <= 0])


def oracle_spend(U, cumsp, budget, c):
    """Per-instance optimal order (index rule: decreasing u_i(x)/R_i(x), success probability
    per expected dollar; equals decreasing alpha_i(x) for geometric curves). Needs p_i(x)
    of the incoming query, so it is an upper bound on what a deployable order achieves."""
    u, R = _cells(U, cumsp, budget, c)
    ratio = np.where(R > 0, u / np.where(R > 0, R, 1.0), -1.0)
    order = np.argsort(-ratio, axis=1)
    logf = np.log(np.clip(1.0 - u, 1e-12, 1.0))
    logf, R = np.take_along_axis(logf, order, axis=1), np.take_along_axis(R, order, axis=1)
    reach = np.exp(np.cumsum(logf, axis=1) - logf)
    return float((reach * R).sum(1).mean())


def best_fixed_order(U, cumsp, budget, c, max_funded=9):
    """Exhaustive search over permutations of the funded models. (order, spend); (None, nan) if too many."""
    import itertools
    M = U.shape[1]
    funded = [i for i in range(M) if c[i] > 0]
    rest = [i for i in range(M) if c[i] <= 0]
    if len(funded) > max_funded:
        return None, np.nan
    u, R = _cells(U, cumsp, budget, c)
    logf = np.log(np.clip(1.0 - u, 1e-12, 1.0))[:, funded]
    Rf = R[:, funded]
    best, best_val = None, np.inf
    for perm in itertools.permutations(range(len(funded))):
        p = list(perm)
        lf, RR = logf[:, p], Rf[:, p]
        val = float((np.exp(np.cumsum(lf, 1) - lf) * RR).sum(1).mean())
        if val < best_val:
            best, best_val = p, val
    return np.array([funded[i] for i in best] + rest), best_val


def local_search_order(U, cumsp, budget, c, order, sweeps=50):
    """Adjacent-swap descent from `order` (when too many funded models to enumerate)."""
    order = np.array(order)
    cur = spend_fixed(U, cumsp, budget, c, order)
    for _ in range(sweeps):
        improved = False
        for k in range(len(order) - 1):
            cand = order.copy(); cand[k], cand[k + 1] = cand[k + 1], cand[k]
            v = spend_fixed(U, cumsp, budget, c, cand)
            if v < cur - 1e-15:
                order, cur, improved = cand, v, True
        if not improved:
            break
    return order


def expected_spend(U, cumsp, budget, c, order="greedy"):
    """Expected spend of allocation c: 'greedy' = fixed Section-3 population order
    (deployable, DEFAULT, matches arbitrage.py); 'oracle' = per-instance index-rule order
    (needs p_i(x) of the query)."""
    if order == "greedy":
        return spend_fixed(U, cumsp, budget, c, greedy_order(U, cumsp, budget, c))
    if order == "oracle":
        return oracle_spend(U, cumsp, budget, c)
    if order == "best":
        # Cost-minimising FIXED population order: exhaustive over the funded models, falling
        # back to adjacent-swap descent from greedy when there are too many to enumerate
        # (so the returned value is an upper bound on the optimum in that regime).
        o, val = best_fixed_order(U, cumsp, budget, c)
        if o is None:
            o = local_search_order(U, cumsp, budget, c, greedy_order(U, cumsp, budget, c))
            val = spend_fixed(U, cumsp, budget, c, o)
        return val
    raise ValueError(f"unknown order convention {order!r}")


def _coord_ascent(U, dUdb, budget, lam, c0, sweeps=40, cap=None, robust=False):
    """Cyclic coordinate ascent of mean_p[1 - prod_m (1-U_pm(c_m))] - lam*sum(c).

    Default step assumes the marginal Mg(b) is decreasing in b (log-convex failure,
    e.g. 'geom') and interpolates the crossing Mg == lam. `robust=True` instead takes
    the exact argmax of the 1-D objective over the budget grid (the interpolated
    objective is piecewise linear, so the grid argmax is exact)
    M = U.shape[1]
    c = c0.copy()
    u_cur = np.stack([_interp_scalar(U[:, m, :], budget, c[m]) for m in range(M)], 1)
    logfbar = np.log(np.clip(1.0 - u_cur, 1e-12, 1.0))
    P = U.shape[0]
    for _ in range(sweeps):
        delta = 0.0
        tot = logfbar.sum(1)
        for i in range(M):
            w = np.exp(tot - logfbar[:, i])                     # others fail (P,)
            if robust:
                gain = (w @ U[:, i, :]) / P - lam * budget      # objective gain vs c_i = 0, per grid point
                if cap is not None:
                    gain = np.where(budget <= cap[i], gain, -np.inf)
                b = int(np.argmax(gain))
                x = float(budget[b]) if gain[b] > 0.0 else 0.0
            else:
                Mg = (w[:, None] * dUdb[:, i, :]).mean(0)       # marginal per grid budget (B,)
                # Mg is decreasing in budget; find c_i where Mg == lam
                if Mg[0] <= lam:
                    x = 0.0
                elif Mg[-1] >= lam:
                    x = float(budget[-1])
                else:
                    b = int(np.searchsorted(-Mg, -lam))         # first index with Mg<lam
                    lo, hi = b - 1, b
                    denom = Mg[lo] - Mg[hi]
                    f = (Mg[lo] - lam) / denom if denom != 0 else 0.0
                    x = float(budget[lo] + f * (budget[hi] - budget[lo]))
                if cap is not None:
                    x = min(x, float(cap[i]))
            delta = max(delta, abs(x - c[i]))
            c[i] = x
            u_i = _interp_scalar(U[:, i, :], budget, x)
            new_lf = np.log(np.clip(1.0 - u_i, 1e-12, 1.0))
            tot += new_lf - logfbar[:, i]
            logfbar[:, i] = new_lf
        if delta < 1e-10:
            break
    return c


def lambda_grid(U, budget, n=70, n_head=0, robust=False):
    """Shadow-price grid of `arbitrage_frontier` (high -> low), from lam_hi = largest mean
    marginal utility per dollar. `n_head` extra points just below lam_hi make the frontier
    continuous down to zero budget (as in arbitrage.arbitrage_frontier)."""
    dUdb = np.gradient(U, budget, axis=2)
    M = U.shape[1]
    lam_hi = float(max(dUdb[:, m, 0].mean() for m in range(M)))
    if robust:                                     # non-monotone marginals: start from the largest grid slope
        lam_hi = max(lam_hi, float(max(dUdb[:, m, :].mean(0).max() for m in range(M))))
    head = lam_hi * (1.0 - np.logspace(-4, np.log10(0.8), n_head)) if n_head > 0 else np.empty(0)
    return np.concatenate([head, np.logspace(np.log10(lam_hi * (0.2 if n_head > 0 else 1.0)), np.log10(lam_hi * 1e-6), n)])


def arbitrage_frontier(U, budget, n=70, cap=None, order="greedy", robust=False, n_head=0,
                       lams=None, c_inits=None, sweeps=40):
    """Trace optimal arbitrage frontier by sweeping lambda. Returns
    (perf, spend, allocations). `cap` (M,) optionally bounds each model's budget.
    `order`: spend convention (see `expected_spend`); `robust`: see `_coord_ascent`;
    `n_head`: see `lambda_grid`. `lams` overrides the grid; `c_inits` (len(lams), M)
    warm-starts each lambda from a given allocation (e.g. the full-data solution when
    refitting on a cross-validation fold) instead of from the previous lambda."""
    dUdb = np.gradient(U, budget, axis=2)
    M = U.shape[1]
    cumsp = [cum_spend(U[:, m, :], budget, _cap(m)) for m in range(M)]
    if lams is None:
        lams = lambda_grid(U, budget, n=n, n_head=n_head, robust=robust)
    n = lams.size
    perf = np.empty(n); spend = np.empty(n); allocs = np.empty((n, M))
    c = np.zeros(M)
    for k, lam in enumerate(lams):
        c0 = c if c_inits is None else np.asarray(c_inits[k], dtype=float)
        c = _coord_ascent(U, dUdb, budget, lam, c0, cap=cap, robust=robust, sweeps=sweeps)
        perf[k] = pool_perf(U, budget, c)
        spend[k] = expected_spend(U, cumsp, budget, c, order=order)
        allocs[k] = c
    return perf, spend, allocs


def invert_to_cost(perf_curve, cost_curve, target_perf):
    order = np.argsort(perf_curve)
    pc, cc = perf_curve[order], cost_curve[order]
    out = np.interp(target_perf, pc, cc, left=np.nan, right=np.nan)
    out[target_perf > pc[-1] + 1e-12] = np.nan
    out[target_perf < pc[0] - 1e-12] = np.nan
    return out
