"""Section-2 arbitrage: optimal query-independent budget allocation.

Implements the allocation problem of Section 2 of the paper:

Under repeated sampling, model i solves query x with a single-attempt success
probability p_i(x) at per-attempt cost q_i(x), so a budget c_i buys k = c_i / q_i(x)
attempts and

    u_i(x; c_i) = 1 - (1 - p_i(x))^k = 1 - exp(-z_i(x; c_i)),
    z_i(x; c_i) = min(alpha_i(x) c_i, zcap_i(x)),
    alpha_i(x)  = -log(1 - p_i(x)) / q_i(x)  >= 0.

p_i(x) is estimated from the n_i(x) attempts we actually observed, and BY DEFAULT
WE DO NOT EXTRAPOLATE PAST THEM: the cap zcap_i(x) = n_i(x) * (-log(1 - p_i(x)))
freezes the solve probability at its value after n_i(x) attempts, so the
(n+1)-th attempt costs money but adds nothing. `build_alpha(..., extrapolate=True)`
sets zcap = +inf and recovers the plain geometric extrapolation 1 - (1-p)^k.

The arbitrageur picks a single allocation c = (c_1, ..., c_N) >= 0 with
sum_i c_i <= C maximizing the portfolio ("at least one model solves it") utility

    U(c) = E_x[ 1 - prod_i (1 - u_i(x; c_i)) ] = E_x[ 1 - exp(-sum_i z_i(x; c_i)) ].

sum_i z_i(x; c_i) is concave in c (a sum of min(linear, constant)) and 1 - e^{-z}
is concave increasing, so U is monotone and jointly concave (Thm. KKT) with or
without the cap; the Lagrangian coordinate ascent below finds the global optimum.

WHAT THE PAPER'S FIGURES USE (Appendix C, decided 2026-08-27). This module is the
closed-form GEOMETRIC model, the object of the theory. The reported figures instead run
the same lambda-sweep on TABULATED curves (`arbitrage_tab.py`) built from the unbiased
pass@k estimator of Chen et al. (`estimator.py`, mode 'chen': unbiased at integer
attempts, constant hazard within an attempt), because the geometric plug-in is biased
pessimistic near k = n and inflates the arbitrageur's margin. Conventions shared by both
paths:
  - deployment order: the greedy population order of the paper's order section
    (`greedy_order` / `deployed_spend(order="greedy")`, the DEFAULT of
    `arbitrage_frontier`); the per-instance order (`expected_spend`) is an oracle that
    needs p_i(x) of the incoming query and is kept only as a reference;
  - evaluation: cross-validated over problems (`cv.py`); in-sample curves are used only
    as optimizer diagnostics.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_results(base_dir: str, rename: dict | None = None) -> dict[str, list[dict]]:
    """Load one `<model>.jsonl` per file. Each row: {id, attempts:[0/1...], mean_cost}."""
    results = {}
    for file in sorted(os.listdir(base_dir)):
        if not file.endswith(".jsonl"):
            continue
        name = file[: -len(".jsonl")]
        if rename is not None:
            if name not in rename:
                continue
            name = rename[name]
        results[name] = [json.loads(line) for line in open(os.path.join(base_dir, file))]
    return results


def build_alpha(
    results: dict[str, list[dict]],
    models: Sequence[str],
    cost_override: dict[str, float] | None = None,
    extrapolate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Return (alpha, zcap, q, problem_ids).

    alpha : (P, N) exponential-model rate alpha_i(x) = -log(1-p_i(x))/q_i(x).
    zcap  : (P, N) cap on the log-hazard z_i(x) = alpha_i(x) c_i, equal to
            n_i(x) * (-log(1-p_i(x))): the solve probability stops improving after
            the n_i(x) observed attempts (no extrapolation). +inf everywhere when
            `extrapolate=True` (plain geometric extrapolation).
    q     : (P, N) per-attempt cost q_i(x).
    p_i(x) is the empirical one-shot accuracy s/n, capped at 1 - 1/(2n) so that a
    fully-solved problem yields a large-but-finite rate rather than +inf.
    """
    by_id = {m: {r["id"]: r for r in results[m]} for m in models}
    ids = sorted(set.intersection(*[set(by_id[m]) for m in models]))

    P, N = len(ids), len(models)
    alpha = np.zeros((P, N))
    zcap = np.full((P, N), np.inf) if extrapolate else np.zeros((P, N))
    q = np.zeros((P, N))
    for j, m in enumerate(models):
        for i, pid in enumerate(ids):
            row = by_id[m][pid]
            att = row["attempts"]
            n = len(att)
            s = int(sum(att))
            p = min(s / n, 1.0 - 1.0 / (2 * n)) if n > 0 else 0.0
            cost = cost_override[m] if cost_override and m in cost_override else row["mean_cost"]
            h = -np.log1p(-p) if p > 0 else 0.0          # per-attempt log-hazard
            q[i, j] = cost
            alpha[i, j] = h / cost
            if not extrapolate:
                zcap[i, j] = n * h
    return alpha, zcap, q, ids


# --------------------------------------------------------------------------- #
# Portfolio utility, gradient, projection
# --------------------------------------------------------------------------- #
def _z(alpha: np.ndarray, zcap: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per-cell log-hazard z_i(x; c_i) = min(alpha_i(x) c_i, zcap_i(x)). (P, N)."""
    return np.minimum(alpha * c[None, :], zcap)


def _xmax(ai: np.ndarray, zi: np.ndarray) -> float:
    """Spend past which no cell of a model improves: max_x zcap/alpha = max_x n_i(x) q_i(x)
    (+inf when uncapped; 0 if the model never solves anything)."""
    m = ai > 0
    return float(np.max(zi[m] / ai[m])) if m.any() else 0.0


def utility(alpha: np.ndarray, zcap: np.ndarray, c: np.ndarray) -> float:
    return float(np.mean(1.0 - np.exp(-_z(alpha, zcap, c).sum(1))))


def grad(alpha: np.ndarray, zcap: np.ndarray, c: np.ndarray) -> np.ndarray:
    z = _z(alpha, zcap, c)
    e = np.exp(-z.sum(1))                        # (P,) failure prob of the pool on x
    active = alpha * c[None, :] < zcap           # cells still below their sample cap
    return (alpha * active * e[:, None]).mean(0)  # E_x[ alpha_i(x) 1{active} e^{-sum z} ]


def project_capped_simplex(v: np.ndarray, C: float) -> np.ndarray:
    """Euclidean projection onto {c >= 0, sum_i c_i <= C}."""
    w = np.maximum(v, 0.0)
    if w.sum() <= C:
        return w
    # Project onto the simplex {c >= 0, sum c = C} (Duchi et al., 2008).
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - C
    idx = np.arange(1, v.size + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def _coord_ascent(alpha: np.ndarray, zcap: np.ndarray, C: float, lam: float, c0: np.ndarray,
                  sweeps: int = 200) -> np.ndarray:
    """Inner solve: cyclic coordinate ascent of the Lagrangian at fixed lambda.

    Each 1-D subproblem max_{x>=0} E_x[1 - e^{-(z_{-i} + min(alpha_i x, zcap_i))}] - lam*x
    is concave with derivative h'(x) = E_x[alpha_i w e^{-alpha_i x} 1{alpha_i x < zcap_i}] - lam
    monotone non-increasing (it drops to -lam once every cell has used up its observed
    attempts, i.e. for x >= max_x zcap_i/alpha_i), solved by bisection (w = failure prob
    of the others).
    """
    P, M = alpha.shape
    c = c0.copy()
    Z = _z(alpha, zcap, c)
    z = Z.sum(1)
    for _ in range(sweeps):
        delta = 0.0
        for i in range(M):
            ai, zi = alpha[:, i], zcap[:, i]
            aw = ai * np.exp(-(z - Z[:, i]))       # alpha_i * P(all other models fail)
            hi = min(C, _xmax(ai, zi))

            def hp(x):
                act = ai * x < zi
                return float((aw * act) @ np.exp(-x * ai)) / P - lam

            if hp(0.0) <= 0.0:
                x = 0.0
            elif hp(hi) > 0.0:
                x = hi                             # a model can absorb the whole budget
            else:
                lo = 0.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if hp(mid) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                x = 0.5 * (lo + hi)
            zi_new = np.minimum(ai * x, zi)
            delta = max(delta, abs(x - c[i]))
            z += zi_new - Z[:, i]
            Z[:, i] = zi_new
            c[i] = x
        if delta < 1e-12 * (C + 1e-12):
            break
    return c


def optimize(alpha: np.ndarray, zcap: np.ndarray, C: float, c0: np.ndarray | None = None) -> np.ndarray:
    """max_{c>=0, sum c<=C} U(c) via Lagrangian dual bisection (KKT-certified;
    the objective is concave so the KKT point is the global maximizer).

    lambda is the shadow price of a dollar (marginal utility per $). We bisect it
    so the budget binds; for lambda above max_i E_x[alpha_i(x)] no model is funded.
    """
    M = alpha.shape[1]
    lam_hi0 = float(max(alpha[:, i].mean() for i in range(M)))
    lam_lo, lam_hi = 0.0, lam_hi0
    c = np.zeros(M) if c0 is None else c0.copy()
    for _ in range(60):
        lam = 0.5 * (lam_lo + lam_hi)
        c = _coord_ascent(alpha, zcap, C, lam, c)  # warm-started
        if c.sum() > C:
            lam_lo = lam                            # too cheap -> raise price
        else:
            lam_hi = lam
    return _coord_ascent(alpha, zcap, C, lam_hi, c)


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def single_model_perf(alpha_col: np.ndarray, zcap_col: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Utility when the whole budget goes to one model, evaluated on `budgets`."""
    return np.array([np.mean(1.0 - np.exp(-np.minimum(alpha_col * b, zcap_col))) for b in budgets])


def _cell_spend(alpha, zcap, c):
    """Expected spend to commit budget c to a model on each problem (early stopping on
    success): int_0^c e^{-z(t)} dt with z(t) = min(alpha t, zcap), i.e.

        (1 - e^{-z}) / alpha  +  (c - z/alpha) e^{-z},      z = min(alpha c, zcap).

    The second term is the money spent on attempts past the observed n, paid at the
    frozen failure probability e^{-zcap} (zero when uncapped or below the cap).
    alpha = 0 (never solves) => c. Broadcasts over (P, N) or (P,)."""
    z = np.minimum(alpha * c, zcap)
    safe = np.where(alpha > 0, alpha, 1.0)
    e = np.exp(-z)
    return np.where(alpha > 0, (1.0 - e) / safe + (c - z / safe) * e, c)


def expected_spend(alpha: np.ndarray, zcap: np.ndarray, c: np.ndarray) -> float:
    """Expected dollars actually paid to *achieve* utility U(c) under the
    cost-minimizing cascade (order.tex).

    For query x the cascade queries models in decreasing alpha_i(x) order and
    stops on success. Reaching model i costs R_i(x, c_i) (`_cell_spend`) in
    expectation, weighted by the probability that every earlier model failed,
    exp(-sum_{j before i} z_j(x; c_j)). The order is chosen per instance (Prop.),
    i.e. the optimal deployment order.
    """
    z = _z(alpha, zcap, c)                                    # (P, N)
    R = _cell_spend(alpha, zcap, c[None, :])
    order = np.argsort(-alpha, axis=1)                        # decreasing alpha per row
    z_sorted = np.take_along_axis(z, order, axis=1)
    R_sorted = np.take_along_axis(R, order, axis=1)
    reach = np.exp(-(np.cumsum(z_sorted, axis=1) - z_sorted))  # prod of earlier failures
    return float((reach * R_sorted).sum(1).mean())


def single_model_spend(alpha_col: np.ndarray, zcap_col: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Expected spend for one model across `budgets` (early stopping on success)."""
    return np.array([_cell_spend(alpha_col, zcap_col, b).mean() for b in budgets])


def spend_fixed(alpha: np.ndarray, zcap: np.ndarray, c: np.ndarray, order) -> float:
    """Expected spend E_x[R^pi(x, c)] under ONE population order `order` (array of model
    indices, first queried first) -- the deployable convention."""
    z = _z(alpha, zcap, c)
    R = _cell_spend(alpha, zcap, c[None, :])
    order = np.asarray(order)
    z, R = z[:, order], R[:, order]
    reach = np.exp(-(np.cumsum(z, 1) - z))                    # prod of earlier failures
    return float((reach * R).sum(1).mean())


def greedy_order(alpha: np.ndarray, zcap: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Section-3 greedy population order (eq. pop_order_greedy): repeatedly append the
    model maximizing E_x[rho_S u_i] / E_x[rho_S R_i], rho_S = prod_{j in S}(1 - u_j).
    Unfunded models (c_i = 0) go last; they contribute nothing either way."""
    z = _z(alpha, zcap, c)
    R = _cell_spend(alpha, zcap, c[None, :])
    N = alpha.shape[1]
    funded = [i for i in range(N) if c[i] > 0]
    rho = np.ones(alpha.shape[0])
    order = []
    while funded:
        best, best_val = None, -np.inf
        for i in funded:
            den = float((rho * R[:, i]).mean())
            val = float((rho * (1.0 - np.exp(-z[:, i]))).mean()) / den if den > 0 else -np.inf
            if val > best_val:
                best, best_val = i, val
        if best is None:                     # no candidate has positive expected spend: order is immaterial
            order += funded
            break
        order.append(best)
        funded.remove(best)
        rho = rho * np.exp(-z[:, best])
    return np.array(order + [i for i in range(N) if c[i] <= 0])


def deployed_spend(alpha: np.ndarray, zcap: np.ndarray, c: np.ndarray, order: str = "greedy") -> float:
    """Expected spend of allocation c under an order convention:
    'greedy' (fixed population order of Section 3, deployable; DEFAULT) or
    'oracle' (per-instance decreasing-alpha order, needs p_i(x) of the query)."""
    if order == "greedy":
        return spend_fixed(alpha, zcap, c, greedy_order(alpha, zcap, c))
    if order == "oracle":
        return expected_spend(alpha, zcap, c)
    raise ValueError(f"unknown order convention {order!r}")


def _coord_ascent_free(alpha: np.ndarray, zcap: np.ndarray, lam: float, c0: np.ndarray,
                       sweeps: int = 100) -> np.ndarray:
    """Unconstrained coordinate ascent of U(c) - lam*sum(c): each coordinate is
    driven to its stationary point argmax_x E_x[1-e^{-(z_{-i}+min(alpha_i x, zcap_i))}] - lam*x.

    No budget cap; the per-coordinate optimum is finite because the marginal
    E_x[alpha_i w e^{-alpha_i x} 1{active}] decreases monotonically to 0 < lam (and is
    exactly 0 beyond max_x zcap_i/alpha_i when capped). Adaptive upper bracket, then bisection.
    """
    P, M = alpha.shape
    c = c0.copy()
    Z = _z(alpha, zcap, c)
    z = Z.sum(1)
    for _ in range(sweeps):
        delta = 0.0
        for i in range(M):
            ai, zi = alpha[:, i], zcap[:, i]
            aw = ai * np.exp(-(z - Z[:, i]))     # alpha_i * failure-of-others
            xmax = _xmax(ai, zi)

            def hp(x):
                act = ai * x < zi
                return float((aw * act) @ np.exp(-x * ai)) / P - lam

            if hp(0.0) <= 0.0:                   # h'(0) <= 0 -> zero funding
                x = 0.0
            else:
                hi = min(1.0, xmax)
                while hp(hi) > 0.0 and hi < xmax:
                    hi = min(2.0 * hi, xmax)
                    if hi > 1e15:
                        break
                lo = 0.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if hp(mid) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                x = 0.5 * (lo + hi)
            zi_new = np.minimum(ai * x, zi)
            delta = max(delta, abs(x - c[i]))
            z += zi_new - Z[:, i]
            Z[:, i] = zi_new
            c[i] = x
        if delta < 1e-11:
            break
    return c


def arbitrage_frontier(alpha: np.ndarray, zcap: np.ndarray, n: int = 70, n_head: int = 16,
                       order: str = "greedy") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace the optimal frontier by sweeping the shadow price lambda (fast).

    Each lambda gives c*(lambda) via one warm-started coordinate ascent; the
    total budget C = sum(c) and utility U increase as lambda falls. Returns
    (perf, spend, allocations) ordered by increasing performance. `spend` is the
    expected spend under the `order` convention (see `deployed_spend`): the
    allocation itself is order-free, only the money actually paid depends on it.

    `n` log-spaced lambdas cover lam_hi -> lam_hi*1e-6; `n_head` extra points
    sit just below lam_hi (lam = lam_hi*(1 - 1e-4 ... 0.8)) so the frontier is
    continuous down to zero budget instead of jumping from c=0 to the first
    log step (whose allocation is already a sizeable budget).
    """
    M = alpha.shape[1]
    lam_hi = float(max(alpha[:, i].mean() for i in range(M)))
    head = lam_hi * (1.0 - np.logspace(-4, np.log10(0.8), n_head)) if n_head > 0 else np.empty(0)
    lams = np.concatenate([head, np.logspace(np.log10(lam_hi * 0.2), np.log10(lam_hi * 1e-6), n)])  # high->low
    n = lams.size
    perf = np.empty(n)
    spend = np.empty(n)
    allocs = np.empty((n, M))
    c = np.zeros(M)
    for k, lam in enumerate(lams):
        c = _coord_ascent_free(alpha, zcap, lam, c)   # warm-started down the sweep
        perf[k] = utility(alpha, zcap, c)
        spend[k] = deployed_spend(alpha, zcap, c, order)
        allocs[k] = c
    return perf, spend, allocs


def invert_to_cost(perf_curve: np.ndarray, cost_curve: np.ndarray, target_perf: np.ndarray) -> np.ndarray:
    """Reparametrize a monotone (budget -> perf, budget -> cost) pair as
    (perf -> cost); NaN above the curve's reach."""
    order = np.argsort(perf_curve)
    pc, cc = perf_curve[order], cost_curve[order]
    out = np.interp(target_perf, pc, cc, left=np.nan, right=np.nan)
    out[target_perf > pc[-1] + 1e-12] = np.nan
    out[target_perf < pc[0] - 1e-12] = np.nan
    return out
