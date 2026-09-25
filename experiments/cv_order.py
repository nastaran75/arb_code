"""Cross-validated cost of the deployment order (Section 3 figure, plots/fig_order.py).

The Section-2 allocation c*(lambda) is order-free; what the arbitrageur pays depends on the cascade
order. Same protocol as cv.py (problem-level folds, policies indexed by the shadow price lambda,
allocation fit on the training problems, evaluated on the held-out problems with the Chen curves),
but every held-out allocation is deployed under four order conventions:

  greedy    : the Section-3 greedy population order (eq. pop_order_greedy), fit on the training problems
  cheapest  : models sorted by their average cost per attempt on the training problems (a fixed
              order that is independent of the outcomes)
  best      : the best fixed order of the funded models -- the optimum of the deployable class (Theorem:
              NP-hard in general), the figure's baseline. Exact by dynamic programming over subsets of the
              funded models (2^n n evaluations; the reach probability depends only on the set of models
              queried before) when <= DP_MAX are funded; beyond that, the best of a local search over
              pairwise swaps and insertions started from the greedy and from the cheapest-first order
  oracle    : the per-instance order of Proposition (order), decreasing u_i(x)/R_i(x) of the held-out
              query itself -- not deployable; kept in the record as a reference, not drawn

The pooled solve rate per lambda is the same under every order. Output plots/fig_order_data/<key>.json
(lambda, pooled solve rate, pooled expected spend per order; in-sample counterparts) and a summary.

  python cv_order.py               both markets of the paper (LOO on Terminal-Bench, 10-fold on LiveCodeBench)
  python cv_order.py KEY [--k K]   one market; --k 0 = leave-one-out
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arbitrage as A                      # noqa: E402
import arbitrage_tab as T                  # noqa: E402
import cv as CV                            # noqa: E402
import make_panel as MP                    # noqa: E402
from make_figure import DATASETS           # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "plots"))
import fig1 as F                           # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "plots", "fig_order_data")
DP_MAX = 16                                # exact best fixed order by subset DP up to 2^16 states per (fold, lambda); local search beyond
ORDERS = ["greedy", "cheapest", "best", "oracle"]


def oracle_per_problem(U, cumsp, budget, c):
    """Per-problem expected spend under the per-instance index-rule order (T.oracle_spend without the mean)."""
    u, R = T._cells(U, list(cumsp), budget, c)
    ratio = np.where(R > 0, u / np.where(R > 0, R, 1.0), -1.0)
    order = np.argsort(-ratio, axis=1)
    logf = np.log(np.clip(1.0 - u, 1e-12, 1.0))
    logf, R = np.take_along_axis(logf, order, axis=1), np.take_along_axis(R, order, axis=1)
    return (np.exp(np.cumsum(logf, axis=1) - logf) * R).sum(1)


def _spend_perm(logf, R, perm):
    lf, RR = logf[:, perm], R[:, perm]
    return float((np.exp(np.cumsum(lf, 1) - lf) * RR).sum(1).mean())


def _local_search(logf, R, start):
    """Descent over pairwise swaps and single-element insertions from `start` (a permutation of range(n))."""
    order, cur, n = list(start), _spend_perm(logf, R, start), len(start)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order.copy(); cand[i], cand[j] = cand[j], cand[i]
                v = _spend_perm(logf, R, cand)
                if v < cur - 1e-15:
                    order, cur, improved = cand, v, True
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = order.copy(); cand.insert(j, cand.pop(i))
                v = _spend_perm(logf, R, cand)
                if v < cur - 1e-15:
                    order, cur, improved = cand, v, True
    return order, cur


def _dp_order(f, R):
    """Exact min-expected-spend order by dynamic programming over subsets: the probability of reaching a
    model depends only on the SET of models queried before it, so
        cost(S) = min_{i in S} cost(S \\ i) + E_x[ prod_{j in S \\ i} (1 - u_j(x)) * R_i(x) ],
    2^n * n evaluations instead of n! (the same recursion as the grid search of bf_grid.py).
    f: (P, n) failure probabilities 1 - u, R: (P, n) expected spend given reached. Returns (perm, cost)."""
    P, n = f.shape
    N = 1 << n
    reach = np.empty((N, P)); reach[0] = 1.0                      # reach[S] = prod_{j in S} f_j  (per problem)
    for S in range(1, N):
        lb = S & -S
        reach[S] = reach[S ^ lb] * f[:, lb.bit_length() - 1]
    step = reach @ R / P                                           # (N, n): E_x[reach_S(x) R_i(x)]
    cost = np.full(N, np.inf); cost[0] = 0.0
    last = np.full(N, -1, dtype=int)
    for S in range(1, N):
        for i in range(n):
            if S >> i & 1:
                v = cost[S ^ (1 << i)] + step[S ^ (1 << i), i]
                if v < cost[S]:
                    cost[S], last[S] = v, i
    perm, S = [], N - 1
    while S:
        perm.append(last[S]); S ^= 1 << last[S]
    return perm[::-1], float(cost[N - 1])


def best_order(U, cumsp, budget, c, starts, dp_max=DP_MAX):
    """Best fixed order of the funded models on the given problems (the optimum of the deployable class):
    exact by the subset DP when <= dp_max models are funded (2^n states), else the best local-search descent
    from each order in `starts` (full orders over all models)."""
    M = U.shape[1]
    funded = [i for i in range(M) if c[i] > 0]
    rest = [i for i in range(M) if c[i] <= 0]
    u, R = T._cells(U, list(cumsp), budget, c)
    f = np.clip(1.0 - u, 0.0, 1.0)[:, funded]
    Rf = R[:, funded]
    if len(funded) <= dp_max:
        perm, _ = _dp_order(f, Rf)
    else:
        logf = np.log(np.clip(f, 1e-12, 1.0))
        pos = {m: k for k, m in enumerate(funded)}
        perm, best_val = None, np.inf
        for s in starts:
            p, val = _local_search(logf, Rf, [pos[m] for m in s if c[m] > 0])
            if val < best_val:
                perm, best_val = p, val
    return np.array([funded[k] for k in perm] + rest)


def spends(U_te, cumsp_te, budget, c, orders):
    """Held-out per-problem solve prob and spend under each order convention. orders: dict name -> order."""
    out = {}
    u = None
    for name in ("greedy", "cheapest", "best"):
        u, out[name] = CV.eval_alloc(U_te, cumsp_te, budget, c, orders[name])
    out["oracle"] = oracle_per_problem(U_te, cumsp_te, budget, c)
    return u, out


def run(key, K, seed=0, verbose=True):
    cfg = DATASETS[key]
    models, ids, U, cumsp, budget = CV.curves(cfg)
    res = A.load_results(cfg["base_dir"])
    _, _, q, _ = A.build_alpha(res, models)                                    # (P, M) per-problem attempt cost
    P, M = U.shape[:2]
    lams = T.lambda_grid(U, budget, n=CV.N_LAM, n_head=CV.N_HEAD, robust=True)
    L = lams.size
    perf_full, spend_full, allocs_full, orders_full = CV.fit(U, cumsp, budget, lams)
    # in-sample reference: full-data allocation and orders, evaluated on all problems
    cheap_full = np.argsort(q.mean(0))
    ins = {name: np.empty(L) for name in ORDERS}
    n_funded = np.array([(c > 0).sum() for c in allocs_full])
    for k in range(L):
        o = dict(greedy=orders_full[k], cheapest=cheap_full,
                 best=best_order(U, cumsp, budget, allocs_full[k], [orders_full[k], cheap_full]))
        _, s = spends(U, cumsp, budget, allocs_full[k], o)
        for name in ORDERS:
            ins[name][k] = s[name].mean()
    # cross-validation
    arb_u = np.full((L, P), np.nan)
    arb_R = {name: np.full((L, P), np.nan) for name in ORDERS}
    fl = CV.folds(P, K, seed)
    for fi, te in enumerate(fl):
        tr = np.setdiff1d(np.arange(P), te)
        _, _, allocs_tr, orders_tr = CV.fit(U[tr], cumsp[:, tr], budget, lams, c_inits=allocs_full, sweeps=10)
        cheap_tr = np.argsort(q[tr].mean(0))
        for k in range(L):
            o = dict(greedy=orders_tr[k], cheapest=cheap_tr,
                     best=best_order(U[tr], cumsp[:, tr], budget, allocs_tr[k], [orders_tr[k], cheap_tr]))
            u, s = spends(U[te], cumsp[:, te], budget, allocs_tr[k], o)
            arb_u[k, te] = u
            for name in ORDERS:
                arb_R[name][k, te] = s[name]
        if verbose and (fi % max(1, len(fl) // 10) == 0 or fi == len(fl) - 1):
            print(f"  [{key}] fold {fi + 1}/{len(fl)}", flush=True)
    title = MP._title(cfg)
    rec = dict(title=title, legend=F.load(os.path.join(F.DATA_DIR, f"{key}.json"))["legend"] if
               os.path.exists(os.path.join(F.DATA_DIR, f"{key}.json")) else title,
               xlabel=cfg["xlabel"], cv="leave-one-out" if K == 0 else f"{K}-fold", n_problems=P, n_models=M,
               orders=ORDERS, dp_max=DP_MAX,
               lam=lams, n_funded=n_funded,
               solve_in=perf_full, solve_cv=arb_u.mean(1))
    for name in ORDERS:
        rec[f"spend_{name}_in"] = ins[name] * P
        rec[f"spend_{name}_cv"] = arb_R[name].mean(1) * P
    return rec


def summary(rec):
    x = rec["solve_cv"] * 100
    g, c, b, o = (np.asarray(rec[f"spend_{n}_cv"]) for n in ORDERS)
    gi, ci, bi, oi = (np.asarray(rec[f"spend_{n}_in"]) for n in ORDERS)
    print(f"\n== {rec['title']} ({rec['cv']}, {rec['n_problems']} problems, {rec['n_models']} models)")
    print("  solve | #funded | CV spend / best fixed order: greedy cheapest | best/oracle | in-sample: greedy cheapest | best/oracle")
    for gval in [30, 40, 50, 60, 70, 80, 90, 95, 98]:
        if gval < x.min() or gval > x.max():
            continue
        k = int(np.argmin(np.abs(x - gval)))
        print(f"  {x[k]:5.1f}% |  {rec['n_funded'][k]:2d}     |   {g[k]/b[k]:6.3f}   {c[k]/b[k]:6.3f}         |   {b[k]/o[k]:6.3f}    "
              f"|   {gi[k]/bi[k]:6.3f}   {ci[k]/bi[k]:6.3f}     |   {bi[k]/oi[k]:6.3f}")
    ok = b > 0
    for lab, gg, cc, bb in [("CV", g, c, b), ("in-sample", gi, ci, bi)]:
        print(f"  {lab} greedy/best over lambda: min {np.min(gg[ok]/bb[ok]):.3f} median {np.median(gg[ok]/bb[ok]):.3f} max {np.max(gg[ok]/bb[ok]):.3f};"
              f"  cheapest/best: min {np.min(cc[ok]/bb[ok]):.3f} median {np.median(cc[ok]/bb[ok]):.3f} max {np.max(cc[ok]/bb[ok]):.3f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    K = None
    if "--k" in argv:
        i = argv.index("--k"); K = int(argv[i + 1]); del argv[i:i + 2]
    keys = [a for a in argv if not a.startswith("--")] or list(CV.DEFAULT)
    os.makedirs(OUT_DIR, exist_ok=True)
    for key in keys:
        rec = run(key, CV.DEFAULT.get(key, 10) if K is None else K)
        F.dump(rec, os.path.join(OUT_DIR, f"{key}.json"))
        summary(rec)
