"""Per-problem replay of cv.py's leave-one-out evaluation, for error bars and revenue.

cv.py stores only the pooled (solve rate, spend) per shadow price. The bar figures also
need, per held-out problem, the solve probability and the money that reaches each system,
so this module re-runs the identical protocol (same Chen curves, budget grid, lambda grid,
folds, warm-started fold fits, greedy orders) and keeps the per-problem arrays. Cached in
cv_replay_<key>.npz next to this file (~30 s to rebuild).

    replay(key)  -> dict(models, lams, solve (L,P), spend (L,P), rev (L,P,M))
                    with rev.sum(2) == spend and pooled means equal to cv.py's curves
    envelope(solve_pool, spend_pool) -> indices of the lower envelope, sorted by spend
    at_budget(B, spend_kept, X_kept)  -> X interpolated (in spend) at budget B
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = "terminal_bench2_priced"


def _compute(key, seed, verbose, k=None):
    import cv
    import arbitrage_tab as T
    from make_figure import DATASETS
    cfg = DATASETS[key]
    models, ids, U, cumsp, budget = cv.curves(cfg)
    P, M = U.shape[:2]
    lams = T.lambda_grid(U, budget, n=cv.N_LAM, n_head=cv.N_HEAD, robust=True)
    L = lams.size
    _, _, allocs_full, _ = cv.fit(U, cumsp, budget, lams)
    rev = np.zeros((L, P, M))
    solve = np.zeros((L, P))
    fl = cv.folds(P, cv.DEFAULT[key] if k is None else k, seed)       # 0 folds = leave-one-out
    for fi, te in enumerate(fl):
        tr = np.setdiff1d(np.arange(P), te)
        _, _, allocs_tr, orders_tr = cv.fit(U[tr], cumsp[:, tr], budget, lams, c_inits=allocs_full, sweeps=10)
        for k in range(L):
            u, R = T._cells(U[te], list(cumsp[:, te]), budget, allocs_tr[k])
            logf = np.log(np.clip(1.0 - u, 1e-12, 1.0))
            order = np.asarray(orders_tr[k])
            lf, RR = logf[:, order], R[:, order]
            rev[k, te[:, None], order[None, :]] = np.exp(np.cumsum(lf, 1) - lf) * RR   # money reaching each system
            solve[k, te] = 1.0 - np.exp(logf.sum(1))
        if verbose and (fi % max(1, len(fl) // 4) == 0 or fi == len(fl) - 1):
            print(f"  [cv replay] fold {fi + 1}/{len(fl)}", flush=True)
    return dict(models=np.array(models), lams=lams, solve=solve, spend=rev.sum(2), rev=rev)


def replay(key=KEY, seed=0, use_cache=True, verbose=True, k=None):
    """k: fold-count override (None = the dataset's cv.DEFAULT; 0 = leave-one-out). A non-default
    k gets its own cache file."""
    cache = os.path.join(HERE, f"cv_replay_{key}{'' if k is None else f'_k{k}'}.npz")
    if use_cache and os.path.exists(cache):
        d = dict(np.load(cache, allow_pickle=False))
        d["models"] = list(d["models"])
        return d
    d = _compute(key, seed, verbose, k)
    np.savez(cache, **d)
    d["models"] = list(d["models"])
    return d


def envelope(solve_pool, spend_pool):
    """Indices of the lower-left envelope of the pooled points (cv.pareto), sorted by spend."""
    keep, best = [], np.inf
    for i in np.argsort(solve_pool)[::-1]:
        if spend_pool[i] < best:
            keep.append(i)
            best = spend_pool[i]
    return np.array(sorted(keep, key=lambda i: spend_pool[i]))


def at_budget(B, spend_kept, X_kept):
    """Linear interpolation of X (first axis along the kept points) at spend B; clamps at the ends."""
    B = float(np.clip(B, spend_kept[0], spend_kept[-1]))
    hi = int(np.searchsorted(spend_kept, B))
    if hi == 0:
        return X_kept[0]
    lo = hi - 1
    t = (B - spend_kept[lo]) / (spend_kept[hi] - spend_kept[lo])
    return (1 - t) * X_kept[lo] + t * X_kept[hi]
