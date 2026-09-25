"""Problem-level cross-validation of the arbitrage frontier and of the market frontier.

Policy = everything chosen from data: the allocation c*(lambda) and its greedy deployment order.
Policies are indexed by the shadow price lambda (one grid shared by all folds), NOT by an in-sample
target solve rate: selecting "the policy nearest to target g" on the training problems is a discrete
choice that, under leave-one-out, correlates with the held-out problem's outcome (removing a solved
problem lowers training solve rates, so the fold picks a bigger policy and scores it on a problem it
solves), which makes pooled points near policy ties spuriously good. For each fold, policies are fit
on the training problems (Chen curves, lambda-sweep warm-started from the full-data solution, greedy
order) and evaluated on the held-out problems with the Chen curves (unbiased for a fixed policy);
pooling over folds gives one test point (solve rate, expected spend) per lambda. The market gets the
same treatment: at each lambda the single (model, budget) maximizing train solve rate - lambda*budget,
i.e. the best single model under the same shadow price. No test-time envelope.

  python cv.py                     LOO on terminal_bench2_priced, 10-fold on livecodebench_priced,
                                   writes fig_panel_cv.{pdf,png} + cv_<key>.json, prints a summary
  python cv.py KEY [--k K]         one market; --k 0 = leave-one-out
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arbitrage as A                      # noqa: E402
import arbitrage_tab as T                  # noqa: E402
import estimator as E                      # noqa: E402
import make_panel as MP                    # noqa: E402
from make_figure import DATASETS           # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "plots"))
import fig1 as F                           # noqa: E402

DEFAULT = {"livecodebench_priced": 10, "terminal_bench2_priced": 0,
           "deepswe_priced": 10, "terminal_bench4_priced": 0}   # folds; 0 = leave-one-out
FROZEN = {"chen", "chen_lin", "chen_int", "geom"}   # estimators frozen at the recorded attempts
N_LAM, N_HEAD = 70, 16                     # shadow-price grid (as in make_panel)


def curves(cfg, estimator="chen", ext_decades=4):
    """Chen curves U[p, m, b] and per-cell cumulative spend on the figure's budget grid (per-problem costs)."""
    base_dir, models = cfg["base_dir"], cfg["models"]
    if models is None:
        models = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(base_dir, "*.jsonl")))
    results = A.load_results(base_dir)
    _, _, q, ids = A.build_alpha(results, models)
    lo, hi = np.log10(cfg["bmin"]) - ext_decades, np.log10(cfg["bmax"])
    budget = np.logspace(lo, hi, int((hi - lo) * MP.PTS_PER_DECADE))
    U, _ = E.pass_curves(results, models, ids, q, budget, estimator)
    # Frozen curves: no money past the n recorded attempts (see arbitrage_tab.SPEND_CAP).
    _, n_rec = E.counts(results, models, ids)
    T.SPEND_CAP = n_rec * np.asarray(q, dtype=float) if estimator in FROZEN else None
    cumsp = np.stack([T.cum_spend(U[:, m, :], budget, T._cap(m)) for m in range(len(models))])      # (M, P, B)
    return models, ids, U, cumsp, budget


def folds(P, K, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(P)
    K = P if K == 0 else K
    return [np.sort(perm[i::K]) for i in range(K)]


def fit(U, cumsp, budget, lams, c_inits=None, sweeps=40):
    """Frontier on the given problems: (perf, spend, allocs, greedy orders)."""
    perf, spend, allocs = T.arbitrage_frontier(U, budget, robust=True, lams=lams, c_inits=c_inits, sweeps=sweeps)
    orders = [T.greedy_order(U, list(cumsp), budget, c) for c in allocs]
    return perf, spend, allocs, orders


def eval_alloc(U, cumsp, budget, c, order):
    """Per-problem solve probability and expected spend of allocation c deployed in `order`."""
    u, R = T._cells(U, list(cumsp), budget, c)
    logf = np.log(np.clip(1.0 - u, 1e-12, 1.0))
    upool = 1.0 - np.exp(logf.sum(1))
    lf, RR = logf[:, order], R[:, order]
    return upool, (np.exp(np.cumsum(lf, 1) - lf) * RR).sum(1)


def pareto(u, R):
    """Lower-left envelope of pooled (solve rate, spend) points: drop any point that another point
    beats on both. A lambda at which the folds choose different policies (e.g. the market switching
    model) pools into a mixture that lies above its neighbours; the in-sample curves are envelopes by
    construction, so the CV curves are reduced to their envelope before inversion as well."""
    o = np.argsort(u)
    u, R = u[o], R[o]
    keep = np.zeros(u.size, dtype=bool)
    best = np.inf
    for i in range(u.size - 1, -1, -1):                                          # from the highest solve rate down
        if R[i] < best:
            keep[i] = True
            best = R[i]
    return u[keep], R[keep]


def market_policy(perf, lam, budget):
    """Best single model under shadow price lam: argmax over (model, grid budget) of perf - lam*budget.
    Returns (model, budget index)."""
    val = perf - lam * budget[None, :]                                          # (M, B)
    m, j = np.unravel_index(int(np.argmax(val)), val.shape)
    return int(m), int(j)


def run(key, K, seed=0, verbose=True, models=None, estimator="chen"):
    """`models`: optional sub-market (list of system names) instead of the registry's list (diagnostics).
    `estimator`: 'chen' (paper Figure 2: curves frozen at the observed attempts) or 'chen_bb'
    (Figure 1's convention: Chen within the observed attempts, the query-level empirical-Bayes
    continuation beyond them). With 'chen_bb' the held-out evaluation is on the SAME extrapolated
    curves -- there is no ground truth past the recorded attempts -- so the frontier is model-based
    in the extrapolated region, and the per-model BB prior (a, b) is fit once on all problems
    (a 2-parameter cross-fold leak; the per-query posterior itself uses only that query's counts)."""
    cfg = dict(DATASETS[key])
    if models is not None:
        cfg["models"] = list(models)
    models, ids, U, cumsp, budget = curves(cfg, estimator=estimator)
    P, M = U.shape[:2]
    # Cheapest-first deployment order: the common heuristic the paper's greedy rule is compared
    # against. It is a fixed permutation (independent of the allocation and of the fold), so the
    # same folds can be scored under it. Reloading is cheap next to the fitting below; curves()
    # has other callers, so its signature is left alone.
    _, _, q_att, _ = A.build_alpha(A.load_results(cfg["base_dir"]), models)
    cheap_order = np.argsort(np.asarray(q_att, dtype=float).mean(0))
    lams = T.lambda_grid(U, budget, n=N_LAM, n_head=N_HEAD, robust=True)
    L = lams.size
    perf_full, spend_full, allocs_full, orders_full = fit(U, cumsp, budget, lams)
    arb_u = np.full((L, P), np.nan); arb_R = arb_u.copy(); mkt_u = arb_u.copy(); mkt_R = arb_u.copy()
    chp_R = arb_u.copy()                      # same allocations, deployed cheapest-first
    fl = folds(P, K, seed)
    for fi, te in enumerate(fl):
        tr = np.setdiff1d(np.arange(P), te)
        _, _, allocs_tr, orders_tr = fit(U[tr], cumsp[:, tr], budget, lams, c_inits=allocs_full, sweeps=10)
        mk_perf = U[tr].mean(0)                                                 # (M, B) single-model curves on train
        for k, lam in enumerate(lams):
            arb_u[k, te], arb_R[k, te] = eval_alloc(U[te], cumsp[:, te], budget, allocs_tr[k], orders_tr[k])
            # solve rate does not depend on the order, only the spend does -- keep the spend only
            _, chp_R[k, te] = eval_alloc(U[te], cumsp[:, te], budget, allocs_tr[k], cheap_order)
            m, j = market_policy(mk_perf, lam, budget)
            mkt_u[k, te], mkt_R[k, te] = U[te, m, j], cumsp[m, te, j]
        if verbose and (fi % max(1, len(fl) // 10) == 0 or fi == len(fl) - 1):
            print(f"  [{key}] fold {fi + 1}/{len(fl)}", flush=True)

    arb_cv, mkt_cv = (arb_u.mean(1), arb_R.mean(1)), (mkt_u.mean(1), mkt_R.mean(1))
    chp_cv = (arb_u.mean(1), chp_R.mean(1))
    G = lams
    # figure record: in-sample curves as in make_panel (haze, arbitrageur, market) + CV curves on the same grid
    rec = MP.as_record(cfg, MP.compute(cfg, estimator=estimator))
    rec["estimator"] = estimator
    grid = rec["solve_rate"] / 100.0
    rec["arb_cost_cv"] = A.invert_to_cost(*pareto(*arb_cv), grid) * P
    rec["market_cost_cv"] = A.invert_to_cost(*pareto(*mkt_cv), grid) * P
    rec["arb_cheap_cost_cv"] = A.invert_to_cost(*pareto(*chp_cv), grid) * P
    rec["cv"] = "leave-one-out" if K == 0 else f"{K}-fold"
    rec["n_problems"] = P
    rec["cv_lambda"] = G                                                        # raw pooled CV curves, per shadow price
    rec["cv_arb_solve"], rec["cv_arb_spend"] = arb_cv[0], arb_cv[1] * P
    rec["cv_market_solve"], rec["cv_market_spend"] = mkt_cv[0], mkt_cv[1] * P
    rec["cv_arb_cheap_solve"], rec["cv_arb_cheap_spend"] = chp_cv[0], chp_cv[1] * P
    return rec, dict(G=G, arb_cv=arb_cv, mkt_cv=mkt_cv, arb_u=arb_u, arb_R=arb_R, mkt_u=mkt_u, mkt_R=mkt_R,
                     perf_full=perf_full, spend_full=spend_full)


def summary(rec):
    x = rec["solve_rate"]
    m_in = np.clip(1 - rec["arb_cost"] / rec["market_cost"], 0, 1) * 100
    m_cv = np.clip(1 - rec["arb_cost_cv"] / rec["market_cost_cv"], 0, 1) * 100
    print(f"\n== {rec['title']} ({rec['cv']}, {rec['n_problems']} problems)")
    print("  solve | arb $ in-sample / CV | market $ in-sample / CV | margin in-sample / CV")
    for g in [30, 40, 50, 60, 70, 80, 90]:
        k = int(np.argmin(np.abs(x - g)))
        if not np.isfinite(rec["arb_cost_cv"][k]):
            continue
        print(f"  {x[k]:4.0f}% | {rec['arb_cost'][k]:8.3f} / {rec['arb_cost_cv'][k]:8.3f} | {rec['market_cost'][k]:8.3f} / {rec['market_cost_cv'][k]:8.3f} | {m_in[k]:5.1f}% / {m_cv[k]:5.1f}%")
    ok = np.isfinite(m_in) & np.isfinite(m_cv)
    print(f"  mean margin over the CV range: in-sample {np.nanmean(m_in[ok]):.1f}%  CV {np.nanmean(m_cv[ok]):.1f}%;"
          f"  reach: in-sample {np.nanmax(x[np.isfinite(rec['arb_cost'])]):.0f}%  CV {np.nanmax(x[np.isfinite(rec['arb_cost_cv'])]):.0f}%")


if __name__ == "__main__":
    argv = sys.argv[1:]
    K = None
    if "--k" in argv:
        i = argv.index("--k"); K = int(argv[i + 1]); del argv[i:i + 2]
    paper = "--paper" in argv                       # also write plots/fig1_data/<key>.json and the paper's Fig. 2
    estimator = "chen"
    if "--estimator" in argv:
        i = argv.index("--estimator"); estimator = argv[i + 1]; del argv[i:i + 2]
    sfx = "" if estimator == "chen" else f"_{estimator}"          # keep the paper's chen records untouched
    if paper and estimator != "chen":
        raise SystemExit("--paper is the chen (no-extrapolation) figure; refusing to overwrite "
                         "plots/fig1_data with extrapolated curves")
    keys = [a for a in argv if not a.startswith("--")] or list(DEFAULT)
    recs = []
    for key in keys:
        rec, raw = run(key, DEFAULT.get(key, 10) if K is None else K, estimator=estimator)
        F.dump({k: v for k, v in rec.items() if k != "models"}, os.path.join(HERE, f"cv_{key}{sfx}.json"))
        if paper:
            F.dump({k: v for k, v in rec.items() if k != "models"}, os.path.join(F.DATA_DIR, f"{key}.json"))
        summary(rec)
        recs.append(rec)
    MP.set_figwidth(5.5, len(recs) + 1)
    F.draw_fig1(recs, out=os.path.join(HERE, f"fig_panel_cv{sfx}"), figsize=(MP.PANEL_W * (len(recs) + 1), MP.PANEL_H), with_profit=True)
    # NOTE: no copy to F.PAPER_COPY here. plots/fig1.py is what the paper's figure is built
    # from (and what mirrors into figures/); this render is a diagnostic at another size.
