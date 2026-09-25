"""Decision-level validation of the pass@k estimators: the replicates behind Figure 5 and Figure 10.

For every benchmark, calibration size n_sub in N_SUBS and replicate rep < R: draw n_sub of the recorded
attempts of every (problem, model) cell (seeded by (n_sub, rep)), estimate the utility curves with the
plug-in ("geom") and with the zero-inflated, hyperprior-regularized empirical-Bayes extension used
by the released Figure 5/10 records ("zibb"), run the
arbitrage on each estimate, and score the chosen allocations under the Chen curves of the HELD-OUT
attempts (validate_estimator.score). Protocol constants live in validate_estimator.

    python validate_calibration.py monkey_math monkey_codecontests resmat2_aime2024 resmat2_aime2025 resmat2_gmmlu resmat2_mmlupro

Writes val_cache/val_calib_<key>_M<models>_R<R>_kcap<KCAP>.pkl (one record per replicate), which
make_fig_passk_data.py turns into the plots/fig_passk_data records drawn by plots/fig_calib.py.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

import arbitrage_tab as at
import validate_estimator as v

MODES = ("geom", "zibb")


def run_one(args):
    key, n_sub, rep = args
    cfg, models, q, c, n, budget = v.load(key)
    rng = np.random.default_rng(v.SEED + 1000 * n_sub + rep)
    kcap = v.kcap_models(cfg, len(models))
    c_sub, n_sub_arr, c_rest, n_rest = v.subsample(c, n, n_sub, rng, fixed=cfg.get("fixed"))
    ncap = cfg.get("ncap")
    U_true = v.curves_from_counts(c_rest, n_rest, q, budget, "chen", kcap, ncap)      # held-out, Chen unbiased
    cumsp_true = [at.cum_spend(U_true[:, m, :], budget) for m in range(len(models))]
    cap = v.model_cap(kcap, q)
    out = dict(key=key, n_sub=n_sub, rep=rep)
    for mode in MODES:
        U_e = v.curves_from_counts(c_sub, n_sub_arr, q, budget, mode, kcap, ncap)
        ap_c, asp_c, allocs = at.arbitrage_frontier(U_e, budget, cap=cap)
        ap_r, asp_r = v.score(U_true, cumsp_true, budget, allocs)
        out[mode] = dict(claimed=(ap_c, asp_c), realized=(ap_r, asp_r), allocs=allocs,
                         model_perf_est=at.single_model(U_e, budget)[0])
    return out


def main(keys):
    os.makedirs(v.CACHE, exist_ok=True)
    for key in keys:
        v.configure(key)
        t0 = time.time()
        cfg, models, *_ = v.load(key)                    # load once; Pool workers inherit it
        cache = os.path.join(v.CACHE, f"val_calib_{key}_M{len(models)}_R{v.R}_kcap{v.KCAP:.0f}{v.cache_tag()}.pkl")
        if os.path.exists(cache):
            print(f"[{key}] cached: {os.path.basename(cache)}"); continue
        jobs = [(key, n_sub, rep) for n_sub in v.N_SUBS for rep in range(v.R)]
        # Serial by default: the replicates are deterministic and take a few minutes in total.
        # VAL_PARALLEL=1 runs them in a fork-based process pool; on macOS the forked workers can
        # return slightly different L-BFGS-B fits (Accelerate is not fork-safe), so the serial path
        # is the reference.
        if os.environ.get("VAL_PARALLEL"):
            with Pool(min(len(jobs), os.cpu_count() or 4)) as pool:
                runs = pool.map(run_one, jobs)
        else:
            runs = [run_one(j) for j in jobs]
        pickle.dump(runs, open(cache, "wb"))
        print(f"[{key}] {len(runs)} replicates in {time.time() - t0:.0f}s -> {os.path.basename(cache)}")


PAPER_KEYS = ("monkey_math", "monkey_codecontests", "resmat2_aime2024", "resmat2_aime2025", "resmat2_gmmlu", "resmat2_mmlupro")

if __name__ == "__main__":
    main(sys.argv[1:] or list(PAPER_KEYS))
