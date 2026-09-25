"""Per-problem pass@k models for the arbitrage figures.

Estimators producing a tabulated pass curve U[p, m, b] = Pr[model m solves problem p
within budget B[b]] on a shared budget grid (`pass_curves`; per-model or per-problem
attempt costs). All are frozen at the n observed attempts (no extrapolation).

- "chen" (THE PAPER'S CHOICE, for fitting and evaluation): the unbiased pass@k of
  Chen et al. (2021), 1 - C(n-c, k)/C(n, k), at integer k, with the continuous-time
  reading between integers (constant hazard within an attempt = log-linear failure
  probability; below one attempt this is exactly the geometric plug-in). Unbiased per
  problem; log-concave in k (sampling without replacement), so the arbitrage objective
  on these curves is not concave -> `arbitrage_tab.arbitrage_frontier(robust=True)`.
  "chen_lin": randomized last attempt (linear interpolation); "chen_int": floor(k), the
  grid-search convention of prior work (step functions).

- "geom": plug-in geometric  u = 1 - (1 - p_hat)^{min(k, n)},  p_hat = c/n clamped at
  1 - 1/(2n), with k = budget / q_m attempts. The model of the theory (concave
  objective) but biased pessimistic for k > 1 (Jensen), most severely near k = n.
  DEFAULT = NO EXTRAPOLATION past the n observed attempts: the curve is frozen at
  k = n (the (n+1)-th attempt costs money but adds no solve probability);
  `extrapolate=True` gives the plain 1 - (1 - p_hat)^k.

- "zibb": a zero-inflated Beta-Binomial extension of the Section-4 empirical-Bayes
  posterior-predictive, used by the released Figure 5/10 records. Unlike the plain
  Beta-Binomial method stated in Section 4, it adds a point mass at zero and a
  concentration hyperprior. Per model fit
  (a, b, pi0) by type-II MAP; per problem with c successes in n:
      c>0 : u(k) = 1 - Beta(a+c, b+n-c+k)/Beta(a+c, b+n-c)
      c=0 : u(k) = (1-ps)*(1 - Beta(a, b+n+k)/Beta(a, b+n)),
            ps = pi0 / (pi0 + (1-pi0)*Beta(a,b+n)/Beta(a,b))   [spike posterior]
  As k->inf the failure floor is ps (c=0) or 0 (c>0): rarely-/never-observed
  problems keep a calibrated positive solve ceiling instead of collapsing to 0.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, betaln, logsumexp, digamma
from scipy.optimize import minimize


def counts(results, models, ids):
    """Return c (P,M) successes and n (P,M) attempts on the shared problem set."""
    by_id = {m: {r["id"]: r for r in results[m]} for m in models}
    P, M = len(ids), len(models)
    c = np.zeros((P, M))
    n = np.zeros((P, M))
    for j, m in enumerate(models):
        for i, pid in enumerate(ids):
            a = by_id[m][pid]["attempts"]
            c[i, j] = int(sum(a))
            n[i, j] = len(a)
    return c, n


def _log_bb(c, n, a, b):
    return (gammaln(n + 1) - gammaln(c + 1) - gammaln(n - c + 1)
            + gammaln(c + a) + gammaln(n - c + b) - gammaln(n + a + b)
            - (gammaln(a) + gammaln(b) - gammaln(a + b)))


LOG_AB_MAX = np.log(1e6)     # bound on a, b: Beta(a,b) with a+b ~ 1e6 is already a point mass for any k of interest;
                              # beyond ~1e8 the gammaln differences lose precision and the "MLE" becomes a float artifact.
LOG_AB_MIN = np.log(1e-4)
PRIOR_POW = 2.5               # hyperprior p(a, b) ∝ (a + b)^(-PRIOR_POW) — Gelman et al., BDA3 §5.3 (rat-tumour Beta-binomial):
                              # uniform on (a/(a+b), (a+b)^(-1/2)). With only {0,1} counts the likelihood is flat along the ridge
                              # a/(a+b) = const; this prior breaks the tie toward small concentration = the heavy-tail (pessimistic)
                              # end, instead of the point-mass ("every problem solvable") end. At n=10k it moves fits by <5%.
                              # Set prior_pow=0 for the pure MLE.


def _zibb_nll_grad(theta, c, n, prior_pow=PRIOR_POW):
    """ZIBB negative log-posterior (nll + prior_pow*log(a+b)) and its gradient wrt (log a, log b, logit pi0)."""
    a, b = np.exp(theta[0]), np.exp(theta[1])
    pi0 = 1.0 / (1.0 + np.exp(-theta[2]))
    lbb = _log_bb(c, n, a, b)
    log_slab = np.log1p(-pi0) + lbb
    log_spike = np.where(c == 0, np.log(pi0), -np.inf)
    lse = logsumexp(np.stack([log_spike, log_slab]), axis=0)
    w_slab = np.exp(log_slab - lse)                     # posterior weight of the slab per problem
    w_spike = np.where(c == 0, np.exp(log_spike - lse), 0.0)
    d_a = digamma(c + a) - digamma(n + a + b) - digamma(a) + digamma(a + b)
    d_b = digamma(n - c + b) - digamma(n + a + b) - digamma(b) + digamma(a + b)
    g = np.array([-np.sum(w_slab * d_a) * a,
                  -np.sum(w_slab * d_b) * b,
                  -np.sum(w_spike * (1.0 - pi0) - w_slab * pi0)])
    nll = -np.sum(lse)
    if prior_pow:
        nll += prior_pow * np.log(a + b)
        g[0] += prior_pow * a / (a + b)
        g[1] += prior_pow * b / (a + b)
    return nll, g


def fit_zibb(c, n, prior_pow=PRIOR_POW):
    """Type-II MAP of the zero-inflated Beta-Binomial (a, b, pi0) under p(a,b) ∝ (a+b)^(-prior_pow)
    (prior_pow=0 gives the MLE).

    Bounded L-BFGS-B with the analytic gradient, 3 starts. Bounds: 1e-4 <= a, b <= 1e6,
    6e-6 <= pi0 <= 0.9997. When the likelihood is flat along the point-mass ridge
    a/(a+b) = const (models with only {0,1} counts) the fit stops at the bound instead of
    running off to a, b -> inf (which took thousands of Nelder-Mead iterations and produced
    numerically invalid likelihoods).
    """
    bounds = [(LOG_AB_MIN, LOG_AB_MAX), (LOG_AB_MIN, LOG_AB_MAX), (-12.0, 8.0)]
    best = None
    for x0 in ([-1, -1, -1], [0, 0, 0], [-2, 0, -0.5]):
        r = minimize(_zibb_nll_grad, x0, args=(c, n, prior_pow), jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x[0]), np.exp(best.x[1]), 1.0 / (1.0 + np.exp(-best.x[2]))


# pi0 pinned this far below its bound is numerically "off": the spike only matters where the
# beta-binomial itself says log P(s=0) < -40, which never happens on these datasets (~log 0.8).
LOGIT_PI0_OFF = -40.0


def fit_bb(c, n, prior_pow=PRIOR_POW):
    """Type-II MAP of the PLAIN Beta-Binomial (a, b) -- `fit_zibb` with the zero-inflation spike
    switched off. Shares the objective and gradient of `_zibb_nll_grad` so the two are comparable
    by construction. `prior_pow=0` additionally drops the (a+b)^-PRIOR_POW hyperprior, leaving the
    pure MLE of Kazdan et al. (2025)."""
    def f(t2):
        nll, g = _zibb_nll_grad(np.array([t2[0], t2[1], LOGIT_PI0_OFF]), c, n, prior_pow)
        return nll, g[:2]
    bounds = [(LOG_AB_MIN, LOG_AB_MAX), (LOG_AB_MIN, LOG_AB_MAX)]
    best = None
    for x0 in ([-1, -1], [0, 0], [-2, 0]):
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x[0]), np.exp(best.x[1])


def fit_zibb_nm(c, n):
    """Previous fitter (unbounded Nelder-Mead, 3 starts, maxiter 4000). Kept for comparison."""
    def nll(theta):
        a, b = np.exp(theta[0]), np.exp(theta[1])
        pi0 = 1.0 / (1.0 + np.exp(-theta[2]))
        log_slab = np.log1p(-pi0) + _log_bb(c, n, a, b)
        log_spike = np.where(c == 0, np.log(pi0), -np.inf)
        return -np.sum(logsumexp(np.stack([log_spike, log_slab]), axis=0))
    best = None
    for x0 in ([-1, -1, -1], [0, 0, 0], [-2, 0, -0.5]):
        r = minimize(nll, x0, method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x[0]), np.exp(best.x[1]), 1.0 / (1.0 + np.exp(-best.x[2]))


def _k2d(k, P):
    """Attempts grid as (P, B): a shared (B,) grid is broadcast, a per-problem (P, B) grid
    (per-problem attempt costs) is used as is."""
    k = np.asarray(k, dtype=float)
    return np.broadcast_to(k[None, :], (P, k.size)) if k.ndim == 1 else k


def geom_curve(c_col, n_col, k, extrapolate=False):
    """Geometric pass curve for one model. c_col,n_col: (P,), k: (B,) attempts.

    By default the curve is frozen past the observed attempts, k -> min(k, n): the
    (n+1)-th attempt is paid for but adds no solve probability. `extrapolate=True`
    gives the plain geometric extrapolation 1 - (1-phat)^k for all k."""
    phat = np.minimum(c_col / n_col, 1.0 - 1.0 / (2.0 * n_col))
    k = _k2d(k, c_col.size)
    kk = k if extrapolate else np.minimum(k, n_col[:, None])
    # U[p,b] = 1 - (1-phat_p)^{k_b}
    log_fail = np.log1p(-phat)[:, None] * kk
    return 1.0 - np.exp(log_fail)


def chen_curve(c_col, n_col, k, integer_k=False, interp="loglinear"):
    """Unbiased pass@k of Chen et al. (2021) at integer k, as used by a grid search over integer attempts:
        u(k) = 1 - C(n-c, k) / C(n, k),   k integer <= n;  = 1 for k > n-c (c>0);  = 0 for c = 0.
    Frozen past k = n (no extrapolation), like `geom_curve`. Between integers (`interp`):
      'loglinear' (DEFAULT): continuous-time reading -- constant hazard within an attempt, i.e. the
                  failure probability is log-linear between floor(k) and ceil(k). Below one attempt this
                  is exactly the geometric plug-in (1-c/n)^k. The attempt whose completion makes success
                  certain (F(ceil k) = 0) would have infinite hazard; that segment is interpolated linearly.
      'linear'  : randomized last attempt (failure probability linear between the integers).
      'integer' : floor(k) step function (the grid-search convention); `integer_k=True` selects it.
    NOTE: at the integers the failure probability is log-CONCAVE in k (sampling without replacement),
    so the arbitrage objective built on these curves is not concave (Theorem KKT does not apply)."""
    if integer_k:
        interp = "integer"
    P = c_col.size
    nmax = int(n_col.max())
    j = np.arange(nmax)[None, :]                                   # attempt index 0..nmax-1
    num = n_col[:, None] - c_col[:, None] - j                      # n-c-j
    den = n_col[:, None] - j                                       # n-j
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where((num > 0) & (den > 0), np.log(num / np.where(den > 0, den, 1)), -np.inf)
    step = np.where(j < n_col[:, None], step, 0.0)                 # past n: frozen (no more attempts)
    logF = np.concatenate([np.zeros((P, 1)), np.cumsum(step, axis=1)], axis=1)   # (P, nmax+1), logF[:, k]
    F = np.exp(logF)
    kk = np.minimum(_k2d(k, P), n_col[:, None])                    # freeze at the observed n
    kf = np.floor(kk).astype(int)
    kc = np.minimum(kf + 1, nmax)
    Ff = np.take_along_axis(F, kf, axis=1)
    if interp == "integer":
        return 1.0 - Ff
    Fc = np.take_along_axis(F, kc, axis=1)
    frac = kk - kf
    lin = Ff + frac * (Fc - Ff)
    if interp == "linear":
        return 1.0 - lin
    if interp != "loglinear":
        raise ValueError(interp)
    lFf = np.take_along_axis(logF, kf, axis=1)
    lFc = np.where(Fc > 0, np.take_along_axis(logF, kc, axis=1), 0.0)
    loglin = np.exp((1.0 - frac) * lFf + frac * lFc)
    return 1.0 - np.where(Fc > 0, loglin, lin)


def zibb_curve(c_col, n_col, k, a, b, pi0):
    """ZIBB posterior-predictive pass curve for one model. Returns (P, B)."""
    kk = _k2d(k, c_col.size)
    c_col = c_col[:, None]
    n_col = n_col[:, None]
    # slab ratio Beta(a+c, b+n-c+k)/Beta(a+c, b+n-c)
    ratio = np.exp(betaln(a + c_col, b + n_col - c_col + kk) - betaln(a + c_col, b + n_col - c_col))
    u = 1.0 - ratio                                    # correct for c>0
    # c==0 rows: apply spike-posterior downweight ps
    is0 = (c_col[:, 0] == 0)
    if np.any(is0):
        bb0 = np.exp(betaln(a, b + n_col[is0, 0]) - betaln(a, b))       # P(c=0 | slab)
        ps = pi0 / (pi0 + (1.0 - pi0) * bb0)                            # posterior spike weight
        u[is0] = (1.0 - ps)[:, None] * u[is0]
    return np.clip(u, 0.0, 1.0)


def bb_curve(c_col, n_col, k, a, b):
    """Beta--Binomial empirical-Bayes posterior-predictive pass curve."""
    return zibb_curve(c_col, n_col, k, a, b, 0.0)


def chen_bb_curve(c_col, n_col, k, a, b):
    """Chen pass@k where observed, then an empirical-Bayes continuation.

    For k <= n this is exactly the unbiased Chen estimator. Beyond n, it
    anchors at Chen(n) and uses the Beta posterior predictive only for the
    additional k-n attempts. This keeps each curve continuous and monotone;
    directly switching from Chen(k) to the full EB curve at k=n can make an
    estimated success probability fall as more attempts are purchased.
    """
    kk = _k2d(k, c_col.size)
    u_chen = chen_curve(c_col, n_col, kk, interp="loglinear")
    extra = np.maximum(kk - n_col[:, None], 0.0)
    post_fail_extra = np.exp(
        betaln(a + c_col[:, None], b + n_col[:, None] - c_col[:, None] + extra)
        - betaln(a + c_col[:, None], b + n_col[:, None] - c_col[:, None])
    )
    # At k=n, Chen's estimated failure is one iff no success was observed.
    fail_at_n = (c_col == 0)[:, None].astype(float)
    u_extended = 1.0 - fail_at_n * post_fail_extra
    return np.where(kk <= n_col[:, None], u_chen, u_extended)


def pass_curves(results, models, ids, q, budget_grid, mode, extrapolate=False):
    """Tabulated U[p, m, b] for the chosen estimator.

    q: per-attempt cost, (M,) per model or (P, M) per problem and model (as in
    `arbitrage.build_alpha`). budget_grid: (B,). `extrapolate` only affects 'geom'
    ('bb', 'chen_bb', and 'zibb' are models of extrapolation by construction).
    """
    c, n = counts(results, models, ids)
    P, M = c.shape
    B = budget_grid.size
    U = np.empty((P, M, B))
    params = {}
    q = np.asarray(q, dtype=float)
    for j, m in enumerate(models):
        # attempts affordable at each budget: (B,) for a per-model cost, (P, B) per problem
        k = budget_grid / q[j] if q.ndim == 1 else budget_grid[None, :] / q[:, j, None]
        if mode == "geom":
            U[:, j, :] = geom_curve(c[:, j], n[:, j], k, extrapolate=extrapolate)
        elif mode == "chen":                        # unbiased pass@k, continuous time within an attempt (DEFAULT)
            U[:, j, :] = chen_curve(c[:, j], n[:, j], k, interp="loglinear")
        elif mode == "chen_lin":                    # unbiased pass@k, randomized last attempt
            U[:, j, :] = chen_curve(c[:, j], n[:, j], k, interp="linear")
        elif mode == "chen_int":                    # unbiased pass@k, floor(k) (grid-search convention)
            U[:, j, :] = chen_curve(c[:, j], n[:, j], k, interp="integer")
        elif mode == "bb":                          # paper's query-level empirical-Bayes estimator
            a, b = fit_bb(c[:, j], n[:, j], prior_pow=0.0)
            params[m] = (a, b)
            U[:, j, :] = bb_curve(c[:, j], n[:, j], k, a, b)
        elif mode == "chen_bb":                     # Chen where identified; EB only beyond n
            a, b = fit_bb(c[:, j], n[:, j], prior_pow=0.0)
            params[m] = (a, b)
            U[:, j, :] = chen_bb_curve(c[:, j], n[:, j], k, a, b)
        elif mode == "zibb":
            a, b, pi0 = fit_zibb(c[:, j], n[:, j])
            params[m] = (a, b, pi0)
            U[:, j, :] = zibb_curve(c[:, j], n[:, j], k, a, b, pi0)
        else:
            raise ValueError(mode)
    return U, params
