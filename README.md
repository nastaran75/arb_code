Anonymous code release. Everything the paper reports is regenerated from the per-problem data in
`experiments/data/` by one script:

```
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python ./reproduce.sh          # ~15 min on a laptop, CPU only, no network
```

`reproduce.sh` writes every figure of the paper into `figures/` and then runs
`check_reproduction.py`, which compares the regenerated plotted records with the records behind
the submitted figures (`reference/`). Both are included so a reader can either re-run everything
or only redraw.

## Figures

| Paper | File in `figures/` | Computed by | Drawn by |
|---|---|---|---|
| Fig. 1 | `fig_tbench_cost_bars.pdf` | `experiments/make_cost_bars.py` | same |
| Fig. 2, Fig. 6 | `fig1.pdf`, `fig1_appendix.pdf` | `experiments/cv.py --paper` → `plots/fig1_data/` | `plots/fig1.py` |
| Fig. 3, Fig. 7 | `fig_rev_main.pdf`, `fig_rev_appendix.pdf` | `experiments/cv_replay.py` (called by the drawer) | `experiments/fig_appendix.py` |
| Fig. 4, Fig. 8 | `fig_order.pdf`, `fig_order_app.pdf` | `experiments/cv_order.py` → `plots/fig_order_data/` | `plots/fig_order.py` |
| Fig. 5, Fig. 10 | `fig_calib_gap.pdf`, `fig_calib_gap_app.pdf` | `experiments/validate_calibration.py` → `experiments/val_cache/`, then `experiments/make_fig_passk_data.py` → `plots/fig_passk_data/` | `plots/fig_calib.py` |
| Fig. 9 | `fig_gap_dist_all.pdf` | `plots/fig_gap_dist.py` (caches in `plots/gap_cache/`) | same |

## Method code

- `experiments/arbitrage_tab.py` — the Section 2 solver on tabulated utility curves: the Lagrangian
  λ-sweep with cyclic coordinate ascent (`arbitrage_frontier`, `_coord_ascent`), expected spend under
  a deployment order (`cum_spend`, `expected_spend`), and the greedy population order of Section 3
  (`greedy_order`). `SPEND_CAP` implements "no money past the recorded attempts" for frozen curves.
- `experiments/arbitrage.py` — data loading (`load_results`) and the closed-form geometric model
  used for per-attempt costs and the single-model baselines.
- `experiments/estimator.py` — pass@k curve tables: the unbiased estimator of Chen et al. with
  log-linear interpolation between integer attempts (`chen`), the plug-in (`geom`), and the
  query-level empirical-Bayes estimator of Section 4 (`zibb`: a Beta prior per model with a point
  mass at p = 0 and the (a+b)^-5/2 hyperprior, fit by `fit_zibb`; the per-query posterior predictive
  by `zibb_curve`), used by Figures 5, 9 and 10. `chen_bb` uses Chen within the recorded attempts
  and the Beta posterior predictive without the point mass beyond them (Figure 1 only).
- `experiments/cv.py`, `cv_order.py`, `cv_replay.py` — the cross-validation protocol of Section 5.1
  (leave-one-out on the Terminal-Bench markets, 10-fold on LiveCodeBench and DeepSWE; policies
  indexed by the shadow price λ; the market baseline is the best single model under the same λ).
  `cv_order.py` also computes the best fixed order by dynamic programming over the funded models.
- `experiments/validate_estimator.py`, `validate_calibration.py`, `make_fig_passk_data.py` — the
  protocol of Section 5.2: exact subsampling of the recorded attempts, the held-out Chen oracle,
  arbitrage on the estimated curves scored on the oracle curves.
- `experiments/make_figure.py` — the dataset registry (budget grids, titles); `make_panel.py`
  tabulates curves on the budget grid; `plots/paperstyle.py` and `plots/fig1.py` hold the figure style.

## Data

`experiments/data/<dataset>/<system>.jsonl`, one line per problem:
`{"id", "attempts": [0/1, ...], "mean_cost": cost of one attempt}` (plus token fields where they
were recorded). The builders in `experiments/datasets_build/` regenerate these from the public
sources; they need network access and are not run by `reproduce.sh`.

| Dataset | Systems × problems × attempts | Cost of an attempt | Source |
|---|---|---|---|
| `terminal_bench2_priced` | 28 × 88 × ≤5 | Submitter-reported `cost_usd` per trial where recorded, otherwise input/output/cached tokens × the frozen price sheets in `data/_price_snapshots/` (LiteLLM and OpenRouter, 2026-09-17; input price scaled by a cache multiplier calibrated on the systems that report both). Systems reporting neither cost nor tokens are excluded (75 → 28). | [HF `harborframework/terminal-bench-2-leaderboard`](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard) — `terminal_bench2.py`, `tb2_refetch_tokens.py`, `tb2_build_priced.py`, `tb2_prices.py` |
| `terminal_bench4_priced` | 12 × 66 × 5 | `cost_usd` recorded per trial; one all-zero-cost submission excluded. | [Terminal-Bench 4.0 leaderboard](https://github.com/harbor-framework/terminal-bench/tree/main/leaderboard) — `tb4_build.py` |
| `livecodebench_priced` | 24 × 713 shared × 10 | Prompt and full completion tokens × the same price sheets (one request per attempt, no caching); model tokenizer where public, `cl100k_base` as proxy otherwise; reasoning models excluded (their billed chain of thought is not stored). Some system files contain additional, nonshared problems; the analysis uses the 713-problem intersection. | [LiveCodeBench submissions](https://github.com/LiveCodeBench/submissions) — `lcb_build_priced.py` |
| `deepswe_priced` | 70 × 111 × ≤4 | `cost_usd` recorded per trial; a handful of pairs have fewer than four trials. | [DeepSWE v1.1 public trials](https://deepswe.datacurve.ai) — `deepswe_build.py` |
| `monkey_math`, `monkey_codecontests` | 13 × 128 × 10,000; 5 × 140 × 10,000 | Compute proxy: 2 × parameters (B) × the mean number of tokens the model generates for that problem (all 10,000 samples, chars/4). The three Pythia models under 1B parameters are excluded from MATH (registry). | [HF `ScalingIntelligence/monkey_business`](https://huggingface.co/datasets/ScalingIntelligence/monkey_business) — `monkey_business.py` |
| `resmat2_aime2024`, `resmat2_aime2025`, `resmat2_gmmlu`, `resmat2_mmlupro` | 12 × 30 × 2,048–2,560 each | Compute proxy: active parameter count (B), constant per model; the release holds outcomes only. | [HF `stair-lab/irsl_testtime_resmat2`](https://huggingface.co/datasets/stair-lab/irsl_testtime_resmat2) — `resmat2_build.py` |

Only cost ratios matter to the arbitrageur, so the units of the compute proxies are arbitrary.
Zero-cost arms are excluded everywhere, since a free model would be infinitely cheap to the arbitrageur.

## Conventions

- Utility curves are tabulated on a log-spaced budget grid of 50 points per decade, extended four
  decades below the cheapest attempt. Figures 2–4 and 6–8 use the unbiased Chen estimator frozen
  at the recorded attempts, with no spend past them (`arbitrage_tab.SPEND_CAP`). The oracle curves
  in Figures 5, 9, and 10 also use Chen; their estimated policies use the plug-in or the empirical-Bayes
  curves described above. Figure 1 continues the curves beyond the recorded attempts with the Beta
  posterior predictive without the point mass (`chen_bb`).
- The frontier sweeps 86 shadow prices (`cv.N_LAM + cv.N_HEAD`); each is solved by cyclic coordinate
  ascent with an exact grid argmax per coordinate (`robust=True`).
- Randomness enters only through the fold assignment (`cv.folds`, seed 0) and the calibration draws
  (`validate_estimator.subsample`, seed `SEED + 1000 * n_sub + rep`), so every number is deterministic.

## Runtime

Tracing one frontier takes 0.03–0.35 s per market; the cross-validation (up to 88 folds × 86 λ)
and the best-order dynamic programs are the slow parts, a few minutes each. The 240 calibration
replicates of `validate_calibration.py` take about a minute and run serially so that the results are
bit-for-bit deterministic (`VAL_PARALLEL=1` uses a process pool, which on macOS can change the last
digits of some beta-binomial fits).

`figures/` ships with the output of one such run; `check_reproduction.py` reported an exact match with
the submitted records on the machine the release was prepared on.
