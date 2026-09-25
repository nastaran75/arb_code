"""Section-2 figure: cost vs performance, with per-model curves (gray),
the market/pareto frontier (green), and the optimal arbitrage policy (red).

Style follows Fig. 2 (left) of "Computational Arbitrage in AI Model Markets":
x = performance (% solved), y = cost (log scale).

Default = NO extrapolation past the observed attempts per (problem, model): a
model's solve probability on a problem is frozen after its n observed samples
(extra attempts cost money but add nothing). `--extrapolate` gives the plain
geometric extrapolation 1 - (1-p)^k instead."""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

import arbitrage as arb

# Paper style (Times-matching serif + STIX math, hairlines, Type-42 fonts).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots"))
import paperstyle as PS   # noqa: E402

GRAY = "0.72"
GREEN = (46 / 255, 139 / 255, 87 / 255)
RED = (214 / 255, 40 / 255, 40 / 255)


def make_figure(base_dir, models, rename, title, out, xlabel="Solve rate", n_budget=70,
                bmin=1e-3, bmax=10.0, perf_lo=0.0, ylabel="Cost ($)", extrapolate=False):
    if models is None:  # auto-discover all systems in the data dir
        import glob
        models = sorted(os.path.basename(p)[:-len(".jsonl")]
                        for p in glob.glob(os.path.join(base_dir, "*.jsonl")))
    results = arb.load_results(base_dir, rename=rename)
    alpha, zcap, q, ids = arb.build_alpha(results, models, extrapolate=extrapolate)
    P = len(ids)
    budgets = np.logspace(np.log10(bmin), np.log10(bmax), n_budget)

    # per-model (perf, expected-spend) and arbitrage (perf, expected-spend) vs budget
    model_perf = np.array([arb.single_model_perf(alpha[:, j], zcap[:, j], budgets) for j in range(len(models))])
    model_spend = np.array([arb.single_model_spend(alpha[:, j], zcap[:, j], budgets) for j in range(len(models))])
    arb_perf, arb_spend, allocs = arb.arbitrage_frontier(alpha, zcap, n_budget)
    # clip the frontier to the models' budget floor (sub-bmin allocations give a
    # misleading low-cost tail below every model)
    keep = allocs.sum(1) >= bmin
    arb_perf, arb_spend = arb_perf[keep], arb_spend[keep]

    max_perf = float(arb_perf.max())
    grid = np.linspace(max(perf_lo, model_perf.min()), max_perf, 300)

    model_cost = np.array([arb.invert_to_cost(model_perf[j], model_spend[j], grid) for j in range(len(models))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # all-NaN slices above every model's reach
        cheapest = np.nanmin(model_cost, axis=0)
    market_cost = np.fmax.accumulate(cheapest)                        # pareto frontier (green), monotone ...
    market_cost[np.isnan(cheapest)] = np.nan                          # ... and it STOPS where the best single model stops
    arb_cost = np.fmax.accumulate(arb.invert_to_cost(arb_perf, arb_spend, grid))  # red, monotone

    fig, ax = plt.subplots(figsize=(PS.width(0.5), 2.4))   # final size for a half-\textwidth slot
    x = grid * 100.0
    scale = P  # total $ over the benchmark, like the reference figure

    for j in range(len(models)):
        ax.plot(x, model_cost[j] * scale, color=GRAY, lw=1.4, zorder=2)
    ax.plot(x, market_cost * scale, color=GREEN, lw=2.6, zorder=4, label="Best single model (market)")
    ax.plot(x, arb_cost * scale, color=RED, lw=2.6, zorder=5, label="Arbitrage (optimal allocation)")

    ax.set_yscale("log")
    ax.set_xlabel(f"{xlabel} (%)", fontsize=PS.FS["label"])
    ax.set_ylabel(ylabel, fontsize=PS.FS["label"])
    ax.set_title(title, fontsize=PS.FS["title"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=PS.GRID, lw=0.5, ls="-")
    ax.set_axisbelow(True)
    ax.legend(fontsize=PS.FS["legend"], frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print(f"wrote {out}  (P={P} problems, N={len(models)} models, max arb perf={max_perf:.3f}, "
          f"{'extrapolated' if extrapolate else 'no extrapolation past observed attempts'})")
    # quick report: how much cheaper is arbitrage at the highest solve rate any single model reaches?
    top = np.nanmax(grid[np.isfinite(market_cost)])
    mc = np.interp(top, grid, market_cost)
    ac = np.interp(top, grid, arb_cost)
    print(f"  best single model reaches {top*100:.1f}% (arbitrage {max_perf*100:.1f}%); at {top*100:.1f}%: "
          f"market ${mc*scale:.1f}  arbitrage ${ac*scale:.1f}  savings {100*(1-ac/mc):.0f}%")


HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "monkey_math": dict(
        base_dir=os.path.join(HERE, "data", "monkey_math"),
        # Pythia < 1B (70M/160M/410M) dropped 2026-08-26: with <=3 successes in 10 samples the
        # pass@k tail is unidentifiable (see fig_val_identifiability), so nothing can be predicted.
        models=["Pythia-1B", "Pythia-1.4B", "Pythia-2.8B", "Pythia-6.9B", "Pythia-12B",
                "Gemma-2B", "Gemma-7B", "Llama-3-8B", "Llama-3-8B-Instruct", "Llama-3-70B-Instruct"],
        title="MATH (Monkey Business)", xlabel="MATH solve rate", ylabel="Compute (FLOPs, arb. units)",
        bmin=30.0, bmax=5e7, perf_lo=0.0,
    ),
    "monkey_codecontests": dict(
        base_dir=os.path.join(HERE, "data", "monkey_codecontests"),
        models=["Gemma-2B", "Gemma-7B", "Llama-3-8B", "Llama-3-8B-Instruct", "Llama-3-70B-Instruct"],
        title="CodeContests (Monkey Business)", xlabel="CodeContests solve rate",
        ylabel="Compute (FLOPs, arb. units)",
        bmin=500.0, bmax=5e8, perf_lo=0.0,
    ),
    # IRSL test-time response matrices (Truong et al.; datasets_build/resmat2_build.py):
    # 12 reasoning models, 60 problems each, 2048-2560 attempts per cell. Costs are the
    # active-parameter proxy (arb. units); budgets span one cheap attempt to kcap expensive ones.
    "resmat2_aime2024": dict(
        base_dir=os.path.join(HERE, "data", "resmat2_aime2024"), models=None,
        title="AIME 2024 (IRSL)", xlabel="AIME 2024 solve rate",
        ylabel="Compute (params, arb. units)", bmin=1.0, bmax=1e5, perf_lo=0.0,
    ),
    "resmat2_aime2025": dict(
        base_dir=os.path.join(HERE, "data", "resmat2_aime2025"), models=None,
        title="AIME 2025 (IRSL)", xlabel="AIME 2025 solve rate",
        ylabel="Compute (params, arb. units)", bmin=1.0, bmax=1e5, perf_lo=0.0,
    ),
    "resmat2_gmmlu": dict(
        base_dir=os.path.join(HERE, "data", "resmat2_gmmlu"), models=None,
        title="Global-MMLU-Lite (IRSL)", xlabel="Global-MMLU-Lite solve rate",
        ylabel="Compute (params, arb. units)", bmin=1.0, bmax=1e5, perf_lo=0.0,
    ),
    "resmat2_mmlupro": dict(
        base_dir=os.path.join(HERE, "data", "resmat2_mmlupro"), models=None,
        title="MMLU-Pro (IRSL)", xlabel="MMLU-Pro solve rate",
        ylabel="Compute (params, arb. units)", bmin=1.0, bmax=1e5, perf_lo=0.0,
    ),
    "livecodebench_priced": dict(
        base_dir=os.path.join(HERE, "data", "livecodebench_priced"),
        models=None,  # auto-discover: the $-priced subset built by datasets_build/lcb_build_priced.py
        title="LiveCodeBench (24 systems)", xlabel="LiveCodeBench solve rate",
        ylabel="Cost ($)",
        bmin=1e-7, bmax=2.0, perf_lo=0.0,
    ),
    "deepswe_priced": dict(
        base_dir=os.path.join(HERE, "data", "deepswe_priced"),
        models=None,  # auto-discover: 70 (model, reasoning-effort) configs from datasets_build/deepswe_build.py
        title="DeepSWE (70 systems)", xlabel="DeepSWE solve rate",
        ylabel="Cost ($)",
        bmin=1e-2, bmax=100.0, perf_lo=0.0,
    ),
    "terminal_bench4_priced": dict(
        base_dir=os.path.join(HERE, "data", "terminal_bench4_priced"),
        models=None,  # auto-discover: leaderboard submissions built by datasets_build/tb4_build.py
        title="Terminal-Bench 4.0", xlabel="Terminal-Bench 4.0 solve rate",
        ylabel="Cost ($)",
        bmin=1e-2, bmax=500.0, perf_lo=0.0,
    ),
    "terminal_bench2_priced": dict(
        base_dir=os.path.join(HERE, "data", "terminal_bench2_priced"),
        models=None,  # auto-discover; all token-priced (self-reported Mux__Opus-4.6 and ensembles dropped)
        title="Terminal-Bench 2.0 (28 systems)", xlabel="Terminal-Bench solve rate",
        ylabel="Cost ($)",
        bmin=3e-4, bmax=30.0, perf_lo=0.0,
    ),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="terminal_bench2_priced", choices=list(DATASETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--extrapolate", action="store_true",
                    help="plain geometric extrapolation 1-(1-p)^k past the observed attempts (default: off)")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    out = args.out or os.path.join(HERE, f"fig_{args.dataset}.pdf")
    make_figure(
        base_dir=cfg["base_dir"], models=cfg["models"], rename=None,
        title=cfg["title"], xlabel=cfg["xlabel"], out=out,
        bmin=cfg["bmin"], bmax=cfg["bmax"], perf_lo=cfg["perf_lo"],
        ylabel=cfg.get("ylabel", "Cost ($)"), extrapolate=args.extrapolate,
    )
