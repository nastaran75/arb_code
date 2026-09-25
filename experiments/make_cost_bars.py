"""Paper figure: total cost (USD) to reach a target Terminal-Bench solve rate, per system.

Two stacked bar panels (one per target solve rate, e.g. 40% and 70%). EVERY system
appears in BOTH panels, in the SAME left->right order, so a reader can look up/down
to compare one system across the two targets (e.g. the cheapest systems at 70% are
mid-pack at 40%). By default the curves are not extrapolated past the observed
attempts. A dataset may instead request the paper's query-level Beta--Binomial
empirical-Bayes estimator, in which case the figure is explicitly marked as predicted.
Systems that cannot reach a target get a grey "NA" instead of a bar; systems that
reach no target are dropped from the figure.

y = total expected $ to reach the target on all tasks via repeated sampling with
early stopping on success  (= expected cost per task x number of tasks).
Bars are coloured by the provider of the underlying model.

No arbitrage overlay -- just the per-system market cost.
"""
import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import arbitrage as arb

# The paper style lives with the other paper figures; importing it applies the rcParams
# (Times-matching serif + STIX math, hairlines, Type-42 fonts) for every bar figure too.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots"))
import paperstyle as PS   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Figures that appear in the paper are mirrored into <release>/figures/ under the same filename.
PAPER_FIGDIR = os.path.join(HERE, "..", "figures")


def mirror_to_paper(out):
    """Copy `out` into figures/ keeping its basename. No-op if that directory is absent."""
    if not os.path.isdir(PAPER_FIGDIR):
        print(f"note: {os.path.normpath(PAPER_FIGDIR)} does not exist; not mirroring")
        return
    dst = os.path.join(PAPER_FIGDIR, os.path.basename(out))
    shutil.copy(out, dst)
    print(f"mirrored to {os.path.normpath(dst)}")

# Datasets the bar figures can be drawn for (make_acc_bars / make_rev_bars / make_paired_diff
# take --dataset). `budgets` = the two per-query budgets shown; `grid` = log10 range of the
# per-task budget grid the single-system curves are tabulated on (make_figure's bmin/bmax);
# `stem` names the output files; `other_label` is the legend entry of the 4th colour.
DATASETS = {
    "terminal_bench2_priced": dict(
        base=os.path.join(HERE, "data", "terminal_bench2_priced"), name="Terminal-Bench 2.0",
        stem="tbench", budgets=[0.15, 1.0], grid=(-4.0, np.log10(10.0)),
        estimator="chen_bb",
        other_label="Other (MiniMax / DeepSeek / Kimi / GLM)"),
    "deepswe_priced": dict(
        base=os.path.join(HERE, "data", "deepswe_priced"), name="DeepSWE",
        stem="deepswe", budgets=[2.0, 10.0], grid=(-2.0, np.log10(200.0)),
        # 70 configs = 28 models x reasoning efforts; efforts of one model cost within ~2x of
        # each other and crowd the bars, so keep ONE config per model: the cheapest that reaches
        # the sort target (dedupe below).
        dedupe=True, estimator="chen_bb",
        other_label="Other (DeepSeek / GLM / Kimi / Grok / Qwen / Muse)"),
    "terminal_bench4_priced": dict(
        base=os.path.join(HERE, "data", "terminal_bench4_priced"), name="Terminal-Bench 4.0",
        stem="tbench4", budgets=[2.0, 10.0], grid=(-2.0, np.log10(500.0)),
        estimator="chen_bb",
        other_label="Other (GLM / Grok / Gemini)"),
    "livecodebench_priced": dict(
        base=os.path.join(HERE, "data", "livecodebench_priced"), name="LiveCodeBench",
        stem="lcb", budgets=[0.0002, 0.01], grid=(-7.0, np.log10(2.0)),   # the two peaks of the arbitrageur's gain
        estimator="chen_bb",
        other_label="Other (Qwen / DeepSeek / Llama / Mistral)"),
}
DEFAULT_KEY = "terminal_bench2_priced"
BASE = DATASETS[DEFAULT_KEY]["base"]

# Per-dataset solve-rate targets, one panel each. Chosen so the first is reachable by nearly every
# system and the second by roughly half: on LiveCodeBench, whose systems top out between 28% and
# 71%, Terminal-Bench's 40/70 pair would make the second panel 23 NAs and one bar.
TARGETS_BY_KEY = {"terminal_bench2_priced": [0.40, 0.70], "livecodebench_priced": [0.30, 0.50],
                  "deepswe_priced": [0.50, 0.80], "terminal_bench4_priced": [0.30, 0.60]}
CAP_X_MEDIAN = 20                # bar-height cap: cost > this multiple of the panel median -> NA
SORT_BY = 0                      # index into TARGETS: which panel's cost fixes the x-order
                                 # (systems that don't reach that target go last, sorted by
                                 # the other target's cost, then by name)
# NOTE: the dataset dir is the single source of truth for which systems are
# included. The lone self-reported-cost system (Mux__Claude-Opus-4.6) was dropped
# there (data/_dropped_selfreported/), so every system here is on our token x
# cache-aware price sheet -- one uniform cost model.

# provider palette (fixed legend order). Brand hues for the big three; the rest
# chosen for separation (checked: all pairs OKLab dE >= 15 normal vision; the only
# sub-8 CVD pairs are the brand green/terracotta (protan) and green/blue (tritan),
# and every bar is also named by its tick label).
PROVIDERS = [  # (key, legend label, face colour, hatch)
    ("openai",    "OpenAI (GPT)",                                "#0F9D7A", None),
    ("anthropic", "Anthropic (Claude)",                          "#D97757", None),
    ("google",    "Google (Gemini)",                             "#3B78E7", None),
    ("other",     "Other (MiniMax / DeepSeek / Kimi / GLM)",   "#7B3FA0", None),
]
NA_COLOR = "0.55"


def providers(ds):
    """PROVIDERS with the dataset's own label for the 'other' group."""
    return [(k, ds["other_label"] if k == "other" else lab, face, hatch) for k, lab, face, hatch in PROVIDERS]

ARB_COLOR = "#B03A2E"            # the arbitrageur (optimal mix over all systems): crimson,
ARB_HATCH = "///"                # white diagonal hatching marks it as a different kind of bar


def provider(name: str) -> str:
    # Classify by the MODEL half of Agent__Model: on TB4 the agent brand collides with the model
    # ("Claude-Code__GLM-5.3" is a Z.ai model run by an Anthropic agent, not an Anthropic system).
    s = (name.split("__", 1)[1] if "__" in name else name).lower()
    if s.startswith("polaris") or "multiple" in s:
        raise ValueError(f"{name!r} is an ensemble; ensembles were dropped from the tbench data "
                         "(data/_dropped_ensembles/) and are not part of the figures")
    if "claude" in s or "fable" in s or "opus" in s or "sonnet" in s:
        return "anthropic"
    if "gemini" in s:
        return "google"
    if "gpt" in s or "codex" in s:
        return "openai"
    if any(k in s for k in ("minimax", "deepseek", "kimi", "glm",            # Terminal-Bench
                            "qwen", "dscoder", "llama", "mistral", "codestral",     # LiveCodeBench
                            "grok", "muse", "kimi")):                               # DeepSWE / TB4
        return "other"
    raise ValueError(f"unknown provider for {name!r}")


EFFORTS = ("low", "medium", "high", "xhigh", "max")


def family(name: str) -> str:
    """DeepSWE config name without its reasoning-effort suffix."""
    for e in EFFORTS:
        if name.endswith("_" + e):
            return name[: -len(e) - 1]
    return name


def short_deepswe(name: str) -> str:
    fam, eff = family(name), ""
    for e in EFFORTS:
        if name.endswith("_" + e):
            eff = e
    parts = fam.split("_")
    label = "-".join(w.capitalize() if w.isalpha() else w for w in parts)
    label = (label.replace("Claude-", "").replace("Gpt", "GPT").replace("Glm", "GLM")
                  .replace("Deepseek", "DeepSeek").replace("-Preview", ""))
    return f"{label} ({eff})" if eff else label


def short(name: str) -> str:
    if "__" not in name and name == name.lower():          # DeepSWE config, not Agent__Model
        return short_deepswe(name)
    a, _, m = name.replace(" (N=10)", "").partition("__")
    m = (m.replace("-Preview", "").replace("gpt-5.3-codex", "GPT-5.3-Codex")
          .replace("Claude-Opus-4-7", "Claude-Opus-4.7"))
    a = a.replace("IndusAGICodingAgent", "IndusAGI").replace("Gemini_CLI", "Gemini-CLI")
    return f"{a} · {m}" if m else a


def load(estimator="chen", ds=None):
    """Single-model (solve rate, expected spend) curves per system on a per-task budget grid.
    estimator='chen' (default): unbiased pass@k with continuous time within an attempt, tabulated
    (estimator.py / arbitrage_tab.py). estimator='chen_bb': Chen where k is supported by the
    observed attempts, followed by the paper's query-level Beta--Binomial empirical-Bayes
    posterior predictive beyond that range.
    Returns (models, perf, spend, n_tasks, alpha, zcap, nq, q, tab) with tab = dict(U, cumsp, budgets)
    for the tabulated path (None for 'geom'). `ds`: a DATASETS entry (default: Terminal-Bench)."""
    import glob
    ds = ds or DATASETS[DEFAULT_KEY]
    base = ds["base"]
    models = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(base, "*.jsonl")))
    results = arb.load_results(base, rename=None)
    alpha, zcap, q, ids = arb.build_alpha(results, models, extrapolate=False)
    # nq[x, j] = n_j(x) * q_j(x): what it costs to run ALL observed attempts of system j
    # on problem x (the budget past which the no-extrapolation cap binds on that problem)
    by_id = {m: {r["id"]: r for r in results[m]} for m in models}
    nq = np.array([[len(by_id[m][pid]["attempts"]) * by_id[m][pid]["mean_cost"] for m in models]
                   for pid in ids])
    budgets = np.logspace(ds["grid"][0], ds["grid"][1], 600)
    perf, spend, tab = {}, {}, None
    if estimator in ("chen", "bb", "chen_bb", "zibb"):
        import estimator as est
        import arbitrage_tab as T
        U, _ = est.pass_curves(results, models, ids, q, budgets, estimator)   # per-problem attempt costs
        cumsp = [T.cum_spend(U[:, j, :], budgets) for j in range(len(models))]
        tab = dict(U=U, cumsp=cumsp, budgets=budgets)
        for j, m in enumerate(models):
            p, s = U[:, j, :].mean(0), cumsp[j].mean(0)
            o = np.argsort(p)
            perf[m], spend[m] = p[o], s[o]
    else:
        for j, m in enumerate(models):
            p = arb.single_model_perf(alpha[:, j], zcap[:, j], budgets)
            s = arb.single_model_spend(alpha[:, j], zcap[:, j], budgets)
            o = np.argsort(p)
            perf[m], spend[m] = p[o], s[o]
    return models, perf, spend, len(ids), alpha, zcap, nq, q, tab


def cost_to(perf, spend, m, x):
    """Expected $ per task to reach solve rate x with system m; NaN if unreachable."""
    pc, sc = perf[m], spend[m]
    return np.nan if x > pc[-1] + 1e-9 else float(np.interp(x, pc, sc))


def main(key=DEFAULT_KEY, estimator_override=None, suffix="", targets=None, paper_order=False):
    ds = DATASETS[key]
    estimator = estimator_override or ds.get("estimator", "chen")
    TARGETS = list(targets) if targets else TARGETS_BY_KEY[key]
    PROVIDERS_DS = providers(ds)                   # 'other' group is named per dataset
    models, perf, spend, n_tasks, *_ = load(estimator, ds)

    # total cost over all tasks, per system, per target (NaN = target unreachable)
    total = {t: {m: cost_to(perf, spend, m, t) * n_tasks for m in models} for t in TARGETS}

    # Outlier cap: a system costing more than CAP_X_MEDIAN times the panel's median is shown as
    # NA rather than as a bar -- one pathological system (AfterQuery's gpt-oss-20b on
    # Terminal-Bench needs ~2 orders of magnitude more than the field at 40%) would otherwise
    # set the y-scale and flatten every other bar. NA here reads as "not at a relevant price",
    # which is the honest summary; the caption should say the cap.
    for t in TARGETS:
        vals = np.array([v for v in total[t].values() if np.isfinite(v)])
        if vals.size:
            cap = CAP_X_MEDIAN * float(np.median(vals))
            for m, v in total[t].items():
                if np.isfinite(v) and v > cap:
                    print(f"capped to NA at {int(t*100)}%: {short(m)} (${v:,.0f} > "
                          f"{CAP_X_MEDIAN}x median ${np.median(vals):,.0f})")
                    total[t][m] = np.nan

    # drop systems that reach NONE of the targets (they'd be NA in every panel)
    if ds.get("dedupe"):
        # one config per model family: the cheapest that reaches the sort target (NaN = +inf),
        # tie-broken by the other target's cost. Removes the near-duplicate effort variants.
        keep = {}
        for m in models:
            f = family(m)
            def rank(mm):
                # Prefer a config that reaches the HARDEST target (else the bottom panel fills
                # with NAs for models whose cheap effort was kept); among those, cheapest there,
                # then cheapest at the sort target.
                cs = [total[t][mm] for t in TARGETS]
                hard = cs[-1]
                return (np.isnan(hard), np.inf if np.isnan(hard) else hard,
                        np.inf if np.isnan(cs[SORT_BY]) else cs[SORT_BY], mm)
            if f not in keep or rank(m) < rank(keep[f]):
                keep[f] = m
        kept = sorted(keep.values())
        print(f"dedupe: {len(models)} configs -> {len(kept)} (one per model)")
        models = kept
    dropped = [] if paper_order else [m for m in models if all(np.isnan(total[t][m]) for t in TARGETS)]
    models = [m for m in models if m not in dropped]
    if dropped:
        print("dropped (NA at every target):", ", ".join(short(m) for m in dropped))

    # one x-order for every panel: by cost at TARGETS[SORT_BY]; unreachables last.
    # With --paper-order the key is the DATASET'S OWN first target, not the overridden one,
    # so a --targets figure lines up bar-for-bar with the paper's (and keeps every system,
    # since dropping the ones unreachable at the new targets would break the alignment).
    t_sort = TARGETS_BY_KEY[key][SORT_BY] if paper_order else TARGETS[SORT_BY]
    t_other = [t for t in TARGETS if t != t_sort]
    if t_sort not in total:
        total[t_sort] = {m: cost_to(perf, spend, m, t_sort) * n_tasks for m in models}

    def sort_key(m):
        c = total[t_sort][m]
        if not np.isnan(c):
            return (0, c, m)
        tie = [total[t][m] for t in t_other if not np.isnan(total[t][m])]
        return (1, min(tie) if tie else np.inf, m)

    order = sorted(models, key=sort_key)
    xs = np.arange(len(order))
    prov = np.array([provider(m) for m in order])

    # independent y-axes: the harder-target costs are several x the easier ones; a
    # shared scale would crush the top panel.
    # Drawn at final size: included at width=\textwidth (5.5in), so LaTeX does not rescale
    # the figure and every point size below is the size it prints at.
    # ~1.1in per panel plus a fixed block for the rotated model names. The block is sized
    # for the 35-degree labels below: their height is (length x sin(35)), ~19% shorter than
    # the 45 degrees this figure used before, so the allowance drops from 1.45in to 1.24in.
    fig, axes = plt.subplots(len(TARGETS), 1, sharex=True,
                             figsize=(PS.width(1.0), 1.10 * len(TARGETS) + 1.24))

    for panel, (ax, t) in enumerate(zip(axes, TARGETS)):
        vals = np.array([total[t][m] for m in order])
        ok = ~np.isnan(vals)
        # The target label is centred, so reserve headroom against the bars it could land on
        # rather than against the tallest bar overall: without this the 70% label sits on top
        # of whichever mid-order system happens to be expensive.
        ymax = np.nanmax(vals) * 1.08
        mid = ok & (xs > 0.25 * len(order)) & (xs < 0.75 * len(order))
        label_y = 0.82 if panel == 0 else 0.99            # first panel: label sits INSIDE the
        floor = 0.88 if panel == 0 else 0.86              # plot, so the legend can sit flush
        if mid.any():
            ymax = max(ymax, float(np.nanmax(vals[mid])) / (label_y * floor))
        for pkey, _, face, hatch in PROVIDERS_DS:
            sel = ok & (prov == pkey)
            if not sel.any():
                continue
            ax.bar(xs[sel], vals[sel], color=face, width=0.78, zorder=3,
                   edgecolor="0.45" if hatch else "white", linewidth=0.5, hatch=hatch)
        for xi in xs[~ok]:
            ax.text(xi, ymax * 0.015, "NA", ha="center", va="bottom",
                    fontsize=PS.FS["dense"], rotation=90, color=NA_COLOR, fontweight="bold")
        target_label = f"Target: {int(t*100)}% solve rate"
        # The paper's default (chen_bb) is NOT tagged in the panel title: the caption already
        # says the curves are extrapolated past the recorded attempts, and "(Chen/EB)" is
        # unexplained jargon in Figure 1. The exploratory estimators stay tagged.
        if estimator in ("bb", "zibb"):
            target_label += " (EB prediction)"
        ax.text(0.5, label_y, target_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=PS.FS["annot"], fontweight="bold")
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Total cost (USD)", fontsize=PS.FS["label"])
        ax.tick_params(axis="y", labelsize=PS.FS["tick"])
        ax.grid(axis="y", color=PS.GRID, lw=0.5, ls="-")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)

    axes[-1].set_xticks(xs)
    # 35 degrees, not 45: a label anchored at its tick rises by (length x sin(angle)), so a
    # FLATTER angle is what buys back figure height -- sin(35)/sin(45) is a ~19% shorter block.
    # Not flatter still: neighbouring baselines are (tick spacing x sin(angle)) apart, which at
    # 30 degrees closes to roughly the text's own line height and the names start to collide.
    axes[-1].set_xticklabels([short(m) for m in order], rotation=35, ha="right",
                             rotation_mode="anchor", fontsize=PS.FS["dense"])

    handles = [Patch(facecolor=face, hatch=hatch, edgecolor="0.45" if hatch else "white",
                     linewidth=0.5, label=lab) for _, lab, face, hatch in PROVIDERS_DS]
    # Shared legend above both panels rather than inside the first: it belongs to the whole
    # figure, and taking it out of the axes frees the top of the 40% panel for the target
    # label. ncol=2 (not 4) because the four provider labels on one row are ~6.3in wide,
    # which would push the saved figure past \textwidth and make LaTeX scale it down.
    fig.tight_layout(h_pad=0.6)
    # Anchored to the TOP PANEL, not the figure: anchoring to the figure would leave
    # tight_layout's top margin as dead space between the legend and the first panel.
    # y=1.0 with borderpad=0 keeps the legend tight against the top spine; any larger
    # offset reads as a floating band because bbox_inches="tight" crops to the legend.
    # y=0.96, i.e. the legend's bottom row dips just INTO the top of the first panel. That
    # band is empty now that the 40% label sits lower in the plot, and overlapping it is what
    # closes the gap: anchored at 1.0 the legend's own row height reads as floating.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.96),
               bbox_transform=axes[0].transAxes,
               fontsize=PS.FS["legend"], frameon=False, ncol=2, handlelength=1.6,
               columnspacing=1.2, labelspacing=0.2, borderpad=0.0)
    out = os.path.join(HERE, f"fig_{ds['stem']}_cost_bars{suffix}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}  ({len(order)} systems x {len(TARGETS)} targets, {n_tasks} tasks)")
    if key == DEFAULT_KEY and not suffix and targets is None:
        mirror_to_paper(out)                       # Figure 1 of the paper; other datasets are not in it


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(TARGETS_BY_KEY), default=DEFAULT_KEY)
    ap.add_argument("--estimator", choices=("chen", "bb", "chen_bb", "zibb", "geom"))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--paper-order", action="store_true",
                    help="order the bars (and keep the systems) exactly as the paper's figure for "
                         "this dataset, so a --targets variant lines up with it bar-for-bar")
    ap.add_argument("--targets", type=float, nargs="+",
                    help="override the per-dataset target list, e.g. --targets 0.4 0.7 0.8 0.9 "
                         "(never mirrored into the paper)")
    args = ap.parse_args()
    main(args.dataset, args.estimator, args.suffix, args.targets, args.paper_order)
