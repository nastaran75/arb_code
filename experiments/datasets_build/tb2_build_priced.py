"""Build Terminal-Bench 2.0 per-system jsonl with cost = tokens x price sheet
(cache-aware), for as many of the 75 systems as we can price.

Reads token records (tb2_tokens.jsonl), prices each trial via tb2_prices, and
cross-checks the computed cost against real cost_usd for the systems that report
it. Per-attempt cost = cache-aware $ per trial.
"""
import json
import os
from collections import defaultdict

import numpy as np

import tb2_prices as P

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.environ.get("ARB_RAW_DIR", os.path.join(HERE, "_raw")), "terminal_bench2")   # raw caches (not committed)
DATA = os.path.join(HERE, "..", "data")
SCRATCH = RAW
TOK = os.path.join(SCRATCH, "tb2_tokens.jsonl")
OUT = os.path.join(DATA, "terminal_bench2_priced")


def _provider(price_or_model):
    return None  # placeholder; provider grouping done by input-price bucket below


def trial_cost(rec, price, eff_in=None):
    """Cache-aware $ for one trial, or None if tokens missing.

    If the trial reports n_cache, price cached reads at the discount. If it does
    NOT (cache-ambiguous), use `eff_in` — an effective input price calibrated
    from systems that report both cost and tokens — instead of the nominal input
    price, to absorb the typical (unreported) caching.
    """
    n_in, n_out = rec.get("n_in"), rec.get("n_out")
    if n_in is None or n_out is None:
        return None, None
    n_cache = rec.get("n_cache")
    if n_cache is not None and n_cache > 0:
        uncached = max(n_in - n_cache, 0)
        cache_price = price["cache_read"] if price["cache_read"] is not None else 0.1 * price["in"]
        return uncached * price["in"] + n_cache * cache_price + n_out * price["out"], "cache"
    # no cache info: use calibrated effective input price if available
    in_price = eff_in if eff_in is not None else price["in"]
    return n_in * in_price + n_out * price["out"], ("calib" if eff_in is not None else "nocache")



# Ensembles (one submission routing across several providers' models) are excluded
# from the Terminal-Bench datasets: they are not a single priced model. Detected by
# a "Multiple" model tag or >= 2 provider names in the model part of agent__model.
_PROVIDER_TOKENS = ("claude", "gpt", "gemini", "minimax", "deepseek", "kimi", "glm", "qwen", "grok")


def is_ensemble(system: str) -> bool:
    model = system.split("__", 1)[1].lower() if "__" in system else system.lower()
    return "multiple" in model or sum(tok in model for tok in _PROVIDER_TOKENS) >= 2


def main():
    os.makedirs(OUT, exist_ok=True)
    # group token records by system -> task -> [rec...]
    recs = defaultdict(lambda: defaultdict(list))
    seen = set()
    for line in open(TOK):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") or (r["sysname"], r["trial"]) in seen:
            continue
        seen.add((r["sysname"], r["trial"]))
        recs[r["sysname"]][r["task"]].append(r)

    systems = sorted(recs)
    sys_price = {s: P.system_price(s) for s in systems}

    # --- calibrate a global cache multiplier from systems reporting BOTH real
    #     cost and tokens: implied_in / nominal_in per trial (absorbs caching) ---
    ratios = []
    for s in systems:
        pr = sys_price[s]
        if pr is None or pr["in"] <= 0:
            continue
        for t in recs[s]:
            for rr in recs[s][t]:
                c, n_in, n_out = rr.get("cost_usd"), rr.get("n_in"), rr.get("n_out")
                if c and n_in and n_out and n_in > 0:
                    imp = (c - n_out * pr["out"]) / n_in
                    if 0 < imp <= pr["in"]:
                        ratios.append(imp / pr["in"])
    cache_mult = float(np.median(ratios)) if ratios else 0.15
    print(f"calibrated cache multiplier (effective/nominal input price): {cache_mult:.3f}  (from {len(ratios)} trials)")

    priced_rows = []
    validation = []
    for s in systems:
        price = sys_price[s]
        real_costs = [rr.get("cost_usd") for t in recs[s] for rr in recs[s][t] if rr.get("cost_usd") is not None]
        has_real = any(c is not None for c in real_costs) and len(real_costs) > 0
        comp, tiers = [], set()
        if price is not None:
            eff_in = price["in"] * cache_mult
            for t in recs[s]:
                for rr in recs[s][t]:
                    c, tier = trial_cost(rr, price, eff_in=eff_in)
                    if c is not None:
                        comp.append(c); tiers.add(tier)
        # cost source priority: real cost > cache-aware/calibrated tokens
        if has_real and np.mean(real_costs) > 0:
            mean_cost, src = float(np.mean(real_costs)), "real"
        elif comp:
            mean_cost, src = float(np.mean(comp)), ("cache" if "cache" in tiers else "calib")
        else:
            mean_cost, src = None, "none"
        if comp and has_real and np.mean(real_costs) > 0:
            validation.append((s, float(np.mean(comp)), float(np.mean(real_costs))))
        priced_rows.append((s, mean_cost, src, len(comp), len(real_costs)))

    # --- telemetry sanity: a token-priced system with implausibly low logged
    #     input (a terminal-agent trial resends large context; real medians are
    #     50k-1.7M) has truncated token logging -> its token-derived cost is
    #     unreliable, so exclude it. Real-cost systems are unaffected. ---
    MIN_IN = 2000
    med_in = {}
    for s in systems:
        vals = [rr.get("n_in") for t in recs[s] for rr in recs[s][t] if rr.get("n_in") is not None]
        med_in[s] = float(np.median(vals)) if vals else 0.0
    src_by_sys = {s: src for s, mc, src, *_ in priced_rows}
    cost_by_sys0 = {s: mc for s, mc, *_ in priced_rows}
    bad_telemetry = sorted(
        s for s in systems
        if cost_by_sys0.get(s) and cost_by_sys0[s] > 0
        and src_by_sys.get(s) in ("cache", "calib") and med_in[s] < MIN_IN
    )
    ensembles = sorted(s for s in systems if is_ensemble(s))
    keep = {s for s, mc, src, *_ in priced_rows
            if mc is not None and mc > 0 and s not in bad_telemetry and s not in ensembles}
    print(f"EXCLUDED ensembles (multi-provider systems): {ensembles}")
    from collections import Counter
    tier_counts = Counter(src for s, mc, src, *_ in priced_rows if mc is not None and mc > 0 and s in keep)
    print(f"cost-source tiers (kept): {dict(tier_counts)}")
    print(f"EXCLUDED for bad telemetry (token-priced, median n_in < {MIN_IN}):")
    for s in bad_telemetry:
        print(f"   {s:46s} src={src_by_sys[s]:6s} median_n_in={med_in[s]:.0f}")
    # shared task intersection across kept systems
    task_sets = {s: set(recs[s]) for s in keep}
    shared = sorted(set.intersection(*task_sets.values())) if task_sets else []

    cost_by_sys = {s: mc for s, mc, *_ in priced_rows}
    written = []
    for s in sorted(keep):
        with open(os.path.join(OUT, f"{s}.jsonl"), "w") as f:
            for task in shared:
                att = [rr["solved"] for rr in recs[s][task]]
                f.write(json.dumps({"id": task, "attempts": att, "mean_cost": cost_by_sys[s]}) + "\n")
        written.append(s)

    print(f"systems with tokens: {len(systems)};  priced+kept: {len(keep)};  shared tasks: {len(shared)}")
    print(f"\n=== VALIDATION: computed (tokens x price, cache-aware) vs real cost_usd ===")
    print(f"{'system':46s} {'computed$':>10} {'real$':>10} {'ratio':>7}")
    for s, comp, real in sorted(validation, key=lambda x: -x[2]):
        print(f"{s:46s} {comp:10.3f} {real:10.3f} {comp/real if real else float('nan'):7.2f}")
    if validation:
        ratios = [c / r for _, c, r in validation if r > 0]
        print(f"\nmedian computed/real ratio: {np.median(ratios):.2f}  (1.0 = perfect)")
    print(f"\nwrote {len(written)} systems to {OUT}")


if __name__ == "__main__":
    main()
