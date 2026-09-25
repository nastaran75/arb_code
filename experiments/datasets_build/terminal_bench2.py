#!/usr/bin/env python3
"""Build per-system jsonl datasets from the Terminal-Bench 2.0 leaderboard.

Source: HF dataset `harborframework/terminal-bench-2-leaderboard`. Submissions
live under `submissions/terminal-bench/2.0/<Agent__Model>/`. Each system holds
one or more *run* directories, and each run directory holds one directory per
trial named `<task_name>__<hash>`, each containing a `result.json`.

Relevant fields inside each trial's result.json:
  - task_name                          -> e.g. "terminal-bench/adaptive-rejection-sampler"
  - verifier_result.rewards.reward     -> 1.0 (solved) or 0.0 (not solved); the
                                          whole verifier_result may be null when
                                          the trial crashed (e.g. Docker pull
                                          rate-limit / environment build error).
  - agent_result.cost_usd              -> REAL per-trial USD cost (populated for
                                          only a subset of systems; null for many).
  - agent_result.metadata.totalTokens  -> per-trial token count (subset of systems).

Output contract (one file per system = <Agent__Model> folder name):
    {"id": "<task_name>", "attempts": [1,0,1,...], "mean_cost": <float|null>}
  - attempts: list over that system's trials for the task, 1 iff reward==1.0
    else 0 (a null/failed verifier counts as 0, per contract). Ordered by trial
    directory name for determinism.
  - mean_cost: REAL mean per-attempt USD cost. Per-task mean of non-null
    cost_usd when available, falling back to the system-wide mean; null when the
    system reports no cost at all. NO FLOPs proxy is used (real dollars only).
  - The id (task) set is the intersection of tasks present in EVERY retained
    system, so the set of ids is identical across all system files (the
    downstream figure intersects task ids).

Fetching: the global repo tree is huge (~100k files), so we do NOT list it.
Instead we enumerate directories with HfFileSystem (fast, per-directory) to get
the exact result.json paths, then fetch them concurrently via the /resolve/
endpoint. Parsed per-trial records are cached to the raw-cache directory so re-runs of
the aggregation/report logic are instant.
"""
import os
import sys
import json
import time
import pickle
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from huggingface_hub import HfFileSystem, get_token

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO = "harborframework/terminal-bench-2-leaderboard"
BASE = f"datasets/{REPO}/submissions/terminal-bench/2.0"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
REPO_PREFIX = f"datasets/{REPO}/"

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.environ.get("ARB_RAW_DIR", os.path.join(HERE, "_raw")), "terminal_bench2")   # raw caches (not committed)
DATA = os.path.join(HERE, "..", "data")
SCRATCH = RAW
OUT_DIR = os.path.join(DATA, "terminal_bench2")
CACHE = os.path.join(SCRATCH, "tb2_records.pkl")

MAX_WORKERS = 32
TOKEN = get_token()
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
ENUM_CACHE = os.path.join(SCRATCH, "tb2_enum.pkl")
SHARD = os.path.join(SCRATCH, "tb2_shard.jsonl")  # resumable per-trial records

_local = threading.local()


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #
def list_systems(fs):
    """Return [(system_name, syspath), ...] excluding non-system entries."""
    out = []
    for p in fs.ls(BASE, detail=False):
        name = p.split("/")[-1]
        if name in (".gitkeep", ".DS_Store") or name.endswith((".yaml", ".yml")):
            continue
        out.append((name, p))
    return sorted(out)


def list_result_paths(fs, syspath):
    """Return [(trial_name, resolve_relative_result_path), ...] for a system.

    Aggregates across all run directories of the system.
    """
    paths = []
    for run in fs.ls(syspath, detail=False):
        rn = run.split("/")[-1]
        if rn in (".DS_Store", "metadata.yaml") or rn.endswith((".yaml", ".yml")):
            continue
        try:
            trials = fs.ls(run, detail=False)
        except Exception:
            continue  # not a directory
        for tr in trials:
            tn = tr.split("/")[-1]
            if tn in (".DS_Store",) or tn.endswith((".yaml", ".yml", ".json", ".txt")):
                continue
            rel = tr[len(REPO_PREFIX):] + "/result.json"
            paths.append((tn, rel))
    return paths


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _session():
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _local.s = s
    return s


def fetch_one(args):
    """Fetch and minimally-parse one result.json.

    Returns dict with keys: sysname, trial_name, status, and (on success)
    task, solved, reward, cost, tokens, verif_null, exc_type.
    status is one of: 'ok', 'missing' (404), 'error'.
    """
    sysname, trial_name, rel = args
    url = RESOLVE + rel
    last = "retries exhausted"
    for attempt in range(6):
        try:
            r = _session().get(url, timeout=60)
            if r.status_code == 200:
                d = r.json()
                rec = _parse(sysname, trial_name, d)
                rec["rel"] = rel
                return rec
            if r.status_code == 404:
                return {"sysname": sysname, "trial_name": trial_name,
                        "status": "missing", "rel": rel}
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"http {r.status_code}"
                time.sleep(min(30.0, 3.0 * (attempt + 1))); continue
            return {"sysname": sysname, "trial_name": trial_name,
                    "status": "error", "detail": f"http {r.status_code}", "rel": rel}
        except Exception as e:  # noqa
            last = f"exc {e}"
            time.sleep(min(30.0, 3.0 * (attempt + 1)))
    return {"sysname": sysname, "trial_name": trial_name, "status": "error",
            "detail": last, "rel": rel}


# A few systems spell the same task differently; normalize to the majority form
# so the cross-system task-id set is consistent (e.g. 89 shared instead of 88).
TASK_ALIASES = {"install-windows-3-11": "install-windows-3.11"}


def norm_task(t):
    return TASK_ALIASES.get(t, t)


def _parse(sysname, trial_name, d):
    tn = d.get("task_name") or ""
    task = norm_task(tn.split("/")[-1] if tn else trial_name.rsplit("__", 1)[0])
    vr = d.get("verifier_result")
    verif_null = vr is None
    reward = None
    if vr is not None:
        rewards = vr.get("rewards") or {}
        reward = rewards.get("reward")
    solved = 1 if reward == 1.0 else 0
    ar = d.get("agent_result") or {}
    cost = ar.get("cost_usd")
    meta = ar.get("metadata") or {}
    tokens = meta.get("totalTokens")
    exc = d.get("exception_info") or {}
    exc_type = exc.get("exception_type") if exc else None
    return {
        "sysname": sysname, "trial_name": trial_name, "status": "ok",
        "task": task, "solved": solved, "reward": reward,
        "cost": cost if isinstance(cost, (int, float)) else None,
        "tokens": tokens if isinstance(tokens, (int, float)) else None,
        "verif_null": verif_null, "exc_type": exc_type,
    }


def fetch_all(use_cache=True):
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"[cache] loaded {len(data['records'])} records for "
              f"{len(data['systems'])} systems from {CACHE}")
        return data

    if os.path.exists(ENUM_CACHE):
        with open(ENUM_CACHE, "rb") as f:
            enum = pickle.load(f)
        system_names, tasks_all = enum["systems"], enum["tasks_all"]
        print(f"[enum-cache] {len(system_names)} systems, {len(tasks_all)} paths")
    else:
        fs = HfFileSystem()
        systems = list_systems(fs)
        system_names = [s for s, _ in systems]
        print(f"[enum] {len(systems)} systems")
        tasks_all = []          # (sysname, trial_name, rel)
        t0 = time.time()
        for i, (sysname, syspath) in enumerate(systems):
            for trial_name, rel in list_result_paths(fs, syspath):
                tasks_all.append((sysname, trial_name, rel))
        print(f"[enum] {len(tasks_all)} result.json paths across systems "
              f"({time.time()-t0:.1f}s)")
        with open(ENUM_CACHE, "wb") as f:
            pickle.dump({"systems": system_names, "tasks_all": tasks_all}, f)

    # --- resumable sharded fetch ---------------------------------------- #
    # Every completed record is appended (as one json line) to SHARD, keyed by
    # 'rel'. A re-run reloads SHARD and only fetches the paths still missing or
    # previously errored, so rate-limit throttling / process kills never lose
    # progress. Only 'ok' and 'missing' are considered final; 'error' is retried.
    done_recs = {}
    if os.path.exists(SHARD):
        with open(SHARD) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") in ("ok", "missing"):
                    done_recs[r["rel"]] = r
        print(f"[shard] resumed {len(done_recs)} finalized records from {SHARD}")

    # Multiple internal passes: each pass only retries paths not yet finalized,
    # so throttling just spreads work over more passes instead of losing it.
    MAX_PASSES = 12
    for pass_i in range(1, MAX_PASSES + 1):
        remaining = [t for t in tasks_all if t[2] not in done_recs]
        if not remaining:
            break
        print(f"[fetch] pass {pass_i}: {len(remaining)} to fetch "
              f"({len(done_recs)}/{len(tasks_all)} already finalized)", flush=True)
        t1 = time.time()
        done = 0
        lock = threading.Lock()
        shard_fh = open(SHARD, "a")
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                for rec in ex.map(fetch_one, remaining):
                    done += 1
                    if rec.get("status") in ("ok", "missing"):
                        done_recs[rec["rel"]] = rec
                        with lock:
                            shard_fh.write(json.dumps(rec) + "\n")
                            if done % 200 == 0:
                                shard_fh.flush()
                    if done % 5000 == 0:
                        print(f"  pass {pass_i}: {done}/{len(remaining)} "
                              f"(finalized {len(done_recs)}/{len(tasks_all)}, "
                              f"{time.time()-t1:.0f}s)", flush=True)
        finally:
            shard_fh.flush(); shard_fh.close()
        n_err = len(tasks_all) - len(done_recs)
        print(f"[fetch] pass {pass_i} done in {time.time()-t1:.0f}s; "
              f"finalized {len(done_recs)}/{len(tasks_all)} (remaining: {n_err})",
              flush=True)
        if n_err > 0 and pass_i < MAX_PASSES:
            time.sleep(15)  # let any rate-limit window reset

    n_err = len(tasks_all) - len(done_recs)
    if n_err > 0:
        print(f"[fetch] STILL {n_err} paths not finalized after {MAX_PASSES} passes "
              f"-- re-run to resume; NOT writing final cache.", flush=True)
        raise SystemExit(2)

    # assemble records in tasks_all order
    records = [done_recs[rel] for (_, _, rel) in tasks_all]
    data = {"systems": system_names, "records": records}
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] wrote {CACHE} ({len(records)} records)")
    return data


# --------------------------------------------------------------------------- #
# Aggregation + write
# --------------------------------------------------------------------------- #
def aggregate(data):
    """system -> {task -> {'attempts': {trial_name: solved}, 'costs': [..]}}."""
    per_sys = defaultdict(lambda: defaultdict(lambda: {"att": {}, "costs": []}))
    counts = defaultdict(lambda: Counter())  # per-system status/quality counters
    for rec in data["records"]:
        s = rec["sysname"]
        st = rec["status"]
        counts[s][st] += 1
        if st != "ok":
            continue
        counts[s]["trials"] += 1
        if rec["verif_null"]:
            counts[s]["verif_null"] += 1
        if rec["cost"] is not None:
            counts[s]["cost_nonnull"] += 1
        if rec["tokens"] is not None:
            counts[s]["tok_nonnull"] += 1
        t = norm_task(rec["task"])  # normalize (also fixes already-cached records)
        per_sys[s][t]["att"][rec["trial_name"]] = rec["solved"]
        if rec["cost"] is not None:
            per_sys[s][t]["costs"].append(rec["cost"])
    return per_sys, counts



# Ensembles (one submission routing across several providers' models) are excluded
# from the Terminal-Bench datasets: they are not a single priced model. Detected by
# a "Multiple" model tag or >= 2 provider names in the model part of agent__model.
_PROVIDER_TOKENS = ("claude", "gpt", "gemini", "minimax", "deepseek", "kimi", "glm", "qwen", "grok")


def is_ensemble(system: str) -> bool:
    model = system.split("__", 1)[1].lower() if "__" in system else system.lower()
    return "multiple" in model or sum(tok in model for tok in _PROVIDER_TOKENS) >= 2


def main(use_cache=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = fetch_all(use_cache=use_cache)
    per_sys, counts = aggregate(data)
    ensembles = sorted(s for s in per_sys if is_ensemble(s))
    print(f"[filter] excluding {len(ensembles)} ensembles (multi-provider systems): {ensembles}")
    systems = sorted(s for s in per_sys if s not in ensembles)

    # --- report enumeration/fetch errors (non-404) ---
    for s in systems:
        if counts[s]["error"]:
            print(f"[WARN] {s}: {counts[s]['error']} hard fetch errors")

    # --- task-count per system + intersection ---
    task_sets = {s: set(per_sys[s].keys()) for s in systems}
    task_counts = {s: len(task_sets[s]) for s in systems}
    tc_dist = Counter(task_counts.values())
    print("\n[tasks] distinct-task-count distribution across systems:",
          dict(sorted(tc_dist.items())))

    # Full intersection across all systems.
    inter = set.intersection(*task_sets.values()) if task_sets else set()
    union = set.union(*task_sets.values()) if task_sets else set()
    print(f"[tasks] intersection across all {len(systems)} systems = {len(inter)} "
          f"(union = {len(union)})")

    # If a few outlier systems shrink the intersection, identify them.
    from collections import Counter as C
    task_freq = C()
    for s in systems:
        task_freq.update(task_sets[s])
    n_sys = len(systems)
    # tasks present in >= n_sys-2 systems (near-universal)
    near = {t for t, c in task_freq.items() if c >= n_sys - 2}
    reducers = {}
    for s in systems:
        missing = near - task_sets[s]
        if missing:
            reducers[s] = len(missing)
    if reducers:
        print(f"[tasks] systems missing >=1 near-universal task: "
              f"{ {k: v for k, v in sorted(reducers.items(), key=lambda x:-x[1])} }")

    shared_ids = sorted(inter)
    print(f"[tasks] writing {len(shared_ids)} shared task ids per system")

    # --- per-system cost summary ---
    def system_mean_cost(s):
        allc = [c for t in per_sys[s] for c in per_sys[s][t]["costs"]]
        return float(np.mean(allc)) if allc else None

    # --- write files + collect stats ---
    written = []
    stats = []
    for s in systems:
        smean = system_mean_cost(s)
        outpath = os.path.join(OUT_DIR, f"{s}.jsonl")
        pass1_vals = []
        ntrials_list = []
        with open(outpath, "w") as f:
            for tid in shared_ids:
                rec = per_sys[s][tid]
                # deterministic order by trial name
                att = [rec["att"][k] for k in sorted(rec["att"].keys())]
                ntrials_list.append(len(att))
                pass1_vals.append(sum(att) / len(att) if att else 0.0)
                # per-task cost -> fall back to system mean -> null
                if rec["costs"]:
                    mc = float(np.mean(rec["costs"]))
                elif smean is not None:
                    mc = smean
                else:
                    mc = None
                f.write(json.dumps({"id": tid, "attempts": att, "mean_cost": mc}) + "\n")
        written.append(outpath)
        ndist = Counter(ntrials_list)
        n_repr = list(ndist)[0] if len(ndist) == 1 else dict(ndist)
        stats.append({
            "system": s,
            "n_tasks_written": len(shared_ids),
            "n_tasks_total": task_counts[s],
            "n": n_repr,
            "mean_pass1": float(np.mean(pass1_vals)) if pass1_vals else 0.0,
            "mean_cost": smean,
            "verif_null": counts[s]["verif_null"],
            "trials": counts[s]["trials"],
            "cost_nonnull": counts[s]["cost_nonnull"],
            "tok_nonnull": counts[s]["tok_nonnull"],
            "missing_404": counts[s]["missing"],
            "hard_errors": counts[s]["error"],
        })

    # --- print summary table sorted by solve rate ---
    stats.sort(key=lambda r: r["mean_pass1"], reverse=True)
    print("\n=== SUMMARY (sorted by mean pass@1) ===")
    hdr = f"{'system':42s} {'#tasks':>6s} {'n':>6s} {'pass@1':>7s} {'$/att':>8s} {'null%':>6s} {'cost?':>5s}"
    print(hdr)
    for r in stats:
        mc = "  n/a" if r["mean_cost"] is None else f"{r['mean_cost']:.3f}"
        nullpct = 100.0 * r["verif_null"] / r["trials"] if r["trials"] else 0.0
        hascost = "yes" if r["cost_nonnull"] > 0 else "no"
        nrep = str(r["n"])
        print(f"{r['system']:42s} {r['n_tasks_written']:6d} {nrep:>6s} "
              f"{r['mean_pass1']:7.3f} {mc:>8s} {nullpct:5.1f} {hascost:>5s}")

    # --- ranges ---
    p1s = [r["mean_pass1"] for r in stats]
    costs = [r["mean_cost"] for r in stats if r["mean_cost"] is not None]
    n_with_cost = sum(1 for r in stats if r["mean_cost"] is not None)
    print("\n=== RANGES ===")
    print(f"systems written           : {len(written)}")
    print(f"pass@1  min/median/max    : {min(p1s):.3f} / {np.median(p1s):.3f} / {max(p1s):.3f}")
    if costs:
        print(f"$/attempt min/median/max  : {min(costs):.4f} / {np.median(costs):.4f} / {max(costs):.4f}")
    print(f"systems WITH real cost    : {n_with_cost}/{len(stats)}")
    print(f"shared task ids           : {len(shared_ids)}")

    # --- machine-readable summary ---
    summary = {
        "n_systems": len(systems),
        "n_shared_ids": len(shared_ids),
        "shared_ids": shared_ids,
        "union_task_count": len(union),
        "task_count_distribution": {str(k): v for k, v in sorted(tc_dist.items())},
        "intersection_reducers": reducers,
        "systems_with_cost": n_with_cost,
        "pass1_range": [float(min(p1s)), float(np.median(p1s)), float(max(p1s))],
        "cost_range": ([float(min(costs)), float(np.median(costs)), float(max(costs))]
                       if costs else None),
        "table": stats,
    }
    with open(os.path.join(OUT_DIR, "_build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote summary ->", os.path.join(OUT_DIR, "_build_summary.json"))
    print("Files written:", len(written))
    return summary


if __name__ == "__main__":
    use_cache = "--no-cache" not in sys.argv
    t0 = time.time()
    main(use_cache=use_cache)
    print("\nElapsed: %.1fs" % (time.time() - t0))
