"""Build DeepSWE v1.1 (datacurve.ai) into per-system {id, attempts, mean_cost} jsonl.

Source: https://deepswe.datacurve.ai/artifacts/v1.1/trials.json (51 MB; the endpoint behind the
public data browser at deepswe.datacurve.ai/data/v1.1). 70 configs = 28 models x reasoning
efforts under the mini-swe-agent harness, 113 tasks, ~4 trials per (config, task) cell, with a
REAL per-trial cost_usd -- so unlike the Monkey Business sets the per-attempt cost here varies
per task (CV 0.4-1.0 within a config), and `mean_cost` is written PER ROW (per task).

Filtering: keep included_in_score == True (equivalently errored == False; the two coincide in
v1.1, 155 trials). Trials with cost_usd null (98) count for the attempts list but not the cost
mean; a cell with no priced trial falls back to the system's median per-task cost.

    python deepswe_build.py [--src /path/to/trials.json]     # else downloads

The raw file embeds a canary GUID -- do not commit it; only the derived jsonl is written.
"""
import argparse
import collections
import json
import os
import urllib.request

import numpy as np

URL = "https://deepswe.datacurve.ai/artifacts/v1.1/trials.json"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "deepswe_priced")
PREFIX = "mini_swe_agent_"


def main(src=None):
    if src is None:
        src = os.path.join(DATA, "_raw_trials.json")
        if not os.path.exists(src):
            os.makedirs(DATA, exist_ok=True)
            print("downloading", URL)
            urllib.request.urlretrieve(URL, src)
    rows = json.load(open(src))["rows"]
    rows = [r for r in rows if str(r["included_in_score"]) == "True"]
    os.makedirs(DATA, exist_ok=True)
    bysys = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        name = r["config"]
        name = name[len(PREFIX):] if name.startswith(PREFIX) else name
        bysys[name][r["task_name"]].append(r)
    tasks = sorted(set.intersection(*[set(v) for v in bysys.values()]))
    print(f"{len(bysys)} systems, {len(tasks)} shared tasks")
    for name, cells in sorted(bysys.items()):
        med = np.median([float(r["cost_usd"]) for rs in cells.values() for r in rs
                         if r["cost_usd"] is not None])
        with open(os.path.join(DATA, f"{name}.jsonl"), "w") as f:
            for t in tasks:
                rs = sorted(cells[t], key=lambda r: str(r.get("run_attempt")))
                att = [int(str(r["passed"]) == "True") for r in rs]
                costs = [float(r["cost_usd"]) for r in rs if r["cost_usd"] is not None]
                f.write(json.dumps({"id": t, "attempts": att,
                                    "mean_cost": float(np.mean(costs)) if costs else float(med)}) + "\n")
        n = [len(cells[t]) for t in tasks]
        print(f"  {name:42s} trials/cell {min(n)}-{max(n)}  med cost ${med:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    main(ap.parse_args().src)
