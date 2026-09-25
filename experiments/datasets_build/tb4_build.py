"""Build Terminal-Bench 4.0 into per-system {id, attempts, mean_cost} jsonl.

Unlike 2.0 there is no HuggingFace submissions dump; the raw trials live behind Harbor Hub job
pages. The leaderboard index (github.com/harbor-framework/terminal-bench, leaderboard/submissions/)
lists one JSON per entry with the hub job URL(s) and the trial UUIDs; each job page's RSC payload
carries per-trial records (task_name, reward, tokens, cost_usd), 100 per page, paginated by
?page=N. This builder walks all of it and writes data/terminal_bench4_priced/.

    python tb4_build.py            # ~13 systems x 66 tasks x 5 trials
"""
import json
import os
import re
import time
import urllib.request

import numpy as np

IDX = "https://api.github.com/repos/harbor-framework/terminal-bench/contents/leaderboard/submissions"
RAW = "https://raw.githubusercontent.com/harbor-framework/terminal-bench/main/leaderboard/submissions/"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "terminal_bench4_priced")


def get(url, rsc=False):
    req = urllib.request.Request(url, headers={"RSC": "1"} if rsc else {})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", "ignore")


def job_trials(job_url):
    """All trial dicts of one hub job, by brace-matching the RSC payload across pages."""
    out, page = {}, 1
    while True:
        t = get(f"{job_url}?page={page}", rsc=True).replace('\\"', '"')
        found = 0
        for m in re.finditer(r'\{"id":"[a-f0-9-]{36}","name":"[^"]+__[a-f0-9]{8}"', t):
            i = m.start()
            depth, j = 0, i
            while j < len(t):
                if t[j] == "{":
                    depth += 1
                elif t[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            try:
                d = json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
            if "task_name" in d and "reward" in d:
                out[d["id"]] = d
                found += 1
        if not found or page > 40:
            break
        page += 1
    return list(out.values())


def main():
    os.makedirs(DATA, exist_ok=True)
    idx = json.loads(get(IDX))
    for entry in idx:
        sub = json.loads(get(RAW + entry["name"]))
        md = sub["metadata"]
        agent = md["agent_display"]["label"].replace(" ", "-")
        model = md["model_display"]["label"].replace(" ", "-")
        eff = md.get("reasoning_effort") or "none"
        name = f"{agent}__{model}" + (f"-{eff}" if eff not in ("none", None) else "")
        want = set(sub["trials"])
        dq = set(sub.get("disqualified_trials") or [])
        trials = []
        out_path = os.path.join(DATA, f"{name}.jsonl")
        if os.path.exists(out_path):
            print(f"{name:44s} exists, skipping")
            continue
        for ju in sub["source_jobs"]:
            if not ju.startswith("http"):                  # newer submissions store bare job UUIDs
                ju = "https://hub.harborframework.com/jobs/" + ju
            trials += job_trials(ju)
            time.sleep(0.3)
        rows = [d for d in trials if d["id"] in want and d["id"] not in dq]
        missing = len(want - {d["id"] for d in rows})
        bytask = {}
        for d in rows:
            tid = d["task_name"].split("/")[-1]
            bytask.setdefault(tid, []).append(d)
        med = np.median([d["cost_usd"] for d in rows if d.get("cost_usd") is not None])
        with open(out_path, "w") as f:
            for tid in sorted(bytask):
                rs = sorted(bytask[tid], key=lambda d: d.get("started_at") or "")
                att = [int((d.get("reward") or 0) >= 1) for d in rs]
                costs = [d["cost_usd"] for d in rs if d.get("cost_usd") is not None]
                f.write(json.dumps({"id": tid, "attempts": att,
                                    "mean_cost": float(np.mean(costs)) if costs else float(med)}) + "\n")
        n = [len(v) for v in bytask.values()]
        acc = np.mean([a for v in bytask.values() for d in v for a in [(d.get("reward") or 0) >= 1]])
        print(f"{name:44s} tasks={len(bytask):3d} trials/task {min(n)}-{max(n)} "
              f"missing={missing:3d} pass@1={acc*100:5.1f}% med ${med:.2f}/trial")


if __name__ == "__main__":
    main()
