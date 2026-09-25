"""Re-fetch Terminal-Bench 2.0 result.json extracting TOKEN counts (missed by the
first pass), so cost can be computed as tokens x price sheet.

Reuses the enumeration cache (tb2_enum.pkl: systems + [(sysname,trial,rel)]).
Resumable: appends one JSON record per trial to tb2_tokens.jsonl and skips
already-done (sysname,trial). Fields per record:
  sysname, trial, task, solved(0/1), cost_usd(or null),
  n_in, n_out, n_cache (token counts, or null), model_hint(from result if present)
"""
import os, sys, json, time, pickle, threading
from concurrent.futures import ThreadPoolExecutor

import requests
from huggingface_hub import get_token

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.environ.get("ARB_RAW_DIR", os.path.join(HERE, "_raw")), "terminal_bench2")   # raw caches (not committed)
DATA = os.path.join(HERE, "..", "data")
SCRATCH = RAW
ENUM = os.path.join(SCRATCH, "tb2_enum.pkl")
OUT = os.path.join(SCRATCH, "tb2_tokens.jsonl")
REPO = "harborframework/terminal-bench-2-leaderboard"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
TOKEN = get_token()
HEAD = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
MAX_WORKERS = 12
_local = threading.local()


def _session():
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session(); s.headers.update(HEAD); _local.s = s
    return s


def parse(sysname, trial, d):
    tn = d.get("task_name") or ""
    task = tn.split("/")[-1] if tn else trial.rsplit("__", 1)[0]
    vr = d.get("verifier_result") or {}
    reward = (vr.get("rewards") or {}).get("reward") if vr else None
    ar = d.get("agent_result") or {}
    def num(x): return x if isinstance(x, (int, float)) else None
    # model hint (for pricing): try common spots
    model = None
    for src in (ar, d.get("agent_info") or {}, d.get("config") or {}):
        for key in ("model", "model_name", "llm", "model_id"):
            if isinstance(src, dict) and isinstance(src.get(key), str):
                model = src[key]; break
        if model: break
    return {"sysname": sysname, "trial": trial, "task": task,
            "solved": 1 if reward == 1.0 else 0,
            "cost_usd": num(ar.get("cost_usd")),
            "n_in": num(ar.get("n_input_tokens")),
            "n_out": num(ar.get("n_output_tokens")),
            "n_cache": num(ar.get("n_cache_tokens")),
            "model_hint": model}


def fetch(args):
    sysname, trial, rel = args
    url = RESOLVE + rel
    for att in range(6):
        try:
            r = _session().get(url, timeout=60)
            if r.status_code == 200:
                return parse(sysname, trial, r.json())
            if r.status_code == 404:
                return {"sysname": sysname, "trial": trial, "status": "missing"}
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30.0, 2.0 * (att + 1))); continue
            return {"sysname": sysname, "trial": trial, "status": f"http{r.status_code}"}
        except Exception:
            time.sleep(min(30.0, 2.0 * (att + 1)))
    return {"sysname": sysname, "trial": trial, "status": "error"}


def main():
    enum = pickle.load(open(ENUM, "rb"))
    tasks_all = enum["tasks_all"]                       # [(sysname, trial, rel)]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); done.add((r["sysname"], r["trial"]))
            except Exception:
                pass
    todo = [t for t in tasks_all if (t[0], t[1]) not in done]
    print(f"[refetch] {len(tasks_all)} total, {len(done)} done, {len(todo)} to fetch", flush=True)
    t0 = time.time(); n = 0
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for rec in ex.map(fetch, todo):
            f.write(json.dumps(rec) + "\n"); n += 1
            if n % 4000 == 0:
                f.flush(); print(f"  {n}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[refetch] done {n} in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
