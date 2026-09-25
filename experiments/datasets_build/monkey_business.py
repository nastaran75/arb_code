"""Build Monkey Business (MATH or CodeContests) into per-system jsonl.

Usage: python monkey_business.py MATH        (13 systems)
       python monkey_business.py CodeContests (5 systems)

Correctness from the `is_corrects` boolean column (parquet projection); cost = FLOPs proxy
    q_i(x) = 2 * params_i * mean_output_tokens_i(x),
PER PROBLEM: the mean length (chars / 4) over all 10,000 generations the model produced for that
problem, streamed one problem at a time from the `samples` column (~200 MB per config, nothing is
kept). So a problem that makes a model write long solutions costs it more, as on the priced
markets. Each row also records `n_out_mean` (the per-problem mean token estimate) for auditing.
(Until 2026-09-25 the cost was a constant per model, from ~400 generations of the first two rows.)
"""
import json
import os
import sys

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

DS = "ScalingIntelligence/monkey_business"
CONVERT = f"datasets/{DS}@refs/convert/parquet"
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CHARS_PER_TOKEN = 4.0

PARAMS_B = {
    "Pythia-70M": 0.070, "Pythia-160M": 0.160, "Pythia-410M": 0.410,
    "Pythia-1B": 1.0, "Pythia-1.4B": 1.4, "Pythia-2.8B": 2.8,
    "Pythia-6.9B": 6.9, "Pythia-12B": 12.0,
    "Gemma-2B": 2.0, "Gemma-7B": 7.0,
    "Llama-3-8B": 8.0, "Llama-3-8B-Instruct": 8.0, "Llama-3-70B-Instruct": 70.0,
}
BENCH_MODELS = {
    "MATH": ["Pythia-70M", "Pythia-160M", "Pythia-410M", "Pythia-1B", "Pythia-1.4B",
             "Pythia-2.8B", "Pythia-6.9B", "Pythia-12B", "Gemma-2B", "Gemma-7B",
             "Llama-3-8B", "Llama-3-8B-Instruct", "Llama-3-70B-Instruct"],
    "CodeContests": ["Gemma-2B", "Gemma-7B", "Llama-3-8B", "Llama-3-8B-Instruct",
                     "Llama-3-70B-Instruct"],
}


def read_cols(fs, config, cols):
    with fs.open(f"{CONVERT}/{config}/test/0000.parquet") as fh:
        return pq.read_table(fh, columns=cols) if cols else None


def tokens_per_problem(fs, config):
    """{orig_dset_idx: mean output tokens (chars / 4) over all generations of that problem}."""
    out = {}
    with fs.open(f"{CONVERT}/{config}/test/0000.parquet") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=1, columns=["orig_dset_idx", "samples"]):
            pid = batch.column("orig_dset_idx")[0].as_py()
            samples = batch.column("samples")[0].as_py()
            out[int(pid)] = (sum(len(s) for s in samples) / max(len(samples), 1)) / CHARS_PER_TOKEN
    return out


def main(bench):
    models = BENCH_MODELS[bench]
    out = os.path.join(DATA, f"monkey_{bench.lower()}")
    os.makedirs(out, exist_ok=True)
    fs = HfFileSystem()
    for m in models:
        cfg = f"{bench}_{m}"
        tbl = read_cols(fs, cfg, ["orig_dset_idx", "is_corrects"])
        ids = tbl.column("orig_dset_idx").to_pylist()
        ic = tbl.column("is_corrects").to_pylist()
        tok = tokens_per_problem(fs, cfg)
        with open(os.path.join(out, f"{m}.jsonl"), "w") as f:
            for pid, attempts in zip(ids, ic):
                t = tok[int(pid)]
                f.write(json.dumps({"id": int(pid), "attempts": [int(bool(x)) for x in attempts],
                                    "mean_cost": 2.0 * PARAMS_B[m] * t, "n_out_mean": t}) + "\n")
        n = len(ic[0])
        p1 = sum(sum(bool(x) for x in a) for a in ic) / (len(ic) * n)
        ts = sorted(tok.values())
        print(f"{m:22s} P={len(ids)} n={n} tok mean {sum(ts)/len(ts):6.0f} "
              f"[min {ts[0]:.0f}, max {ts[-1]:.0f}] pass@1={p1:.4f}", flush=True)
    print("wrote", len(models), "systems to", os.path.abspath(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MATH")
