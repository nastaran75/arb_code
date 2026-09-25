"""Build data/resmat2_aime and data/resmat2_mmlu from the IRSL test-time response tensor.

Source: https://huggingface.co/datasets/stair-lab/irsl_testtime_resmat2 (resmat.pt), the
test-time scaling data of the IRSL paper (Truong et al.): 12 reasoning models x 120 questions
(30 each from AIME 2024, AIME 2025, Global-MMLU-Lite, MMLU-Pro) x 2048-2560 binary attempts.
LICENSE: none declared as of 2026-09 (repo says TBD) -- ask the authors before publishing
figures built on it.

Two benchmarks of 60 problems each (30-question originals are too small on their own):
  resmat2_aime = AIME 2024 + AIME 2025;  resmat2_mmlu = Global-MMLU-Lite + MMLU-Pro.

Costs are a FLOPs-style proxy, constant per model, like the Monkey Business builders: active
parameter count in billions (Qwen3-30B-A3B counts its ~3B ACTIVE parameters). Arbitrary units;
only ratios matter to the arbitrageur.

    ../../.venv/bin/python resmat2_build.py /path/to/resmat.pt
"""
import collections
import io
import json
import os
import pickle
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# active params (B) -- the per-attempt cost proxy
PARAMS_B = {
    "DeepSeek-R1-Distill-Llama-70B": 70, "DeepSeek-R1-Distill-Llama-8B": 8,
    "DeepSeek-R1-Distill-Qwen-14B": 14, "DeepSeek-R1-Distill-Qwen-32B": 32,
    "DeepSeek-R1-Distill-Qwen-7B": 7, "QwQ-32B": 32, "Qwen3-14B": 14,
    "Qwen3-30B-A3B": 3, "Qwen3-32B": 32, "Qwen3-4B": 4, "Qwen3-8B": 8,
    "gemma-3-27b-it": 27,
}
GROUPS = {"resmat2_aime": ("aime2024", "aime2025"),
          "resmat2_mmlu": ("global_mmlu_lite", "mmlu_pro"),
          # the four 30-question originals, unpooled (noisier: half the problems per benchmark)
          "resmat2_aime2024": ("aime2024",), "resmat2_aime2025": ("aime2025",),
          "resmat2_gmmlu": ("global_mmlu_lite",), "resmat2_mmlupro": ("mmlu_pro",)}


def load_pt(path):
    """torch.save tensor archive -> python objects, numpy-only (no torch dependency)."""
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]

    def make_stub(nm):
        return type("Stub_" + nm, (), {})

    class Unp(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, *a):
                    arr, dt = storage
                    n = int(np.prod(size)) if size else 1
                    flat = np.frombuffer(arr, dtype=dt, count=n,
                                         offset=offset * np.dtype(dt).itemsize)
                    st = np.array(stride) * np.dtype(dt).itemsize
                    return np.lib.stride_tricks.as_strided(flat, shape=size, strides=st).copy()
                return rebuild
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            return make_stub(name)

        def persistent_load(self, pid):
            return (z.read(f"{prefix}/data/{pid[2]}"), np.float64)

    return Unp(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()


def main(path):
    d = load_pt(path)
    T, models, ds = d["data_tensor"], d["models"], np.array(d["datasets"])
    assert set(models) == set(PARAMS_B), sorted(set(models) ^ set(PARAMS_B))
    for key, parts in GROUPS.items():
        out = os.path.join(DATA, key)
        os.makedirs(out, exist_ok=True)
        qidx = np.nonzero(np.isin(ds, parts))[0]
        for j, m in enumerate(models):
            rows = []
            for pid in qidx:
                a = T[j, pid]
                a = a[np.isfinite(a)].astype(int)
                rows.append(dict(id=int(pid), attempts=a.tolist(), mean_cost=float(PARAMS_B[m])))
            with open(os.path.join(out, f"{m}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
        n_att = np.isfinite(T[:, qidx]).sum(2)
        print(f"{key}: {len(models)} models x {len(qidx)} problems, "
              f"attempts {n_att.min()}-{n_att.max()}, -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
