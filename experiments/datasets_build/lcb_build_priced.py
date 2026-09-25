"""Build the $-priced LiveCodeBench subset -> data/livecodebench_priced/<system>.jsonl

Per (system, problem) row:
  {"id": question_id, "attempts": [0/1 x10], "mean_cost": $ per attempt (mean over the 10 attempts),
   "n_in": prompt tokens, "n_out_mean": mean completion tokens}
(`arbitrage.load_results` reads id / attempts / mean_cost; the token fields are for auditing.)

COST MODEL (single-turn code generation, one request per attempt):
    q(x, attempt) = n_in(x) * p_in + n_out(x, attempt) * p_out
  n_in(x)  = tokens of the LCB chat prompt: system message + user message
             ("### Question:\\n<question_content>\\n\\n### Format: ...[starter code]... ### Answer: ...")
             reproduced from lcb_runner/prompts/code_generation.py, plus CHAT_OVERHEAD role/format tokens.
  n_out    = tokens of the stored `output_list[i]` -- the FULL visible completion (explanation + code), not
             just the extracted `code_list[i]`.
  INPUT IS BILLED ON EVERY ATTEMPT, NO CACHING. The arbitrage cascade samples one attempt at a time and stops
  on success, so every attempt is its own request; the prompts (~600 tokens) are below every provider's
  prompt-cache minimum (OpenAI / Anthropic 1024 tokens, Gemini 1024-32k). (LCB's own OpenAI and Gemini
  runners asked for n=10 candidates in ONE request, which bills the prompt once per 10 samples -- that is not
  the deployable-arm setting we model; it would lower per-attempt cost by ~25% for those systems.)

TOKENIZERS: exact where public -- tiktoken (OpenAI), HF `tokenizer.json` (open-weight models, downloaded from
the Hub; gated repos fall back) -- otherwise cl100k_base as a PROXY (Claude, Gemini, Mistral: tokenizers not
public / gated; +-15% on counts). The tokenizer actually used is recorded per system in _build_summary.json.

PRICES ($/Mtok): LiteLLM `model_prices_and_context_window.json` + OpenRouter `/api/v1/models`, fetched live
(cached in --raw-dir), looked up by an EXPLICIT id per system (no fuzzy matching); models no longer on either
sheet get their last published list price hardcoded in FIXED_PRICES. Sheets hold CURRENT prices.

EXCLUDED (see EXCLUDED): reasoning models (the billed chain-of-thought is not in output_list -- R1-0528's
10,550 completions contain zero <think> blocks), private / unreleased models, models with no hosted price,
and free experimental endpoints.

Usage:  .venv/bin/python datasets_build/lcb_build_priced.py [--raw-dir DIR] [--download]
  --download fetches any missing Scenario.codegeneration_10_*_eval_all.json from
  github.com/LiveCodeBench/submissions into --raw-dir (34 files, ~1.2 GB).
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "livecodebench_priced")
BASE_SYSTEMS = os.path.join(HERE, "..", "data", "livecodebench")      # the 34 n=10 systems (FLOPs-proxy build)
DEFAULT_RAW = os.environ.get("LCB_RAW_DIR", os.path.join(os.environ.get("ARB_RAW_DIR", os.path.join(HERE, "_raw")), "livecodebench"))
CHAT_OVERHEAD = 8            # role / message-boundary tokens of a 2-message chat request (OpenAI: ~3 per message + 3)

# ---- LCB prompt (lcb_runner/prompts/code_generation.py, PromptConstants + get_generic_question_template_answer)
SYSTEM_GENERIC = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
                  "generate a correct Python program that matches the specification and passes all tests.")
SYSTEM_GEMINI = SYSTEM_GENERIC + (" Do NOT use system calls like `exit` in the generated program. Ensure that the "
                                  "first code block contains the solution.")
SYSTEM_DEEPSEEK = ("You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek "
                   "Company, and you answer questions related to computer science.")
FMT_WITH_STARTER = ("You will use the following starter code to write the solution to the problem and enclose your "
                    "code within delimiters.")
FMT_WITHOUT_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
                       "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
                       "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")


def user_prompt(q):
    p = f"### Question:\n{q['question_content']}\n\n"
    if q.get("starter_code"):
        p += f"### Format: {FMT_WITH_STARTER}\n```python\n{q['starter_code']}\n```\n\n"
    else:
        p += f"### Format: {FMT_WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    p += "### Answer: (use the provided format with backticks)\n\n"
    return p


# ---- system registry ------------------------------------------------------------------------------------------
# tok   : ("tiktoken", encoding) | ("hf", repo_id)  [falls back to PROXY if the repo is gated] | ("proxy",)
# price : list of explicit ids tried in order -- "lit:<key>" (LiteLLM key) or "or:<id>" (OpenRouter id) -- or
#         "fixed" -> FIXED_PRICES[name]
# style : system message variant used by the LCB runner for that model family
PROXY = ("proxy",)
SYSTEMS = {
    "GPT-4-0613":                 dict(tok=("tiktoken", "cl100k_base"), price=["lit:gpt-4-0613"]),
    "GPT-4-Turbo-1106":           dict(tok=("tiktoken", "cl100k_base"), price=["lit:gpt-4-1106-preview"]),
    "GPT-4-Turbo-2024-04-09":     dict(tok=("tiktoken", "cl100k_base"), price=["lit:gpt-4-turbo-2024-04-09"]),
    "GPT-4O-2024-05-13":          dict(tok=("tiktoken", "o200k_base"), price=["lit:gpt-4o-2024-05-13"]),
    "GPT-4O-2024-08-06":          dict(tok=("tiktoken", "o200k_base"), price=["lit:gpt-4o-2024-08-06"]),
    "GPT-4O-mini-2024-07-18":     dict(tok=("tiktoken", "o200k_base"), price=["lit:gpt-4o-mini-2024-07-18"]),
    "Claude-3-Haiku":             dict(tok=PROXY, price=["lit:claude-3-haiku-20240307"]),
    "Claude-3.5-Sonnet-20240620": dict(tok=PROXY, price=["lit:anthropic.claude-3-5-sonnet-20240620-v1:0"]),
    "Claude-3.5-Sonnet-20241022": dict(tok=PROXY, price=["lit:anthropic.claude-3-5-sonnet-20241022-v2:0"]),
    "Gemini-Flash-1.5-002":       dict(tok=PROXY, price="fixed", style="gemini"),
    "Gemini-Pro-1.5-002":         dict(tok=PROXY, price="fixed", style="gemini"),
    "Gemini-Flash-2.0-Exp (N=10)": dict(tok=PROXY, price=["lit:gemini/gemini-2.0-flash-001", "lit:gemini-2.0-flash-001"],
                                        style="gemini", note="GA price of Gemini 2.0 Flash; the -exp endpoint itself was free"),
    "Codestral-Latest":           dict(tok=("hf", "mistralai/Codestral-22B-v0.1"), price=["lit:mistral/codestral-latest"]),
    "Mistral-Large":              dict(tok=("hf", "mistralai/Mistral-Large-Instruct-2411"), price=["lit:mistral/mistral-large-2411", "lit:mistral-large-2411"]),
    "DeepSeek-V3":                dict(tok=("hf", "deepseek-ai/DeepSeek-V3"), price="fixed"),
    "DSCoder-6.7b-Ins":           dict(tok=("hf", "deepseek-ai/deepseek-coder-6.7b-instruct"), style="deepseek",
                                       price=["lit:llamagate/deepseek-coder-6.7b"], note="only hosted price on either sheet (LlamaGate)"),
    "DSCoder-33b-Ins":            dict(tok=("hf", "deepseek-ai/deepseek-coder-33b-instruct"), style="deepseek",
                                       price=["lit:fireworks_ai/accounts/fireworks/models/deepseek-coder-33b-instruct"]),
    "LLama3.3-70b-Ins":           dict(tok=("hf", "meta-llama/Llama-3.3-70B-Instruct"), price=["or:meta-llama/llama-3.3-70b-instruct"]),
    "Qwen2-Ins-72B":              dict(tok=("hf", "Qwen/Qwen2-72B-Instruct"), price=["or:qwen/qwen-2-72b-instruct", "lit:*qwen2-72b-instruct"]),
    "Qwen2.5-Ins-7B":             dict(tok=("hf", "Qwen/Qwen2.5-7B-Instruct"), price=["or:qwen/qwen-2.5-7b-instruct", "lit:*qwen2.5-7b-instruct"]),
    "Qwen2.5-Ins-32B":            dict(tok=("hf", "Qwen/Qwen2.5-32B-Instruct"), price=["or:qwen/qwen2.5-32b-instruct", "lit:*qwen2.5-32b-instruct"]),
    "Qwen2.5-Ins-72B":            dict(tok=("hf", "Qwen/Qwen2.5-72B-Instruct"), price=["or:qwen/qwen-2.5-72b-instruct", "lit:*qwen2.5-72b-instruct"]),
    "Qwen2.5-Coder-Ins-7B":       dict(tok=("hf", "Qwen/Qwen2.5-Coder-7B-Instruct"), price=["or:qwen/qwen2.5-coder-7b-instruct", "lit:*qwen2.5-coder-7b-instruct"]),
    "Qwen2.5-Coder-Ins-32B":      dict(tok=("hf", "Qwen/Qwen2.5-Coder-32B-Instruct"), price=["or:qwen/qwen-2.5-coder-32b-instruct", "lit:*qwen2.5-coder-32b-instruct"]),
}
# last published list prices ($/Mtok) for models that are no longer on the live sheets
FIXED_PRICES = {
    "Gemini-Flash-1.5-002": (0.075, 0.30, "Google list price for Gemini 1.5 Flash, prompts <=128k (deprecated Sep 2025, off the sheets)"),
    "Gemini-Pro-1.5-002":   (1.25, 5.00, "Google list price for Gemini 1.5 Pro, prompts <=128k (deprecated Sep 2025, off the sheets)"),
    "DeepSeek-V3":          (0.27, 1.10, "DeepSeek API list price for deepseek-chat = V3 (Feb 2025); the endpoint now serves V3.2 at $0.28/$0.42"),
}
EXCLUDED = {
    "DeepSeek-R1-0528":          "reasoning model: billed chain-of-thought not in output_list (0 <think> blocks in 10,550 completions)",
    "DeepSeek-R1-Preview":       "reasoning model: chain-of-thought not in output_list",
    "DeepSeek-R1-Lite-Preview":  "reasoning model; never had a public API price",
    "Gemini-Flash-2.0-Thinking": "reasoning model; free experimental endpoint, no price",
    "Kimi-k1.6-IOI":             "reasoning model; private / unreleased",
    "OpenReasoning-Nemotron-32B": "reasoning model; no hosted price",
    "MetaStone-L1-7B":           "reasoning model; no hosted price",
    "AzeroGPT-64b":              "private model, no price",
    "DSCoder-1.3b-Ins":          "no hosted price on either sheet",
    "Gemini-Exp-1206":           "experimental endpoint, never priced",
}


# ---- raw data -------------------------------------------------------------------------------------------------
def _get(url, dest=None, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "lcb-build-priced"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if dest:
        with open(dest, "wb") as f:
            f.write(data)
    return data


def download_raw(raw_dir, systems):
    os.makedirs(raw_dir, exist_ok=True)
    for name in systems:
        dest = os.path.join(raw_dir, f"{name}.json")
        if os.path.exists(dest):
            continue
        enc = urllib.parse.quote(name)
        listing = json.loads(_get(f"https://api.github.com/repos/LiveCodeBench/submissions/contents/{enc}", timeout=60))
        cands = [x["name"] for x in listing if "eval_all" in x["name"] and "_10_" in x["name"]]
        if not cands:
            print(f"  !! no n=10 eval_all file for {name}"); continue
        print(f"  downloading {name} ({cands[0]}) ...")
        _get(f"https://raw.githubusercontent.com/LiveCodeBench/submissions/main/{enc}/{urllib.parse.quote(cands[0])}", dest + ".part")
        os.replace(dest + ".part", dest)


def load_raw(raw_dir, name):
    """Problems of one system. graded_list must be length 10 everywhere; a problem whose output_list is EMPTY
    (the runner recorded no completions -- e.g. 30 problems of DeepSeek-V3) is kept as 10 failed attempts with
    zero completion tokens (input-only cost), matching the graded_list the file carries."""
    data = json.load(open(os.path.join(raw_dir, f"{name}.json")))
    assert isinstance(data, list) and all(len(q["graded_list"]) == 10 for q in data), name
    n_empty = 0
    for q in data:
        if len(q["output_list"]) != 10:
            assert len(q["output_list"]) == 0 and not any(q["graded_list"]), (name, q["question_id"])
            q["output_list"] = [""] * 10
            n_empty += 1
    return data, n_empty


# ---- tokenizers -----------------------------------------------------------------------------------------------
_TOK_CACHE = {}


def get_tokenizer(spec, raw_dir):
    """Return (count_fn(list[str]) -> list[int], description)."""
    key = tuple(spec)
    if key in _TOK_CACHE:
        return _TOK_CACHE[key]
    import tiktoken
    if spec[0] == "tiktoken":
        enc = tiktoken.get_encoding(spec[1])
        res = (lambda texts: [len(t) for t in enc.encode_ordinary_batch(texts)], f"tiktoken/{spec[1]} (exact)")
    elif spec[0] == "hf":
        repo = spec[1]
        tdir = os.path.join(raw_dir, "tokenizers"); os.makedirs(tdir, exist_ok=True)
        path = os.path.join(tdir, repo.replace("/", "__") + ".tokenizer.json")
        if not os.path.exists(path):
            try:
                _get(f"https://huggingface.co/{repo}/resolve/main/tokenizer.json", path, timeout=120)
            except Exception as e:                        # gated repo (401/403) or no network
                res = get_tokenizer(PROXY, raw_dir)
                res = (res[0], f"{res[1]} [HF {repo} unavailable: {getattr(e, 'code', e)}]")
                _TOK_CACHE[key] = res
                return res
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(path)
        res = (lambda texts: [len(e.ids) for e in tk.encode_batch(texts, add_special_tokens=False)], f"hf/{repo} (exact)")
    else:
        enc = tiktoken.get_encoding("cl100k_base")
        res = (lambda texts: [len(t) for t in enc.encode_ordinary_batch(texts)], "cl100k_base PROXY (tokenizer not public)")
    _TOK_CACHE[key] = res
    return res


# ---- prices ---------------------------------------------------------------------------------------------------
def load_sheets(raw_dir):
    lp = os.path.join(raw_dir, "litellm_prices_live.json"); op = os.path.join(raw_dir, "openrouter_models.json")
    if not os.path.exists(lp):
        _get("https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json", lp, timeout=120)
    if not os.path.exists(op):
        _get("https://openrouter.ai/api/v1/models", op, timeout=120)
    lit = {k: (float(v.get("input_cost_per_token") or 0) * 1e6, float(v.get("output_cost_per_token") or 0) * 1e6)
           for k, v in json.load(open(lp)).items() if isinstance(v, dict)}
    orr = {m["id"]: (float(m["pricing"].get("prompt") or 0) * 1e6, float(m["pricing"].get("completion") or 0) * 1e6)
           for m in json.load(open(op))["data"]}
    return lit, orr


def resolve_price(name, cfg, lit, orr):
    """Return (p_in, p_out, source) in $/Mtok."""
    if cfg["price"] == "fixed":
        pin, pout, note = FIXED_PRICES[name]
        return pin, pout, f"fixed: {note}"
    for pid in cfg["price"]:
        src, key = pid.split(":", 1)
        table = lit if src == "lit" else orr
        if key.startswith("*"):                           # suffix match, cheapest positive entry across providers
            hits = [(v, k) for k, v in table.items() if k.lower().endswith(key[1:].lower()) and v[0] > 0 and v[1] > 0]
            if hits:
                (pin, pout), k = min(hits)
                return pin, pout, f"{src}:{k} (cheapest provider entry)"
        elif key in table and table[key][0] > 0 and table[key][1] > 0:
            pin, pout = table[key]
            return pin, pout, pid
    raise KeyError(f"no price found for {name}: tried {cfg['price']}")


# ---- build ----------------------------------------------------------------------------------------------------
def build(raw_dir):
    os.makedirs(OUT, exist_ok=True)
    base = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(BASE_SYSTEMS, "*.jsonl")))
    unknown = [s for s in base if s not in SYSTEMS and s not in EXCLUDED]
    assert not unknown, f"systems without a registry decision: {unknown}"
    lit, orr = load_sheets(raw_dir)
    summary = {"cost_model": "q = n_in*p_in + n_out*p_out per attempt; input billed on every attempt, no caching",
               "chat_overhead_tokens": CHAT_OVERHEAD, "excluded": EXCLUDED, "systems": {}}
    rows_hdr = f"{'system':28s} {'#prob':>5s} {'pass@1':>6s} {'in/M':>6s} {'out/M':>6s} {'n_in':>5s} {'n_out':>5s} {'$/attempt':>9s} {'$ run':>7s}  tokenizer | price source"
    print(rows_hdr); print("-" * len(rows_hdr))
    for name in base:
        if name in EXCLUDED:
            continue
        cfg = SYSTEMS[name]
        data, n_empty = load_raw(raw_dir, name)
        count, tok_desc = get_tokenizer(cfg["tok"], raw_dir)
        pin, pout, psrc = resolve_price(name, cfg, lit, orr)
        system_msg = {"gemini": SYSTEM_GEMINI, "deepseek": SYSTEM_DEEPSEEK}.get(cfg.get("style"), SYSTEM_GENERIC)
        n_sys = count([system_msg])[0]
        n_user = count([user_prompt(q) for q in data])
        n_out = np.array(count([o for q in data for o in q["output_list"]]), float).reshape(len(data), 10)
        n_in = np.array(n_user, float) + n_sys + CHAT_OVERHEAD
        cost = (n_in[:, None] * pin + n_out * pout) / 1e6                        # (P, 10) $ per attempt
        with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
            for q, ni, no, c in zip(data, n_in, n_out, cost):
                f.write(json.dumps({"id": q["question_id"], "attempts": [int(bool(g)) for g in q["graded_list"]],
                                    "mean_cost": float(c.mean()), "n_in": int(ni), "n_out_mean": float(no.mean())}) + "\n")
        p1 = float(np.mean([np.mean(q["graded_list"]) for q in data]))
        summary["systems"][name] = dict(n_problems=len(data), pass1=p1, p_in=pin, p_out=pout, price_source=psrc,
                                        tokenizer=tok_desc, n_in_mean=float(n_in.mean()), n_out_mean=float(n_out.mean()),
                                        cost_per_attempt_mean=float(cost.mean()), cost_run_total=float(cost.sum()),
                                        n_problems_empty_output=n_empty, note=cfg.get("note", ""))
        print(f"{name:28s} {len(data):5d} {p1:6.3f} {pin:6.2f} {pout:6.2f} {n_in.mean():5.0f} {n_out.mean():5.0f} "
              f"{cost.mean():9.5f} {cost.sum():7.2f}  {tok_desc} | {psrc}" + (f" | {n_empty} empty-output problems" if n_empty else ""))
    json.dump(summary, open(os.path.join(OUT, "_build_summary.json"), "w"), indent=1)
    print(f"\n{len(summary['systems'])} systems written to {os.path.abspath(OUT)}; excluded {len(EXCLUDED)}:")
    for k, v in EXCLUDED.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    ap.add_argument("--download", action="store_true", help="fetch missing raw submission files first")
    args = ap.parse_args()
    if args.download:
        download_raw(args.raw_dir, sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(BASE_SYSTEMS, "*.jsonl"))))
    build(args.raw_dir)
