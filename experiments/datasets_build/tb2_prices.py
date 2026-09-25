"""Unified, cache-aware price sheet for Terminal-Bench system names.

Sources (fetched live to the raw-cache directory):
  - litellm_prices_live.json   (BerriAI/litellm model_prices_and_context_window)
  - openrouter_models.json     (openrouter.ai/api/v1/models)

Exposes price_for(model_str) -> {in, out, cache_read} in $/token, and
parse_models(system_name) -> [model_str, ...] for agent__model (and combos).
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.environ.get("ARB_RAW_DIR", os.path.join(HERE, "_raw")), "terminal_bench2")   # raw caches (not committed)
DATA = os.path.join(HERE, "..", "data")
SCRATCH = RAW


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Frozen copies of both price sheets (see data/_price_snapshots/README.md). Both upstreams
# move constantly -- the LiteLLM file lands ~17 commits a day -- so a fresh fetch does NOT
# reproduce the paper's numbers. A live cache in ARB_RAW_DIR wins if present; otherwise these
# pinned snapshots are used, which makes a clean checkout reproducible with no network access.
SNAPSHOTS = os.path.join(DATA, "_price_snapshots")


def _sheet(live_name, frozen_name):
    """Path to a price sheet: the live cache if it exists, else the frozen snapshot, else None."""
    live = os.path.join(SCRATCH, live_name)
    if os.path.exists(live):
        return live
    frozen = os.path.join(SNAPSHOTS, frozen_name)
    return frozen if os.path.exists(frozen) else None


def build_index():
    """Return {normalized_model_id: {in,out,cache_read}} from both sources.

    Longer (more specific) normalized keys are preferred at match time.
    """
    idx = {}

    def add(key, cin, cout, cread):
        # skip placeholders (0 prices) and absurd unit-bug entries (> $1000/Mtok)
        if not key or cin is None or cout is None or cin <= 0 or cout <= 0 or cin > 1e-3:
            return
        if cread is not None and cread <= 0:      # a genuine $0 cache-read is implausible -> treat as missing
            cread = None
        cur = idx.get(key)
        if cur is None or (cread is not None and cur["cache_read"] is None):
            idx[key] = {"in": cin, "out": cout, "cache_read": cread}

    # litellm: strip provider prefixes, keep base model id
    p = _sheet("litellm_prices_live.json", "litellm_prices_2026-09-17.json")
    if p:
        for k, v in json.load(open(p)).items():
            if not isinstance(v, dict):
                continue
            cin = v.get("input_cost_per_token"); cout = v.get("output_cost_per_token")
            cread = v.get("cache_read_input_token_cost")
            base = k.split("/")[-1]                      # drop vertex_ai/, bedrock/reg/, etc.
            base = re.sub(r"[-:]v?\d+:\d+$", "", base)    # bedrock version suffixes
            add(_norm(base), cin, cout, cread)

    # openrouter: id vendor/model, pricing strings $/token
    p = _sheet("openrouter_models.json", "openrouter_models_2026-09-17.json")
    if p:
        data = json.load(p and open(p))
        for e in data.get("data", data):
            pid = e.get("id", ""); pr = e.get("pricing") or {}
            def f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None
            cin = f(pr.get("prompt")); cout = f(pr.get("completion"))
            cread = f(pr.get("input_cache_read"))
            model = pid.split("/")[-1].split(":")[0]
            add(_norm(model), cin, cout, cread)
    return idx


_IDX = None

# manual aliases for names the fuzzy matcher can't reach
ALIASES = {
    "opus46": "claudeopus46", "opus45": "claudeopus45", "opus47": "claudeopus47",
    "claude46opus": "claudeopus46", "claude45opus": "claudeopus45", "claude47opus": "claudeopus47",
    "claude46sonnet": "claudesonnet46", "claude45sonnet": "claudesonnet45",
}
# models with no public price -> excluded from the priced set
UNPRICEABLE = {"termigen32b", "multiple"}


def _lookup(q):
    if q in ALIASES:
        q = ALIASES[q]
    if q in _IDX:
        return _IDX[q]
    cands = [k for k in _IDX if k.startswith(q) or q.startswith(k)]
    return _IDX[max(cands, key=len)] if cands else None


def price_for(model_str):
    """Best cache-aware price for a model string, or None. Tries the full string
    and progressively-shorter trailing token spans (to peel agent prefixes)."""
    global _IDX
    if _IDX is None:
        _IDX = build_index()
    raw = re.sub(r"(?i)[-_.]?(preview|thinking|reasoning|highspeed|latest|instruct|chat|exp)$", "", model_str)
    if _norm(raw) in UNPRICEABLE:
        return None
    toks = re.split(r"[-_ ]", raw)
    # try full, then drop leading tokens one at a time (keep >=1 token)
    for start in range(len(toks)):
        q = _norm("".join(toks[start:]))
        if not q:
            continue
        pr = _lookup(q)
        if pr:
            return pr
    return None


_MODEL_RE = re.compile(r"(?i)(gpt-?5[.\d]*-?codex|gpt-?5[.\d]*-?nano|gpt-?5[.\d]*-?mini|gpt-?5[.\d]*|"
                       r"claude-?opus-?4[.\-]?\d|claude-?sonnet-?4[.\-]?\d|opus-?4[.\-]?\d|"
                       r"gemini-?3[.\d]*-?pro|gemini-?3[.\d]*-?flash|glm-?[45][.\d]*|minimax-?m2[.\d]*|"
                       r"kimi-?k2[.\d]*|deepseek-?v3[.\d]*|qwen3[.\d]*-?coder-?\d*b?|grok-?4[.\d]*)")


def parse_models(system_name):
    """Return candidate model string(s). Splits agent__model, and for combos
    (several models in one name) returns each detected model."""
    chunk = system_name.split("__", 1)[1] if "__" in system_name else system_name
    combos = _MODEL_RE.findall(chunk)
    return combos if len(combos) > 1 else [chunk]


def system_price(system_name):
    """Cache-aware price for a system. For multi-model combos, use the most
    expensive (highest output cost) constituent as a conservative proxy."""
    prices = [price_for(m) for m in parse_models(system_name)]
    prices = [p for p in prices if p]
    if not prices:
        # last resort: price the whole chunk
        p = price_for(system_name.split("__", 1)[-1])
        prices = [p] if p else []
    return max(prices, key=lambda p: p["out"]) if prices else None


if __name__ == "__main__":
    import glob
    idx = build_index()
    print(f"price index: {len(idx)} normalized model ids")
    systems = sorted(os.path.basename(p)[:-6]
                     for p in glob.glob(os.path.join(DATA, "terminal_bench2", "*.jsonl")))
    matched, unmatched = [], []
    for s in systems:
        pr = system_price(s)
        (matched if pr else unmatched).append((s, parse_models(s), pr))
    print(f"\nMATCHED {len(matched)}/{len(systems)}:")
    for s, m, pr in matched:
        print(f"  {s:52s} in={pr['in']*1e6:.2f} out={pr['out']*1e6:.2f} cache={('%.3f'%(pr['cache_read']*1e6)) if pr['cache_read'] else 'n/a'} $/Mtok")
    print(f"\nUNMATCHED {len(unmatched)}:")
    for s, m, _ in unmatched:
        print(f"  {s:52s} model='{m}'")
