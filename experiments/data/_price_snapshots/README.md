# Price snapshots used by `terminal_bench2_priced` (and re-usable by the other priced sets)

Frozen copies of the two live price sources. Both upstreams move constantly — the LiteLLM
file takes ~17 commits/day — so the figure is only reproducible against these copies.

| file | source | pinned to |
|---|---|---|
| `litellm_prices_2026-09-17.json` | BerriAI/litellm `model_prices_and_context_window.json` | commit `3d0fd127d5a151f7f094462b16ac9c2a01a047b6` (2026-09-17T07:20:16Z) |
| `openrouter_models_2026-09-17.json` | `https://openrouter.ai/api/v1/models` | fetched 2026-09-17; the API is unversioned, so this copy IS the only record |

Verify:

    shasum -a 256 litellm_prices_2026-09-17.json
    # d32e6215072dd9b99c9c18e3e21cb128beed1a96089c1859707e5b0ca46a9cc1
    shasum -a 256 openrouter_models_2026-09-17.json
    # b97d80d5b8902849034a021edef463c4e93ca1757348b6373c0bb83d5ed7295e

Re-fetch the pinned LiteLLM file exactly:

    curl -sO https://raw.githubusercontent.com/BerriAI/litellm/3d0fd127d5a151f7f094462b16ac9c2a01a047b6/model_prices_and_context_window.json

`tb2_prices.py` reads `$ARB_RAW_DIR/terminal_bench2/{litellm_prices_live,openrouter_models}.json`
first and falls back to these copies, so a clean checkout reproduces the paper's prices with no
network access. Delete the raw cache to force the frozen ones.

NOTE: only the 18 token-priced systems use these sheets. The other 10 take the submitter-reported
`agent_result.cost_usd`, which carries each submission's own run date instead.
