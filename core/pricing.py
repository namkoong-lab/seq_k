"""Per-model cost derivation, in USD.

Cost comes from the first of four sources that can answer, most authoritative
first. Every summary records which one was used, per model, as `cost_source`:

  "reported" — the PROVIDER told us what it actually charged, summed per call.
               OpenRouter returns `usage.cost` on every response (and we persist
               the whole response), which makes this exact rather than an estimate:
               it accounts for which upstream provider served the request, BYOK,
               and promotional / :free tiers that no static table can model.
               Used only when EVERY call for that model reported a cost.
  "table"    — the hand-maintained PRICING dict below: public list prices as of
               PRICING_LAST_UPDATED. Edit it to pin a price you disagree with.
  "litellm"  — litellm's bundled price map, deliberately scoped to OPENROUTER
               MODELS ONLY (see _LITELLM_SCOPE_PREFIX). Rationale: OpenRouter
               spans hundreds of vendor models that would otherwise all need
               hand-entry, and its ids are unambiguous. For direct providers we
               keep PRICING authoritative — those prices are few, stable, and
               verified by hand, and litellm's map moves on package upgrade,
               which we don't want silently rewriting headline direct-API costs.
               Summaries record `litellm_version` for the runs this does apply to.
  null       — nobody knew (one-line stderr warning, deduped per session).

Token counts are the source of truth regardless; only the derived `cost_usd`
depends on any of this.

Coverage note: litellm's OpenRouter coverage is itself partial (it maps
openrouter/openai/gpt-4o but not openrouter/openai/gpt-4o-mini), which is exactly
why provider-reported cost ranks above it — reported cost covers every OpenRouter
model, mapped or not.
"""

from __future__ import annotations

import sys

PRICING_LAST_UPDATED = "2026-06-20"


def litellm_version():
    """Version of the bundled price map, recorded alongside any "litellm" cost.
    Read from package metadata — litellm exposes no __version__ attribute."""
    try:
        import importlib.metadata
        return importlib.metadata.version("litellm")
    except Exception:
        return None

# All values are USD per 1,000,000 tokens.
#   input        — uncached prompt tokens (treats cache writes as plain input;
#                  Anthropic's 25% cache-write premium is intentionally ignored
#                  for simplicity — see core/pricing.py docstring history).
#   cached_input — cache READ tokens (cheaper than input).
#   output       — completion tokens.
PRICING = {
    # ---- Anthropic Claude 4.x ------------------------------------------------
    "anthropic/claude-opus-4-8":    {"input": 15.00, "cached_input": 1.50,  "output": 75.00},
    "anthropic/claude-opus-4-7":    {"input": 15.00, "cached_input": 1.50,  "output": 75.00},
    "anthropic/claude-sonnet-4-7":  {"input":  3.00, "cached_input": 0.30,  "output": 15.00},
    "anthropic/claude-sonnet-4-6":  {"input":  3.00, "cached_input": 0.30,  "output": 15.00},
    "anthropic/claude-haiku-4-5":   {"input":  1.00, "cached_input": 0.10,  "output":  5.00},

    # ---- OpenAI --------------------------------------------------------------
    "openai/gpt-4o":                {"input":  2.50, "cached_input": 1.25,  "output": 10.00},
    "openai/gpt-4o-mini":           {"input":  0.15, "cached_input": 0.075, "output":  0.60},
    "openai/o1":                    {"input": 15.00, "cached_input": 7.50,  "output": 60.00},
    "openai/o1-mini":               {"input":  3.00, "cached_input": 1.50,  "output": 12.00},

    # ---- Google Gemini -------------------------------------------------------
    # Judges reached over the DIRECT OpenAI API rather than OpenRouter, so
    # nothing reports a per-call cost and _LITELLM_SCOPE_PREFIX does not cover
    # them. Rates are litellm's own for these ids, copied rather than guessed —
    # without them a direct-API re-judge records null cost and silently
    # understates what the run spent.
    "openai/gpt-5.4":               {"input":  2.50, "cached_input": 0.25,  "output": 15.00},
    "openai/gpt-5.2":               {"input":  1.75, "cached_input": 0.175, "output": 14.00},
    "gemini/gemini-2.0-flash":      {"input":  0.10, "cached_input": 0.025, "output":  0.40},
    "gemini/gemini-1.5-pro":        {"input":  1.25, "cached_input": 0.3125, "output":  5.00},
}

_WARNED_MISSING = set()
_LITELLM_CACHE = {}     # model -> rate dict | None (get_model_info isn't free)

# litellm's price map is consulted for these model ids ONLY. Everything else is
# PRICING-or-null, so a litellm upgrade can never silently move a direct-API cost.
# Widen this tuple if you want litellm to cover another provider prefix.
_LITELLM_SCOPE_PREFIX = ("openrouter/",)


def cost_for(model, input_tokens, cached_tokens, output_tokens, reported_cost=None):
    """Cost in USD for one model's usage, plus the source that produced it.

    Returns (cost_usd, source) where source is "reported" | "table" | "litellm"
    | None. See the module docstring for the precedence and why.

    `reported_cost` is the provider's own charge summed over the calls in this
    bucket. Callers pass it ONLY when every call in the bucket reported one —
    a partial sum would silently undercount, which is worse than an estimate.
    """
    if reported_cost is not None:
        return round(float(reported_cost), 8), "reported"   # already exact; keep sub-micro digits
    p, source = PRICING.get(model), "table"
    if p is None and str(model).startswith(_LITELLM_SCOPE_PREFIX):
        p, source = _litellm_rates(model), "litellm"
    if p is None:
        if model not in _WARNED_MISSING:
            _WARNED_MISSING.add(model)
            print(f"⚠ no pricing for {model!r} — cost_usd will be null. Add it to "
                  f"core/pricing.py (litellm's map is consulted for "
                  f"{'/'.join(_LITELLM_SCOPE_PREFIX)}* only).", file=sys.stderr)
        return None, None
    uncached_input = max(0, int(input_tokens) - int(cached_tokens))
    return round(
        (uncached_input * p["input"]
         + int(cached_tokens) * p["cached_input"]
         + int(output_tokens) * p["output"]) / 1_000_000,
        6,   # micro-dollar precision is plenty
    ), source


def _litellm_rates(model):
    """litellm's bundled prices for `model`, in the same USD-per-million shape as
    PRICING. None if litellm doesn't map it. get_model_info resolves prefixed ids
    (openai/o3-mini, openrouter/openai/gpt-4o) that the raw model_cost dict misses.

    Falls back to the plain input rate when a model has no separate cache-read
    price — that overstates cached cost slightly, which is the safe direction.
    """
    if model in _LITELLM_CACHE:
        return _LITELLM_CACHE[model]
    rates = None
    try:
        import litellm
        info = litellm.get_model_info(model) or {}
        in_cost, out_cost = info.get("input_cost_per_token"), info.get("output_cost_per_token")
        if in_cost is not None and out_cost is not None:
            cached = info.get("cache_read_input_token_cost")
            rates = {"input": in_cost * 1_000_000,
                     "cached_input": (cached if cached is not None else in_cost) * 1_000_000,
                     "output": out_cost * 1_000_000}
    except Exception:
        rates = None    # unmapped model, or litellm changed its API — fall through to null
    _LITELLM_CACHE[model] = rates
    return rates
