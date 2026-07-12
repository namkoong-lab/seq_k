"""Per-model token pricing in USD per million tokens.

All prices are public list prices as of `PRICING_LAST_UPDATED`. Update the table
and bump the date when you notice them change — `pricing_last_updated` is
recorded in every summary.json so future readers know which snapshot was used.

`cost_for()` returns None for models not in the table (and prints a one-line
warning, deduped per session so you can spot it without spam). Token counts
remain the source of truth either way.
"""

from __future__ import annotations

import sys

PRICING_LAST_UPDATED = "2026-06-20"

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
    "gemini/gemini-2.0-flash":      {"input":  0.10, "cached_input": 0.025, "output":  0.40},
    "gemini/gemini-1.5-pro":        {"input":  1.25, "cached_input": 0.3125, "output":  5.00},
}

_WARNED_MISSING = set()


def cost_for(model, input_tokens, cached_tokens, output_tokens):
    """Compute cost in USD for one model's usage.

    Returns None if `model` isn't in PRICING (and warns to stderr once per
    session). Token counts already in the data stay authoritative; only the
    derived `cost_usd` field is missing in that case.
    """
    if model not in PRICING:
        if model not in _WARNED_MISSING:
            _WARNED_MISSING.add(model)
            print(f"⚠ no pricing for {model!r} — cost_usd will be null. "
                  f"Add it to core/pricing.py", file=sys.stderr)
        return None
    p = PRICING[model]
    uncached_input = max(0, int(input_tokens) - int(cached_tokens))
    return round(
        (uncached_input * p["input"]
         + int(cached_tokens) * p["cached_input"]
         + int(output_tokens) * p["output"]) / 1_000_000,
        6,   # micro-dollar precision is plenty
    )
