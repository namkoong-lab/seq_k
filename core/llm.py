"""All model calls go through here.

LiteLLM picks the provider from the model prefix and reads that provider's key
(openai/* -> OPENAI_API_KEY, plus anthropic/*, gemini/*, deepseek/*, dashscope/*
for Qwen). No retries or fallbacks — let errors surface.

record()/phase() let the harness capture every (prompt, output) sent here,
tagged by which role (actor/judge/critic) issued it. The harness then packs
those into the actor / judge / critic sections of the saved attempt JSON —
see the schema at the top of core/results.py.
"""

from __future__ import annotations

import contextlib

import litellm

_sink = None        # list to append calls to while recording, else None
_phase = "actor"    # which agent the current complete() call belongs to


@contextlib.contextmanager
def record(sink):
    """While active, append {phase, model, prompt, output} for each complete() call to `sink`."""
    global _sink
    prev, _sink = _sink, sink
    try:
        yield
    finally:
        _sink = prev


@contextlib.contextmanager
def phase(name):
    """Tag complete() calls in this block as coming from agent `name`."""
    global _phase
    prev, _phase = _phase, name
    try:
        yield
    finally:
        _phase = prev


_ACTOR_TIMEOUT_SECONDS = 1200    # 20 min — actor calls on long-context benchmarks (ARC, big rubrics) can run hot.
_DEFAULT_TIMEOUT_SECONDS = 600   # litellm's default — fine for typical judge/critic calls.


def complete(model: str, prompt: str, temperature: float, *, reasoning_effort=None) -> str:
    """One LLM call. `reasoning_effort` opts into the provider's reasoning /
    extended-thinking mode (OpenAI o1/o3, Anthropic Extended Thinking, Gemini
    thinking). LiteLLM translates the "low"/"medium"/"high" alias into whatever
    the provider expects; if the model doesn't support reasoning, LiteLLM
    raises — we let it, fail-loud (research reproducibility over silent no-op).
    """
    # The actor phase gets a longer timeout than judge/critic phases since the
    # actor often processes much larger prompts (full retry trajectories, large
    # rubric sets, ARC grid contexts). Judge/critic prompts are typically short.
    timeout = _ACTOR_TIMEOUT_SECONDS if _phase == "actor" else _DEFAULT_TIMEOUT_SECONDS
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "timeout": timeout,
        "num_retries": 0,   # no hidden retries — let timeouts/errors surface
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    response = litellm.completion(**kwargs)
    output = response.choices[0].message.content
    if _sink is not None:
        # Record AFTER the call so we capture the verbatim response and the
        # provider-reported token usage (most precise — no tokenizer estimates).
        usage = _extract_usage(response)
        entry = {
            "phase": _phase, "model": model, "prompt": prompt, "output": output,
            **usage,
            "finish_reason": _finish_reason(response),
            "raw_response": _serialize_response(response),
        }
        # Reasoning-related fields are only saved on actor calls — that's the
        # only role wired to accept reasoning_effort (judge/critic never do).
        if _phase == "actor":
            entry["reasoning_effort"] = reasoning_effort
            entry["thinking_content"] = _thinking_content(response)
        _sink.append(entry)
    return output


def _finish_reason(response):
    """Provider-reported stop reason for the completion.
    "stop" (natural end), "length" (max_tokens truncation), "content_filter"
    (safety block), "tool_calls" (function/tool use). Useful for detecting
    silent truncation post-hoc.
    """
    try:
        return response.choices[0].finish_reason
    except (AttributeError, IndexError):
        return None


def _thinking_content(response):
    """Extract reasoning / extended-thinking prose if the provider returned it.

    Anthropic Extended Thinking : message.thinking_blocks = [{type, thinking, ...}]
                                  — a list, we join them with blank lines.
    OpenAI o1/o3 (some tiers)   : message.reasoning_content (single string).
    Providers that don't return the prose (only counts) leave this as None.
    Storing None on non-reasoning calls keeps the schema uniform.
    """
    try:
        msg = response.choices[0].message
    except (AttributeError, IndexError):
        return None
    blocks = getattr(msg, "thinking_blocks", None) or []
    if blocks:
        parts = [b.get("thinking", "") for b in blocks if isinstance(b, dict) and b.get("thinking")]
        if parts:
            return "\n\n".join(parts)
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        return str(reasoning)
    return None


def _serialize_response(response):
    """Dict-form of the LiteLLM ModelResponse, safe to json.dumps.

    LiteLLM uses Pydantic v2 (model_dump); fall back to Pydantic v1 (dict).
    Storing the whole response captures provider-specific fields (finish_reason,
    exact served model version, system_fingerprint, etc.) that we'd otherwise
    lose to the current-fields-only recording — cheap insurance, ~500 bytes/call.
    """
    for method in ("model_dump", "dict"):
        fn = getattr(response, method, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return None


def _extract_usage(response):
    """Pull input / cached / thinking / output tokens from the provider's
    response.usage. Returns a uniform dict with four integer keys.

    Provider notes:
      - input_tokens:    total prompt tokens (litellm's `prompt_tokens`). For
                         Anthropic this INCLUDES cache reads; subtract cached_tokens
                         to get the uncached portion priced at the regular input rate.
      - cached_tokens:   prompt tokens served from cache (`prompt_tokens_details.cached_tokens`).
                         Priced at the lower cache-read rate.
      - thinking_tokens: a SUBSET of output_tokens for visibility — DOES NOT add to cost.
                         OpenAI o1/o3:  completion_tokens_details.reasoning_tokens
                         Gemini 2.x+ :  usage_metadata.thoughts_token_count
                         Anthropic   :  0 (no separate breakout; thinking is in output_tokens)
      - output_tokens:   total completion tokens (litellm's `completion_tokens`).
                         For reasoning models this includes thinking_tokens.
    Defaults to zero on any missing field so the schema is uniform across providers.
    """
    usage = getattr(response, "usage", None) or {}

    def _get(obj, key, default=0):
        if isinstance(obj, dict):
            return obj.get(key, default) or default
        return getattr(obj, key, default) or default

    prompt_tokens = _get(usage, "prompt_tokens")
    completion_tokens = _get(usage, "completion_tokens")

    # Cached prompt tokens (Anthropic + OpenAI both nest under prompt_tokens_details).
    details_in = _get(usage, "prompt_tokens_details", default=None)
    cached_tokens = _get(details_in, "cached_tokens") if details_in else 0

    # Thinking / reasoning tokens (OpenAI). For Anthropic they're in output_tokens already.
    details_out = _get(usage, "completion_tokens_details", default=None)
    thinking_tokens = _get(details_out, "reasoning_tokens") if details_out else 0

    # Gemini exposes them under usage_metadata on the raw response.
    if not thinking_tokens:
        meta = getattr(response, "usage_metadata", None) or {}
        thinking_tokens = _get(meta, "thoughts_token_count") or _get(meta, "thinking_tokens")

    return {
        "input_tokens":    int(prompt_tokens),
        "cached_tokens":   int(cached_tokens),
        "thinking_tokens": int(thinking_tokens),
        "output_tokens":   int(completion_tokens),
    }
