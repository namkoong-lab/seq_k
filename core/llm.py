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


def complete(model: str, prompt: str, temperature: float) -> str:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        num_retries=0,   # no hidden retries
    )
    output = response.choices[0].message.content
    if _sink is not None:
        # Record AFTER the call so we capture the verbatim response and the
        # provider-reported token usage (most precise — no tokenizer estimates).
        usage = _extract_usage(response)
        _sink.append({"phase": _phase, "model": model, "prompt": prompt, "output": output, **usage})
    return output


def _extract_usage(response):
    """Pull input / cached / output tokens from the provider's response.usage.

    Returns a dict with three integer keys: input_tokens, cached_tokens, output_tokens.
    Defaults to zero on any missing field so the schema is uniform.
    """
    usage = getattr(response, "usage", None) or {}
    # litellm normalizes the field names but we still accept either via getattr/dict access.
    def _get(obj, key, default=0):
        if isinstance(obj, dict):
            return obj.get(key, default) or default
        return getattr(obj, key, default) or default

    prompt_tokens = _get(usage, "prompt_tokens")
    completion_tokens = _get(usage, "completion_tokens")
    # Cached tokens live in nested details on both Anthropic and OpenAI responses.
    details = _get(usage, "prompt_tokens_details", default=None)
    cached_tokens = _get(details, "cached_tokens") if details else 0
    return {
        "input_tokens": int(prompt_tokens),
        "cached_tokens": int(cached_tokens),
        "output_tokens": int(completion_tokens),
    }
