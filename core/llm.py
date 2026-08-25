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
import os
import re
import sys

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


# Timeouts are env-overridable because they interact badly with retries under
# concurrency: a hung call costs timeout x (retries+1), and core/parallel.py's
# pmap blocks on its slowest member, so ONE stuck judge call stalls a whole
# attempt. Judge/critic calls normally return in seconds — a judge call still
# running after ~2 min is pathological, and failing it fast (then retrying) beats
# waiting 10 min. Keep the actor generous: long research answers are legitimate.
_ACTOR_TIMEOUT_SECONDS = int(os.environ.get("SEQK_ACTOR_TIMEOUT", "1200"))
_DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("SEQK_TIMEOUT", "600"))

# Retries are OFF by default: this repo prefers errors to surface over silent
# recovery. The exception is deliberate high concurrency — core/parallel.py fans
# out ~25 judge calls per attempt, and several runs may execute at once, so
# provider 429s become an expected consequence of our own load rather than a
# signal about the experiment. Set SEQK_NUM_RETRIES (litellm retries rate limits,
# timeouts and 5xx with backoff) when running the grid; leave it unset for
# single runs so failures stay loud.
_NUM_RETRIES = int(os.environ.get("SEQK_NUM_RETRIES", "0"))

# Forwarded as OpenRouter's `provider` routing preference, to steer away from
# cheap upstreams that return engine_overloaded/429 under load. e.g.
#     '{"sort":"throughput","allow_fallbacks":true}'
# Changes only WHICH host serves the request; the model id is untouched, so a
# run restarted with it still resumes. openrouter/* only.
_OPENROUTER_PROVIDER = os.environ.get("SEQK_OPENROUTER_PROVIDER") or None

# How many times to re-issue a call that came back with empty content. Unlike
# _NUM_RETRIES this defaults ON: an empty completion is never a useful result,
# and litellm cannot retry it because the HTTP call itself succeeded.
_EMPTY_RETRIES = int(os.environ.get("SEQK_EMPTY_RETRIES", "3"))


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
        "num_retries": _NUM_RETRIES,   # see _NUM_RETRIES above (0 unless SEQK_NUM_RETRIES set)
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if _OPENROUTER_PROVIDER and str(model).startswith("openrouter/"):
        import json as _json
        kwargs["extra_body"] = {"provider": _json.loads(_OPENROUTER_PROVIDER)}
    # Empty-completion retry. Providers occasionally return HTTP 200 with no
    # content — litellm sees a success, so num_retries never fires, and the empty
    # string reaches a caller that reasonably assumes it got text (the rubric
    # judges json.loads() it and die, taking a multi-hour grid run with them).
    # One bad response in thousands should not be fatal, so retry it here; if it
    # is STILL empty after _EMPTY_RETRIES we return it and let the caller fail
    # loud, which keeps a genuinely-refusing model distinguishable from a glitch.
    for _attempt in range(_EMPTY_RETRIES + 1):
        try:
            response = litellm.completion(**kwargs)
        except litellm.exceptions.UnsupportedParamsError as exc:
            if not _drop_rejected_param(kwargs, exc):
                raise
            response = litellm.completion(**kwargs)
        output = response.choices[0].message.content
        if output and output.strip():
            break
        if _attempt < _EMPTY_RETRIES:
            print(f"⚠ empty completion from {model} (phase={_phase}), "
                  f"retry {_attempt + 1}/{_EMPTY_RETRIES}", file=sys.stderr)
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


_DROP_WARNED = set()
_PARAM_RE = re.compile(r"\b(temperature|reasoning_effort|top_p|max_tokens)\b")


def _drop_rejected_param(kwargs, exc):
    """Drop the one param a provider refused, and SAY SO. True if we can retry.

    The refusal is often about the VALUE, not the parameter: gpt-5 accepts
    `temperature` but only `temperature=1`, so a run recorded at 0.7 dies in
    litellm's parameter mapping before a token is spent — even though the call
    went through when the run was originally made. Asking
    `get_supported_openai_params` does not see that; only the raised error does.

    litellm's own `drop_params=True` handles it silently, which is the wrong
    trade here: the run would execute at a different temperature than its own
    config claims and nothing downstream would show it. Drop it, print it, and
    leave the recorded config honest about what was ASKED for.
    """
    msg = str(exc)
    hit = _PARAM_RE.search(msg)
    if not hit:
        return False
    key = hit.group(1)
    if key not in kwargs:
        return False
    val = kwargs.pop(key)
    model = kwargs.get("model")
    tag = (model, key)
    if tag not in _DROP_WARNED:
        _DROP_WARNED.add(tag)
        print(f"⚠ {model} rejected `{key}={val!r}` — {msg.strip().splitlines()[0][:120]}\n"
              f"   Dropping it and retrying; the provider default applies. The run's "
              f"recorded config still says {key}={val!r}.", file=sys.stderr)
    return True


def complete_parsed(model, prompt, parse, *, temperature=0.0, tries=3, retry_temperature=0.3):
    """complete(), but retried until `parse` accepts the output.

    Rubric judges are asked for JSON and almost always comply. The rare failures
    are generation glitches, not disagreements: a bare ```json fence with nothing
    inside, a truncated object, a refusal in prose. Each one is a single bad
    sample out of tens of thousands — but the benchmark parsers are deliberately
    fail-loud, so one of them takes down a whole multi-hour run. (On the
    ResearchRubrics grid this accounted for 22 of 25 supervisor restarts.)

    Retrying the CALL is the right level to fix that: it covers every
    malformation at once, where patching the parser only ever covers the shape
    you happened to see. The first attempt uses `temperature` (0.0 for judges —
    the correct setting); retries nudge it up, because re-issuing a deterministic
    request that just failed would likely reproduce the same broken output.

    Re-raises the LAST parse error if every try fails, so a judge that genuinely
    cannot answer stays as loud as before.
    """
    last = None
    for i in range(tries):
        out = complete(model, prompt, temperature if i == 0 else retry_temperature)
        try:
            return parse(out)
        except Exception as exc:      # parser decides what counts as unusable
            last = exc
            if i < tries - 1:
                print(f"⚠ judge output unparseable from {model} "
                      f"({type(exc).__name__}), retry {i + 1}/{tries - 1}", file=sys.stderr)
    raise last


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
