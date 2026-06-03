"""All model calls go through here.

LiteLLM picks the provider from the model prefix and reads that provider's key
(openai/* -> OPENAI_API_KEY, plus anthropic/*, gemini/*, deepseek/*, dashscope/*
for Qwen). No retries or fallbacks — let errors surface.

record()/phase() let the harness capture every prompt sent here, tagged by which
agent (actor/judge/critic) issued it — so the prompt view is uniform across
benchmarks without any of them recording prompts by hand.
"""

from __future__ import annotations

import contextlib

import litellm

_sink = None        # list to append calls to while recording, else None
_phase = "actor"    # which agent the current complete() call belongs to


@contextlib.contextmanager
def record(sink):
    """While active, append {phase, model, prompt} for each complete() call to `sink`."""
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
    if _sink is not None:
        _sink.append({"phase": _phase, "model": model, "prompt": prompt})
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        num_retries=0,   # no hidden retries
    )
    return response.choices[0].message.content
