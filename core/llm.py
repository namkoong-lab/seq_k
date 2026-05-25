"""The single entry point for model calls.

LiteLLM talks to each provider's native API using that provider's own key. The
model prefix selects the provider and which env key is read:

    openai/gpt-4o-mini        -> OPENAI_API_KEY
    anthropic/claude-opus-4-7 -> ANTHROPIC_API_KEY
    gemini/gemini-3-flash     -> GEMINI_API_KEY
    deepseek/deepseek-chat    -> DEEPSEEK_API_KEY
    dashscope/qwen-max        -> DASHSCOPE_API_KEY

No retries, no fallbacks: any provider error raises so failures are visible.
"""

from __future__ import annotations

import litellm


def complete(model: str, prompt: str, temperature: float) -> str:
    """Send one user message and return the model's text response."""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        num_retries=0,   # fail loud — no hidden retries
    )
    return response.choices[0].message.content
