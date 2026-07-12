"""tau2 feedback modes:

    binary           — pass/fail note only            (template, no LLM)
    raw              — public per-check failure summary (template, no LLM)
    retry_diagnostics — failure summary + a checklist  (template, no LLM)
    critic           — LLM reads the transcript + public summary and writes
                       adaptive, targeted feedback     (LLM call, see prompts.CRITIC_PROMPT)

Leak-safety:
- `raw_eval_output` is the public failure summary (check outcomes + missed action
  *names*, never gold arguments). Safe to feed back.
- The `critic` mode is given ONLY `attempt.output` (the agent's own transcript) and
  `raw_eval_output`. It is NEVER given `result.details` (which holds the gold
  arguments), and the prompt forbids inventing values. So it can diagnose behavior
  without access to the answer key.

pass@k never calls this.
"""

from __future__ import annotations

from core import llm

from . import prompts


def feedback(task, attempt, result, mode, *, judge_model=None,
             critic_model=None, **_extra):
    # critic_model: the LLM used to WRITE feedback (only the `critic` mode uses it).
    #   Template modes ignore it. **_extra future-proofs against new core kwargs.
    if mode == "binary":
        return ("Your previous attempt did not fully satisfy the task. Handle the "
                "next conversation more carefully and complete every required "
                "action and communication.")
    if mode == "raw":
        return result.raw_eval_output or _binary_fallback()
    if mode == "retry_diagnostics":
        return _retry_diagnostics(result.raw_eval_output)
    if mode == "critic":
        return _llm_critic(attempt, result, critic_model)
    raise ValueError(f"unknown feedback mode: {mode!r}")


# Transcript can be long; cap what we send the critic to keep prompt size / cost
# bounded. The tail of the conversation (where the failure happens) matters most.
_CRITIC_TRANSCRIPT_CHARS = 12000


def _llm_critic(attempt, result, critic_model):
    """LLM critic: reads the failed transcript + public summary, writes feedback.

    LEAK-SAFE: sees only `attempt.output` (agent's own transcript) and
    `result.raw_eval_output` (public summary). NEVER `result.details` (gold args).
    Falls back to the template `raw` summary if the model call fails/returns empty."""
    transcript = (attempt.output or "")
    if len(transcript) > _CRITIC_TRANSCRIPT_CHARS:
        transcript = "…[earlier turns truncated]…\n" + transcript[-_CRITIC_TRANSCRIPT_CHARS:]
    eval_summary = (result.raw_eval_output or "").strip() or _binary_fallback()

    critic_prompt = prompts.CRITIC_PROMPT.format(
        eval_summary=eval_summary,
        transcript=transcript,
    )
    try:
        out = llm.complete(critic_model, critic_prompt, temperature=0.7)
    except Exception:
        out = ""
    out = (out or "").strip()
    if not out:
        return eval_summary  # fall back to the public summary rather than nothing
    return out


def _binary_fallback():
    return "Your previous attempt did not pass the task evaluation."


def _retry_diagnostics(raw_eval_output):
    summary = (raw_eval_output or "").strip() or _binary_fallback()
    return "\n".join([
        "what_went_wrong:",
        summary,
        "",
        "retry_focus:",
        "- Re-read the relevant domain policy before acting; confirm every "
        "precondition with the user.",
        "- Make sure each required write action is performed exactly once with "
        "the correct arguments.",
        "- Explicitly state any information the task requires you to communicate "
        "to the user.",
        "- Do not stop until the customer's request is fully resolved.",
    ]).strip()
