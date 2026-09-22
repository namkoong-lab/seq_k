"""ResearchRubrics feedback modes:
    binary  — pass/fail bit only
    raw     — the verifier's prioritised diagnostic (mandatory failures + penalties)
    critic  — rubric-blind LLM critic (sees only the task and the response)

pass@k never calls this.
"""

from __future__ import annotations

from core import llm

from . import prompts


def feedback(task, attempt, result, mode, *, critic_model):
    if mode == "binary":
        return ("Your previous research response did not satisfy all mandatory rubric "
                "criteria (or triggered a penalty). Revise it to address every requirement.")
    if mode == "raw":
        return result.raw_eval_output
    if mode == "critic":
        critic_prompt = prompts.CRITIC.format(task_prompt=task.prompt, response=attempt.output)
        return llm.complete(critic_model, critic_prompt, temperature=0.7)
    if mode == "compact_eval_output":
        critic_prompt = prompts.COMPACT_EVAL_OUTPUT.format(raw_output=result.raw_eval_output or "")
        out = (llm.complete(critic_model, critic_prompt, temperature=0.7) or "").strip()
        return out or result.raw_eval_output
    raise ValueError(f"unknown feedback mode: {mode!r}")
