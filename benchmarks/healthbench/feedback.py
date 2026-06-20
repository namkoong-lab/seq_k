"""HealthBench feedback modes:
    binary  — pass/fail bit only
    raw     — the verifier's diagnostic (unmet positive + triggered negative + safety gaps)
    judge   — rubric-grounded LLM guidance derived from the verifier output

The paper's self_judge_feedback / self_blind_feedback modes are intentionally not
here yet: "self" feedback must be written by the ACTOR model, but the harness only
hands feedback() the critic_model. Add them once the feedback contract also passes
the actor model. pass@k never calls this.
"""

from __future__ import annotations

from core import llm

from . import prompts


def feedback(task, attempt, result, mode, *, critic_model):
    if mode == "binary":
        return ("Your previous answer did not pass HealthBench grading (score below threshold, "
                "or a safety/negative criterion was violated). Revise it.")
    if mode == "raw":
        return result.raw_eval_output
    if mode == "judge":
        critic_prompt = (
            f"{prompts.JUDGE_FEEDBACK_SYSTEM}\n\n"
            f"# Verifier output\n{result.raw_eval_output}\n\n"
            f"# Assistant response\n{attempt.output or ''}\n\n"
            "Write concise revision guidance:"
        )
        return llm.complete(critic_model, critic_prompt, temperature=0.7)
    raise ValueError(f"unknown feedback mode: {mode!r}")
