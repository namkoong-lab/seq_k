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
    if mode in ("judge", "judge_feedback", "self_judge_feedback"):
        # judge_feedback is the legacy name for `judge`; self_judge_feedback is the
        # same prompt with the ACTOR as critic -- the harness decides who that is,
        # and the imported runs recorded critic.model == actor.model on all of them.
        critic_prompt = (
            f"{prompts.JUDGE_FEEDBACK_SYSTEM}\n\n"
            f"# Verifier output\n{result.raw_eval_output}\n\n"
            f"# Assistant response\n{attempt.output or ''}\n\n"
            "Write concise revision guidance:"
        )
        return llm.complete(critic_model, critic_prompt, temperature=0.7)
    if mode == "compact_eval_output":
        critic_prompt = (
            f"{prompts.JUDGE_FEEDBACK_SYSTEM}\n\n"
            f"# Verifier output\n{result.raw_eval_output}\n\n"
            "Compact this into the criteria that FAILED, one line each, then one line "
            "on what passed. Plain text, under 150 words:"
        )
        return llm.complete(critic_model, critic_prompt, temperature=0.7)
    if mode == "self_blind_feedback":
        # Blind: the critic never sees the verifier output, only the response.
        critic_prompt = (f"{prompts.BLIND_FEEDBACK}\n\n"
                         f"# Assistant response\n{attempt.output or ''}\n\n"
                         "Write your feedback:")
        return llm.complete(critic_model, critic_prompt, temperature=0.7)
    raise ValueError(f"unknown feedback mode: {mode!r}")
