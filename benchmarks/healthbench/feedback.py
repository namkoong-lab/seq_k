"""HealthBench feedback modes:
    binary  — pass/fail bit only
    raw     — the verifier's diagnostic (unmet positive + triggered negative + safety gaps)
    judge       — rubric-grounded LLM guidance derived from the verifier output
    self_blind  — actor-written self-review seeing only its own prior response
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
    protocol_spec = prompts.protocol(task.grading.get("protocol"))
    if mode == "judge":
        critic_prompt = (
            f"{protocol_spec.judge_feedback_system}\n\n"
            f"# Verifier output\n{result.raw_eval_output}\n\n"
            f"# Assistant response\n{attempt.output or ''}\n\n"
            "Write concise revision guidance:"
        )
        return _complete_nonempty(critic_model, critic_prompt, mode=mode)
    if mode == "self_blind":
        # Deliberately exclude task.prompt, rubrics, verifier output, score, and
        # outcome. The harness resolves critic_model to the actor for this mode.
        critic_prompt = (
            f"{protocol_spec.self_blind_feedback_system}\n\n"
            f"# Your previous response\n{attempt.output or ''}\n\n"
            "Write concise self-review guidance:"
        )
        return _complete_nonempty(critic_model, critic_prompt, mode=mode)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _complete_nonempty(model, prompt, *, mode):
    output = llm.complete(model, prompt, temperature=0.7, reject_truncated=True)
    if not str(output or "").strip():
        raise RuntimeError(f"HealthBench {mode} feedback model returned empty output")
    return output
