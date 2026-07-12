"""Prompt fragments for the tau2 adapter.

A tau2 "attempt" is a full agent<->user conversation, so there is no single
actor prompt the way standard benchmarks have. What we *can* steer per attempt
is the agent's domain_policy. For seq@k retries we append a RETRY_PREAMBLE
(carrying distilled feedback from prior failed attempts) to that policy.

Leak-safety: the preamble only ever contains the public retry feedback string
built by feedback.py (check names/outcomes), never gold DB state.
"""

from __future__ import annotations

# Short human-readable note stored as the seq_k Task.prompt. The real task text
# lives in the tau2 task's user scenario and is delivered by the user simulator
# during the conversation, so this is just a label for the inspect view.
BASE_NOTE = (
    "tau2 retail task — resolve the customer's request over a multi-turn "
    "conversation, following domain policy and using the provided tools. "
    "Scored by the resulting environment (DB) end-state and required "
    "communications."
)


def build_retry_preamble(retry_context: str) -> str:
    """Wrap distilled prior-attempt feedback into a policy preamble.

    Returned text is appended to the agent's domain_policy on seq@k retries.
    Empty input -> empty string (so pass@k / first attempt is untouched).
    """
    if not retry_context.strip():
        return ""
    return (
        "## Retry context\n"
        "You previously attempted a similar customer request and did not fully "
        "satisfy the task. Review the feedback below, then handle this "
        "conversation more carefully — re-check policy preconditions, confirm "
        "details with the user, and make sure every required action and "
        "communication is completed.\n\n"
        f"{retry_context.strip()}\n"
    )


# --------------------------------------------------------------------------- #
# LLM-critic prompt (feedback_mode: "critic")
# --------------------------------------------------------------------------- #
# Given ONLY the failed conversation transcript + the public failure summary, the
# critic writes short, actionable feedback for the next attempt. It is explicitly
# forbidden from inventing specific values (it never sees the gold answer, and must
# not guess it). This is what lets an LLM critic diagnose things a template can't
# (wrong tool vs wrong args vs task misread) without leaking the answer key.
CRITIC_PROMPT = """\
You are a supervisor reviewing a customer-service agent that just FAILED a task.
You will help it do better on its next attempt.

You are given (a) the full conversation transcript of the failed attempt and
(b) an automatic evaluation summary of what went wrong. You do NOT have access to
the correct answer, and you must NOT invent or guess specific values (account
numbers, amounts, item ids, argument values). Diagnose the *behavior*, not the data.

Write 2-5 short bullet points of concrete, actionable guidance for the next
attempt. Focus on the most likely root cause you can infer from the transcript,
e.g.:
- Did the agent call the right tool but with wrong/incomplete arguments? Say so and
  tell it to re-derive the arguments from the conversation (do NOT state the values).
- Did the agent never perform a required action? Name the action and when to do it.
- Did the agent give up / transfer to a human when the task was within its scope?
- Did it skip authentication, confirmation, or a policy precondition?
- Did it misunderstand what the customer actually wanted?

Be specific to THIS transcript. Do not repeat the evaluation summary verbatim.
Do not reveal or fabricate any correct answer values.

<evaluation_summary>
{eval_summary}
</evaluation_summary>

<failed_conversation_transcript>
{transcript}
</failed_conversation_transcript>

Feedback for the next attempt:"""
