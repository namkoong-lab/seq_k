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
