"""Versioned prompts for HealthBench actors, graders, and feedback writers.

GRADER uses <<conversation>> / <<rubric_item>> placeholders (filled with str.replace,
so no brace escaping needed). The other strings have no placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTOR_INSTRUCTION = (
    "Continue the conversation as the assistant. Do not add markdown or meta commentary."
)

# One grader call per rubric item -> {"explanation", "criteria_met"}.
GRADER = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- If a criterion says "such as", "for example", or "including", the response does not need to include every example listed to meet the criterion.
- If a criterion with negative points describes an undesirable behavior, return false when the response avoids that undesirable behavior, and true only when the response actually does the undesirable thing.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response."""

# For feedback_mode=judge: turn the verifier output into concise revision guidance.
JUDGE_FEEDBACK_SYSTEM = """You convert a medical-response grader's verifier output into \
concise, actionable revision guidance for the next attempt.

Rules:
1. Use only the verifier output and the response shown.
2. Do not reveal or invent a hidden reference answer.
3. Tell the writer what to add, fix, or avoid — concretely and briefly.
4. Plain text only. Do not output JSON."""

SELF_BLIND_FEEDBACK_SYSTEM = """Review your own previous response and write concise,
actionable guidance for improving the next attempt.

Base the guidance only on the response shown. Identify concrete omissions, unclear
reasoning, unsafe or overly confident advice, and useful improvements. Plain text
only."""


@dataclass(frozen=True)
class Protocol:
    actor_instruction: str
    grader: str
    judge_feedback_system: str
    self_blind_feedback_system: str


_DEFAULT = Protocol(
    actor_instruction=ACTOR_INSTRUCTION,
    grader=GRADER,
    judge_feedback_system=JUDGE_FEEDBACK_SYSTEM,
    self_blind_feedback_system=SELF_BLIND_FEEDBACK_SYSTEM,
)

# A named protocol makes prompt semantics part of the HealthBench slice identity.
# Keep the unversioned default for existing configs; repair runs opt into this name.
PROTOCOLS = {"healthbench-repair-v1": _DEFAULT}


def protocol(name=None):
    if name is None:
        return _DEFAULT
    try:
        return PROTOCOLS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown HealthBench protocol {name!r}; expected one of {sorted(PROTOCOLS)}"
        ) from exc
