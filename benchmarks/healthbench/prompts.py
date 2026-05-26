"""Prompts for HealthBench: the per-rubric grader, the actor framing, and the
rubric-grounded judge-feedback writer. Ported from the original seq_k_eval adapter.

GRADER uses <<conversation>> / <<rubric_item>> placeholders (filled with str.replace,
so no brace escaping needed). The other strings have no placeholders.
"""

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
