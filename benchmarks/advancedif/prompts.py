"""Prompt for AdvancedIF's per-question instruction-following judge.

`.format()` placeholders: {conversation}, {response}, {requirements}
(literal JSON braces are doubled so .format leaves them alone).
"""

JUDGE = """You are grading whether an assistant's response satisfies a set of \
instruction-following requirements for a conversation.

# Conversation
{conversation}

# Assistant response to grade
{response}

# Requirements
{requirements}

# Instructions
For EACH numbered requirement, decide whether the assistant's response fully satisfies it.
A requirement is met only if the response clearly and completely satisfies it; if any part
is unmet, mark it not met.

Return ONLY a JSON object in this exact schema (no other text):
{{
  "verdicts": [
    {{"question": 1, "met": true, "reason": "<one short sentence>"}},
    {{"question": 2, "met": false, "reason": "<one short sentence>"}}
  ]
}}
"""
