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
Each numbered requirement is a yes/no check. For EACH one, judge it in two steps:
1. "expected": from the conversation and system instructions above, work out what the
   assistant was asked to do — i.e. for a correct response, should the answer to this check
   be "yes" or "no"? Some checks describe behavior the assistant was told to AVOID, so the
   expected answer is "no".
2. "actual": look at the assistant's response and determine the actual answer to this
   check — "yes" or "no". If the check only applies under a condition ("If X, ..." /
   "When X, ...") and that condition does not arise this turn, set "actual" to "n/a".
The requirement is met ("met": true) when "actual" equals "expected", or when "actual" is
"n/a" (the check does not apply this turn). Otherwise it is not met.

Return ONLY a JSON object in this exact schema (no other text):
{{
  "verdicts": [
    {{"question": 1, "expected": "no", "actual": "no", "met": true, "reason": "<one short sentence>"}},
    {{"question": 2, "expected": "yes", "actual": "no", "met": false, "reason": "<one short sentence>"}},
    {{"question": 3, "expected": "yes", "actual": "n/a", "met": true, "reason": "<one short sentence>"}}
  ]
}}
"""
