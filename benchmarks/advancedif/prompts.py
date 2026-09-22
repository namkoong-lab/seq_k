"""Prompts for AdvancedIF.

JUDGE    — per-question instruction-following judge.
           `.format()` placeholders: {conversation}, {response}, {requirements}
           (literal JSON braces are doubled so .format leaves them alone).
CRITIQUE — `critique` feedback mode: a reviewer LLM that sees ONLY the task text
           and the model's answer (no rubric, no verifier output), ported verbatim
           from seq_k_eval. `.format()` placeholders: {task}, {answer}.
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


# The reviewer sees ONLY the task and the model's answer — no ground-truth, no
# verifier output, no rubric. System instructions are folded into this single
# user prompt because core.llm.complete takes one message. Ported from
# seq_k_eval's ADVANCEDIF_CRITIQUE_SYSTEM_PROMPT.
CRITIQUE = """You are a careful reviewer. Read the TASK and the MODEL'S ANSWER below, then write actionable suggestions to help the model improve on a retry.

How to review:
1. Read the task carefully and identify what kind of response it expects.
2. Read the model's answer and consider where it could be improved — for example, missing content, factual issues, weak reasoning, format problems, or anything that does not match what the task is asking for.
3. Tell the model what to change. Be prescriptive and concrete — point to the specific part of the answer that needs revision and say what to do differently.

Hard constraints:
- You do NOT have access to a ground-truth answer or external verifier. Base your feedback only on the task text and the model's answer.
- Do not invent facts, guess the correct answer, or restate the task.
- Stay under 150 words. Plain text. No headers, no preamble.

If the answer looks fully correct on its face, say so briefly and stop.

TASK:
{task}

MODEL'S ANSWER:
{answer}

Provide your feedback now."""



# COMPACT_EVAL_OUTPUT — `compact_eval_output` feedback mode. Recovered from the
# imported seq_k_eval runs rather than reconstructed from a spec: across 410
# LLM-written feedbacks in those runs there are 388 distinct opening lines, so the
# original used no fixed template, and every one names which requirements failed
# and which passed. Unlike CRITIQUE the reviewer DOES see the verifier verdicts;
# unlike the `compact` template it paraphrases them instead of echoing question_N.
COMPACT_EVAL_OUTPUT = """You are compacting a verifier's report so a model can retry.

Below are per-requirement verdicts for the model's previous answer. Rewrite them as a
short brief:
- List the requirements that FAILED, one per line, phrased as what the answer did or
  did not do. Paraphrase; do not echo the question numbers verbatim.
- Then, in one sentence, note what PASSED so the model does not regress on it.

Be terse. Plain text. No preamble, no headers beyond the two sections. Under 150 words.

VERIFIER VERDICTS:
{verdicts}

Write the brief now."""


# GUIDED — recovered mode. The imported runs recorded prose that names the unmet
# requirement BY NUMBER and says concretely what to add ("you need to suggest that
# the user consult a personal trainer ... (Requirement 7)"). So unlike CRITIQUE it
# sees the verdicts, and unlike COMPACT_EVAL_OUTPUT it prescribes the fix.
GUIDED = """You are guiding a model through a retry.

Below are per-requirement verdicts for its previous answer. Write short, concrete
guidance: name each unmet requirement by number and say plainly what the answer must
do to satisfy it. If the answer is close, say so. Plain text, under 150 words.

VERIFIER VERDICTS:
{verdicts}

MODEL'S ANSWER:
{answer}

Write the guidance now."""
