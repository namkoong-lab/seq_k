"""Judge + critic prompt text. .format() placeholders:
  JUDGE     : rubrics_text, model_output
  SOCRATIC  : rubrics_text, failed_requirement_count, requirement_status, raw_output
  DIRECTIVE : same as SOCRATIC
JSON braces in JUDGE are doubled {{ }} so .format leaves them alone.
"""

# Known problem with JUDGE under openai/gpt-5.4 via OpenRouter (September 2026):
# the refusals depend on which provider OpenRouter routes to. OpenAI's own
# endpoint answers many sends with HTTP 400 "flagged as potentially violating our
# usage policy" (about half on 09-21, all of them on the evening of 09-22), while
# Azure graded the same requests; on 09-21 Azure instead returned empty
# content_filter replies. Some replies also stop mid-rationale as invalid JSON.
# None of these is a verdict, and re-sending the identical request is safe. With
# Azure excluded, DKR tasks 11, 24, 28 and RSA tasks 3, 4, 6 never graded. Editing
# this text makes a different judge, so runs graded before and after would not be
# comparable.

JUDGE = """Starting now, you are a rigorous instruction-following grading teacher. \
Your task is to accurately grade and score student answers based on the 【Rubrics】.

Grading Criteria
This is a strict, all-or-nothing grading system. The final score is binary.
To receive a score of 1, the student's answer must perfectly satisfy every single \
requirement listed in the 【Rubrics】.
If even one requirement is not fully met, the final score will be 0.
Grading Process
Please strictly follow the steps below for analysis—no steps may be skipped:
Step 1: Analyze the Standard Answer
List all explicit requirements in the 【Rubrics】 item by item (including format, content, \
quantity, order, etc.).
Identify implicit requirements in the 【Rubrics】 (e.g., language style, logical structure).
Define specific evaluation criteria for each requirement (e.g., "must include X," \
"must not exceed Y").
Step 2: Check Each Requirement Against the Student's Answer
For every requirement in the 【Rubrics】, verify one by one whether the student's answer \
fully satisfies it.
Step 3: Self-Reflection
Before giving the final score, you must conduct the following checks:
  Completeness Check: Whether all requirements in the standard answer have been reviewed \
with no omissions.
  Strictness Check: Whether the evaluation strictly adheres to the "fully satisfied" \
standard without relaxing requirements due to subjective judgment.
  Consistency Check: Whether the grading rationale aligns logically with the final score.
  Objectivity Check: Whether judgments are based on objective facts rather than \
subjective speculation.
Output Format Requirements
【Grading Rationale】: xxx
【List of Requirement Satisfaction Status】: [x1, x2, ..., xi, ..., xn] \
(where n is the total number of requirements in the 【Rubrics】, and xi indicates whether \
the student's answer meets the i-th requirement, with values "yes"/"no")
【Overall Score】: x points (x is an integer, either 0 or 1.)

Content to Be Graded
【Rubrics】:
{rubrics_text}

【Student Response】:
{model_output}

Please strictly output ONLY the following JSON format (do not output any other content):
{{
  "Grading Rationale": "Your detailed grading rationale",
  "List of Requirement Satisfaction Status": ["yes", "no", ...],
  "Overall Score": 0 or 1
}}
"""


SOCRATIC = """You are providing Socratic feedback on a student's attempt. You have access to
the grading rubric internally, but your feedback must help the student discover
what to improve WITHOUT giving away the answers.

STRICT ANTI-LEAKAGE RULES — you MUST follow all of these:
- NEVER quote or paraphrase any specific rubric requirement
- NEVER reveal exact phrases, words, sign-offs, or formatting that the rubric expects
- NEVER name specific emojis, slang terms, stylistic elements, or structural requirements
- NEVER say things like "the rubric requires X" or "you need to include Y"
- Instead, ask questions that point the student toward the RIGHT CATEGORY of improvement
  (e.g. "Does your tone match the persona?" NOT "You need to use Gen Z slang like 'bet'")
- Refer to requirements only by NUMBER (e.g. "Requirement 3 was not met") and by
  general category (e.g. "closing format", "tone consistency", "visual elements")

Here is the rubric (for your internal reference only — do NOT reproduce):
{rubrics_text}

Failed Requirement Count: {failed_requirement_count}
Requirement Satisfaction Status: {requirement_status}

Student's Attempt:
{raw_output}

Provide 2-4 Socratic questions that guide the student to reflect on what's missing.
Each question should hint at the AREA of improvement without revealing the specific answer.
"""


DIRECTIVE = """You are providing direct feedback on a student's attempt. You have access to
the grading rubric, but you must guide the student without giving away answers.

STRICT ANTI-LEAKAGE RULES — you MUST follow all of these:
- NEVER quote or reproduce any rubric requirement text
- NEVER reveal exact phrases, words, sign-offs, or specific content the rubric expects
- NEVER name specific emojis, slang terms, or formatting that the rubric requires
- NEVER provide copy-pasteable solutions (e.g. don't say "add the phrase 'XYZ'")
- Instead, describe the CATEGORY of each failed requirement
  (e.g. "Your closing does not match the expected format" NOT a verbatim sign-off)
- You may say things like: "Requirement 3 (closing format) was not met — re-read
  the persona instructions to find the expected sign-off"

Here is the rubric (for your internal reference only — do NOT reproduce):
{rubrics_text}

Failed Requirement Count: {failed_requirement_count}
Requirement Satisfaction Status: {requirement_status}

Student's Attempt:
{raw_output}

For each failed requirement, provide:
1. The requirement number and its general CATEGORY (e.g. "tone", "structure", "closing")
2. What is wrong in general terms (without revealing the specific expected content)
3. Where the student should look to figure out the correct answer (e.g. "re-read the system instructions")

Be concise and specific about what's wrong, but never provide the fix directly.
"""


# ---------------------------------------------------------------------------
# Modes recovered from imported seq_k_eval runs. Each was reconstructed from the
# feedback those runs actually recorded (see scripts + the audit in the session
# notes), NOT from an upstream spec — the original prompts were never saved,
# because those runs recorded critic.calls == [].
#
# The axes are: what the critic SEES (rubric vs blind) and what it PRODUCES
# (questions, directives, judge-style guidance, or bare counts).
# ---------------------------------------------------------------------------

SOCRATIC_NO_RUBRIC = """You are providing Socratic feedback on a student's attempt. You do NOT have access to the grading rubric. Work only from the attempt itself
and the count of failed requirements.

Failed Requirement Count: {failed_requirement_count}
Requirement Satisfaction Status: {requirement_status}

Student's Attempt:
{attempt}

Provide 2-4 Socratic questions that guide the student to reflect on what's missing.
Each question should hint at the AREA of improvement without revealing the specific answer.
"""

SOCRATIC_BLIND = """You are providing Socratic feedback on a student's attempt. You do NOT have
access to the grading rubric, the grader's verdicts, or how many requirements failed.
Work only from the attempt itself.

Student's Attempt:
{attempt}

Provide 2-4 Socratic questions that guide the student to reflect on what's missing.
Each question should hint at the AREA of improvement without revealing the specific answer.
"""

DIRECTIVE_NO_RUBRIC = """You are providing direct feedback on a student's attempt. You do NOT have access to the grading rubric. Work only from the attempt itself
and the count of failed requirements.

Failed Requirement Count: {failed_requirement_count}
Requirement Satisfaction Status: {requirement_status}

Student's Attempt:
{attempt}

For each failed requirement, provide:
1. The requirement number and its general CATEGORY (e.g. "tone", "structure", "closing")
2. What is wrong in general terms (without revealing the specific expected content)
3. Where the student should look to figure out the correct answer (e.g. "re-read the system instructions")

Be concise and specific about what's wrong, but never provide the fix directly.
"""

COMPACT_EVAL_OUTPUT = """You are compacting a rubric grader's report so a model can retry.

Below is the grader's per-requirement outcome. Restate it as a brief:
- How many requirements failed, out of how many.
- One line per failed requirement, phrased as what the answer did or did not do.
Do not quote rubric text. Be terse, plain text, under 120 words.

GRADER OUTPUT:
{raw_output}

Failed: {failed_requirement_count}   Status: {requirement_status}

Write the brief now."""

JUDGE_FEEDBACK = """You are turning a rubric grader's output into revision guidance.

Read the grader's per-requirement outcome and the student's attempt, then write
concrete guidance for the retry. Name each requirement by NUMBER and describe the
category of the gap. Do not reproduce rubric text verbatim. Under 200 words.

GRADER OUTPUT:
{raw_output}

Failed Requirement Count: {failed_requirement_count}
Requirement Satisfaction Status: {requirement_status}

STUDENT'S ATTEMPT:
{attempt}

Write the guidance now."""

JUDGE_FEEDBACK_NO_POINTS = """You are turning a rubric grader's output into revision guidance.

Read the grader's outcome and the student's attempt, then write concrete guidance
for the retry. Do NOT mention requirement numbers, point values, or scores — describe
only WHAT to improve and why. Do not reproduce rubric text. Under 200 words.

GRADER OUTPUT:
{raw_output}

STUDENT'S ATTEMPT:
{attempt}

Write the guidance now."""
