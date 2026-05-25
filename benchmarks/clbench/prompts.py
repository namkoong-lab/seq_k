"""Prompt text for CLBench DKR: the rubric judge and the two LLM critics.

Ported verbatim (trimmed) from the original seq_k_eval CLBench adapter and its
clbench_feedback.yaml. `.format()` placeholders:
  JUDGE     : {rubrics_text}, {model_output}
  SOCRATIC  : {rubrics_text}, {failed_requirement_count}, {requirement_status}, {raw_output}
  DIRECTIVE : same as SOCRATIC
(The literal JSON braces in JUDGE are doubled `{{ }}` so .format leaves them alone.)
"""

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
