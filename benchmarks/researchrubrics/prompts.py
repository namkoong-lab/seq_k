"""Prompts for ResearchRubrics: the actor framing, the per-criterion judge, and
the rubric-blind critic. Ported from the original seq_k_eval adapter.

`.format()` placeholders:
  JUDGE   : {response_text}, {criterion}, {axis}, {weight}   (literal JSON braces doubled)
  CRITIC  : {task_prompt}, {response}
"""

# Prepended to each task's research prompt so the actor knows the expected form.
ACTOR_INSTRUCTION = (
    "You are a research assistant. Write a thorough, well-structured response to the "
    "research prompt below. Use markdown formatting. Be comprehensive, accurate, and "
    "address all aspects of the prompt."
)

# One judge call per rubric criterion -> binary Satisfied / Not Satisfied.
JUDGE = """You are an expert evaluator assessing whether a document satisfies a specific \
rubric criterion. Your evaluation must be precise, objective, and based solely on evidence \
present in the document.

## Evaluation Framework
Evaluate the criterion on a binary satisfaction scale:
1. Not Satisfied (Score: 0.0): The document fails to meet the criterion.
2. Satisfied (Score: 1.0): The document fully meets the criterion.

## Important Guidelines
- Base your evaluation ONLY on what is explicitly present in the document
- Do not make assumptions about implied or missing content
- Be consistent in your evaluation standards across all criteria
- Provide specific examples from the document to support your verdict
- Example lists in rubrics are guidance, not exhaustive

## Document Content
{response_text}

## Rubric Criterion to Evaluate
**Title**: {criterion}
**Category**: {axis}
**Weight**: {weight}

## Your Task
Evaluate whether the above document satisfies this specific rubric criterion.

## Required Response Format
Return ONLY valid JSON with this schema:
{{
  "verdict": "Not Satisfied" or "Satisfied",
  "score": 0.0 or 1.0,
  "confidence": 0.0 to 1.0,
  "reasoning": "Detailed explanation with specific evidence",
  "missing_elements": ["Element 1", "..."]
}}
"""

# Rubric-blind critic (sees only the task and the response), for feedback_mode=critic.
CRITIC = """You are a critical evaluator of research assistant responses. Given a task and \
a response, identify the most important gaps, errors, or weaknesses. Be direct and specific. \
Focus on what would most help the writer improve on a retry. Plain text only — 2-4 focused points.

Task:
{task_prompt}

Response:
{response}

Provide your critique:"""
