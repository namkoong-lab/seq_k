"""CLBench: task loading + rubric-judge verifier.

Loads the Domain Knowledge Reasoning slice of CL-bench and is fail-loud: a judge
response that does not parse raises a ValueError (with the raw judge text) rather
than defaulting to a score of 0.
"""

from __future__ import annotations

import json
import re

from huggingface_hub import hf_hub_download

from core import llm
from core.types import Task, VerifierResult

from . import prompts

DATASET = "tencent/CL-bench"
FILENAME = "CL-bench.jsonl"
CATEGORY = "Domain Knowledge Reasoning"


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def load_tasks():
    """Download CL-bench and return the Domain Knowledge Reasoning tasks."""
    path = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=FILENAME)
    tasks = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            task = _normalize(json.loads(line), idx)
            if task.grading["context_category"] == CATEGORY:
                tasks.append(task)
    return tasks


def _normalize(record, idx):
    raw_messages = record.get("messages") or []
    messages = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user").strip().lower() or "user"
        messages.append({"role": role, "content": coerce_chat_content(m.get("content"))})
    if not messages:
        raise ValueError(f"CL-bench record {idx} has no messages")

    rubrics = record.get("rubrics") or []
    if not isinstance(rubrics, list):
        rubrics = [rubrics]

    meta = record.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    task_id = (meta.get("task_id") or record.get("task_id")
               or record.get("idx") or f"clbench_{idx:05d}")

    prompt = (
        "Conversation history (oldest to newest):\n"
        f"{render_chat_messages(messages)}\n\n"
        "Write the next assistant response to continue this conversation. "
        "Return only that assistant response."
    )
    return Task(
        id=str(task_id),
        prompt=prompt,
        grading={
            "rubrics": rubrics,
            "context_category": str(meta.get("context_category") or "").strip(),
            "messages": messages,
        },
    )


# --------------------------------------------------------------------------- #
# Verifier (rubric judge)
# --------------------------------------------------------------------------- #
def verify(task, attempt, *, judge_model):
    rubrics = task.grading["rubrics"]
    total = len(rubric_criteria_list(rubrics))
    judge_prompt = prompts.JUDGE.format(
        rubrics_text=build_rubrics_text(rubrics),
        model_output=attempt.output or "",
    )
    raw = llm.complete(judge_model, judge_prompt, temperature=0.0)
    parsed = _parse_judge(raw)

    score = parsed["score"]
    success = score == 1
    status = parsed["requirement_status"]
    failed = (sum(1 for s in status if s == "no")
              if total and len(status) == total else None)

    return VerifierResult(
        success=success,
        score=float(score),
        raw_eval_output=(raw.strip() if not success else ""),
        private={
            "judge_prompt": judge_prompt,
            "requirement_status": status,
            "grading_rationale": parsed["rationale"],
            "judge_raw_output": raw,
            "failed_requirement_count": failed,
            "total_requirements": total,
        },
    )


def _parse_judge(raw):
    """Strict parse of the judge JSON. Raises (fail-loud) if it can't be read."""
    payload = json.loads(extract_json_text(strip_code_fence(raw)))
    if not isinstance(payload, dict):
        raise ValueError(f"judge did not return a JSON object:\n{raw}")

    raw_score = payload.get("Overall Score")
    if raw_score is None:
        raw_score = payload.get("overall_score")
    if raw_score is None:
        raw_score = payload.get("score")
    score = parse_score(raw_score)
    if score is None:
        raise ValueError(f"judge response missing/invalid 'Overall Score':\n{raw}")

    status = payload.get("List of Requirement Satisfaction Status")
    if status is None:
        status = payload.get("requirement_status")
    if not isinstance(status, list):
        status = []
    status = [str(s).strip().lower() for s in status]

    rationale = payload.get("Grading Rationale") or payload.get("grading_rationale") or ""
    return {"score": score, "requirement_status": status, "rationale": str(rationale).strip()}


# --------------------------------------------------------------------------- #
# Shared helpers (also used by feedback.py)
# --------------------------------------------------------------------------- #
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)


def coerce_chat_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict) and block.get("text") is not None:
                chunks.append(str(block["text"]))
        return "".join(chunks)
    return str(content)


def render_chat_messages(messages):
    lines = []
    for idx, m in enumerate(messages):
        role = str(m.get("role") or "unknown").strip().lower()
        lines.append(f"[{idx:02d}] {role}:")
        lines.append(coerce_chat_content(m.get("content")).strip())
    return "\n\n".join(lines).strip()


def _rubric_criteria(rubric):
    if isinstance(rubric, dict):
        return str(rubric.get("rubric_criteria") or "").strip()
    return str(rubric or "").strip()


def rubric_criteria_list(rubrics):
    return [c for c in (_rubric_criteria(r) for r in rubrics) if c]


def build_rubrics_text(rubrics):
    crit = rubric_criteria_list(rubrics)
    if not crit:
        return "No explicit rubrics provided."
    return "\n".join(f"{i}. {c}" for i, c in enumerate(crit, 1))


def strip_code_fence(text):
    t = str(text or "").strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def extract_json_text(response):
    m = _JSON_BLOCK.search(response)
    if m:
        return m.group(1).strip()
    start, end = response.find("{"), response.rfind("}")
    if start != -1 and end > start:
        return response[start:end + 1].strip()
    start, end = response.find("["), response.rfind("]")
    if start != -1 and end > start:
        return response[start:end + 1].strip()
    return response.strip()


def parse_score(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        iv = int(value)
        return iv if iv in (0, 1) else None
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    if re.fullmatch(r"0(?:\.0+)?(?:\s+points?)?", text):
        return 0
    if re.fullmatch(r"1(?:\.0+)?(?:\s+points?)?", text):
        return 1
    return None
