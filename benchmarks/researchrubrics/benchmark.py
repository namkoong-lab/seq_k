"""ResearchRubrics task loading + per-criterion rubric judge (soft score).

Each research response is graded one rubric criterion at a time by an LLM judge
(binary Satisfied/Not Satisfied). The task score is the weighted compliance
fraction; success = every mandatory (positive-weight) criterion satisfied AND no
penalty (negative-weight) criterion triggered.

Ported from the original seq_k_eval adapter, made fail-loud: a judge verdict that
cannot be parsed raises (with the raw text) instead of defaulting to "Not Satisfied".
Judging is sequential per rubric for simplicity; parallelize with a thread pool if
it becomes a bottleneck.
"""

from __future__ import annotations

import json
import re

from huggingface_hub import hf_hub_download

from core import llm
from core.types import Task, VerifierResult

from . import prompts

DATASET = "ScaleAI/researchrubrics"
FILENAME = "processed_data.jsonl"


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def load_tasks(domains=None):
    """Download ResearchRubrics and return its tasks.

    `domains` (optional list) filters to those domain labels.
    """
    path = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=FILENAME)
    tasks = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            task = _normalize(json.loads(line), idx)
            if domains and task.grading["domain"] not in set(domains):
                continue
            tasks.append(task)
    return tasks


def _normalize(record, idx):
    raw_rubrics = record.get("rubrics") or []
    if not isinstance(raw_rubrics, list):
        raw_rubrics = [raw_rubrics]
    rubrics = []
    for r in raw_rubrics:
        if not isinstance(r, dict) or "criterion" not in r:
            raise ValueError(f"ResearchRubrics record {idx} has a rubric without 'criterion'")
        rubrics.append({
            "criterion": str(r["criterion"]),
            "weight": float(r.get("weight", 1.0)),
            "axis": str(r.get("axis") or "Explicit Criteria"),
        })

    research_prompt = str(record.get("prompt") or "")
    if not research_prompt:
        raise ValueError(f"ResearchRubrics record {idx} has no prompt")

    sample_id = str(record.get("sample_id") or f"rr_{idx:05d}")
    return Task(
        id=sample_id,
        prompt=f"{prompts.ACTOR_INSTRUCTION}\n\n{research_prompt}",
        grading={"rubrics": rubrics, "domain": str(record.get("domain") or "")},
    )


# --------------------------------------------------------------------------- #
# Verifier (per-criterion LLM judge -> weighted compliance)
# --------------------------------------------------------------------------- #
def verify(task, attempt, *, judge_model):
    rubrics = task.grading["rubrics"]
    response = attempt.output or ""
    verdicts = [_judge_criterion(r, response, judge_model) for r in rubrics]

    compliance = _compliance(verdicts, rubrics)
    mandatory_pass = all(v["score"] == 1.0 for v, r in zip(verdicts, rubrics) if r["weight"] > 0)
    no_penalties = all(v["score"] == 0.0 for v, r in zip(verdicts, rubrics) if r["weight"] < 0)
    success = mandatory_pass and no_penalties

    return VerifierResult(
        success=success,
        score=float(compliance),
        raw_eval_output=("" if success else build_rubric_feedback(verdicts, rubrics)),
        private={
            "verdicts": verdicts,
            "compliance": compliance,
            "mandatory_total": sum(1 for r in rubrics if r["weight"] > 0),
            "mandatory_passed": sum(1 for v, r in zip(verdicts, rubrics)
                                    if v["score"] == 1.0 and r["weight"] > 0),
            "penalty_triggered": sum(1 for v, r in zip(verdicts, rubrics)
                                     if v["score"] == 1.0 and r["weight"] < 0),
        },
    )


def _judge_criterion(rubric, response_text, judge_model):
    prompt = prompts.JUDGE.format(
        response_text=response_text,
        criterion=rubric["criterion"],
        axis=rubric["axis"],
        weight=rubric["weight"],
    )
    raw = llm.complete(judge_model, prompt, temperature=0.0)
    return _parse_verdict(raw)


def _parse_verdict(raw):
    """Parse one criterion's judge JSON; raise if it can't be read."""
    payload = json.loads(extract_json_text(strip_code_fence(raw)))
    if not isinstance(payload, dict):
        raise ValueError(f"judge did not return a JSON object:\n{raw}")

    raw_score = payload.get("score")
    if raw_score is None:
        verdict = str(payload.get("verdict") or "").strip().lower()
        if "satisfied" in verdict and "not" not in verdict:
            raw_score = 1.0
        elif "not" in verdict and "satisfied" in verdict:
            raw_score = 0.0
        else:
            raise ValueError(f"judge verdict missing 'score' and unrecognized 'verdict':\n{raw}")
    score = 1.0 if float(raw_score) >= 0.5 else 0.0

    missing = payload.get("missing_elements") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    return {
        "verdict": "Satisfied" if score == 1.0 else "Not Satisfied",
        "score": score,
        "reasoning": str(payload.get("reasoning") or "").strip(),
        "missing_elements": missing,
    }


def _compliance(verdicts, rubrics):
    """Σ(score × weight) / Σ(positive weights)."""
    numerator = sum(v["score"] * r["weight"] for v, r in zip(verdicts, rubrics))
    denominator = sum(r["weight"] for r in rubrics if r["weight"] > 0)
    return numerator / denominator if denominator > 0 else 0.0


def build_rubric_feedback(verdicts, rubrics):
    """Mandatory failures first, then triggered penalties (the public diagnostic)."""
    mandatory_failures, penalty_triggers = [], []
    for v, r in zip(verdicts, rubrics):
        weight = r["weight"]
        if v["score"] == 0.0 and weight > 0:
            entry = f"[{r['axis']}] {r['criterion']}"
            if v["reasoning"]:
                entry += f"\n   Judge: {v['reasoning'][:250]}"
            missing = ", ".join(str(m) for m in (v.get("missing_elements") or []))
            if missing:
                entry += f"\n   Missing: {missing}"
            mandatory_failures.append(entry)
        elif v["score"] == 1.0 and weight < 0:
            penalty_triggers.append(f"[{r['axis']}] {r['criterion']}\n   Judge: {v['reasoning'][:200]}")

    lines = []
    if mandatory_failures:
        lines.append(f"Your response failed {len(mandatory_failures)} mandatory criteria:")
        lines += [f"  {i}. {e}" for i, e in enumerate(mandatory_failures, 1)]
    if penalty_triggers:
        lines.append(f"\n{len(penalty_triggers)} penalty criteria were triggered:")
        lines += [f"  {i}. {e}" for i, e in enumerate(penalty_triggers, 1)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


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
    return response.strip()
