"""AdvancedIF task loading + per-question instruction-following judge.

The model continues a conversation; an LLM judge then decides, requirement by
requirement, whether the response satisfies each. Scoring is all-or-nothing
(score 1.0 only if every requirement is met) — matching AdvancedIF's binary
treatment in the paper.

NOTE: the original seq_k_eval adapter shelled out to the upstream AdvancedIF
repo's `cli.py evaluate` for grading. To keep this repo self-contained (no
external checkout, no subprocess), the per-question judging is reimplemented here
with a single LLM judge call. The concept matches, but scores will not be
bit-identical to the upstream evaluator; wire the upstream CLI back in if you need
paper-exact AdvancedIF numbers.

Data is a prepared AdvancedIF JSONL (conversation_history + rubrics per record),
passed via a variant's `options: {data_path: ...}`. A grader response that cannot
be parsed raises (fail-loud).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core import llm
from core.types import Task, VerifierResult

from . import prompts

_BENCHMARK_NAME_ALIASES = {
    "system_steerability_v2": "if_system_steerability_oss",
    "carried_context_multi_turn_eval_v5": "if_carried_context_oss",
    "complex_if_single_turn_v5": "if_complex_if_oss",
}
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def load_tasks(data_path):
    """Load AdvancedIF tasks from a prepared JSONL (set via options.data_path)."""
    path = Path(data_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"AdvancedIF data path does not exist: {path}")
    tasks, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(_normalize(json.loads(line), idx))
            except ValueError:
                skipped += 1   # structurally unsupported record (e.g. unhandled multi-turn shape)
    if not tasks:
        raise ValueError(f"no compatible AdvancedIF tasks found in {path}")
    if skipped:
        print(f"AdvancedIF: skipped {skipped} incompatible records in {path}")
    return tasks


def _normalize(record, idx):
    conversation = _normalize_conversation(record.get("conversation_history") or [])
    if not conversation:
        raise ValueError(f"AdvancedIF record {idx} has no usable conversation content")

    prompt_metadata = _as_dict(record.get("prompt_metadata"))
    rubrics = _normalize_rubrics(prompt_metadata.get("rubrics") or record.get("rubrics"))
    if not rubrics:
        raise ValueError(f"AdvancedIF record {idx} is missing rubrics")

    benchmark_name = str(_BENCHMARK_NAME_ALIASES.get(
        str(record.get("benchmark_name") or "").strip(),
        str(record.get("benchmark_name") or "").strip()))
    if not benchmark_name and len(conversation) != 1:
        raise ValueError(f"AdvancedIF record {idx} has unsupported multi-turn shape")

    source_row = int(record.get("source_row") or idx + 2)
    task_id = str(record.get("task_id") or f"advancedif_{source_row:05d}")
    return Task(
        id=task_id,
        prompt=_build_actor_prompt(benchmark_name, conversation),
        grading={"rubrics": rubrics, "conversation": _transcript(conversation),
                 "benchmark_name": benchmark_name},
    )


# --------------------------------------------------------------------------- #
# Verifier (per-question LLM judge, all-or-nothing)
# --------------------------------------------------------------------------- #
def verify(task, attempt, *, judge_model):
    rubrics = task.grading["rubrics"]
    requirements = "\n".join(f"{i}. {r}" for i, r in enumerate(rubrics, 1))
    judge_prompt = prompts.JUDGE.format(
        conversation=task.grading["conversation"],
        response=attempt.output or "",
        requirements=requirements,
    )
    judge_output = llm.complete(judge_model, judge_prompt, temperature=0.0)
    verdicts = _parse_verdicts(judge_output, len(rubrics))
    success = all(v["met"] for v in verdicts)
    return VerifierResult(
        success=success,
        score=1.0 if success else 0.0,
        raw_eval_output=("" if success else _format_verdicts(verdicts)),
        details={"verdicts": verdicts,
                       "rubric_count": len(rubrics),
                       "met_count": sum(1 for v in verdicts if v["met"])},
    )


def _parse_verdicts(judge_output, expected_count):
    """Strict parse of the judge's per-question verdicts; raise if unreadable."""
    payload = json.loads(_extract_json(judge_output))
    if not isinstance(payload, dict) or not isinstance(payload.get("verdicts"), list):
        raise ValueError(f"judge did not return a 'verdicts' list:\n{judge_output}")
    verdicts = []
    for i, v in enumerate(payload["verdicts"], 1):
        if not isinstance(v, dict) or "met" not in v:
            raise ValueError(f"verdict {i} missing 'met':\n{judge_output}")
        verdicts.append({"question": v.get("question", i),
                         "met": _as_bool(v["met"]),
                         "reason": str(v.get("reason") or "").strip()})
    if len(verdicts) != expected_count:
        raise ValueError(f"judge returned {len(verdicts)} verdicts for "
                         f"{expected_count} requirements:\n{judge_output}")
    return verdicts


def _format_verdicts(verdicts):
    parts = []
    for v in verdicts:
        line = f"question_{v['question']}: {'Yes' if v['met'] else 'No'}"
        if not v["met"] and v["reason"]:
            line += f" — {v['reason']}"
        parts.append(line)
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Prompt building + normalization helpers
# --------------------------------------------------------------------------- #
def _build_actor_prompt(benchmark_name, conversation):
    bn = benchmark_name.lower()
    if _is_single_turn(bn, conversation):
        return conversation[0]["content"]
    transcript = _transcript(conversation)
    if "system_steer" in bn:
        return ("You are the assistant. Follow the system instructions and continue the "
                "conversation below. Write only the next assistant response.\n\n" + transcript)
    return ("You are the assistant. Continue the conversation below. "
            "Write only the next assistant response.\n\n" + transcript)


def _is_single_turn(benchmark_name, conversation):
    if "single_turn" in benchmark_name:
        return True
    return len(conversation) == 1 and conversation[0]["role"] == "user"


def _transcript(conversation):
    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in conversation)


def _normalize_conversation(value):
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    if not isinstance(value, list):
        raise ValueError("conversation_history must be a list")
    out = []
    for m in value:
        if not isinstance(m, dict):
            raise ValueError("conversation_history entries must be objects")
        content = _coerce_text(m.get("content")).strip()
        if not content:
            continue
        role = str(m.get("role") or "user").strip().lower()
        out.append({"role": role if role in ("system", "user", "assistant") else "user",
                    "content": content})
    return out


def _normalize_rubrics(value):
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [text]
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item or "").strip()]


def _as_dict(value):
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else {}
    return dict(value) if isinstance(value, dict) else {}


def _coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1", "met")


def _extract_json(response):
    m = _JSON_BLOCK.search(str(response or ""))
    if m:
        return m.group(1).strip()
    text = str(response or "")
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1].strip()
    return text.strip()
