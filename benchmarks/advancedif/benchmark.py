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

Data downloads at run time from the official HF dataset (facebook/AdvancedIF,
if_oss_full_data.csv) — same pattern as clbench/healthbench/researchrubrics, no
local file needed. Before indexing, the CSV's benchmark blocks are reordered
hardest-first (system-steerability → carried-context → complex-IF), each block
keeping its internal CSV order — so `max_tasks` subsets hit the hard tasks
(every system-steerability task carries a system prompt). task_id is
advancedif_<1-based position in the REORDERED list> — the same numbering the
prepared JSONL used, so every existing run's task ids still line up. A grader
response that cannot be parsed raises (fail-loud).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from core import llm
from core.types import Task, VerifierResult

from . import prompts

_BENCHMARK_NAME_ALIASES = {
    "system_steerability_v2": "if_system_steerability_oss",
    "carried_context_multi_turn_eval_v5": "if_carried_context_oss",
    "complex_if_single_turn_v5": "if_complex_if_oss",
}
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

DATASET = "facebook/AdvancedIF"
FILENAME = "if_oss_full_data.csv"
CATEGORY_CACHE = "data/advancedif/advancedIF.categories.json"   # repo-relative default

# Task order: the CSV's three benchmark blocks, hardest first (system-steerability
# tasks all carry a system prompt). Unknown names sort last. Within a block the
# CSV order is kept. This matches the prepared-JSONL ordering every existing run
# was numbered with — do not change, task_id/canonical_index stability depends on it.
_BLOCK_ORDER = {
    "system_steerability_v2":              0,
    "carried_context_multi_turn_eval_v5":  1,
    "complex_if_single_turn_v5":           2,
}

# The CSV cells hold full conversations — far past csv's default 128K field cap.
csv.field_size_limit(sys.maxsize)


# --------------------------------------------------------------------------- #
# Path-layout declarations (consumed by core/results.py)
# --------------------------------------------------------------------------- #
VERIFIER = "llm"               # an LLM judge grades the response against rubrics
# binary/raw/compact are template-only; these call the critic LLM.
# `guided` is not implemented in this file — it comes from a newer version of
# this benchmark used by collaborators, and is declared here so imported runs
# get a correct identity: two of them differ ONLY in critic_model, which is
# only possible if the mode consults the critic. Omitting it silently merged them.
LLM_CRITIC_MODES = {"critique", "guided", "compact_eval_output"}


def slice_name(_options):
    """Single slice — the JSONL file is the dataset; subset is by `task_indices`."""
    return "advancedif"


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def load_tasks(category_cache_path=None):
    """Download the official AdvancedIF CSV from HF and return tasks, hard-first.
    Rows are stably reordered by benchmark block (_BLOCK_ORDER, hardest first),
    each block keeping its internal CSV order, then numbered 1-based; that
    position is the task_id and (for compatible records) the canonical_index.

    If a rubric-category cache exists (see scripts/classify_rubrics.py), attach each
    task's per-rubric categories to grading['rubric_categories'] for the `category`
    feedback mode. Missing cache is fine here — only feedback(mode='category') requires
    it, and it fails loud there. category_cache_path defaults to CATEGORY_CACHE."""
    path = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=FILENAME)
    with open(path, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    # Hard-first block reorder; sort() is stable, so each block keeps its CSV order.
    records.sort(key=lambda r: _BLOCK_ORDER.get(str(r.get("benchmark_name") or "").strip(), len(_BLOCK_ORDER)))
    # Number AFTER the reorder — task_id is the 1-based position in the reordered
    # list, matching the prepared-JSONL numbering every existing run was scored with.
    for i, row in enumerate(records, 1):
        row["source_row"] = i
        row["task_id"] = f"advancedif_{i:05d}"
    tasks, skipped = [], 0
    for record in records:
        try:
            tasks.append(_normalize(record, record["source_row"] - 1, canonical_index=len(tasks) + 1))
        except ValueError:
            skipped += 1   # structurally unsupported record (e.g. unhandled multi-turn shape)
    if not tasks:
        raise ValueError(f"no compatible AdvancedIF tasks found in {DATASET}")
    if skipped:
        print(f"AdvancedIF: skipped {skipped} incompatible records in {DATASET}")
    _attach_rubric_categories(tasks, category_cache_path)
    return tasks


def _attach_rubric_categories(tasks, category_cache_path=None):
    cache_path = (Path(category_cache_path).expanduser() if category_cache_path
                  else Path(CATEGORY_CACHE))
    if not cache_path.exists():
        return
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    for task in tasks:
        labels = cache.get(task.id)
        if labels is not None:
            task.grading["rubric_categories"] = labels   # list[list[str]] aligned to rubrics


def _normalize(record, idx, *, canonical_index):
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
        canonical_index=canonical_index,
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
