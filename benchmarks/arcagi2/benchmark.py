"""ARC-AGI-2 task loading + deterministic exact-grid verifier.

No LLM judge: the model emits output grids as JSON, we parse them (with light
JSON / digit-stream repair) and compare cell-for-cell against the held-out test
outputs. success = every test grid matches exactly; score is binary (1.0/0.0).
The informative feedback is a bounded per-grid cell-match summary that never
reveals the target grids.

A malformed model answer is a real task failure here (score 0 + a format hint),
not a swallowed infrastructure error — distinct from the LLM-judge benchmarks
where a judge that won't parse raises. Needs a local ARC-AGI-2 data checkout,
passed via a variant's `options: {data_dir: ...}`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.types import Task, VerifierResult

from . import prompts

FEEDBACK_MAX_MISMATCHES = 8          # cap on per-grid status lines in the cell-match summary
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Path-layout declarations (consumed by core/results.py)
# --------------------------------------------------------------------------- #
VERIFIER = "deterministic"     # exact-grid match — no LLM judge
LLM_CRITIC_MODES = set()       # binary / cell_match are template-only


def slice_name(options):
    """Each split (evaluation / training) is its own slice with its own
    canonical task numbering."""
    return f"arcagi2-{options.get('split', 'evaluation')}"


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def load_tasks(data_dir, split="evaluation"):
    """Load ARC-AGI-2 tasks from a local checkout. canonical_index = 1-based
    position in the deterministic _task_ids() order (file list or sorted glob).

    `data_dir` points at the repo's `data/` directory (it contains
    `<split>/<task_id>.json`). Pass it via a variant's `options: {data_dir: ...}`.
    """
    data_dir = Path(data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"ARC-AGI-2 data directory not found: {data_dir}. Clone "
            "https://github.com/arcprize/ARC-AGI-2 and point options.data_dir at its data/ folder."
        )
    tasks = []
    for i, task_id in enumerate(_task_ids(data_dir, split), 1):
        raw = json.loads((data_dir / split / f"{task_id}.json").read_text(encoding="utf-8"))
        tasks.append(_normalize(raw, task_id, canonical_index=i))
    return tasks


def _task_ids(data_dir, split):
    list_path = data_dir / f"{split}.txt"
    if list_path.exists():
        return [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted(p.stem for p in (data_dir / split).glob("*.json"))


def _normalize(raw, task_id, *, canonical_index):
    expected_outputs = [pair.get("output") for pair in raw.get("test", [])]
    return Task(
        id=str(task_id),
        canonical_index=canonical_index,
        prompt=format_problem(raw),
        grading={"expected_outputs": expected_outputs},
    )


def format_problem(task):
    train_lines = []
    for idx, pair in enumerate(task.get("train", [])):
        train_lines += [f"--Example {idx}--", "", "INPUT:", "", json.dumps(pair.get("input")),
                        "", "OUTPUT:", "", json.dumps(pair.get("output")), ""]
    test_pairs = task.get("test", [])
    if len(test_pairs) == 1:
        test_lines = [json.dumps(test_pairs[0].get("input"))]
        head, tail = "--Test Input--", "--End of Test Input--"
    else:
        test_lines = []
        for idx, pair in enumerate(test_pairs):
            test_lines += [f"--Test Input {idx}--", "", json.dumps(pair.get("input")), ""]
        head, tail = "--Test Inputs--", "--End of Test Inputs--"
    return "\n".join([
        prompts.PUZZLE_INTRO, "",
        "--Training Examples--", "\n".join(train_lines).rstrip(), "--End of Training Examples--", "",
        head, "\n".join(test_lines).rstrip(), tail, "",
        prompts.OUTPUT_INSTRUCTION,
    ])


# --------------------------------------------------------------------------- #
# Verifier (deterministic exact grid match)
# --------------------------------------------------------------------------- #
def verify(task, attempt, *, judge_model=None):   # judge_model unused (deterministic)
    expected = task.grading["expected_outputs"]
    try:
        prediction, parse_method = parse_prediction(attempt.output or "", expected)
    except Exception as exc:
        return VerifierResult(
            success=False, score=0.0,
            raw_eval_output=_parse_error_feedback(str(exc), expected),
            details={"parse_method": "failed", "parse_error": str(exc)},
        )

    report = compare_exact(prediction, expected)
    success = bool(report["all_correct"])
    raw_eval = "" if success else build_cell_match_feedback(report, FEEDBACK_MAX_MISMATCHES)
    warnings = _check_grid_format(prediction, expected)
    if warnings and not success:
        raw_eval = warnings + "\n" + raw_eval
    return VerifierResult(
        success=success,
        score=1.0 if success else 0.0,
        raw_eval_output=raw_eval,
        details={"parse_method": parse_method, "comparison": report, "prediction": prediction},
    )


# --------------------------------------------------------------------------- #
# Prediction parsing
# --------------------------------------------------------------------------- #
def parse_prediction(raw_output, expected_outputs, allow_digit_stream_repair=True):
    try:
        payload = _load_json_payload(raw_output)
        return _normalize_prediction(payload, len(expected_outputs)), "json"
    except Exception:
        recovered = _from_digit_stream(raw_output, expected_outputs)
        if recovered is None or not allow_digit_stream_repair:
            raise
        return recovered, "digit_stream_repair"


def _extract_json_text(response):
    text = str(response or "")
    m = _JSON_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            return text[start:end + 1].strip()
    return text.strip()


def _repair_json_text(text):
    repaired = str(text or "").strip()
    if not repaired:
        return repaired
    repaired = repaired.translate(str.maketrans({
        "“": '"', "”": '"', "‘": "'", "’": "'",
        " ": " ", "，": ",", "：": ":",
    }))
    for _ in range(4):
        prev = repaired
        repaired = re.sub(r"(\]|\})\s*(\[|\{)", r"\1,\2", repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r",\s*,+", ",", repaired)
        if repaired == prev:
            break
    square_diff = repaired.count("[") - repaired.count("]")
    if square_diff > 0:
        m = re.search(r"\}+$", repaired)
        if m:
            repaired = repaired[:m.start()] + ("]" * square_diff) + repaired[m.start():]
        else:
            repaired += "]" * square_diff
    curly_diff = repaired.count("{") - repaired.count("}")
    if curly_diff > 0:
        repaired += "}" * curly_diff
    return repaired


def _load_json_payload(raw_output):
    for candidate in (_extract_json_text(raw_output), str(raw_output or "").strip()):
        if not candidate:
            continue
        for variant in (candidate, _repair_json_text(candidate)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(variant.lstrip())
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except json.JSONDecodeError:
                    continue
    raise ValueError("could not parse a JSON object/array from the response")


def _is_grid(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(row, list) and bool(row) for row in value))


def _validate_grid(grid):
    if not _is_grid(grid):
        raise ValueError("grid must be a non-empty list of non-empty rows")
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("grid rows must all be the same length")
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                raise ValueError("grid cells must be integers 0-9")
    return grid


def _shape(grid):
    return (len(grid), len(grid[0]) if grid else 0)


def _normalize_prediction(value, expected_count):
    def is_grid_list(c):
        return isinstance(c, list) and bool(c) and all(_is_grid(x) for x in c)

    if isinstance(value, dict):
        value = value.get("test", value.get("output", value))
    if isinstance(value, list):
        if value and all(isinstance(x, dict) and "output" in x for x in value):
            value = [x["output"] for x in value]
        if expected_count == 1 and _is_grid(value) and not is_grid_list(value):
            return [_validate_grid(value)]
        if len(value) != expected_count:
            raise ValueError(f"expected {expected_count} test outputs, got {len(value)}")
        return [_validate_grid(g) for g in value]
    if _is_grid(value) and expected_count == 1:
        return [_validate_grid(value)]
    raise ValueError("output JSON must be a list of grids or an object with 'test'")


def _from_digit_stream(raw_output, expected_outputs):
    try:
        shapes = [_shape(_validate_grid(g)) for g in expected_outputs]
    except Exception:
        return None
    total = sum(r * c for r, c in shapes)
    if total <= 0:
        return None
    nums = [int(t) for t in re.findall(r"-?\d+", _extract_json_text(raw_output))]
    nums = [n for n in nums if 0 <= n <= 9]
    if len(nums) < total:
        return None
    stream, offset, prediction = nums[:total], 0, []
    for rows, cols in shapes:
        grid = []
        for _ in range(rows):
            row = stream[offset:offset + cols]
            if len(row) != cols:
                return None
            offset += cols
            grid.append(row)
        prediction.append(grid)
    return prediction


# --------------------------------------------------------------------------- #
# Comparison + feedback
# --------------------------------------------------------------------------- #
def compare_exact(prediction, expected):
    report = {"all_correct": True, "tests": []}
    for idx, exp in enumerate(expected):
        pred = prediction[idx]
        ps, es = _shape(pred), _shape(_validate_grid(exp))
        t = {"test_index": idx, "size_match": ps == es, "exact_match": False, "mismatch_count": 0}
        if ps != es:
            report["all_correct"] = False
            report["tests"].append(t)
            continue
        total = es[0] * es[1]
        mismatches = sum(1 for r in range(es[0]) for c in range(es[1]) if pred[r][c] != exp[r][c])
        t["total_cells"] = total
        if mismatches:
            report["all_correct"] = False
            t["mismatch_count"] = mismatches
            t["cell_accuracy"] = (total - mismatches) / total if total else 0.0
        else:
            t["exact_match"] = True
            t["cell_accuracy"] = 1.0
        report["tests"].append(t)
    return report


def build_cell_match_feedback(report, max_lines):
    if report.get("all_correct"):
        return "correct_output"
    tests = report.get("tests", [])
    lines = ["incorrect_output"]
    for used, t in enumerate(tests):
        if used >= max_lines:
            lines.append(f"...truncated remaining grid status (cap={max_lines}).")
            break
        label = "Output grid" if len(tests) == 1 else f"Output grid {int(t['test_index']) + 1}"
        if t.get("exact_match"):
            lines.append(f"{label}: size correct: True; content match accuracy: 100.00%.")
        elif not t.get("size_match"):
            lines.append(f"{label}: size correct: False.")
        else:
            total = int(t.get("total_cells", 0))
            mm = int(t.get("mismatch_count", 0))
            lines.append(f"{label}: size correct: True; content match accuracy: "
                         f"{t.get('cell_accuracy', 0.0):.2%} ({total - mm} of {total} cells match).")
    return "\n".join(lines)


def _parse_error_feedback(error_str, expected_outputs):
    lines = ["format_error: could not parse your response as valid grid JSON.",
             'hint: return a JSON object with key "test" mapping to a list of grids. '
             "Each grid is a list of lists of integers 0-9. No markdown."]
    if len(expected_outputs) != 1:
        lines.append(f"hint: expected {len(expected_outputs)} test output grid(s).")
    return "\n".join(lines)


def _check_grid_format(prediction, expected_outputs):
    warnings = []
    for idx, pred in enumerate(prediction):
        ph, pw = _shape(pred)
        eh, ew = _shape(expected_outputs[idx])
        if (ph, pw) != (eh, ew):
            warnings.append(f"format_warning: Output grid {idx + 1} is {ph}x{pw}, "
                            "but the expected size differs — check the transformation rule.")
        if len({len(row) for row in pred}) > 1:
            warnings.append(f"format_warning: Output grid {idx + 1} has inconsistent row widths.")
    return "\n".join(warnings)
