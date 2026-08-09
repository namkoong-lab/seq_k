"""MBPP Pro deterministic benchmark for the seq_k harness.

The actor sees the base and self-invoking problem statements but never the
reference solutions or hidden tests. The verifier appends the hidden test code
to the candidate and executes it in a bounded subprocess.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from core.types import Task, VerifierResult

from . import prompts

VERIFIER = "deterministic"
LLM_CRITIC_MODES: set[str] = set()

DATASET_NAME = "CodeEval-Pro/mbpp-pro"
DATA_REVISION = "36f292eafc597b38535fbc8b4aafea8e5c654e3c"
DATA_URL = (
    "https://raw.githubusercontent.com/CodeEval-Pro/CodeEval-Pro/"
    f"{DATA_REVISION}/dataset/mbpp_pro.json"
)
DATA_SHA256 = "bb2edf4f2da393403083977e94cb4e4bbbe68fcb85495dfdf500885a945e8b3b"
_DEFAULT_TIMEOUT_SECONDS = 30
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_NO_REPLAY_ERRORS = {"SyntaxError", "IndentationError", "TabError", "EarlyExit"}


@dataclass(frozen=True)
class ScriptResult:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    error_type: str


def slice_name(options: dict | None = None) -> str:
    """Name used by ``core.results.build_run_path`` for this benchmark slice."""
    return "mbpppro"


def load_tasks(
    data_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    exclude_canonical_indices: list[int] | tuple[int, ...] = (),
) -> list[Task]:
    records = _load_records(data_path=data_path, cache_dir=cache_dir)

    tasks: list[Task] = []
    for canonical_index, record in enumerate(_sorted_records(records), start=1):
        record_id = _record_id(record, fallback=canonical_index - 1)
        tasks.append(
            Task(
                id=f"mbpp-{record_id}",
                canonical_index=canonical_index,
                prompt=_actor_prompt(record),
                grading={
                    "dataset": "mbpp",
                    "source_dataset": DATASET_NAME,
                    "source_revision": DATA_REVISION,
                    "record": record,
                    "test_code": str(record["test_code"]),
                    "timeout_seconds": int(timeout_seconds),
                },
            )
        )
    excluded = {int(index) for index in exclude_canonical_indices}
    if excluded:
        missing = excluded - {task.canonical_index for task in tasks}
        if missing:
            raise ValueError(
                f"exclude_canonical_indices not found: {sorted(missing)}"
            )
        tasks = [task for task in tasks if task.canonical_index not in excluded]
    return tasks


def _load_records(
    *,
    data_path: str | Path | None,
    cache_dir: str | Path | None,
) -> list[dict]:
    path = Path(data_path).expanduser() if data_path else _cached_dataset_path(cache_dir)
    if data_path:
        if not path.exists():
            raise FileNotFoundError(f"MBPP Pro data file does not exist: {path}")
    elif not _valid_pinned_cache(path):
        _download_pinned_dataset(path)

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"expected MBPP Pro JSON list, got {type(records).__name__}")

    required = {"raw_problem", "new_problem", "test_code"}
    for index, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"MBPP Pro record {index} missing fields: {sorted(missing)}")
    return records


def _cached_dataset_path(cache_dir: str | Path | None) -> Path:
    if cache_dir:
        root = Path(cache_dir).expanduser()
    else:
        root = Path.home() / ".cache" / "seq_k" / "mbpppro"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            root = Path(tempfile.gettempdir()) / "seq_k-cache" / "mbpppro"
    return root / "mbpp_pro.json"


def _download_pinned_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            payload = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the pinned MBPP Pro JSON. Download it manually "
            f"from {DATA_URL} and set options.data_path."
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(
            f"MBPP Pro checksum mismatch: expected {DATA_SHA256}, got {digest}"
        )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _valid_pinned_cache(path: Path) -> bool:
    """Return whether an existing default cache is complete and pinned."""
    if not path.exists():
        return False
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
            return False
        records = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(records, list)


def _actor_prompt(record: dict) -> str:
    return prompts.ACTOR_PROMPT.format(
        base_problem=str(record["raw_problem"]).rstrip(),
        main_problem=str(record["new_problem"]).rstrip(),
    )


def _sorted_records(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda record: _record_id(record, fallback=10**12))


def _record_id(record: dict, *, fallback: int) -> int:
    try:
        return int(record.get("id", fallback))
    except (TypeError, ValueError):
        return fallback


def verify(task, attempt, *, judge_model=None):
    """Grade one candidate solution by running it against hidden Python tests."""

    timeout_seconds = int(task.grading.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
    code = extract_code(attempt.output)
    test_code = task.grading["test_code"]

    result = _run_guarded(f"{code}\n\n{test_code}\n", timeout_seconds=timeout_seconds)
    success = result.status == "passed"

    cases = _split_replay_cases(test_code)
    if success:
        failure_cases = []
        passed_cases = len(cases)
    elif result.error_type in _NO_REPLAY_ERRORS:
        failure_cases = []
        passed_cases = 0
    else:
        failure_cases, passed_cases = _failure_cases(
            code, cases, timeout_seconds=timeout_seconds,
        )

    return VerifierResult(
        success=success,
        score=1.0 if success else 0.0,
        raw_eval_output=_public_feedback(result, failure_cases),
        details={
            "dataset": task.grading["dataset"],
            "source_dataset": task.grading["source_dataset"],
            "source_revision": task.grading["source_revision"],
            "status": result.status,
            "returncode": result.returncode,
            "error_type": result.error_type,
            "stdout": _clip(result.stdout, 4000),
            "stderr": _clip(result.stderr, 4000),
            "failure_cases": failure_cases,
            "passed_cases": passed_cases,
            "total_cases": len(cases),
        },
    )


def extract_code(output: str) -> str:
    """Extract Python code from model output, accepting fenced or raw code."""

    if not output:
        return ""

    blocks = _CODE_FENCE_RE.findall(output)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    return str(output).strip()


def _run_guarded(script: str, *, timeout_seconds: float) -> ScriptResult:
    """Run the script and detect clean exits that occur before tests finish."""
    token = f"__seqk_{secrets.token_hex(8)}__"
    epilogue = (f"\n\nimport os as __seqk_os\n"
                f"__seqk_os.write(1, {token!r}.encode())\n")
    result = _run_python(script + epilogue, timeout_seconds=timeout_seconds)
    if result.status == "passed" and token not in result.stdout:
        result = replace(result, status="failed", error_type="EarlyExit")
    return replace(result, stdout=result.stdout.replace(token, "").rstrip("\n"))


def _run_python(script: str, *, timeout_seconds: float) -> ScriptResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "candidate.py"
        path.write_text(script, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmpdir,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return ScriptResult(
                status="failed",
                returncode=None,
                stdout=stdout,
                stderr=stderr,
                error_type="Timeout",
            )

    if completed.returncode == 0:
        return ScriptResult(
            status="passed",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error_type="Passed",
        )

    return ScriptResult(
        status="failed",
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        error_type=_error_type(completed.stderr),
    )


def _failure_cases(code: str, cases: list[dict], *, timeout_seconds: float) -> tuple[list[dict], int]:
    """Diagnose failed test cases for sequential feedback."""
    if not cases: 
        return [], 0

    failures = []
    passed_cases = 0

    for case in cases:
        script = f"{code}\n\n{case['setup']}\n\n{case['test']}\n"
        result = _run_guarded(script, timeout_seconds=timeout_seconds)
        if result.status == "passed":
            passed_cases += 1
            continue

        failure = {
            "name": f"test_{case['index']}",
            "index": case["index"],
            "exception_type": result.error_type,
            "assertion": case["test"],
            "summary": _case_summary(result),
            "message": _tail_line(result.stderr),
        }
        if result.error_type != "Timeout":
            details = _assert_details(code, case, timeout_seconds=timeout_seconds)
            failure.update({k: v for k, v in details.items() if v is not None})
        failures.append(failure)

    return failures, passed_cases


def _split_replay_cases(test_code: str) -> list[dict]:
    """Split top-level asserts and try-wrapped asserts into replay cases."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    case_ranges = []
    for node in tree.body:
        if isinstance(node, ast.Assert) or (
            isinstance(node, ast.Try)
            and any(isinstance(child, ast.Assert) for child in ast.walk(node))
        ):
            
            end_lineno = getattr(node, "end_lineno", None) or node.lineno
            case_ranges.append((node.lineno, end_lineno))

    if not case_ranges:
        return []

    lines = test_code.splitlines()
    cases = []

    for index, (start, end) in enumerate(case_ranges, start=1):
        setup = "\n".join(
            line for line_no, line in enumerate(lines[: start - 1], start=1)
            if not any(
                case_start <= line_no <= case_end
                for case_start, case_end in case_ranges)).strip()

        cases.append({
            "index": index,
            "setup": setup,
            "test": "\n".join(lines[start - 1:end]).strip(),
        })

    return cases


def _assert_details(code: str, case: dict, *, timeout_seconds: float) -> dict:
    parsed = _parse_simple_equality_assert(case["test"])
    if not parsed:
        return {}

    left_expr, right_expr = parsed
    probe = (
        f"{code}\n\n{case['setup']}\n\n"
        "import json as __evalpro_json\n"
        "try:\n"
        f"    __actual = {left_expr}\n"
        f"    __expected = {right_expr}\n"
        "    print(__evalpro_json.dumps({\n"
        "        'actual': repr(__actual),\n"
        "        'expected': repr(__expected),\n"
        "    }))\n"
        "except Exception as __exc:\n"
        "    print(__evalpro_json.dumps({'eval_error': type(__exc).__name__ + ': ' + str(__exc)}))\n"
    )
    result = _run_python(probe, timeout_seconds=timeout_seconds)
    payload = _last_json_line(result.stdout)

    details = {
        "function_call": left_expr,
        "expected": None,
        "actual": None,
    }
    if payload:
        details["expected"] = payload.get("expected")
        details["actual"] = payload.get("actual")
        if payload.get("eval_error"):
            details["eval_error"] = payload["eval_error"]
    return details


def _parse_simple_equality_assert(assertion: str) -> tuple[str, str] | None:
    """Return ``(actual_expr, expected_expr)`` for a diagnosable assert.
    Supports ``assert a == b``, ``assert a is b``, and
    ``assert (math.)isclose(a, b, ...)``. Returns ``None`` otherwise.
    """
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None

    expr = tree.body[0].test
    if (
        isinstance(expr, ast.Compare)
        and len(expr.ops) == 1
        and len(expr.comparators) == 1
        and isinstance(expr.ops[0], (ast.Eq, ast.Is))
    ):
        return ast.unparse(expr.left), ast.unparse(expr.comparators[0])

    # (math.)isclose(actual, expected, ...)
    if isinstance(expr, ast.Call) and len(expr.args) >= 2:
        func = expr.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "isclose":
            return ast.unparse(expr.args[0]), ast.unparse(expr.args[1])

    return None


def _last_json_line(stdout: str) -> dict | None:
    for line in reversed(str(stdout or "").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _public_feedback(result: ScriptResult, failure_cases: list[dict]) -> str:
    if result.status == "passed":
        return "All tests passed."
    if result.error_type == "Timeout":
        return "Your code timed out, possibly due to an infinite loop or slow implementation."
    if failure_cases:
        first = failure_cases[0]
        return (
            "Your code ran but failed a correctness test. "
            f"{first['name']} failed with {first['exception_type']}."
        )
    tail = _tail_line(result.stderr)
    return f"Your code raised {result.error_type} before the tests could pass." + (
        f" ({tail})" if tail else ""
    )


def _case_summary(result: ScriptResult) -> str:
    if result.error_type == "AssertionError":
        return "The function returned the wrong value for one case."
    if result.error_type == "Timeout":
        return "The testcase timed out."
    return "The testcase raised an exception before it could pass."


_FRAME_RE = re.compile(r'^[ \t]*File "')
_EXC_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)[ \t]*(?::|$)")


def _error_type(stderr: str) -> str:
    lines = str(stderr or "").splitlines()
    frames = [i for i, line in enumerate(lines) if _FRAME_RE.match(line)]
    head = ""
    if frames:
        for line in lines[frames[-1] + 1:]:
            if line.strip() and line[:1] not in (" ", "\t"):
                head = line.strip()
                break
    if not head:                       # no traceback at all (killed, or bare stderr)
        nonblank = [line.strip() for line in lines if line.strip()]
        head = nonblank[-1] if nonblank else ""
    match = _EXC_LINE_RE.match(head)
    return match.group(1).rsplit(".", 1)[-1] if match else "Error"


def _tail_line(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _clip(text: str, limit: int) -> str:
    text = "" if text is None else str(text)
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
    return text
