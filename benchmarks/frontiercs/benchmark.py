"""FrontierCS (algorithmic track) task loading + official Docker-judge verifier.

Grading is fully execution-based: the extracted C++ program is submitted over
HTTP to the official Frontier-CS judge server (go-judge + per-problem testlib
checkers, running in Docker) and the per-case results come back as JSON. The
submit/poll protocol is the upstream AlgorithmicLocalRunner's two endpoints,
reimplemented here directly with requests — importing the upstream package
buys nothing over two HTTP calls. No LLM judge anywhere, hence
VERIFIER = "deterministic" — judge.model is null in saved attempts.

Score is CONTINUOUS, healthbench-style: each case's checker ratio is clamped
to [0, 1] (the judge itself never clamps, and a few upstream checkers — the
beat-the-jury heuristic problems 182-189 — can emit ratios > 1 when a solution
beats the jury reference), then averaged → score ∈ [0, 1]. success =
score >= success_threshold (default 1.0 = full marks on every case). This
deliberately breaks with seq_k_eval's `score > 0` pass rule, which ended
trajectories on any partial credit (32/100 counted as solved) and left the
heuristic half of the benchmark with no improvement pressure. The migrated
runs under run_old/frontiercs keep the old binarized scores — the two curve
families are different metrics and must not share a plot axis.

Tasks, judge code and grading data all come from ONE source: a clone of the
official Frontier-CS repo at the pinned commit 55cde54 (statements, per-problem
testlib checkers, testdata and the Docker judge server live side by side in its
algorithmic/ tree). The pin matters: upstream keeps drifting, and grading
depends on the exact checkers/testdata. The official HF dataset mirror is NOT
a substitute — relative to this commit it is missing files for 61 problems and
has content drift on 9 more. Problem ids are non-contiguous upstream (12,
18-21, ... never existed); everything is keyed by explicit id, never position.

PREREQS (external, fail-loud if missing): the pinned checkout
  git clone https://github.com/FrontierCS/Frontier-CS.git experiments/third_party/Frontier-CS
  git -C experiments/third_party/Frontier-CS checkout 55cde54
and a running judge server: `docker compose up -d` in
<frontiercs_root>/algorithmic/. Configure via a variant's `options:`.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
import yaml

from core.types import Task, VerifierResult

# --------------------------------------------------------------------------- #
# Path-layout declarations (consumed by core/results.py)
# --------------------------------------------------------------------------- #
VERIFIER = "deterministic"     # official Docker test-case judge — no LLM
LLM_CRITIC_MODES = set()       # binary / case_summary / raw are template-only

DEFAULT_JUDGE_URL = "http://localhost:8081"
DEFAULT_SUCCESS_THRESHOLD = 1.0   # full marks; seq_k_eval used > 0.0 (any partial credit)
DEFAULT_JUDGE_TIMEOUT = 1000      # seconds; matches the upstream runner's DEFAULT_TIMEOUT
_POLL_INTERVAL = 2.0
_SUBMIT_TIMEOUT = 30
_DEFAULT_TIME_LIMIT_S = 2.0

# Same extraction the seq_k_eval adapter used: fenced code block, else full text.
_CODE_BLOCK = re.compile(r"```(?:cpp|c\+\+|cc|python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_ERROR_TEXT_LIMIT = 4000       # compile errors keep their g++ formatting up to this
_CASE_MSG_LIMIT = 500          # per-case checker message stored in details
_CASE_OUTPUT_EXCERPT = 400     # per-case program-stdout excerpt stored in details
# The judge holds a case's full stdout (up to 128 MB); details keeps only the
# excerpt above — anything more would bloat attempt files for no analysis value.


def slice_name(_options):
    return "frontiercs"


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def load_tasks(frontiercs_root, judge_url=DEFAULT_JUDGE_URL,
               success_threshold=DEFAULT_SUCCESS_THRESHOLD,
               judge_timeout=DEFAULT_JUDGE_TIMEOUT):
    """Load FrontierCS algorithmic tasks straight from the pinned checkout:
    one task per <root>/algorithmic/problems/<id>/ directory, sorted by
    int(id). canonical_index = 1-based position in that order — the same
    order seq_k_eval selected its first-50 subset from, so migrated runs and
    future runs share task numbering.

    Judge configuration rides in task.grading: verify()/feedback() receive no
    options, so grading (verify-only by contract) is the channel. The actor
    prompt is statement + the C++17 base prompt (same wording seq_k_eval used).
    """
    from . import prompts

    root = Path(frontiercs_root).expanduser()
    problems_dir = root / "algorithmic" / "problems"
    if not problems_dir.is_dir():
        raise FileNotFoundError(
            f"frontiercs_root has no algorithmic/problems/: {root}. "
            "Clone the official Frontier-CS repository at the pinned commit:\n"
            "  git clone https://github.com/FrontierCS/Frontier-CS.git "
            "experiments/third_party/Frontier-CS\n"
            "  git -C experiments/third_party/Frontier-CS checkout 55cde54"
        )
    judge_cfg = {
        "url": str(judge_url).rstrip("/"),
        "success_threshold": float(success_threshold),
        "timeout": float(judge_timeout),
        "frontiercs_root": str(root),
    }
    tasks = []
    problem_dirs = sorted(
        (d for d in problems_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    for pdir in problem_dirs:
        statement_path = pdir / "statement.txt"
        if not statement_path.is_file():
            raise FileNotFoundError(
                f"problem {pdir.name} has no statement.txt: {statement_path}. "
                "The checkout is incomplete or not at the pinned commit 55cde54."
            )
        tag_path = pdir / "tag.txt"   # gitignored upstream; usually absent
        tag = tag_path.read_text(encoding="utf-8").strip() if tag_path.is_file() else ""
        tasks.append(Task(
            id=pdir.name,
            canonical_index=len(tasks) + 1,
            prompt=f"{statement_path.read_text(encoding='utf-8')}\n\n{prompts.BASE_PROMPT}",
            grading={"tag": tag, "language": "cpp", "judge_cfg": judge_cfg},
        ))
    return tasks


def extract_code(text):
    """The program to submit, pulled out of the actor's reply.

    Taking the FIRST fenced block (what seq_k_eval did) breaks on reasoning-style
    actors: they emit thought fragments and one-line placeholders as blocks along
    the way and put the real program last, so the judge compiles a stub and
    reports a bogus "undefined reference to `main`". Prefer the last block that
    defines main, then the last block, and only then the whole reply — that final
    fallback is for an actor that emitted no fence at all, which the prompt asks
    it not to do.
    """
    blocks = [b.strip() for b in _CODE_BLOCK.findall(str(text or ""))]
    blocks = [b for b in blocks if b]
    for block in reversed(blocks):
        if "int main" in block:
            return block
    if blocks:
        return blocks[-1]
    return str(text or "").strip()


# --------------------------------------------------------------------------- #
# Verifier (official judge server over HTTP)
# --------------------------------------------------------------------------- #
def verify(task, attempt, *, judge_model=None):   # judge_model unused (deterministic)
    cfg = task.grading["judge_cfg"]
    code = extract_code(attempt.output)
    # A response with no code is a real task failure (same policy as arcagi2's
    # malformed answers), not an infrastructure error.
    if not code:
        return VerifierResult(
            success=False, score=0.0,
            raw_eval_output=("Your response contained no code. Return a single-file "
                            "self-contained C++17 program (optionally in a ``` code block)."),
            details={"status": "invalid_output", "problem_id": task.id},
        )

    _check_judge_or_die(cfg)
    result = _submit_and_poll(cfg, task.id, code, lang=task.grading.get("language", "cpp"))

    # status=="error": the judge ran but the submission never produced case
    # results — dominated by compile failures, which are the model's fault and
    # score 0 with the compiler's own diagnostics as feedback.
    if result.get("status") == "error":
        error_text = str(result.get("error") or "Unknown judge error").strip()
        return VerifierResult(
            success=False, score=0.0,
            raw_eval_output=error_text[:_ERROR_TEXT_LIMIT],
            details={"status": "error", "problem_id": task.id, "error": error_text[:_ERROR_TEXT_LIMIT]},
        )

    cases = result.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError(
            f"frontiercs judge returned status={result.get('status')!r} with no cases "
            f"for problem {task.id} — unexpected response shape: {list(result)}"
        )

    # Per-case clamp is the repo-side guarantee that score stays in [0, 1]:
    # upstream checkers only bound their printed ratio by convention, and the
    # 182-189 beat-the-jury batch can exceed 1.
    ratios = [min(max(float(c.get("scoreRatio") or 0.0), 0.0), 1.0) for c in cases]
    score = sum(ratios) / len(ratios)
    threshold = cfg["success_threshold"]
    success = score >= threshold

    time_limit_s = _time_limit_s(cfg, task.id)
    summary = _case_summary(cases, time_limit_s=time_limit_s)
    raw_eval = "" if success else _headline(score, threshold, summary)

    return VerifierResult(
        success=success,
        score=score,
        raw_eval_output=raw_eval,
        details={
            "status": "done",
            "problem_id": task.id,
            "passed": bool(result.get("passed")),          # judge's own all-cases-perfect flag
            "judge_score": float(result.get("score") or 0.0),               # 0-100, unclamped
            "judge_score_unbounded": float(result.get("scoreUnbounded") or 0.0),  # 0-100+, uncapped
            "score_unbounded": float(result.get("scoreUnbounded") or 0.0) / 100.0,
            "threshold": threshold,
            "time_limit_s": time_limit_s,
            "cases": [_slim_case(c, r) for c, r in zip(cases, ratios)],
        },
    )


_JUDGE_CHECKED = set()   # judge URLs already probed this process


def _check_judge_or_die(cfg):
    url = cfg["url"]
    if url in _JUDGE_CHECKED:
        return
    try:
        resp = requests.get(f"{url}/problems", timeout=5)
        ok = resp.status_code == 200
    except requests.RequestException:
        ok = False
    if not ok:
        root = cfg.get("frontiercs_root") or "<frontiercs_root>"
        raise RuntimeError(
            f"FrontierCS judge server not reachable at {url}. "
            f"Start it with: cd {root}/algorithmic && docker compose up -d"
        )
    _JUDGE_CHECKED.add(url)


def _submit_and_poll(cfg, problem_id, code, *, lang):
    url = cfg["url"]
    resp = requests.post(
        f"{url}/submit",
        files={"code": (f"solution.{ 'cpp' if lang == 'cpp' else lang }", code)},
        data={"pid": str(problem_id), "lang": lang},
        timeout=_SUBMIT_TIMEOUT,
    )
    resp.raise_for_status()
    sid = resp.json().get("sid")
    if sid is None:
        raise RuntimeError(f"frontiercs judge /submit returned no sid for problem {problem_id}: {resp.text[:500]}")

    deadline = time.time() + cfg["timeout"]
    while time.time() < deadline:
        try:
            poll = requests.get(f"{url}/result/{sid}", timeout=10)
            if poll.status_code == 404:      # not judged yet
                time.sleep(_POLL_INTERVAL)
                continue
            poll.raise_for_status()
            result = poll.json()
            if result.get("status") in ("done", "error"):
                return result
        except requests.RequestException:
            pass                             # transient — keep polling until the deadline
        time.sleep(_POLL_INTERVAL)
    raise RuntimeError(
        f"frontiercs judge timed out after {cfg['timeout']:.0f}s "
        f"(problem {problem_id}, sid {sid}) — server hung or the submission is stuck"
    )


def _slim_case(case, clamped_ratio):
    """Per-case record for details: everything the raw feedback mode renders,
    with the two unbounded-size fields (checker msg, program stdout) excerpted."""
    output = case.get("output")
    return {
        "ok": bool(case.get("ok")),
        "status": str(case.get("status") or "Unknown"),
        "time_s": round((case.get("time") or 0) / 1e9, 4),
        "memory_bytes": case.get("memory"),
        "score_ratio": clamped_ratio,
        "score_ratio_unbounded": float(case.get("scoreRatioUnbounded") or 0.0),
        "msg": str(case.get("msg") or "")[:_CASE_MSG_LIMIT],
        "output_excerpt": (str(output)[:_CASE_OUTPUT_EXCERPT] if output else ""),
    }


# --------------------------------------------------------------------------- #
# Case summary (the case_summary feedback mode's text; ported from seq_k_eval)
# --------------------------------------------------------------------------- #
def _full_credit(case):
    """Whether a case earned full marks. Deliberately NOT case["ok"]: that is
    testlib's strict accept flag, and partial-credit problems award marks via
    quitp(), whose non-zero exit leaves ok false even at Ratio 1.0000. The judge
    itself ignores ok the same way — judge_engine derives both the per-case
    status and the overall passed flag from scoreRatio == 1.0 — and so does our
    score. Counting ok would contradict the verdict tally in the same report
    ("0/3 cases passed" next to "{'Correct': 2, 'Wrong Answer': 1}").
    """
    return float(case.get("scoreRatio") or 0.0) >= 1.0


def _headline(score, threshold, summary):
    head = f"score={score:.3f}; threshold={threshold:.3f} (score must be >= threshold)"
    return f"{head}\n\n{summary}" if summary else head


def _case_summary(cases, *, time_limit_s, max_sample_msgs=3, max_msg_chars=200):
    """Aggregate per-case results into a short diagnostic answering four
    questions and nothing more: how many tests ran, how many the submission
    passed, how many it failed, and which distinct reasons account for those
    failures. Everything else here supports those four — the TLE heuristic
    (>=95% of the limit with no captured stdout; the sandbox kills the program
    before the checker sees anything, so a timeout would otherwise be
    indistinguishable from a wrong answer), the time and score-ratio stats, and
    a closing diagnosis.

    Deliberately withholds WHICH tests failed and what their data was — that is
    the raw mode's job, and the gap between the two modes is the experiment.
    """
    total = len(cases)
    ok_count = sum(1 for c in cases if _full_credit(c))

    tle_threshold_s = 0.95 * time_limit_s
    likely_tle = sum(1 for c in cases
                     if (c.get("time") or 0) / 1e9 >= tle_threshold_s and not c.get("output"))

    times_s = [(c.get("time") or 0) / 1e9 for c in cases]
    ratios = [float(c.get("scoreRatio") or 0.0) for c in cases]

    lines = [
        f"{total} test case(s) run: {ok_count} passed, {total - ok_count} failed "
        f"(a case counts as passed only at ratio 1.000).",
    ]
    if likely_tle:
        lines.append(f"Likely TLE: {likely_tle}/{total} cases hit ≥{tle_threshold_s:.2f}s "
                     f"with no captured output (time limit {time_limit_s:.1f}s).")
    lines.append(f"Time per case (s): min={min(times_s):.3f}, max={max(times_s):.3f}, "
                 f"mean={sum(times_s) / total:.3f} (limit={time_limit_s:.1f}s).")
    if max(ratios) > 0:
        lines.append(f"Score ratio per case: min={min(ratios):.3f}, max={max(ratios):.3f}, "
                     f"mean={sum(ratios) / total:.3f}.")

    # Group the failures by (verdict, numeric-normalised checker message) so the
    # actor learns which distinct failure MODES it hit and how many tests each
    # accounts for — "Overlap at cell (7,0)" and "(5,3)" are one mode, not two,
    # and "6 tests timed out, 1 wrong answer" is a different fix from the
    # reverse. Passing cases' messages ("ok") carry no signal.
    reasons, order = {}, []
    for c in cases:
        if _full_credit(c):
            continue
        msg = (c.get("msg") or "").strip().split("\n")[0]
        status = str(c.get("status") or "Unknown")
        key = (status, re.sub(r"\d+(?:\.\d+)?", "<num>", msg)[:120])
        if key not in reasons:
            reasons[key] = {"n": 0, "status": status, "msg": msg[:max_msg_chars]}
            order.append(key)
        reasons[key]["n"] += 1
    if order:
        lines += ["", f"Why the {total - ok_count} failing case(s) failed:"]
        for key in order[:max_sample_msgs]:
            r = reasons[key]
            lines.append(f"  - {r['n']} case(s): {r['status']}"
                         + (f" — {r['msg']}" if r["msg"] else ""))
        # Say what was dropped rather than silently truncating the list.
        if len(order) > max_sample_msgs:
            lines.append(f"  - ... and {len(order) - max_sample_msgs} further distinct "
                         f"failure mode(s), not shown")

    hints = []
    if likely_tle and likely_tle >= 0.5 * total:
        hints.append("most cases hit the time limit → reduce algorithmic complexity")
    elif likely_tle:
        hints.append("some cases hit the time limit → check worst-case complexity")
    # Correctness vs strategy are opposite fixes, so they must not both fire: a
    # solution earning partial credit is valid-but-suboptimal, and telling it to
    # "review correctness" points it the wrong way. Only a shutout (no test
    # earned any credit) is a correctness verdict.
    if ok_count == 0 and not likely_tle and order and max(ratios) == 0:
        hints.append("no case earned any credit → review correctness before tuning")
    if max(ratios) > 0 and ok_count < total:
        hints.append("this problem gives partial credit; a non-zero score ratio means the "
                     "solution is valid but suboptimal — improve the strategy")
    if hints:
        lines += ["", "Diagnosis: " + "; ".join(hints) + "."]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Problem metadata from the Frontier-CS checkout
# --------------------------------------------------------------------------- #
_TIME_LIMIT_CACHE = {}


def _time_limit_s(cfg, problem_id):
    """Global time limit from the problem's config.yaml (per-subtask overrides
    exist upstream but the global value is what the summary's TLE heuristic
    needs). Falls back to 2.0s when frontiercs_root isn't configured."""
    root = cfg.get("frontiercs_root")
    if not root:
        return _DEFAULT_TIME_LIMIT_S
    key = (root, str(problem_id))
    if key in _TIME_LIMIT_CACHE:
        return _TIME_LIMIT_CACHE[key]
    path = Path(root) / "algorithmic" / "problems" / str(problem_id) / "config.yaml"
    limit = _DEFAULT_TIME_LIMIT_S
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            limit = _parse_time_limit(loaded.get("time_limit") or loaded.get("time"))
        except yaml.YAMLError:
            pass                             # malformed config.yaml → default limit
    _TIME_LIMIT_CACHE[key] = limit
    return limit


def _parse_time_limit(value, default=_DEFAULT_TIME_LIMIT_S):
    """'2s' / '500ms' / bare number → seconds."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    try:
        if text.endswith("ms"):
            return float(text[:-2]) / 1000.0
        if text.endswith("s"):
            return float(text[:-1])
        return float(text)
    except ValueError:
        return default


def testdata_paths(cfg, problem_id, case_number):
    """(input, answer) paths for 1-based case N. The judge builds its case list
    from config.yaml subtask n_cases counts as sequentially numbered files, so
    cases[i] in the result maps to testdata/(i+1).in / .ans."""
    root = cfg.get("frontiercs_root")
    if not root:
        return None, None
    base = Path(root) / "algorithmic" / "problems" / str(problem_id) / "testdata"
    return base / f"{case_number}.in", base / f"{case_number}.ans"
