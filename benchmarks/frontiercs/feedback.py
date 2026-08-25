"""FrontierCS feedback modes:
    binary       — pass/fail sentence only
    case_summary — the verifier's aggregate diagnostic (verdict distribution, TLE
                   heuristic, time/ratio stats, deduplicated checker messages);
                   the mode seq_k_eval ran
    raw          — per-case verdicts verbatim (one line per case: verdict, time,
                   ratio, checker message), plus input/expected/got excerpts for
                   the first failing cases, read from the Frontier-CS checkout's
                   testdata (requires options.frontiercs_root)

No LLM call — all modes derive from the deterministic verifier output.
pass@k never calls this.
"""

from __future__ import annotations

_MAX_CASE_LINES = 100        # per-case listing cap (largest observed problem has 70 cases)
_MAX_DIFF_CASES = 2          # failing cases that get input/expected/got excerpts
_EXCERPT_CHARS = 400


def feedback(task, attempt, result, mode, *, critic_model=None):
    if mode == "binary":
        return ("Your previous solution did not pass the test cases. "
                "Revise it and provide a corrected or stronger solution.")
    if mode == "case_summary":
        return result.raw_eval_output
    if mode == "raw":
        return _raw(task, result)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _raw(task, result):
    details = result.details or {}
    cases = details.get("cases")
    # No per-case data (compile error, empty submission) — the verifier's
    # diagnostic (full compiler output / format hint) is already the best signal.
    if not cases:
        return result.raw_eval_output

    lines = [f"score={result.score:.3f}; "
             f"threshold={details.get('threshold', 1.0):.3f} (score must be >= threshold)", ""]
    # case_summary states conclusions; raw dumps the judge's fields verbatim, and
    # several of them read the wrong way round without a key (a ratio is partial
    # credit, not a percentage of cases; a case at the limit with no output was
    # killed rather than judged wrong). Without this the two modes differ in how
    # hard they are to READ, not just in how much they say.
    limit = details.get("time_limit_s")
    lines += [
        "How to read this: your score is the mean of the per-case `ratio` values, so "
        "you pass only when every case reaches ratio 1.000. `ratio` is the credit this "
        "problem's checker gave that case; `unbounded` above 1.000 means you beat the "
        "reference solution there. "
        + (f"The time limit is {limit:.1f}s — a case at or near it with no output was "
           "killed for exceeding it, not judged wrong. " if limit else "")
        + "Text after the dash is the checker's own message.",
        "",
    ]
    # Full credit, not case["ok"] — see benchmark._full_credit for why they differ.
    lines.append(f"Per-case results ({sum(1 for c in cases if c['score_ratio'] >= 1.0)}"
                 f"/{len(cases)} passed):")
    for i, c in enumerate(cases[:_MAX_CASE_LINES], start=1):
        entry = f"  case {i}: {c['status']}, time={c['time_s']:.3f}s, ratio={c['score_ratio']:.3f}"
        if c.get("score_ratio_unbounded", 0.0) > 1.0:
            entry += f" (unbounded {c['score_ratio_unbounded']:.3f})"
        if c.get("msg"):
            entry += f" — {c['msg'].splitlines()[0]}"
        lines.append(entry)
    if len(cases) > _MAX_CASE_LINES:
        lines.append(f"  ... and {len(cases) - _MAX_CASE_LINES} more cases")

    diffs = _failing_case_diffs(task, details, cases)
    if diffs:
        lines.append("")
        lines.extend(diffs)
    return "\n".join(lines)


def _failing_case_diffs(task, details, cases):
    """input / expected / got excerpts for the first failing cases. Skipped
    silently when frontiercs_root isn't configured or a testdata file is
    missing — the per-case listing above still stands on its own."""
    from . import benchmark as bm

    cfg = task.grading.get("judge_cfg") or {}
    out, shown = [], 0
    for i, c in enumerate(cases, start=1):
        if c["score_ratio"] >= 1.0 or shown >= _MAX_DIFF_CASES:
            continue
        in_path, ans_path = bm.testdata_paths(cfg, details.get("problem_id", task.id), i)
        if in_path is None or not in_path.exists():
            continue
        out.append(f"Failing case {i} detail:")
        out.append(f"  input (excerpt): {_excerpt(in_path.read_text(encoding='utf-8', errors='replace'))}")
        if ans_path.exists():
            # Checker-validated problems often ship a placeholder .ans (any valid
            # answer is accepted) — label accordingly so the actor doesn't chase it.
            out.append("  reference answer (excerpt; may be a placeholder — the checker "
                       f"accepts any valid answer): {_excerpt(ans_path.read_text(encoding='utf-8', errors='replace'))}")
        got = c.get("output_excerpt") or ""
        out.append(f"  your output (excerpt): {_excerpt(got) if got else '<no output captured>'}")
        shown += 1
    return out


def _excerpt(text, limit=_EXCERPT_CHARS):
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} more chars]"
