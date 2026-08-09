"""Deterministic feedback modes for MBPP Pro seq@k runs."""

from __future__ import annotations


def feedback(task, attempt, result, mode, *, critic_model):
    """Return feedback for the next seq@k attempt after a failed attempt."""

    if mode == "binary":
        return "Your previous solution failed the hidden tests. Please revise the code."

    if mode == "error_summary":
        return _error_summary(result)

    if mode == "test_details":
        return _test_details(result)

    raise ValueError(
        f"unknown feedback mode for mbpppro: {mode!r}; use 'binary', "
        "'error_summary', or 'test_details'"
    )


def _failure_cases(result):
    return result.details.get("failure_cases") or []


def _error_summary(result):
    cases = _failure_cases(result)
    lines = [
        "Your previous solution failed.",
        "",
        "Failure summary:",
    ]
    if cases:
        for case in cases[:3]:
            name = case.get("name") or f"test_{case.get('index', '?')}"
            error = case.get("exception_type") or result.details.get("error_type") or "Error"
            summary = case.get("summary") or "The function returned the wrong value for one case."
            lines.append(f"- {name} failed with {error}. {summary}")
    else:
        error = result.details.get("error_type") or "Error"
        lines.append(f"- The verifier failed with {error}.")

    lines += [
        "- No full expected output is shown.",
        "",
        "Use this information to fix the general logic. Return only the revised Python code.",
    ]
    return "\n".join(lines)


def _test_details(result):
    cases = _failure_cases(result)
    if not cases:
        return result.raw_eval_output

    lines = [
        "Your previous solution failed.",
        "",
        "Failed test details:",
    ]
    for case in cases[:2]:
        name = case.get("name") or f"test_{case.get('index', '?')}"
        lines.append(f"- {name}")
        call = case.get("function_call")
        expected = case.get("expected")
        actual = case.get("actual")
        if call:
            lines.append(f"  function call: {_clip(call)}")
        if expected is not None:
            lines.append(f"  expected: {_clip(expected)}")
        if actual is not None:
            lines.append(f"  got: {_clip(actual)}")
        if not any((call, expected is not None, actual is not None)):
            lines.append(f"  error: {_clip(case.get('message') or case.get('exception_type') or 'failed')}")

    lines += [
        "",
        "Use this feedback to fix the general logic. Do not hard-code these cases.",
        "Return only the revised Python code.",
    ]
    return "\n".join(lines)


def _clip(value, limit=300):
    text = str(value)
    if len(text) > limit:
        return text[:limit] + f"...[truncated {len(text) - limit} chars]"
    return text
