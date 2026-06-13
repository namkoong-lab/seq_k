"""TerminalBench feedback modes (all built from the parsed Harbor trial, no extra
LLM call):
    binary           — pass/fail bit only
    raw              — the verifier summary verbatim (reward + remaining issues)
    retry_diagnostics — verifier status + last command + last output excerpt +
                        public failure signals + a retry focus checklist

Secret-looking values were already redacted during artifact parsing. pass@k never
calls this.

Truncation here is intentional: the stored attempt JSON keeps full trajectory /
verifier / terminal output, but the retry-feedback string the next actor sees is
trimmed to keep the prompt manageable.
"""

from __future__ import annotations

# Caps applied only when building the next-attempt prompt; they do NOT affect
# what's stored in the result JSON.
RETRY_LAST_OUTPUT_CHARS = 2000
RETRY_ERROR_SIGNAL_COUNT = 4
RETRY_ERROR_SIGNAL_CHARS = 240


def feedback(task, attempt, result, mode, *, judge_model=None):
    if mode == "binary":
        return ("Your previous Harbor attempt did not pass the verifier. Use the terminal "
                "feedback to make a stronger next attempt.")
    if mode == "raw":
        return result.raw_eval_output
    if mode == "retry_diagnostics":
        return _retry_diagnostics(result.judge_details, result.raw_eval_output)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _retry_diagnostics(p, raw_eval_output):
    # verifier_summary is the same string as raw_eval_output; we accept either source
    # so this also works for older runs where it lived inside judge_details.
    verifier_status = p.get("verifier_summary") or raw_eval_output or "reward=0.0"
    lines = [f"task={p.get('task_id', '')}", "",
             "verifier_status:", verifier_status, "",
             "observed_terminal_state:"]
    last_command = p.get("last_command")
    lines.append(f"last_command: {last_command}" if last_command else "last_command: (none captured)")
    # Back-compat: pre-rename files stored this as "last_output_excerpt".
    last_output = _trim(p.get("last_output") or p.get("last_output_excerpt"), RETRY_LAST_OUTPUT_CHARS)
    lines.append(f"last_output:\n{last_output}" if last_output
                 else "last_output: (none captured)")

    lines += ["", "public_failure_signals:"]
    signals = [_trim(s, RETRY_ERROR_SIGNAL_CHARS)
               for s in (p.get("error_signals") or []) if str(s).strip()][:RETRY_ERROR_SIGNAL_COUNT]
    if signals:
        lines += [f"- {s}" for s in signals]
    else:
        lines.append("- No explicit runtime error captured; the remaining issue is likely an "
                     "end-state mismatch rather than a crash.")

    lines += ["", "retry_focus:",
              "- Re-check task-specific paths, filenames, service state, and output format.",
              "- Run a concrete terminal verification command before stopping.",
              "- Trust observed terminal output over self-reported success."]
    return "\n".join(lines).strip()


def _trim(text, limit):
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "\n…[truncated; full text in result file]"
