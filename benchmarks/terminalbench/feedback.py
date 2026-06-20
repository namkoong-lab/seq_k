"""TerminalBench feedback modes (all built from the parsed Harbor trial, no extra
LLM call):
    binary            — pass/fail bit only
    raw               — the FULL pytest stdout verbatim (no abridgement)
    retry_diagnostics — verifier status + last command + FULL last output +
                        EVERY public failure signal + a retry focus checklist

Nothing in this module is summarized or truncated — the next attempt's agent
sees the verifier output exactly as it was emitted.

Secret-looking values were already redacted during artifact parsing. pass@k never
calls this.
"""

from __future__ import annotations


def feedback(task, attempt, result, mode, *, critic_model=None):
    if mode == "binary":
        return ("Your previous Harbor attempt did not pass the verifier. Use the terminal "
                "feedback to make a stronger next attempt.")
    if mode == "raw":
        return result.raw_eval_output
    if mode == "retry_diagnostics":
        return _retry_diagnostics(result.details, result.raw_eval_output)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _retry_diagnostics(details, raw_eval_output):
    """Structured retry diagnostics; full content, no abridgement."""
    verifier_status = raw_eval_output or "reward=0.0"
    lines = [f"task={details.get('task_id', '')}", "",
             "verifier_status:", verifier_status, "",
             "observed_terminal_state:"]
    last_command = details.get("last_command")
    lines.append(f"last_command: {last_command}" if last_command else "last_command: (none captured)")
    last_output = details.get("last_output") or ""
    lines.append(f"last_output:\n{last_output}" if last_output
                 else "last_output: (none captured)")

    lines += ["", "public_failure_signals:"]
    signals = [str(s) for s in (details.get("error_signals") or []) if str(s).strip()]
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
