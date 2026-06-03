"""TerminalBench feedback modes (all built from the parsed Harbor trial, no extra
LLM call):
    binary           — pass/fail bit only
    raw              — the verifier summary verbatim (reward + remaining issues)
    retry_diagnostics — verifier status + last command + last output excerpt +
                        public failure signals + a retry focus checklist

Secret-looking values were already redacted during artifact parsing. pass@k never
calls this.
"""

from __future__ import annotations


def feedback(task, attempt, result, mode, *, judge_model=None):
    if mode == "binary":
        return ("Your previous Harbor attempt did not pass the verifier. Use the terminal "
                "feedback to make a stronger next attempt.")
    if mode == "raw":
        return result.raw_eval_output
    if mode == "retry_diagnostics":
        return _retry_diagnostics(result.private)
    raise ValueError(f"unknown feedback mode: {mode!r}")


def _retry_diagnostics(p):
    lines = [f"task={p.get('task_id', '')}", "",
             "verifier_status:", p.get("verifier_summary") or "reward=0.0", "",
             "observed_terminal_state:"]
    last_command = p.get("last_command")
    lines.append(f"last_command: {last_command}" if last_command else "last_command: (none captured)")
    last_output = p.get("last_output_excerpt")
    lines.append(f"last_output_excerpt:\n{last_output}" if last_output
                 else "last_output_excerpt: (none captured)")

    lines += ["", "public_failure_signals:"]
    signals = [s for s in (p.get("error_signals") or []) if str(s).strip()]
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
