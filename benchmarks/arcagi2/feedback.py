"""ARC-AGI-2 feedback modes:
    binary     — pass/fail bit only
    cell_match — the verifier's bounded per-grid cell-match summary (size + accuracy,
                 never the target grids); falls back to format hints on a parse error

No LLM call — both modes are derived from the deterministic verifier output.
pass@k never calls this.
"""

from __future__ import annotations


def feedback(task, attempt, result, mode, *, judge_model=None):
    if mode == "binary":
        return ("Your previous answer did not exactly match the expected output grid(s). "
                "Re-examine the transformation and try again.")
    if mode == "cell_match":
        return result.raw_eval_output
    raise ValueError(f"unknown feedback mode: {mode!r}")
