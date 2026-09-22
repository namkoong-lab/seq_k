"""Saved attempt JSON -> database rows.

One implementation for both the live harness and `db_sync --rebuild`, so a
rebuilt database cannot disagree with an incrementally written one. Reads only
what is on disk, so it works the same on a run pulled down from S3.
"""

from __future__ import annotations

from core import db, ids, pricing, results

_TOKENS = ("input_tokens", "cached_tokens", "thinking_tokens", "output_tokens")


def call_rows_for_attempt(attempt):
    """One row per LLM call in one attempt.

    `call_index` numbers the calls within an (attempt, phase) so the natural key
    is stable across re-extraction: the actor is always call_index 0, and
    judge/critic/summarizer calls keep their stored order. (It is NOT called
    `seq` — in this repo `seq` means seq@k.)
    """
    out = []
    actor = attempt.get("actor") or {}
    # A reconstructed attempt has no observed call: no tokens, no cost, no
    # provider response. `llm_calls` is the ledger of calls we saw, and a null
    # row here would be indistinguishable from a real call with unreported
    # usage. The attempt and the run's claim on it are still recorded.
    # A REUSED attempt is the same artifact as its source: this run CLAIMED an
    # existing generation instead of drawing its own, so it made no actor call and
    # must carry no actor cost. Emitting one here would bill every re-judge for
    # generations it never paid for — the exact double-count the attempts /
    # run_attempts split exists to prevent.
    if not attempt.get("reconstructed") and not attempt.get("reused_from"):
        out.append(_call_row("actor", 0, actor))
    for phase in ("judge", "critic", "summarizer"):
        for i, call in enumerate((attempt.get(phase) or {}).get("calls") or []):
            out.append(_call_row(phase, i, call))
    return out


def task_rows(run_path, canonical_index, *, ident, k, seq, task_id=None, prompt=None,
              storage_key=None, run_id=None):
    """Everything one task contributes, in the six-table shape.

    Returns (task, attempts, run_attempts, calls) where `task` is the dimension
    row, `attempts` are ARTIFACTS (actor generations), `run_attempts` are this
    run's claims on them plus its judge's verdicts, and `calls` is the ledger.

    Whether a claimed attempt was newly generated or reused is decided at write
    time by the DB (an artifact with a matching actor_fingerprint already
    exists); here every attempt on disk belongs to this run, because a run only
    writes files for attempts it generated.
    """
    saved = results.load_task_attempts(run_path, canonical_index)
    if not saved:
        return None, [], [], []
    task = {
        "slice_key": ident["slice_key"],
        "task_index": canonical_index,
        "task_id": task_id or saved[0].get("task_id") or f"task-{canonical_index}",
        "prompt": prompt,
        "meta": {},
    }
    attempts, claims, calls = [], [], []
    for a in saved:
        idx = a["attempt_index"]
        judge = a.get("judge") or {}
        actor = a.get("actor") or {}
        # `reused_from` marks an attempt this run CLAIMED rather than generated
        # (core/rejudge.py). The artifact belongs to the source run, so the row
        # below is attributed there and upserts onto the SOURCE's existing
        # attempts row via its (generated_by_run, task_uid, attempt_index) key —
        # one artifact, now with two run_attempts claims on it.
        #
        # This has to live on disk, not just in the writing code path: db_sync
        # --rebuild re-derives every row from these files, so attributing a
        # claimed artifact to the claiming run here would quietly rewrite
        # provenance into "freshly generated" on every rebuild.
        reused = a.get("reused_from") or None
        # The source run can be PURGED after this run claimed its generations.
        # `attempts.generated_by_run` is a real foreign key, so pointing at a
        # deleted run fails the insert and the whole task silently drops out of
        # the mirror — 30 tasks of one run did exactly that. When the claim is
        # marked orphaned, attribute the artifact to the run that still HOLDS the
        # file, which is the only place it survives. `reused_from` itself stays:
        # it is the historical record, and call_rows_for_attempt keys the
        # actor-cost suppression off it, so dropping it would bill this run for a
        # generation it never made.
        orphaned = bool(reused and reused.get("source_purged"))
        attempts.append({
            "actor_fingerprint": ids.actor_fingerprint(ident, idx),
            # The attempt number in the run that GENERATED this artifact. A
            # reusing run does not create artifacts, so it never writes this.
            "attempt_index": idx,
            "generated_by_run": (reused.get("run_id") if reused and not orphaned else run_id),
            "output_key": ((reused.get("output_key") if reused and not orphaned else None)
                           or (f"{storage_key}/task-{canonical_index}/attempt-{idx}.json"
                               if storage_key else None)),
            "finish_reason": actor.get("finish_reason"),
            "created_at": a.get("timestamp") or a.get("created_at"),
        })
        claims.append({
            "attempt_index": idx,
            "solved": bool(judge.get("success")),
            "score": _f(judge.get("score")),
            "extra": {},
        })
        calls.append(call_rows_for_attempt(a))
    return task, attempts, claims, calls


def run_rollup(run_path, *, k, seq, ident=None, early_stop_ok=False, tasks_requested=None):
    """Aggregate counts and cost for a whole run, straight off disk.

    This is the MANIFEST's rollup — the on-disk record, which must stand alone
    because S3 has to be self-describing. Postgres does not store these numbers
    at all; there they are the `run_summary` view over llm_calls. Same
    definition, computed in two places for two different consumers.
    """
    total = done = partial = solved = n_attempts = 0
    cost = 0.0
    for idx in _task_indices(run_path):
        saved = results.load_task_attempts(run_path, idx)
        if not saved:
            continue
        wins = [a["attempt_index"] for a in saved if (a.get("judge") or {}).get("success")]
        is_done = results.is_done(saved, k, seq=seq, early_stop_ok=early_stop_ok)
        total += 1
        done += bool(is_done)
        partial += (not is_done)
        solved += bool(wins)
        n_attempts += len(saved)
        for a in saved:
            cost += sum(c["cost_usd"] or 0.0 for c in call_rows_for_attempt(a))
    # `tasks_total` counts the task directories that EXIST. That is the right
    # number for per-task arithmetic and the wrong one for "is this run
    # finished?": a 30-task run that died after three has three directories, all
    # of them done, and reports 3/3. `tasks_requested` is what the run was asked
    # to cover (manifest `scope`), so the gap between them is the tasks that were
    # never started at all — invisible in every count above.
    out = {"tasks_total": total, "tasks_done": done, "tasks_partial": partial,
           "tasks_success": solved, "attempts_total": n_attempts,
           "cost_usd": round(cost, 6)}
    if tasks_requested is not None:
        out["tasks_requested"] = int(tasks_requested)
        out["tasks_never_started"] = max(0, int(tasks_requested) - total)
    return out


def iter_task_indices(run_path):
    return _task_indices(run_path)


def mirror_task(run_id, run_path, canonical_index, *, ident, k, seq, task_id=None,
                prompt=None, storage_key=None):
    """Extract one task and push it to the DB. Never raises."""
    task, attempts, claims, calls = task_rows(
        run_path, canonical_index, ident=ident, k=k, seq=seq, task_id=task_id,
        prompt=prompt, storage_key=storage_key, run_id=run_id)
    if task is None:
        return False
    return db.record_task(run_id, task, attempts, claims, calls, run_path=run_path)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _call_row(phase, call_index, src):
    cost, source = _cost(src)
    row = {
        "phase": phase,
        "call_index": call_index,
        "model": src.get("model") or "unknown",
        "cost_usd": cost,
        "cost_source": source if source in db.COST_SOURCES else "unknown",
    }
    for t in _TOKENS:
        row[t] = int(src.get(t) or 0)
    return row


def _cost(src):
    """Provider-reported charge when present, rate-table math otherwise.

    Per CALL, unlike results._tokens_across_attempts which decides per model
    bucket — at this granularity there is no mixing to guard against, so a call
    that reported its own cost always uses it.
    """
    reported = results._reported_cost(src)
    return pricing.cost_for(src.get("model"), int(src.get("input_tokens") or 0),
                            int(src.get("cached_tokens") or 0),
                            int(src.get("output_tokens") or 0),
                            reported_cost=reported)


def _task_indices(run_path):
    import os
    import re
    pat = re.compile(r"^task-(\d+)$")
    if not os.path.isdir(run_path):
        return []
    out = []
    for entry in os.listdir(run_path):
        m = pat.match(entry)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


__all__ = ["call_rows_for_attempt", "task_rows", "run_rollup", "mirror_task",
           "iter_task_indices"]
