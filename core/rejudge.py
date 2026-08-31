"""Re-grade an existing run's generations with a different judge.

    python -m core rejudge <run> --judge openrouter/openai/gpt-5.4
    python -m core rejudge <run> --judge ... --apply

Self-contained: nothing else in `core` imports this module. It reaches the rest
of the system only through the same public helpers the harness uses, plus ONE
piece of shared vocabulary — the `reused_from` block it writes into an attempt
file, which core/rows.py reads. That block has to be on disk rather than only in
this code path because `db_sync --rebuild` re-derives every database row from
the files; a claimed artifact recorded anywhere else would be relabelled as
freshly generated on the next rebuild.

WHY. Re-judging does not change what the actor wrote, so those generations
should be CLAIMED again rather than paid for again. That is what `attempts`
being unowned by a run buys you (db/schema.sql) — this is the caller that was
missing.

WHAT IT REFUSES. `ids.actor_fingerprint` already says when two generations are
interchangeable:

    pass@k, any attempt   -> judge excluded (no feedback exists)   reusable
    seq@k, attempt 1      -> judge excluded (nothing graded yet)   reusable
    seq@k, attempt 2+     -> judge/critic/feedback_mode INCLUDED   NOT reusable

A seq@k attempt after the first was prompted with feedback derived from the OLD
judge's verdict; reusing it would measure that judge's influence while claiming
to measure the new one's. So there is no metric check here — it compares the two
fingerprints and refuses when they differ. The hash enforces the rule.

WHAT IT PRODUCES. A new identity (the judge changed), so `registry.resolve`
decides the destination exactly as it does for a normal run — which means
re-judging one half of a judge-split run lands its attempts in the OTHER half,
merging them. Each attempt file is copied with its actor section verbatim and a
fresh judge section, so the run stays self-describing on disk and in S3.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass

from core import db, ids, llm, registry, results, rows, s3sync
from core.types import Attempt


@dataclass
class Plan:
    """What a re-judge would do. Built without side effects, so `--apply` and a
    dry run take exactly the same path up to this point."""
    src_path: str
    src: dict                  # source manifest
    benchmark: object
    k: int
    dst_ident: dict
    dst_cands: list
    claimable: list            # [(task_index, attempt_index)] safe to claim
    blocked: list              # [(task_index, attempt_index)] must be regenerated


def _find(runs_root, selector):
    """The one run matching `selector` — a run_id prefix, a storage key, or a
    by-label path, the same forms `runs.py show` accepts."""
    hits = [(p, m) for p, m in registry.iter_manifests(runs_root)
            if m["run_id"].startswith(selector) or selector in m["storage_key"]]
    if len(hits) != 1:
        raise SystemExit(f"{selector!r} matched {len(hits)} runs; be more specific")
    return hits[0]


def _identity(m, benchmark, *, judge_model, k):
    """The manifest's identity with only the judge swapped. Goes through
    ids.candidates so the result is a real current-version fingerprint and the
    re-judge can itself be resumed."""
    c = m["config"]
    return ids.candidates(
        benchmark_module=benchmark, options=m.get("options") or {},
        metric=c["metric"], k=k, model=c["model"], judge_model=judge_model,
        critic_model=c.get("critic_model"), feedback_mode=c["feedback_mode"],
        context=c["context"], prompt_variant=c["prompt_variant"],
        temperature=c["temperature"], seed=c.get("seed"),
        reasoning_effort=c.get("reasoning_effort"),
        output_budget=c.get("output_budget"),
        summarizer_model=c.get("summarizer_model"))


def plan(runs_root, selector, *, judge_model):
    src_path, m = _find(runs_root, selector)
    benchmark = importlib.import_module(m["config"]["benchmark"])
    k = m.get("k_target") or m["config"]["k"]

    src = _identity(m, benchmark, judge_model=m["config"]["judge_model"], k=k)[0][2]
    dst_cands = _identity(m, benchmark, judge_model=judge_model, k=k)
    dst = dst_cands[0][2]
    if src["judge_model"] == dst["judge_model"]:
        raise SystemExit(f"already judged by {dst['judge_model']!r} — nothing to do")

    claimable, blocked = [], []
    for t in rows.iter_task_indices(src_path):
        for a in results.load_task_attempts(src_path, t):
            i = int(a["attempt_index"])
            ok = ids.actor_fingerprint(src, i) == ids.actor_fingerprint(dst, i)
            (claimable if ok else blocked).append((t, i))
    return Plan(src_path, m, benchmark, k, dst, dst_cands, claimable, blocked)


def _claim(p, out, task, ai, *, judge_model):
    """Judge one existing generation and write it into `out` as a claimed attempt."""
    src_a = json.loads(open(results.attempt_file(p.src_path, task.canonical_index, ai),
                            encoding="utf-8").read())
    output = (src_a.get("actor") or {}).get("output") or ""
    calls = []
    with llm.record(calls), llm.phase("judge"):
        v = p.benchmark.verify(task, Attempt(ai, output), judge_model=judge_model)

    a = src_a                                  # a fresh parse; nothing else aliases it
    a["judge"] = {"model": judge_model, "success": bool(v.success),
                  "score": float(v.score), "raw_eval_output": v.raw_eval_output,
                  "details": v.details,
                  "calls": [{f: x for f, x in c.items() if f != "phase"}
                            for c in calls if c.get("phase") == "judge"]}
    # critic and summarizer are DERIVED from the judge — the critic writes about a
    # verdict, the summarizer compresses that. Carrying the source's forward would
    # attach the old judge's reasoning to the new judge's verdict.
    a.pop("summarizer", None)
    a["critic"] = {"model": None, "feedback": None, "calls": []}
    a["reused_from"] = {
        "run_id": p.src["run_id"],
        "storage_key": p.src["storage_key"],
        "output_key": f"{p.src['storage_key']}/task-{task.canonical_index}/attempt-{ai}.json",
        "note": "actor output claimed verbatim; this run paid for the judge only",
    }
    path = results.attempt_file(out, task.canonical_index, ai)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(a, indent=1, ensure_ascii=False))
    return v


def rejudge(selector, *, judge_model, runs_root="runs", apply=False, s3_sync=None):
    p = plan(runs_root, selector, judge_model=judge_model)
    print(f"source : {p.src['storage_key']}  judge={p.src['config']['judge_model']}")
    print(f"target : judge={p.dst_ident['judge_model']}  "
          f"metric={p.dst_ident['metric']} k={p.k}")
    print(f"  claimable {len(p.claimable)} attempt(s) · blocked {len(p.blocked)}")
    if p.blocked:
        print("  BLOCKED: seq@k attempt 2+ carries the old judge's feedback in its own\n"
              "  prompt, so it is not a sample from the same distribution under a new\n"
              "  judge. Those attempts must be REGENERATED, not re-judged.")
    if not p.claimable:
        raise SystemExit("nothing claimable under this judge change — see above.")
    if not apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    s3sync.check_auth_or_die(s3_sync=s3_sync)
    tasks = {t.canonical_index: t for t in p.benchmark.load_tasks(**(p.src.get("options") or {}))}
    d = p.dst_ident
    out, manifest, created = registry.resolve(
        runs_root, p.dst_cands,
        labels={"slice": d["slice_key"], "metric": d["metric"], "k": p.k,
                "agent": d["model"], "judge": d["judge_model"], "fb": d["feedback_mode"],
                "critic": d["critic_model"], "context": d["context"],
                "prompt": d["prompt_variant"], "temp": d["temperature"],
                "seed": d["seed"], "reason": d["reasoning_effort"]},
        options=p.src.get("options") or {},
        code={**results.code_provenance(), "rejudged_from": p.src["storage_key"],
              "rejudged_from_judge": p.src["config"]["judge_model"]},
        label_path=results.build_run_path(
            runs_root="", benchmark_module=p.benchmark,
            options=p.src.get("options") or {}, metric=d["metric"], model=d["model"],
            judge_model=d["judge_model"], critic_model=d["critic_model"],
            feedback_mode=d["feedback_mode"], k=p.k, context=d["context"],
            prompt_variant=d["prompt_variant"], temperature=d["temperature"],
            seed=d["seed"], reasoning_effort=d["reasoning_effort"]),
        k_target=p.k)
    run_id = manifest["run_id"]
    print(f"Run path: {out}/   ({'new' if created else 'resuming'} | run_id={run_id})")
    db.upsert_run(manifest, run_path=out)

    seq = d["metric"] == "seq@k"
    by_task = {}
    for t, i in p.claimable:
        by_task.setdefault(t, []).append(i)

    for n, tidx in enumerate(sorted(by_task), 1):
        task = tasks.get(tidx)
        if task is None:
            print(f"  task-{tidx}: not in this slice any more — skipped")
            continue
        results.save_task_meta(out, task)
        done = {int(a["attempt_index"]) for a in results.load_task_attempts(out, tidx)}
        print(f"\n[{n}/{len(by_task)}] task-{tidx} ({task.id})")
        for ai in sorted(by_task[tidx]):
            if ai in done:
                print(f"    attempt-{ai}: already re-judged, skipping")
                continue
            v = _claim(p, out, task, ai, judge_model=judge_model)
            print(f"    attempt-{ai}: success={v.success} score={v.score}")
        results._refresh_task_summary(out, tidx, task_id=task.id)
        results.save_summary(out, k=p.k)
        rows.mirror_task(run_id, out, tidx, ident=d, k=p.k, seq=seq,
                         task_id=task.id, prompt=task.prompt,
                         storage_key=manifest["storage_key"])

    rollup = rows.run_rollup(out, k=p.k, seq=seq)
    status = "complete" if rollup["tasks_total"] and not rollup["tasks_partial"] else "partial"
    registry.update_manifest(out, status=status, finished_at=ids.iso(ids.utc_now()),
                             rollup=rollup)
    db.finish_run(run_id, status=status, finished_at=ids.iso(ids.utc_now()), run_path=out)
    print(f"\n     {status}: {rollup['tasks_done']}/{rollup['tasks_total']} tasks, "
          f"{rollup['attempts_total']} attempts, ${rollup['cost_usd']:.4f} "
          f"(judge only — the actor was not re-paid for)")
    s3sync.upload_run(out, s3_sync=s3_sync)
