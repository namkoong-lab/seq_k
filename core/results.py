"""Layout: deterministic path derived from (slice, metric, agent, verifier, feedback).

Same config → same folder → re-running auto-resumes. To start fresh, `rm -rf` the
target path or change the config (model / feedback / metric / etc).

    runs/                                          (mirrored to s3://seq-k/, except config.json stays local)
    └── <slice>/                                   benchmark.slice_name(options) — e.g. terminalbench, clbench-dkr
        └── <metric>/                              passk | seqk
            └── <agent>/                           actor model id, "/" → "__"
                └── <verifier>/                    judge_model id, OR "harbor", OR "deterministic"
                    └── <feedback>/                template mode name (raw/binary/...) OR critic model id
                        ├── config.json            LOCAL ONLY — frozen variant config
                        ├── summary.json           aggregate pass@k / seq@k + last_updated
                        ├── task-1/                Task.canonical_index — stable across configs
                        │   ├── task_meta.json     { task_id, prompt, canonical_index }
                        │   ├── summary.json       per-task + last_updated
                        │   ├── attempt-1.json     actor/judge/critic shape (see types.Step)
                        │   ├── attempt-2.json
                        │   └── …
                        └── task-2/ …

Saved attempt JSON shape — IDENTICAL across every benchmark:

    {
      "task_id":        benchmark task id,
      "task_index":     canonical 1-based index (matches folder),
      "metric":         "pass@k" | "seq@k",
      "feedback_mode":  "binary" | "raw" | …,
      "attempt_index":  1-based,

      "actor":   {model, prompt, output},
      "judge":   {model, success, score, raw_eval_output, details, calls},
      "critic":  {model, feedback, calls}
    }
"""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone

from core import pricing

# UTC iso-ish, filesystem-safe (": " → "-")
_ISO_UTC = "%Y-%m-%dT%H-%M-%SZ"


# --------------------------------------------------------------------------- #
# Path construction
# --------------------------------------------------------------------------- #
def build_run_path(*, runs_root, benchmark_module, options, metric, model, judge_model,
                   critic_model, feedback_mode):
    """Compose the 5-level run directory.

    runs_root/<slice>/<metric_short>/<agent>/<verifier>/<feedback>/
    """
    mod = importlib.import_module(benchmark_module) if isinstance(benchmark_module, str) else benchmark_module
    slice_part = mod.slice_name(options or {})
    metric_part = _metric_short(metric)
    agent_part = _safe(model)
    verifier_part = _safe(judge_model) if mod.VERIFIER == "llm" else mod.VERIFIER
    feedback_part = _safe(critic_model) if feedback_mode in mod.LLM_CRITIC_MODES else feedback_mode
    return os.path.join(runs_root, slice_part, metric_part, agent_part, verifier_part, feedback_part)


def task_dir(run_path, canonical_index):
    return os.path.join(run_path, f"task-{canonical_index}")


def attempt_file(run_path, canonical_index, attempt_index):
    return os.path.join(task_dir(run_path, canonical_index), f"attempt-{attempt_index}.json")


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def init_run(run_path, *, continue_run=False, **config):
    """Create the run dir and freeze config.json.

    Path-derivation already guarantees that benchmark / metric / model /
    feedback_mode are identical for any two configs that land at the same
    path — those fields can never collide here.

    The only field that can mismatch at a shared path is k. Policy:
        new k == old k  -> resume (skip already-done tasks)
        new k >  old k  -> default: CLOBBER the path (local + S3 mirror later),
                           restart from attempt 1.
                           continue_run=True: KEEP attempts 1..old_k, extend to old_k+1..new_k.
        new k <  old k  -> raise (truncation is undefined intent).
    """
    cfg_path = os.path.join(run_path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            existing = json.load(f)
        existing_k = int(existing.get("k", 0))
        new_k = int(config.get("k", 0))
        if new_k < existing_k:
            raise ValueError(
                f"k={new_k} requested but {cfg_path} already has k={existing_k}. "
                f"Truncating an existing run is not supported; `rm -rf` the path to start fresh."
            )
        if new_k > existing_k and not continue_run:
            # Clobber: discard the old run entirely.
            print(f"→ clobbering existing run at {run_path} (k went {existing_k} → {new_k})")
            import shutil
            shutil.rmtree(run_path)
        # else: extend (continue_run) or no-op (new_k == existing_k) — keep attempts.
    os.makedirs(run_path, exist_ok=True)
    _write(cfg_path, _json(config))


def save_task_meta(run_path, task):
    """Write task-<i>/task_meta.json once. Skips if already present."""
    path = os.path.join(task_dir(run_path, task.canonical_index), "task_meta.json")
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write(path, _json({
        "task_id": task.id,
        "task_index": task.canonical_index,
        "prompt": task.prompt,
    }))


def save_attempt(*, run_path, task, step, metric, feedback_mode):
    """Write task-<i>/attempt-<j>.json AND refresh that task's per-task summary."""
    path = attempt_file(run_path, task.canonical_index, step.attempt_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "task_id": task.id,
        "task_index": task.canonical_index,
        "metric": metric,
        "feedback_mode": feedback_mode,
        **asdict(step),
    }
    _write(path, _json(payload))
    _refresh_task_summary(run_path, task.canonical_index, task_id=task.id)


def save_summary(run_path, k):
    """Write run-level summary.json (aggregate; per-task lives in task-*/summary.json)."""
    trajs = load(run_path)
    summary = _summary_view(trajs, k)
    summary["last_updated"] = _now()
    _write(os.path.join(run_path, "summary.json"), _json(summary))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def load(run_path):
    """All per-task trajectories under run_path, sorted by canonical_index."""
    if not os.path.isdir(run_path):
        return []
    trajs = []
    for entry in sorted(os.listdir(run_path)):
        m = _TASK_DIR.match(entry)
        if not m:
            continue
        idx = int(m.group("idx"))
        attempts = _load_task_attempts(run_path, idx)
        if attempts:
            trajs.append(_assemble(attempts))
    trajs.sort(key=lambda t: t["task_index"])
    return trajs


def load_task_attempts(run_path, canonical_index):
    """Saved attempt files for one task, sorted by attempt_index."""
    return _load_task_attempts(run_path, canonical_index)


def is_done(prior_attempts, k, *, seq):
    """Whether a task has produced enough attempts for `metric`.

    seq@k: stop retrying as soon as ONE attempt succeeded — there's nothing left
           to improve, and any extra retry would just waste compute on a solved task.
    pass@k: K INDEPENDENT samples — never stop early on success. The point is the
            per-attempt success rate, so a task with one passing attempt out of
            (so far) 2 is NOT done at k=5; we still owe 3 more independent draws.
    """
    if not prior_attempts:
        return False
    if not seq:
        return len(prior_attempts) >= k
    return any(a["judge"]["success"] for a in prior_attempts) or len(prior_attempts) >= k


def load_task(run_path, canonical_index):
    """Reconstructed trajectory for one task, or None."""
    attempts = _load_task_attempts(run_path, canonical_index)
    return _assemble(attempts) if attempts else None


def inspect(run_path, *, task_index=None, task_id=None):
    """Print a task's trajectory step-by-step. Pass either task_index OR task_id."""
    if task_index is None and task_id is None:
        raise ValueError("inspect() needs task_index or task_id")
    if task_index is None:
        for entry in sorted(os.listdir(run_path)):
            m = _TASK_DIR.match(entry)
            if not m:
                continue
            meta_path = os.path.join(run_path, entry, "task_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    if json.load(f).get("task_id") == task_id:
                        task_index = int(m.group("idx"))
                        break
        if task_index is None:
            raise KeyError(f"task_id {task_id!r} not found in {run_path}")
    traj = load_task(run_path, task_index)
    if traj is None:
        raise KeyError(f"task-{task_index} has no attempts in {run_path}")
    print(f"task-{task_index} | task_id={traj['task_id']} | model={traj['model']} | "
          f"success={traj['success']} | best_score={traj['best_score']}")
    for step in traj["steps"]:
        print(_render_step(step, limit=0))


# --------------------------------------------------------------------------- #
# Console rendering
# --------------------------------------------------------------------------- #
def print_step(step, limit=3000):
    print(_render_step(asdict(step), limit))


def cumulative_best_by_attempt(traj, k):
    best, curve = 0.0, []
    steps = traj["steps"]
    for t in range(k):
        if t < len(steps):
            best = max(best, steps[t]["judge"]["score"])
        curve.append(best)
    return curve


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
_TASK_DIR = re.compile(r"^task-(?P<idx>\d+)$")
_ATTEMPT_FILE = re.compile(r"^attempt-(?P<idx>\d+)\.json$")


def _metric_short(metric):
    return {"pass@k": "passk", "seq@k": "seqk"}.get(metric, metric.replace("@", ""))


def _safe(name):
    """Make a model id filesystem-safe: '/' → '__'. Round-trippable."""
    return str(name).replace("/", "__")


def _now():
    return datetime.now(timezone.utc).strftime(_ISO_UTC)


def _load_task_attempts(run_path, canonical_index):
    d = task_dir(run_path, canonical_index)
    if not os.path.isdir(d):
        return []
    attempts = []
    for name in os.listdir(d):
        m = _ATTEMPT_FILE.match(name)
        if not m:
            continue
        with open(os.path.join(d, name), encoding="utf-8") as f:
            attempts.append(json.load(f))
    return sorted(attempts, key=lambda a: a["attempt_index"])


def _refresh_task_summary(run_path, canonical_index, *, task_id):
    """Rewrite task-<i>/summary.json from current attempt files. Includes
    per-model token totals (input / cached / output) summed across every LLM
    call in every attempt of this task."""
    attempts = _load_task_attempts(run_path, canonical_index)
    if not attempts:
        return
    summary = {
        "task_id": task_id,
        "task_index": canonical_index,
        "success": any(a["judge"]["success"] for a in attempts),
        "best_score": max(a["judge"]["score"] for a in attempts),
        "last_updated": _now(),
        "attempts": [
            {"attempt": a["attempt_index"], "success": a["judge"]["success"], "score": a["judge"]["score"]}
            for a in attempts
        ],
        "tokens": _tokens_across_attempts(attempts),
        "pricing_last_updated": pricing.PRICING_LAST_UPDATED,
    }
    _write(os.path.join(task_dir(run_path, canonical_index), "summary.json"), _json(summary))


def _tokens_across_attempts(attempts):
    """Aggregate token usage across attempts, keyed by model id.

    Combines actor (one count per attempt) + each judge call + each critic call.
    Same model used by multiple roles → merged into one entry. Each entry gets
    a derived `cost_usd` from core/pricing.py (null if model not in table)."""
    by_model = {}
    for a in attempts:
        actor = a["actor"]
        _add(by_model, actor["model"], actor)
        for call in a["judge"].get("calls") or []:
            _add(by_model, call["model"], call)
        for call in a["critic"].get("calls") or []:
            _add(by_model, call["model"], call)
    _annotate_costs(by_model)
    return by_model


def _annotate_costs(by_model):
    """In-place: attach `cost_usd` to each entry (None if model isn't priced)."""
    for model, bucket in by_model.items():
        bucket["cost_usd"] = pricing.cost_for(
            model, bucket["input_tokens"], bucket["cached_tokens"], bucket["output_tokens"]
        )


_TOKEN_FIELDS = ("input_tokens", "cached_tokens", "thinking_tokens", "output_tokens")


def _add(by_model, model, src):
    """Sum the four token fields from `src` into the per-model bucket."""
    bucket = by_model.setdefault(model, {k: 0 for k in _TOKEN_FIELDS})
    for k in _TOKEN_FIELDS:
        bucket[k] += int(src.get(k, 0))


def _assemble(attempts):
    attempts = sorted(attempts, key=lambda a: a["attempt_index"])
    first = attempts[0]
    steps = [
        {
            "attempt_index": a["attempt_index"],
            "actor": a["actor"],
            "judge": a["judge"],
            "critic": a["critic"],
        }
        for a in attempts
    ]
    return {
        "task_id": first["task_id"],
        "task_index": first["task_index"],
        "metric": first["metric"],
        "feedback_mode": first["feedback_mode"],
        "model": first["actor"]["model"],
        "steps": steps,
        "success": any(s["judge"]["success"] for s in steps),
        "best_score": max(s["judge"]["score"] for s in steps),
    }


def _summary_view(trajs, k):
    if not trajs:
        return {"tasks": 0, "k": k}
    metric = trajs[0]["metric"]
    label = "seq" if metric == "seq@k" else "pass"
    curves = [cumulative_best_by_attempt(t, k) for t in trajs]
    at = [sum(c[t] for c in curves) / len(curves) for t in range(k)]
    summary = {
        "metric": metric,
        "model": trajs[0]["model"],
        "feedback_mode": trajs[0]["feedback_mode"],
        "tasks": len(trajs),
        "k": k,
    }
    for t in range(k):
        summary[f"{label}@{t + 1}"] = round(at[t], 4)
    if metric == "seq@k" and k >= 2:
        delta = at[k - 1] - at[0]
        summary["delta"] = round(delta, 4)
        if delta > 0:
            summary["EGS"] = round((at[1] - at[0]) / delta, 3)
            summary["LGS"] = round((at[k - 1] - at[k - 2]) / delta, 3)
    # Tokens summed across every attempt of every task, grouped by model.
    by_model = {}
    for t in trajs:
        for step in t["steps"]:
            _add(by_model, step["actor"]["model"], step["actor"])
            for c in step["judge"].get("calls") or []:
                _add(by_model, c["model"], c)
            for c in step["critic"].get("calls") or []:
                _add(by_model, c["model"], c)
    _annotate_costs(by_model)
    summary["tokens"] = by_model
    summary["pricing_last_updated"] = pricing.PRICING_LAST_UPDATED
    return summary


def _json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def _write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _truncate(text, limit):
    text = "" if text is None else str(text)
    if limit and len(text) > limit:
        return text[:limit] + f"\n…[truncated {len(text) - limit} chars; full text in results folder]"
    return text


def _render_step(step, limit):
    j = step["judge"]
    c = step["critic"]
    status = j["details"].get("requirement_status")
    verdict = "PASS" if j["success"] else "FAIL"
    lines = [
        f"\n--- attempt {step['attempt_index']} ---",
        f"ACTOR ({step['actor']['model']}) PROMPT (what the model saw):",
        _truncate(step["actor"]["prompt"], limit),
        "\nACTOR OUTPUT:",
        _truncate(step["actor"]["output"], limit),
        f"\nJUDGE ({j['model']}) VERDICT: {verdict}  score={j['score']}",
    ]
    if status:
        lines.append(f"rubrics: {status}")
    for i, call in enumerate(j["calls"], 1):
        lines.append(f"\nJUDGE CALL {i} ({call['model']}):")
        lines.append("  prompt:"); lines.append(_truncate(call["prompt"], limit))
        lines.append("  output:"); lines.append(_truncate(call["output"], limit))
    for i, call in enumerate(c["calls"], 1):
        lines.append(f"\nCRITIC CALL {i} ({call['model']}):")
        lines.append("  prompt:"); lines.append(_truncate(call["prompt"], limit))
        lines.append("  output:"); lines.append(_truncate(call["output"], limit))
    if c["feedback"]:
        lines.append(f"\nCRITIC ({c['model']}) FEEDBACK INTO NEXT ATTEMPT:")
        lines.append(_truncate(c["feedback"], limit))
    return "\n".join(lines)
