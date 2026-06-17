"""Each run writes a folder runs/<name>/ holding:

    config.json                          the frozen run config
    tasks/<slug>_attempt_NN.json         one file per (task, attempt) — self-contained
    summary.json                         derived: pass@k/seq@k + per-task scores

Per-attempt files mean a crash only loses the in-flight attempt, and re-running
the same config skips finished tasks and resumes partial ones from the next attempt.

Saved attempt JSON shape (IDENTICAL across every benchmark). Three sections —
`actor`, `judge`, `critic` — each independent, each with its own `model`:

    {
      # run identity (one per attempt file)
      "task_id":         benchmark task id,
      "metric":          "pass@k" | "seq@k",
      "feedback_mode":   "binary" | "raw" | "compact" | "socratic" | "directive" | "retry_diagnostics" | ...,
      "attempt_index":   1-based (attempt 1 of K, 2 of K, ...),

      # ACTOR — the model being evaluated
      "actor": {
        "model":   the actor model id,
        "prompt":  the EXACT text the actor saw. For seq@k attempt N>=2 this includes
                   "This is attempt N of K" + every prior actor.output + every prior critic.feedback,
        "output":  the actor's raw response (VERBATIM). For multi-step agentic benchmarks
                   (terminalbench), this is the agent's FINAL message; the full trajectory
                   lives at judge.details.trajectory_full.
      },

      # JUDGE — produces success/score; runs on every attempt
      "judge": {
        "model":            the judge model id,
        "success":          bool,
        "score":            float,
        "raw_eval_output":  judge's PUBLIC diagnostic — safe to show next attempt (empty on success),
        "details":          judge's INTERNAL scratch (per-rubric verdicts, harbor trial details, etc.) —
                            benchmark-specific keys, never fed back into the actor,
        "calls":            list of every LLM call the judge made this attempt:
                            [{model, prompt, output}, ...] — VERBATIM each.
                            Empty for non-LLM judges (terminalbench, arcagi2).
                            Multi-element for benchmarks that grade per-rubric
                            (healthbench, researchrubrics).
      },

      # CRITIC — produces feedback for the NEXT attempt's actor (seq@k failed non-final only)
      "critic": {
        "model":     the critic model id,
        "feedback":  the EXACT string the next attempt's actor.prompt will include, or null.
                     For LLM critic modes this matches critic.calls[-1].output.
                     For template-only modes it's derived from judge.raw_eval_output / judge.details.
                     null on: pass@k, successful attempts, or the last attempt.
        "calls":     list of every LLM call the critic made this attempt:
                     [{model, prompt, output}, ...] — VERBATIM each.
                     Empty for template-only feedback modes (binary/raw/compact/cell_match/
                     retry_diagnostics), pass@k, success, last attempt.
      }
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def init_run(out, **config):
    """Create the run dir and freeze config.json. Refuse to resume into a dir
    whose existing config disagrees on metric/k/feedback_mode/model/benchmark."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        for key in ("benchmark", "metric", "k", "feedback_mode", "model"):
            if existing.get(key) != config.get(key):
                raise ValueError(
                    f"config mismatch in {path}: {key}={existing.get(key)!r} on disk, "
                    f"{config.get(key)!r} in new run. Use a different out: directory."
                )
        return
    _write(path, _json(config))


def save_attempt(*, task, step, k, model, metric, feedback_mode, out):
    """Write tasks/<slug>_attempt_NN.json for one attempt. Schema is documented
    at the top of this module."""
    tasks_dir = os.path.join(out, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    width = max(2, len(str(k)))
    name = f"{_slug(task.id)}_attempt_{step.attempt_index:0{width}d}.json"
    payload = {
        "task_id": task.id,
        "metric": metric,
        "feedback_mode": feedback_mode,
        **asdict(step),
    }
    _write(os.path.join(tasks_dir, name), _json(payload))


def save_summary(out, k):
    trajs = load(out)
    _write(os.path.join(out, "summary.json"),
           _json({"summary": _summary_view(trajs, k), "tasks": _tasks_view(trajs)}))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def load(out):
    """All trajectories in `out` (a run folder), as dicts."""
    tasks_dir = os.path.join(out, "tasks")
    if not os.path.isdir(tasks_dir):
        return []
    by_slug = {}
    for name in sorted(os.listdir(tasks_dir)):
        m = _ATTEMPT_FILE.match(name)
        if not m:
            continue
        with open(os.path.join(tasks_dir, name), encoding="utf-8") as f:
            by_slug.setdefault(m.group("slug"), []).append(json.load(f))
    return [_assemble(att) for att in by_slug.values()]


def load_task_attempts(out, task_id):
    """Saved attempt files for one task, sorted by attempt_index. [] if none."""
    tasks_dir = os.path.join(out, "tasks")
    if not os.path.isdir(tasks_dir):
        return []
    slug = _slug(task_id)
    prefix = f"{slug}_attempt_"
    attempts = []
    for name in os.listdir(tasks_dir):
        if name.startswith(prefix) and name.endswith(".json"):
            with open(os.path.join(tasks_dir, name), encoding="utf-8") as f:
                attempts.append(json.load(f))
    return sorted(attempts, key=lambda a: a["attempt_index"])


def is_done(prior_attempts, k):
    return bool(prior_attempts) and (
        any(a["judge"]["success"] for a in prior_attempts) or len(prior_attempts) >= k
    )


def load_task(out, task_id):
    """One reconstructed trajectory, or None if absent."""
    attempts = load_task_attempts(out, task_id)
    return _assemble(attempts) if attempts else None


def inspect(out, task_id):
    traj = load_task(out, task_id)
    if traj is None:
        raise KeyError(f"task {task_id!r} not found in {out}")
    print(f"task {traj['task_id']} | metric={traj['metric']} | actor={traj['model']} "
          f"| feedback={traj['feedback_mode']} | success={traj['success']} "
          f"| best_score={traj['best_score']}")
    for step in traj["steps"]:
        print(_render_step(step, limit=0))


# --------------------------------------------------------------------------- #
# Console rendering
# --------------------------------------------------------------------------- #
def print_step(step, limit=3000):
    print(_render_step(asdict(step), limit))


def cumulative_best_by_attempt(traj, k):
    """Best score by attempt t, carried forward (monotonically non-decreasing).

    If a task succeeded at attempt N (and the harness stopped), the carried-forward
    best stays at that score for attempts N+1..K — which is what makes pass@k /
    seq@k monotonic.
    """
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
_ATTEMPT_FILE = re.compile(r"^(?P<slug>.+)_attempt_(?P<n>\d+)\.json$")


def _slug(task_id):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("_") or "task"
    if s == task_id and len(s) <= 120:
        return s
    return f"{s[:120]}_{hashlib.sha1(task_id.encode()).hexdigest()[:8]}"


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
        "metric": first["metric"],
        "model": first["actor"]["model"],     # actor model — the one being evaluated
        "feedback_mode": first["feedback_mode"],
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
    return summary


def _tasks_view(trajs):
    return [
        {
            "task_id": t["task_id"],
            "metric": t["metric"],
            "model": t["model"],
            "feedback_mode": t["feedback_mode"],
            "success": t["success"],
            "best_score": t["best_score"],
            "attempts": [
                {
                    "attempt": s["attempt_index"],
                    "success": s["judge"]["success"],
                    "score": s["judge"]["score"],
                    "requirement_status": s["judge"]["details"].get("requirement_status"),
                    "failed_requirement_count": s["judge"]["details"].get("failed_requirement_count"),
                    "total_requirements": s["judge"]["details"].get("total_requirements"),
                }
                for s in t["steps"]
            ],
        }
        for t in trajs
    ]


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
