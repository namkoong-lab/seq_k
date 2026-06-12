"""Each run writes a folder runs/<name>/ holding:

    config.json                          the frozen run config
    tasks/<slug>_attempt_NN.json         one file per (task, attempt) — self-contained
    summary.json                         derived: pass@k/seq@k + per-task scores

Per-attempt files mean a crash only loses the in-flight attempt, and re-running
the same config skips finished tasks and resumes partial ones from the next
attempt.
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
    """Write tasks/<slug>_attempt_NN.json for one attempt."""
    tasks_dir = os.path.join(out, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    width = max(2, len(str(k)))
    name = f"{_slug(task.id)}_attempt_{step.attempt_index + 1:0{width}d}.json"
    payload = {
        "task_id": task.id,
        "task_prompt": task.prompt,
        "model": model,
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
        any(a["result"]["success"] for a in prior_attempts) or len(prior_attempts) >= k
    )


def load_task(out, task_id):
    """One reconstructed trajectory, or None if absent."""
    attempts = load_task_attempts(out, task_id)
    return _assemble(attempts) if attempts else None


def inspect(out, task_id):
    traj = load_task(out, task_id)
    if traj is None:
        raise KeyError(f"task {task_id!r} not found in {out}")
    print(f"task {traj['task_id']} | metric={traj['metric']} | model={traj['model']} "
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
            best = max(best, steps[t]["result"]["score"])
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
            "prompt": a["prompt"],
            "output": a["output"],
            "result": a["result"],
            "feedback": a["feedback"],
            "calls": a.get("calls", []),
        }
        for a in attempts
    ]
    return {
        "task_id": first["task_id"],
        "metric": first["metric"],
        "model": first["model"],
        "feedback_mode": first["feedback_mode"],
        "task_prompt": first["task_prompt"],
        "steps": steps,
        "success": any(s["result"]["success"] for s in steps),
        "best_score": max(s["result"]["score"] for s in steps),
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
                    "attempt": s["attempt_index"] + 1,
                    "success": s["result"]["success"],
                    "score": s["result"]["score"],
                    "requirement_status": (s["result"].get("private") or {}).get("requirement_status"),
                    "failed_requirement_count": (s["result"].get("private") or {}).get("failed_requirement_count"),
                    "total_requirements": (s["result"].get("private") or {}).get("total_requirements"),
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
    r = step["result"]
    status = (r.get("private") or {}).get("requirement_status")
    verdict = "PASS" if r["success"] else "FAIL"
    lines = [
        f"\n--- attempt {step['attempt_index'] + 1} ---",
        "PROMPT:",
        _truncate(step["prompt"], limit),
        "\nOUTPUT:",
        _truncate(step["output"], limit),
        f"\nVERDICT: {verdict}  score={r['score']}",
    ]
    if status:
        lines.append(f"rubrics: {status}")
    if step.get("feedback"):
        lines.append("\nFEEDBACK:")
        lines.append(_truncate(step["feedback"], limit))
    return "\n".join(lines)
