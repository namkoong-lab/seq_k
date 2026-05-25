"""Each run writes one folder, `runs/<name>/`, holding three files:

    full.json     every trajectory in full — prompts, outputs, grading
    results.json  scores and per-rubric verdicts only (no prompts)
    prompts.txt   each agent's exact prompt at every step, labelled ACTOR /
                  JUDGE / CRITIC, for reviewing what each model could see

All three are rewritten after each task and swapped in atomically, so a crash
leaves a valid folder with every finished task. `load`, `inspect`, and `metrics`
read full.json — pass them either the run folder or the file directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict


def save(traj, out):
    os.makedirs(out, exist_ok=True)
    full = os.path.join(out, "full.json")
    trajs = load(out) if os.path.exists(full) else []
    trajs.append(asdict(traj))
    _write(full, _json(trajs))
    _write(os.path.join(out, "results.json"), _json(_results_view(trajs)))
    _write(os.path.join(out, "prompts.txt"), _prompts_view(trajs))


def load(out):
    """Read every trajectory as dicts. `out` is the run folder or its full.json."""
    path = os.path.join(out, "full.json") if os.path.isdir(out) else out
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def inspect(out, task_id):
    """Reprint one saved trajectory, step by step, in full (no truncation)."""
    for traj in load(out):
        if traj["task_id"] == task_id:
            print(f"task {traj['task_id']} | metric={traj['metric']} | model={traj['model']} "
                  f"| feedback={traj['feedback_mode']} | success={traj['success']} "
                  f"| best_score={traj['best_score']}")
            for step in traj["steps"]:
                print(_render_step(step, limit=0))
            return
    raise KeyError(f"task {task_id!r} not found in {out}")


def print_step(step, limit=3000):
    print(_render_step(asdict(step), limit))


def _results_view(trajs):
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


def _prompts_view(trajs):
    out = []
    for t in trajs:
        out += ["=" * 72,
                f"TASK {t['task_id']}  |  metric={t['metric']}  |  feedback={t['feedback_mode']}",
                "=" * 72]
        for s in t["steps"]:
            priv = s["result"].get("private") or {}
            out += [f"\n{'-' * 72}", f"ATTEMPT {s['attempt_index'] + 1}", "-" * 72]
            out.append(f"\n[ACTOR PROMPT]  (shown to the model being evaluated: {t['model']})\n\n{s['prompt']}")
            if priv.get("judge_prompt"):
                out.append(f"\n[JUDGE PROMPT]  (shown to the grader)\n\n{priv['judge_prompt']}")
            if priv.get("critic_prompt"):
                out.append(f"\n[CRITIC PROMPT — {t['feedback_mode']}]  (shown to the feedback model)\n\n{priv['critic_prompt']}")
        out.append("")
    return "\n".join(out)


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
