"""The run loop. One metric per run: pass@k OR seq@k.

Pass@k and seq@k share this loop; the difference is the prompt. Under pass@k every
attempt sees only the original task prompt (feedback-blind; feedback() is never
called). Under seq@k every attempt also gets a horizon note ("attempt t of K") —
including the first — plus the history of prior attempts and their feedback.
Because the horizon note is present from attempt 1, seq@1 differs from pass@1: the
model knows it is in a multi-attempt retry loop.
"""

from __future__ import annotations

from core import llm, results
from core.types import Attempt, Step, Trajectory


def run(benchmark, *, metric, k, feedback_mode, model, judge_model=None,
        temperature=0.7, max_tasks=None, out, console_char_limit=3000):
    if metric not in ("pass@k", "seq@k"):
        raise ValueError(f"metric must be 'pass@k' or 'seq@k', got {metric!r}")
    judge_model = judge_model or model

    tasks = benchmark.load_tasks()
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    print(f"Loaded {len(tasks)} tasks | benchmark={benchmark.__name__} | metric={metric} "
          f"| k={k} | model={model} | judge={judge_model} | feedback={feedback_mode}")

    for i, task in enumerate(tasks, 1):
        print(f"\n{'=' * 72}\n{metric} | task {task.id} ({i}/{len(tasks)})\n{'=' * 72}")
        traj = run_task(benchmark, task, metric=metric, k=k, feedback_mode=feedback_mode,
                        model=model, judge_model=judge_model, temperature=temperature,
                        console_char_limit=console_char_limit)
        results.save(traj, out)   # append per task: a later crash keeps earlier results
        print(f"--> task {task.id}: success={traj.success} best_score={traj.best_score}")

    print(f"\nDone. {len(tasks)} trajectories -> {out}/  (full.json, results.json, prompts.txt)")


def run_task(benchmark, task, *, metric, k, feedback_mode, model, judge_model,
             temperature, console_char_limit):
    seq = metric == "seq@k"
    steps, history = [], []
    for t in range(k):
        prompt = build_prompt(task, history, t, k, seq=seq)
        output = llm.complete(model, prompt, temperature)        # raises on API error — we want that
        attempt = Attempt(t, output)
        result = benchmark.verify(task, attempt, judge_model=judge_model)

        # pass@k is feedback-blind, so it never even calls feedback().
        fb = None
        if seq and not result.success and t < k - 1:
            fb = benchmark.feedback(task, attempt, result, feedback_mode, judge_model=judge_model)

        step = Step(t, prompt, output, result, fb)
        steps.append(step)
        results.print_step(step, limit=console_char_limit)      # live debug output
        if result.success:
            break
        history.append((attempt, fb))

    return Trajectory(
        task_id=task.id, metric=metric, model=model, feedback_mode=feedback_mode,
        steps=steps,
        success=any(s.result.success for s in steps),
        best_score=max(s.result.score for s in steps),
    )


def build_prompt(task, history, t, k, *, seq):
    parts = [task.prompt]
    if seq:
        # Horizon-aware: tell the model "attempt t of K" on every attempt,
        # including the first. This makes seq@1 differ from pass@1 — the model
        # knows it is in a retry loop with feedback coming.
        note = f"This is attempt {t + 1} of {k}."
        if history:
            note += " Review your previous attempt(s) and the feedback below, then provide an improved answer."
        else:
            note += " If this attempt does not pass, you will receive feedback and can revise it on the remaining attempts."
        parts.append(note)
        for past_attempt, past_feedback in history:
            parts.append(f"<PreviousAttempt>\n{past_attempt.output}\n</PreviousAttempt>")
            if past_feedback:
                parts.append(f"<Feedback>\n{past_feedback}\n</Feedback>")
    return "\n\n".join(parts)
