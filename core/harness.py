"""Run loop, one metric per run.

pass@k: every attempt sees only the task prompt, no feedback. seq@k: every attempt
also gets an "attempt t of K" note (from the first) plus prior attempts and their
feedback — so seq@1 != pass@1.
"""

from __future__ import annotations

from core import llm, results
from core.types import Attempt, Step, Trajectory


def run(benchmark, *, metric, k, feedback_mode, model, judge_model=None,
        temperature=0.7, max_tasks=None, out, console_char_limit=3000, options=None):
    if metric not in ("pass@k", "seq@k"):
        raise ValueError(f"metric must be 'pass@k' or 'seq@k', got {metric!r}")
    judge_model = judge_model or model

    tasks = benchmark.load_tasks(**(options or {}))   # benchmark-specific knobs from the config's `options:`
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    print(f"Loaded {len(tasks)} tasks | benchmark={benchmark.__name__} | metric={metric} "
          f"| k={k} | model={model} | judge={judge_model} | feedback={feedback_mode}")

    for i, task in enumerate(tasks, 1):
        print(f"\n{'=' * 72}\n{metric} | task {task.id} ({i}/{len(tasks)})\n{'=' * 72}")
        traj = run_task(benchmark, task, metric=metric, k=k, feedback_mode=feedback_mode,
                        model=model, judge_model=judge_model, temperature=temperature,
                        console_char_limit=console_char_limit, options=options or {}, out=out)
        results.save(traj, out, k=k)   # write per task so a crash keeps the finished ones
        print(f"--> task {task.id}: success={traj.success} best_score={traj.best_score}")

    print(f"\nDone. {len(tasks)} trajectories -> {out}/  (full.json, results.json, prompts.md)")


def run_task(benchmark, task, *, metric, k, feedback_mode, model, judge_model,
             temperature, console_char_limit, options=None, out=None):
    seq = metric == "seq@k"
    options = options or {}
    # Agentic benchmarks (e.g. TerminalBench) own their attempt: they build their own
    # prompt, run it in an environment, and verify it. Everything else uses the
    # standard llm.complete + verify path below — unchanged.
    owns_attempt = hasattr(benchmark, "run_attempt")
    steps, history = [], []
    for t in range(k):
        calls = []
        with llm.record(calls):
            if owns_attempt:
                prompt, output, result = benchmark.run_attempt(
                    task, history, t, k, seq=seq, model=model, judge_model=judge_model,
                    temperature=temperature, options=options, out=out)
            else:
                prompt = build_prompt(task, history, t, k, seq=seq)
                output = llm.complete(model, prompt, temperature)        # actor
                with llm.phase("judge"):
                    result = benchmark.verify(task, Attempt(t, output), judge_model=judge_model)
            attempt = Attempt(t, output)

            fb = None
            if seq and not result.success and t < k - 1:   # pass@k never asks for feedback
                with llm.phase("critic"):
                    fb = benchmark.feedback(task, attempt, result, feedback_mode, judge_model=judge_model)

        # actor prompt is already Step.prompt; keep only the auxiliary (judge/critic) calls
        step = Step(t, prompt, output, result, fb,
                    calls=[c for c in calls if c["phase"] != "actor"])
        steps.append(step)
        results.print_step(step, limit=console_char_limit)
        if result.success:
            break
        history.append((attempt, fb))

    return Trajectory(
        task_id=task.id, metric=metric, model=model, feedback_mode=feedback_mode,
        task_prompt=task.prompt, steps=steps,
        success=any(s.result.success for s in steps),
        best_score=max(s.result.score for s in steps),
    )


def build_prompt(task, history, t, k, *, seq):
    parts = [task.prompt]
    if seq:
        # "attempt t of K" on every attempt, including the first — this is what
        # makes seq@1 differ from pass@1.
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
