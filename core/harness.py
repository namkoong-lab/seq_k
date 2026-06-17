"""Run loop, one metric per run.

pass@k: every attempt sees only the task prompt, no feedback. seq@k: every attempt
also gets an "attempt t of K" note (from the first) plus prior attempts and their
feedback — so seq@1 != pass@1.

Folder naming: every invocation stamps `<out>-<UTC-timestamp>/` so two people
(or two machines) running the same variant don't stomp each other's results.
To continue a crashed run, pass `--resume <existing-folder>` on the CLI — that
skips the stamp and reuses the folder verbatim, and init_run's config-mismatch
guard refuses to mix incompatible configs.

Each attempt is written to its own file under runs/<name>/tasks/, so a crash
only loses the in-flight attempt.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core import llm, results, s3sync
from core.types import Attempt, Step, Trajectory

# UTC so timestamps sort and compare cleanly across machines/timezones.
# Filesystem-safe: ISO 8601 basic with `:` swapped to `-`.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%SZ"


def run(benchmark, *, metric, k, feedback_mode, model, judge_model=None, critic_model=None,
        temperature=0.7, max_tasks=None, out, console_char_limit=3000, options=None,
        s3_sync=None, resume=None):
    if metric not in ("pass@k", "seq@k"):
        raise ValueError(f"metric must be 'pass@k' or 'seq@k', got {metric!r}")
    # Default chain: actor model → judge_model → critic_model. Each role gets its
    # own model field in the saved JSON; you can mix-and-match by setting any of
    # them in the variant YAML.
    judge_model = judge_model or model
    critic_model = critic_model or judge_model
    options = options or {}

    out = _resolve_out(out, resume)

    tasks = benchmark.load_tasks(**options)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    results.init_run(
        out, benchmark=benchmark.__name__, metric=metric, k=k,
        feedback_mode=feedback_mode, model=model, judge_model=judge_model,
        critic_model=critic_model, temperature=temperature, options=options,
    )

    print(f"Loaded {len(tasks)} tasks | benchmark={benchmark.__name__} | metric={metric} "
          f"| k={k} | actor={model} | judge={judge_model} | critic={critic_model} | feedback={feedback_mode}")

    priors = [results.load_task_attempts(out, task.id) for task in tasks]
    n_done = sum(1 for p in priors if results.is_done(p, k))
    n_partial = sum(1 for p in priors if p and not results.is_done(p, k))
    if n_done or n_partial:
        print(f"Resume: {n_done} done, {n_partial} partial, {len(tasks) - n_done - n_partial} fresh")

    for i, (task, prior) in enumerate(zip(tasks, priors), 1):
        if results.is_done(prior, k):
            print(f"\n[{i}/{len(tasks)}] task {task.id}: skip (already done)")
            continue
        print(f"\n{'=' * 72}\n{metric} | task {task.id} ({i}/{len(tasks)})\n{'=' * 72}")
        traj = run_task(benchmark, task, prior=prior, metric=metric, k=k,
                        feedback_mode=feedback_mode, model=model,
                        judge_model=judge_model, critic_model=critic_model,
                        temperature=temperature, console_char_limit=console_char_limit,
                        options=options, out=out)
        results.save_summary(out, k=k)
        print(f"--> task {task.id}: success={traj.success} best_score={traj.best_score}")

    print(f"\nDone. {len(tasks)} tasks -> {out}/  (tasks/, summary.json, config.json)")
    s3sync.upload_run(out, s3_sync=s3_sync)


def run_task(benchmark, task, *, prior, metric, k, feedback_mode, model, judge_model, critic_model,
             temperature, console_char_limit, options=None, out=None):
    seq = metric == "seq@k"
    options = options or {}
    # Agentic benchmarks (e.g. TerminalBench) own their attempt: they build their own
    # prompt, run it in an environment, and verify it. Everything else uses the
    # standard llm.complete + verify path below — unchanged.
    owns_attempt = hasattr(benchmark, "run_attempt")

    steps = [_step_from_saved(a) for a in prior]
    history = [(Attempt(a["attempt_index"], a["actor"]["output"]), a["critic"]["feedback"]) for a in prior] if seq else []

    for t in range(len(prior), k):
        calls = []
        with llm.record(calls):
            if owns_attempt:
                prompt, output, result = benchmark.run_attempt(
                    task, history, t, k, seq=seq, model=model,
                    judge_model=judge_model, critic_model=critic_model,
                    temperature=temperature, options=options, out=out)
            else:
                prompt = build_prompt(task, history, t, k, seq=seq)
                output = llm.complete(model, prompt, temperature)        # actor
                with llm.phase("judge"):
                    result = benchmark.verify(task, Attempt(t + 1, output), judge_model=judge_model)
            attempt = Attempt(t + 1, output)

            fb = None
            # Critic runs on every failed seq@k attempt — including the last one,
            # so a `--resume` with a higher k has bridging feedback. pass@k never asks.
            if seq and not result.success:
                with llm.phase("critic"):
                    fb = benchmark.feedback(task, attempt, result, feedback_mode, critic_model=critic_model)

        # Group every recorded LLM call by role into its own section dict.
        judge_calls = [_strip_phase(c) for c in calls if c["phase"] == "judge"]
        critic_calls = [_strip_phase(c) for c in calls if c["phase"] == "critic"]
        step = Step(
            attempt_index=t + 1,
            actor={"model": model, "prompt": prompt, "output": output},
            judge={"model": judge_model, "success": result.success, "score": result.score,
                   "raw_eval_output": result.raw_eval_output, "details": result.details,
                   "calls": judge_calls},
            critic={"model": critic_model, "feedback": fb, "calls": critic_calls},
        )
        steps.append(step)
        results.save_attempt(task=task, step=step, k=k, model=model,
                             metric=metric, feedback_mode=feedback_mode, out=out)
        results.print_step(step, limit=console_char_limit)
        if result.success:
            break
        history.append((attempt, fb))

    return Trajectory(
        task_id=task.id, metric=metric, model=model, feedback_mode=feedback_mode,
        task_prompt=task.prompt, steps=steps,
        success=any(s.judge["success"] for s in steps),
        best_score=max(s.judge["score"] for s in steps),
    )


def _strip_phase(call):
    return {"model": call["model"], "prompt": call["prompt"], "output": call.get("output", "")}


def _resolve_out(out, resume):
    """Pick the actual run folder: explicit --resume wins, else stamp a fresh one.

    out      — the base name from the YAML (e.g. "runs/terminalbench.passk")
    resume   — an existing folder path; if set, used verbatim (no stamping)
    returns  — "runs/terminalbench.passk-2026-06-15T22-38-45Z" or `resume` as given
    """
    if resume:
        return resume
    stamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    return f"{out}-{stamp}"


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


def _step_from_saved(a):
    return Step(
        attempt_index=a["attempt_index"],
        actor=a["actor"],
        judge=a["judge"],
        critic=a["critic"],
    )
