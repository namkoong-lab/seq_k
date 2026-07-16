"""Run loop, one metric per run.

pass@k: every attempt sees only the task prompt, no feedback. seq@k: every attempt
also gets an "attempt t of K" note (from the first) plus prior attempts and their
feedback — so seq@1 != pass@1.

Folder naming: the run path is deterministically derived from
    (slice_name(options), metric, model, judge_model, critic_model, feedback_mode)
by core.results.build_run_path. Same config → same folder → re-running auto-resumes.
init_run's config-mismatch guard refuses to mix incompatible k/temperature/etc.
into an existing folder. To start fresh: `rm -rf` the path.

Each attempt is written to its own task-N/attempt-M.json file, so a crash
only loses the in-flight attempt.
"""

from __future__ import annotations

from core import llm, results, s3sync
from core.types import Attempt, Step, Trajectory, VerifierResult


def run(benchmark, *, metric, k, feedback_mode, model, judge_model=None, critic_model=None,
        temperature=0.7, max_tasks=None, runs_root="runs",
        console_char_limit=3000, options=None, s3_sync=None, task_indices=None,
        continue_run=False, output_budget=None, reasoning_effort=None):
    if metric not in ("pass@k", "seq@k"):
        raise ValueError(f"metric must be 'pass@k' or 'seq@k', got {metric!r}")
    # output_budget is a run-level cap on cumulative actor output tokens per task.
    # It only makes sense for seq@k (a shared, decrementing budget across attempts);
    # pass@k draws independent samples, so fail loud rather than silently ignore it.
    if output_budget is not None and metric != "seq@k":
        raise ValueError(f"output_budget applies to seq@k only, got metric={metric!r}")
    # reasoning_effort opts the actor into the provider's reasoning / extended-thinking
    # mode. Judge and critic never get it — they're graders, not the agent under test.
    # Agentic benchmarks (terminalbench) run the actor inside their own subprocess,
    # so we can't thread reasoning_effort through — fail loud rather than silently ignore.
    if reasoning_effort is not None and hasattr(benchmark, "run_attempt"):
        raise ValueError(
            f"reasoning_effort not supported for agentic benchmark {benchmark.__name__}: "
            f"the actor call lives inside benchmark.run_attempt and can't be threaded through"
        )
    # Default chain: actor model → judge_model → critic_model. Each role gets its
    # own field in the saved JSON; mix-and-match by setting any of them in the YAML.
    judge_model = judge_model or model
    critic_model = critic_model or judge_model
    options = options or {}

    out = results.build_run_path(
        runs_root=runs_root, benchmark_module=benchmark, options=options,
        metric=metric, model=model, judge_model=judge_model,
        critic_model=critic_model, feedback_mode=feedback_mode,
        output_budget=output_budget, reasoning_effort=reasoning_effort,
    )

    # Fail fast if S3 sync is enabled but auth is bad — otherwise we'd discover
    # it after the entire run (potentially hours of Docker work) is done.
    s3sync.check_auth_or_die(s3_sync=s3_sync)

    tasks = benchmark.load_tasks(**options)
    if task_indices:
        wanted = set(task_indices)
        tasks = [t for t in tasks if t.canonical_index in wanted]
        missing = wanted - {t.canonical_index for t in tasks}
        if missing:
            raise ValueError(f"task_indices not found in this slice: {sorted(missing)}")
    elif max_tasks is not None:
        tasks = tasks[:max_tasks]

    results.init_run(
        out, continue_run=continue_run,
        benchmark=benchmark.__name__, metric=metric, k=k,
        feedback_mode=feedback_mode, model=model, judge_model=judge_model,
        critic_model=critic_model, temperature=temperature, options=options,
        output_budget=output_budget, reasoning_effort=reasoning_effort,
    )

    print(f"Loaded {len(tasks)} tasks | benchmark={benchmark.__name__} | metric={metric} "
          f"| k={k} | actor={model} | judge={judge_model} | critic={critic_model} | feedback={feedback_mode}")
    print(f"Run path: {out}/")

    seq = metric == "seq@k"
    priors = [results.load_task_attempts(out, task.canonical_index) for task in tasks]
    n_done = sum(1 for p in priors if results.is_done(p, k, seq=seq))
    n_partial = sum(1 for p in priors if p and not results.is_done(p, k, seq=seq))
    if n_done or n_partial:
        print(f"Resume: {n_done} done, {n_partial} partial, {len(tasks) - n_done - n_partial} fresh")

    for i, (task, prior) in enumerate(zip(tasks, priors), 1):
        results.save_task_meta(out, task)
        if results.is_done(prior, k, seq=seq):
            print(f"\n[{i}/{len(tasks)}] task-{task.canonical_index} ({task.id}): skip (already done)")
            continue
        print(f"\n{'=' * 72}\n{metric} | task-{task.canonical_index} {task.id} ({i}/{len(tasks)})\n{'=' * 72}")
        # Always refresh the run-level summary, even if run_task crashes mid-task
        # (e.g. provider timeout) — otherwise the run summary would lag behind
        # partial per-task data that's already on disk.
        try:
            traj = run_task(benchmark, task, prior=prior, metric=metric, k=k,
                            feedback_mode=feedback_mode, model=model,
                            judge_model=judge_model, critic_model=critic_model,
                            temperature=temperature, console_char_limit=console_char_limit,
                            options=options, out=out, output_budget=output_budget,
                            reasoning_effort=reasoning_effort)
        finally:
            results.save_summary(out, k=k)
        print(f"--> task-{task.canonical_index} {task.id}: success={traj.success} best_score={traj.best_score}")

    print(f"\nDone. {len(tasks)} tasks -> {out}/")
    s3sync.upload_run(out, s3_sync=s3_sync)


def run_task(benchmark, task, *, prior, metric, k, feedback_mode, model, judge_model, critic_model,
             temperature, console_char_limit, options=None, out=None, output_budget=None,
             reasoning_effort=None):
    seq = metric == "seq@k"
    options = options or {}
    # Agentic benchmarks (e.g. TerminalBench) own their attempt: they build their own
    # prompt, run it in an environment, and verify it. Everything else uses the
    # standard llm.complete + verify path below.
    owns_attempt = hasattr(benchmark, "run_attempt")

    steps = [_step_from_saved(a) for a in prior]
    history = [(Attempt(a["attempt_index"], a["actor"]["output"]), a["critic"]["feedback"]) for a in prior] if seq else []
    # Cumulative actor output tokens spent on this task so far (for output_budget).
    # Seeded from resumed attempts so a resumed run keeps counting where it left off.
    used_output = (sum(int(s.actor.get("output_tokens", 0)) for s in steps)
                   if output_budget is not None else 0)

    for t in range(len(prior), k):
        # Re-load prior from disk each iteration so the latest just-finished
        # attempt's saved data is visible to the next attempt's retry context.
        # (The static `prior` from before the loop only reflects attempts that
        # existed BEFORE this run_task call.)
        current_prior = results.load_task_attempts(out, task.canonical_index) if owns_attempt else prior
        calls = []
        over_budget = False
        with llm.record(calls):
            if owns_attempt:
                prompt, output, result = benchmark.run_attempt(
                    task, history, t, k, seq=seq, model=model,
                    judge_model=judge_model, critic_model=critic_model,
                    temperature=temperature, options=options, out=out,
                    prior=current_prior)
            else:
                remaining = output_budget - used_output if output_budget is not None else None
                prompt = build_prompt(task, history, t, k, seq=seq, remaining_budget=remaining)
                # Actor is the ONLY role to receive reasoning_effort — judge/critic are
                # graders, not the agent under test.
                output = llm.complete(model, prompt, temperature,
                                      reasoning_effort=reasoning_effort)
                if output_budget is not None:
                    used_output += _round_actor_output_tokens(calls)
                # Budget check runs AFTER the actor generates: if this attempt's output
                # pushed cumulative usage over the run-level budget, the attempt fails
                # immediately. The judge is skipped (over budget == fail regardless of
                # answer quality); so is the critic below (there is no next attempt).
                if output_budget is not None and used_output > output_budget:
                    over_budget = True
                    result = VerifierResult(
                        success=False, score=0.0,
                        raw_eval_output=(f"over budget: used {used_output} output tokens "
                                         f"> budget {output_budget}; judge skipped"),
                        details={"over_budget": True})
                else:
                    with llm.phase("judge"):
                        result = benchmark.verify(task, Attempt(t + 1, output), judge_model=judge_model)
            attempt = Attempt(t + 1, output)

            fb = None
            # Critic runs on every failed seq@k attempt — including the last one —
            # so a future re-run with a higher k has bridging feedback. pass@k never asks.
            # An over-budget attempt is terminal, so it skips the critic too.
            if seq and not result.success and not over_budget:
                with llm.phase("critic"):
                    fb = benchmark.feedback(task, attempt, result, feedback_mode, critic_model=critic_model)

        # Group every recorded LLM call by role into its own section dict.
        judge_calls = [_strip_phase(c) for c in calls if c["phase"] == "judge"]
        critic_calls = [_strip_phase(c) for c in calls if c["phase"] == "critic"]
        actor_tokens = _actor_tokens(calls, result, owns_attempt)
        # judge.model is null when the verifier isn't an LLM (e.g. terminalbench's
        # harbor or arcagi2's deterministic verifier) OR when the attempt went over
        # budget (the judge never ran). critic.model is null when the feedback_mode
        # is template-only, or the critic was skipped (over budget = no next attempt).
        judge_model_saved = None if over_budget else (
            judge_model if getattr(benchmark, "VERIFIER", "llm") == "llm" else None)
        critic_model_saved = critic_model if (
            feedback_mode in getattr(benchmark, "LLM_CRITIC_MODES", set()) and not over_budget) else None
        actor_section = {"model": model, "prompt": prompt, "output": output, **actor_tokens}
        # Merge provider metadata from the actor's llm.complete call: finish_reason
        # (stop / length / content_filter / tool_calls), full raw_response (safety
        # net for anything else the provider returned), and reasoning-specific
        # fields (reasoning_effort setting, thinking_content prose) when present.
        # Skipped for agentic benchmarks (owns_attempt) — Harbor runs the actor
        # inside its own subprocess so llm.record() never sees the call.
        if not owns_attempt:
            actor_section.update(_actor_metadata(calls))
        # budget bookkeeping is written only when the run is budget-enabled, so
        # budget-off runs keep the exact same attempt schema as before.
        if output_budget is not None:
            actor_section["budget"] = {"total": output_budget, "used": used_output, "over": over_budget}
        step = Step(
            attempt_index=t + 1,
            actor=actor_section,
            judge={"model": judge_model_saved, "success": result.success, "score": result.score,
                   "raw_eval_output": result.raw_eval_output, "details": result.details,
                   "calls": judge_calls},
            critic={"model": critic_model_saved, "feedback": fb, "calls": critic_calls},
        )
        steps.append(step)
        results.save_attempt(run_path=out, task=task, step=step,
                             metric=metric, feedback_mode=feedback_mode)
        results.print_step(step, limit=console_char_limit)
        # Stop on: over budget (task failed, terminal — nothing left to try), or
        # seq@k success (nothing left to improve, extra attempts waste compute).
        # pass@k NEVER breaks: it wants K INDEPENDENT samples so pass@1..@k from the
        # same data are meaningful — not just "we got lucky on attempt 1".
        if over_budget or (seq and result.success):
            break
        history.append((attempt, fb))

    return Trajectory(
        task_id=task.id, metric=metric, model=model, feedback_mode=feedback_mode,
        task_prompt=task.prompt, steps=steps,
        success=any(s.judge["success"] for s in steps),
        best_score=max(s.judge["score"] for s in steps),
    )


def _strip_phase(call):
    """Per-call record stored under judge.calls / critic.calls. Uniform schema.
    Includes provider-reported finish_reason and the raw serialized response so
    we can audit stop reasons, exact served model versions, safety refusals, etc.
    """
    return {
        "model": call["model"], "prompt": call["prompt"], "output": call["output"],
        "input_tokens":    call.get("input_tokens", 0),
        "cached_tokens":   call.get("cached_tokens", 0),
        "thinking_tokens": call.get("thinking_tokens", 0),
        "output_tokens":   call.get("output_tokens", 0),
        "finish_reason":   call.get("finish_reason"),
        "raw_response":    call.get("raw_response"),
    }


def _actor_metadata(calls):
    """Provider metadata for the actor call on this attempt (non-agentic path).
    Returns finish_reason, raw_response, and reasoning-related fields as a dict
    ready to merge into actor_section. If no actor call was recorded (shouldn't
    happen on the non-agentic path), returns an empty dict — actor_section keeps
    the pre-change schema.
    """
    for c in calls:
        if c["phase"] == "actor":
            return {
                "finish_reason":    c.get("finish_reason"),
                "raw_response":     c.get("raw_response"),
                "reasoning_effort": c.get("reasoning_effort"),
                "thinking_content": c.get("thinking_content"),
            }
    return {}


def _actor_tokens(calls, result, owns_attempt):
    """Provider-reported token counts for the actor's LLM call(s) this attempt.

    Non-agentic benchmarks: one llm.complete tagged phase="actor"; just read it.
    Agentic benchmarks (terminalbench): the agent runs inside Harbor/Docker, so
    our llm.record() never sees its calls. Token usage comes from the verifier
    result's details (Harbor reports aggregate counts across all agent steps;
    thinking_tokens is 0 because Harbor doesn't expose it).
    """
    keys = ("input_tokens", "cached_tokens", "thinking_tokens", "output_tokens")
    if owns_attempt:
        usage = (result.details or {}).get("actor_token_usage") or {}
        return {k: int(usage.get(k, 0)) for k in keys}
    for c in calls:
        if c["phase"] == "actor":
            return {k: int(c.get(k, 0)) for k in keys}
    return {k: 0 for k in keys}


def _round_actor_output_tokens(calls):
    """Output tokens the actor produced on THIS attempt (standard non-agentic path).
    Read straight after the actor's llm.complete, before judge/critic run — so the
    only recorded call is the actor's — but filter by phase to be safe."""
    return sum(int(c.get("output_tokens", 0)) for c in calls if c["phase"] == "actor")


def build_prompt(task, history, t, k, *, seq, remaining_budget=None):
    parts = [task.prompt]
    if seq:
        # "attempt t of K" on every attempt, including the first — this is what
        # makes seq@1 differ from pass@1.
        note = f"This is attempt {t + 1} of {k}."
        if history:
            note += " Review your previous attempt(s) and the feedback below, then provide an improved answer."
        else:
            note += " If this attempt does not pass, you will receive feedback and can revise it on the remaining attempts."
        # Spatial analog of horizon awareness: tell the actor how much of its
        # output-token budget is left for the whole task. Going over fails the task.
        if remaining_budget is not None:
            note += (f" You have about {max(0, remaining_budget)} output tokens left for this entire "
                     f"task across all your remaining attempts. If your cumulative output exceeds this "
                     f"budget the task is failed immediately, so budget your response length accordingly.")
        parts.append(note)
        # Numbered tags so the agent can disambiguate "attempt 1" vs "attempt 2" etc.
        for i, (past_attempt, past_feedback) in enumerate(history, 1):
            parts.append(f"<PreviousAttempt {i}>\n{past_attempt.output}\n</PreviousAttempt {i}>")
            if past_feedback:
                parts.append(f"<Feedback {i}>\n{past_feedback}\n</Feedback {i}>")
    return "\n\n".join(parts)


def _step_from_saved(a):
    return Step(
        attempt_index=a["attempt_index"],
        actor=a["actor"],
        judge=a["judge"],
        critic=a["critic"],
    )
