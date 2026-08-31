"""Run loop, one metric per run.

pass@k: every attempt sees only the task prompt, no feedback. seq@k: every attempt
also gets an "attempt t of K" note (from the first) plus prior attempts and their
feedback — so seq@1 != pass@1.

Run identity: core.ids.fingerprint hashes every field that changes results, and
core.registry maps that fingerprint to a `runs/<benchmark>/<run_id>/` folder.
Same config → same fingerprint → same folder → re-running auto-resumes. A
differing field (k, temperature, output_budget, …) is a different fingerprint and
therefore a different folder, so incompatible attempts can no longer be mixed.
`runs/by-label/` carries the readable v2 names as symlinks. To start fresh:
change any field, or `rm -rf` the run dir and drop it from runs/.registry.json.

Each attempt is written to its own task-N/attempt-M.json file, so a crash
only loses the in-flight attempt.
"""

from __future__ import annotations

from core import db, ids, llm, prompts, registry, results, rows, s3sync, summarizer
from core.types import Attempt, Step, Trajectory, VerifierResult


def run(benchmark, *, metric, k, feedback_mode, model, judge_model=None, critic_model=None,
        temperature=0.7, max_tasks=None, runs_root="runs",
        console_char_limit=3000, options=None, s3_sync=None, task_indices=None,
        continue_run=False, output_budget=None, reasoning_effort=None,
        summarize=False, summarizer_model=None, seed=None,
        context=None, prompt_variant=prompts.DEFAULT_VARIANT):
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
    # `context` is WHAT the agent gets on a retry — nothing, the verbatim prior
    # attempts, or self-written summaries of them. Three values, content only.
    # HOW it is worded is `prompt_variant` (core/prompts.py). The legacy
    # `summarize` flag still works; `context` wins if both are given.
    if context is not None:
        summarize = results.summarizes(context)
        if metric == "pass@k" and context != "na":
            raise ValueError(f"pass@k has no retry context; use context=na, got {context!r}")
    else:
        context = results.context_from_legacy(metric, summarize)

    # prompt_variant selects the template, including whether the retry note names
    # the horizon. Unknown names fail loudly rather than silently defaulting.
    spec = prompts.spec(prompt_variant)
    if not (spec.horizon and spec.retry_frame) and metric != "seq@k":
        raise ValueError(f"prompt_variant {prompt_variant!r} alters the RETRY framing, "
                         f"which only exists for seq@k; got metric={metric!r}")
    if not (spec.horizon and spec.retry_frame) and hasattr(benchmark, "run_attempt"):
        raise ValueError(
            f"prompt_variant {prompt_variant!r} not supported for agentic benchmark "
            f"{benchmark.__name__}: the retry framing is built inside "
            f"benchmark.run_attempt, not build_prompt")
    # Self-summarization compresses prior attempts for the NEXT attempt's prompt,
    # so it only means anything when there IS a next attempt carrying history.
    # pass@k attempts are independent and see no history at all.
    if summarize and metric != "seq@k":
        raise ValueError(f"summarize applies to seq@k only, got metric={metric!r}")
    # Each role defaults to the actor model when unset (judge and critic both fall
    # back to `model`, not to each other). Each gets its own field in the saved
    # JSON; mix-and-match by setting any of them in the YAML.
    judge_model = judge_model or model
    critic_model = critic_model or model
    # Same default, but for a different reason: the agent summarizes its own attempt
    # for its own future self, so the summarizer IS the actor unless overridden.
    summarizer_model = summarizer_model or model
    options = options or {}

    # Identity, not location. The fingerprint over every result-affecting field
    # is what decides whether this is a NEW run or a RESUME of an existing one;
    # the directory is just where the registry happens to have put it.
    def _identity_for(seed_value):
        c = ids.candidates(
            benchmark_module=benchmark, options=options, metric=metric, k=k, model=model,
            judge_model=judge_model, critic_model=critic_model, feedback_mode=feedback_mode,
            context=context, prompt_variant=prompt_variant, temperature=temperature,
            seed=seed_value, reasoning_effort=reasoning_effort, output_budget=output_budget,
            summarizer_model=summarizer_model,
        )
        lp = results.build_run_path(
            runs_root="", benchmark_module=benchmark, options=options,
            metric=metric, model=model, judge_model=judge_model,
            critic_model=critic_model, feedback_mode=feedback_mode,
            k=k, context=context, prompt_variant=prompt_variant,
            temperature=temperature, seed=seed_value, reasoning_effort=reasoning_effort,
        )
        return c, c[0][2], lp

    cands, ident, label_path = _identity_for(seed)

    # AUTO-SEED. At a non-zero temperature a run is a SAMPLE, and `seed` is the
    # field that lets a second sample of one config exist beside the first —
    # it is part of the fingerprint, so two seeds hash differently, both stay
    # live, and their draws pool. Uniqueness on the fingerprint is total (there
    # is no supersede), so an unseeded config has exactly one slot; seeding from
    # the start is what keeps the second run possible at all.
    #
    # ONLY when this would be a NEW run. If the config already has a run — every
    # pre-existing unseeded run included — resolve it unchanged and resume. The
    # alternative forks all of them: their variant YAMLs would hash to a seed=1
    # fingerprint nothing owns, start fresh, and pay for finished attempts twice.
    if seed is None and temperature != 0 and not registry.exists(runs_root, cands):
        seed = 1
        cands, ident, label_path = _identity_for(seed)
        print(f"no seed given at temperature={temperature}; assigning seed={seed}. "
              f"For an independent replicate of this config, pass seed: 2.")

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

    out, manifest, created = registry.resolve(
        runs_root, cands, labels=_labels(ident, k), options=options,
        code=results.code_provenance(), label_path=label_path, k_target=k,
        continue_run=continue_run,
    )
    run_id = manifest["run_id"]
    results.init_run(
        out, continue_run=continue_run,
        benchmark=benchmark.__name__, metric=metric, k=k,
        feedback_mode=feedback_mode, model=model, judge_model=judge_model,
        critic_model=critic_model, temperature=temperature, options=options,
        output_budget=output_budget, reasoning_effort=reasoning_effort,
        context=context, prompt_variant=prompt_variant, seed=seed,
        summarizer_model=summarizer_model if summarize else None,
        run_id=run_id, fingerprint=manifest["fingerprint"],
    )
    db.upsert_run(manifest, run_path=out)

    print(f"Loaded {len(tasks)} tasks | benchmark={benchmark.__name__} | metric={metric} "
          f"| k={k} | actor={model} | judge={judge_model} | critic={critic_model} | feedback={feedback_mode}"
          + (f" | summarizer={summarizer_model}" if summarize else "")
          + f" | context={context} | prompt={prompt_variant}"
          + (f" | seed={seed}" if seed is not None else ""))
    print(f"Run path: {out}/   ({'new' if created else 'resuming'} | run_id={run_id})")
    print(f"Label:    {runs_root}/{registry.BY_LABEL}/{label_path}/")

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
                            reasoning_effort=reasoning_effort, summarize=summarize,
                            summarizer_model=summarizer_model, prompt_variant=prompt_variant)
        finally:
            results.save_summary(out, k=k)
            # Mirror this task into the DB. Batched per task, never per call, so
            # Neon latency stays out of the inner loop; failures spool to
            # <run>/.db_pending.jsonl and never interrupt the run.
            rows.mirror_task(run_id, out, task.canonical_index, ident=ident, k=k, seq=seq,
                             task_id=task.id, prompt=task.prompt,
                             storage_key=manifest["storage_key"])
        print(f"--> task-{task.canonical_index} {task.id}: success={traj.success} best_score={traj.best_score}")

    print(f"\nDone. {len(tasks)} tasks -> {out}/")
    _finalize(out, run_id, k=k, seq=seq)
    s3sync.upload_run(out, s3_sync=s3_sync)


def _labels(ident, k):
    """The denormalised view stored in the manifest, so S3 is self-describing
    and the database can be rebuilt from the bucket alone. `k` is passed
    separately because it is not always part of identity (see core/ids.py)."""
    return {"slice": ident["slice_key"], "metric": ident["metric"], "k": k,
            "agent": ident["model"], "judge": ident["judge_model"],
            "fb": ident["feedback_mode"], "critic": ident["critic_model"],
            "context": ident["context"], "prompt": ident["prompt_variant"],
            "temp": ident["temperature"], "seed": ident["seed"],
            "reason": ident["reasoning_effort"]}


def _finalize(out, run_id, *, k, seq):
    """Stamp the terminal status onto the manifest, then the DB.

    Manifest first: it is the source of truth and must be correct even if the
    database write fails. The rollup is written to the MANIFEST only — S3 has to
    be self-describing — while Postgres derives the same numbers from the
    `run_summary` view, so there is no cached copy there to go stale.
    """
    rollup = rows.run_rollup(out, k=k, seq=seq)
    status = "complete" if rollup["tasks_total"] and not rollup["tasks_partial"] else "partial"
    registry.update_manifest(out, status=status, finished_at=ids.iso(ids.utc_now()),
                             rollup=rollup)
    db.finish_run(run_id, status=status, finished_at=ids.iso(ids.utc_now()), run_path=out)
    print(f"     {status}: {rollup['tasks_done']}/{rollup['tasks_total']} tasks done, "
          f"{rollup['tasks_success']} solved, {rollup['attempts_total']} attempts, "
          f"${rollup['cost_usd']:.4f}")


def run_task(benchmark, task, *, prior, metric, k, feedback_mode, model, judge_model, critic_model,
             temperature, console_char_limit, options=None, out=None, output_budget=None,
             reasoning_effort=None, summarize=False, summarizer_model=None,
             prompt_variant=prompts.DEFAULT_VARIANT):
    seq = metric == "seq@k"
    options = options or {}
    # Agentic benchmarks (e.g. TerminalBench) own their attempt: they build their own
    # prompt, run it in an environment, and verify it. Everything else uses the
    # standard llm.complete + verify path below.
    owns_attempt = hasattr(benchmark, "run_attempt")

    steps = [_step_from_saved(a) for a in prior]
    # One record per prior attempt: the verbatim output, the feedback it drew, and
    # (summarize runs only) the summary that stands in for BOTH in the next prompt.
    # `.get("summarizer")` keeps this readable on attempt files written before the
    # summarizer existed — they simply have no such key.
    history = [{"attempt": Attempt(a["attempt_index"], a["actor"]["output"]),
                "feedback": a["critic"]["feedback"],
                "summary": (a.get("summarizer") or {}).get("summary")}
               for a in prior] if seq else []
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
                prompt = build_prompt(task, history, t, k, seq=seq, remaining_budget=remaining,
                                      prompt_variant=prompt_variant)
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

            # Summarizer runs LAST — it compresses this attempt's output together
            # with the feedback the critic just produced, so it has to see both.
            # Same gating as the critic (failed, not over budget, seq@k), and for
            # the same reason: the summary only matters if another attempt can use
            # it, and running it on the final attempt too means a later k-extension
            # resumes with an unbroken summary log.
            summary = summary_error = None
            if summarize and seq and not result.success and not over_budget:
                sum_task, sum_output = _summarizer_inputs(benchmark, task, output, result)
                # Deliberate exception to this repo's fail-loud rule, and the ONLY
                # place it applies. By this point the attempt is finished and scored
                # — for an agentic benchmark that's minutes of Docker and real money
                # — but it isn't on disk yet (save_attempt is below). Letting a
                # summarizer failure propagate would throw all of that away to lose
                # an auxiliary field. So we degrade instead: summary stays None,
                # summarizer.render_history / terminalbench._retry_context both fall
                # back to the verbatim attempt, and the reason is recorded in
                # summarizer.error so a half-summarized run is visible in the data
                # rather than silent.
                with llm.phase("summarizer"):
                    try:
                        summary = summarizer.summarize(
                            summarizer_model, task_prompt=sum_task, output=sum_output,
                            feedback=fb,
                            template=getattr(benchmark, "SUMMARIZER_PROMPT", None))
                    except Exception as exc:
                        summary_error = f"{type(exc).__name__}: {exc}"
                        print(f"⚠ summarizer failed on attempt {t + 1} "
                              f"(attempt kept, history falls back to verbatim): {summary_error}")

        # Group every recorded LLM call by role into its own section dict.
        judge_calls = [_strip_phase(c) for c in calls if c["phase"] == "judge"]
        critic_calls = [_strip_phase(c) for c in calls if c["phase"] == "critic"]
        summarizer_calls = [_strip_phase(c) for c in calls if c["phase"] == "summarizer"]
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
        # The summarizer section exists only on summarize-on runs (flag off → the key
        # is dropped in results.save_attempt, preserving the pre-feature schema).
        # Within such a run it's always present, with nulls when the summarizer was
        # skipped (attempt succeeded, or went over budget), so every attempt file in
        # the run has the same shape.
        summarizer_section = None
        if summarize:
            summarizer_section = {"model": summarizer_model if summary is not None else None,
                                  "summary": summary, "calls": summarizer_calls}
            # Only present when the summarizer raised — absent on healthy attempts.
            if summary_error:
                summarizer_section["error"] = summary_error
        step = Step(
            attempt_index=t + 1,
            actor=actor_section,
            judge={"model": judge_model_saved, "success": result.success, "score": result.score,
                   "raw_eval_output": result.raw_eval_output, "details": result.details,
                   "calls": judge_calls},
            critic={"model": critic_model_saved, "feedback": fb, "calls": critic_calls},
            summarizer=summarizer_section,
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
        history.append({"attempt": attempt, "feedback": fb, "summary": summary})

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


def _summarizer_inputs(benchmark, task, output, result):
    """(task_prompt, output) for the summarizer — what the task WAS, and what to compress.

    Defaults to (task.prompt, actor.output), correct whenever the actor's prompt is
    the task and its output is what the next prompt would carry. Agentic benchmarks
    can differ on BOTH: TerminalBench's task.prompt is only scaffolding (Harbor
    injects the real instruction inside the sandbox) and its retry context carries
    the full multi-step trajectory rather than the final message. Such a benchmark
    exposes `summarizer_inputs(task, output, result) -> {"task_prompt", "output"}`;
    either key may be omitted to keep the default.
    """
    hook = getattr(benchmark, "summarizer_inputs", None)
    if callable(hook):
        got = hook(task, output, result) or {}
        return (got.get("task_prompt") or task.prompt, got.get("output") or output)
    return task.prompt, output


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
    Read straight after the actor's llm.complete, before judge/critic/summarizer run
    — so the only recorded call is the actor's — but filter by phase to be safe.

    The phase filter is also what keeps SUMMARIZER output out of output_budget: the
    summary is harness overhead, not answer effort, and it's short by construction,
    so it never eats the actor's token allowance. It is still counted for cost —
    see results._tokens_across_attempts."""
    return sum(int(c.get("output_tokens", 0)) for c in calls if c["phase"] == "actor")




def build_prompt(task, history, t, k, *, seq, remaining_budget=None,
                 prompt_variant=prompts.DEFAULT_VARIANT):
    """Compose the actor prompt.

    The retry framing — the thing that makes seq@1 differ from pass@1 — comes
    from the TEMPLATE, selected by `prompt_variant` (see core/prompts.py). What
    history is shown is `context`, and the caller has already resolved that into
    the `history` argument.
    """
    spec = prompts.spec(prompt_variant)
    parts = [task.prompt]
    if seq:
        # The horizon count goes on every attempt including the first — that is
        # precisely what makes seq@1 != pass@1 under a horizon-bearing variant.
        note = prompts.retry_note(spec, t=t, k=k, has_history=bool(history))
        # Spatial analog of horizon awareness: tell the actor how much of its
        # output-token budget is left for the whole task. Going over fails the task.
        # Emitted even by the no-frame variant — the budget is a hard rule the
        # actor is scored against, not retry framing, so suppressing it would
        # change the task rather than the ablation.
        if remaining_budget is not None:
            note = (note + " " if note else "") + (
                f"You have about {max(0, remaining_budget)} output tokens left for this entire "
                f"task across all your remaining attempts. If your cumulative output exceeds this "
                f"budget the task is failed immediately, so budget your response length accordingly.")
        if note:
            parts.append(note)
        # Prior attempts are shown by every variant — the ablation removes
        # the FRAMING, not the history. Numbered tags so the actor can disambiguate
        # "attempt 1" vs "attempt 2"; on summarize runs each pair collapses to a
        # single <AttemptSummary i> (core.summarizer.render_history owns that).
        parts.extend(summarizer.render_history(history))
    return "\n\n".join(parts)


def _step_from_saved(a):
    return Step(
        attempt_index=a["attempt_index"],
        actor=a["actor"],
        judge=a["judge"],
        critic=a["critic"],
        summarizer=a.get("summarizer"),   # absent on pre-feature / summarize-off files
    )
