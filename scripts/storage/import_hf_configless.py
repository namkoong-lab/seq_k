"""Import the HuggingFace runs that have no `events.jsonl`.

    python scripts/storage/import_hf_configless.py                 # dry run
    python scripts/storage/import_hf_configless.py --apply
    python scripts/storage/import_hf_configless.py --apply --only clbench_dkr

HOW IT FINDS THE CONFIG. An earlier importer read the run config from the first
line of `events.jsonl` and skipped any run without one — 93 of the 138 run
directories in the bucket. Those were parked as un-importable because the path
omits k, judge and context.

That was wrong. The `seqk.attempt.v2` record carries the config on EVERY
attempt, including the field that importer refused to guess:

    verifier.evaluator_input.judge_model      the judge, recorded not inferred
    metric_mode                               seqk / passk
    feedback_type                             the channel
    agent_name, temperature, seed
    actor.history_was_summarized              context: full vs summary
    additional_info.api_raw_request.reasoning_effort
    task_metadata.context_category            CL-bench DKR vs RSA, per task
    dataset.dataset_key / dataset_family

`k` comes from `trajectories_metrics.json` when present — it reports seq@1..seq@N
and the highest N is the run's horizon. Deriving k from the largest observed
attempt_index instead would UNDERSTATE it whenever every task happened to
succeed early, so that path is a fallback and is recorded as such.

DEPRECATED ATTEMPTS ARE SKIPPED. The collaborators' own QA marks bad attempts
in place:

    additional_info.deprecated = true
    additional_info.deprecation_reason = "significant QA issue(s): ..."

Measured over a 271-attempt sample: 4.4% of attempts, touching 5 of 91 runs.
Reasons range from harmless (`attempt_json_unexpected_extra`) to disqualifying
(`actor_output_empty, task_error_present`, `prior_feedback_missing_from_prompt`).
We do not grade them — any marker means skip, and the count lands in the
manifest under `code.deprecated_skipped` so a run that lost attempts is never
mistaken for one that never had them.

A task whose attempts are ALL skipped is not written at all, and a run left with
no tasks is reported and not imported.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core import ids, registry, results  # noqa: E402

BUCKET = "namkoong-lab/seq-k"

def convert_attempt(v2, *, task_index, metric, feedback_mode, model, judge_model,
                    critic_model, reasoning_effort):
    """One seqk.attempt.v2 record -> this repo's attempt schema.

    Token counts come from `additional_info.metadata`, which mirrors the
    provider's usage block and also carries `cost` — so imported calls keep
    provider-reported pricing instead of falling back to rate tables.
    """
    ai = v2.get("additional_info") or {}
    md = ai.get("metadata") or {}
    actor_src = v2.get("actor") or {}
    ver = v2.get("verifier") or {}
    fbp = v2.get("feedback_provider") or {}

    # judge.score has to mean what the native benchmark writes: the SOFT score
    # for HealthBench (normalized_score) and ResearchRubrics (compliance_score),
    # 0/1 elsewhere. Reading the binary `correctness` annotation first is what
    # stored 0.0 for an attempt whose own raw_output says 0.4889, and it is why
    # the seed-42 HealthBench/ResearchRubrics runs cannot reproduce Tables 7-8
    # from the database. The value was never lost — it is in the file.
    raw_out = ver.get("raw_output") if isinstance(ver.get("raw_output"), dict) else {}
    score = next((raw_out[k] for k in ("normalized_score", "compliance_score")
                  if isinstance(raw_out.get(k), (int, float))), None)
    if score is None:
        for an in v2.get("annotations") or []:
            if an.get("name") == "judge_score" and isinstance(an.get("score"), (int, float)):
                score = an["score"]
                break
    if score is None:
        for an in v2.get("annotations") or []:
            if an.get("name") == "correctness" or an.get("evaluator_key") == "correctness":
                score = an.get("score")
                break
    if score is None:
        score = 1.0 if ver.get("success") else 0.0

    actor = {
        "model": model,
        "prompt": actor_src.get("input_text") or "",
        "output": actor_src.get("raw_text_output") or "",
        "input_tokens": int(md.get("prompt_tokens") or 0),
        "cached_tokens": int(md.get("cached_tokens") or 0),
        "thinking_tokens": int(md.get("reasoning_tokens") or 0),
        "output_tokens": int(md.get("completion_tokens") or 0),
        "finish_reason": md.get("finish_reason"),
        "raw_response": ai.get("api_raw_response"),
        "reasoning_effort": reasoning_effort,
        "thinking_content": actor_src.get("reasoning_content") or "",
    }
    # The provider's own charge lives in metadata.cost here rather than in
    # usage.cost; surface it where core/rows.py looks for it.
    if md.get("cost") is not None:
        rr = dict(actor["raw_response"] or {})
        usage = dict(rr.get("usage") or {})
        usage.setdefault("cost", md["cost"])
        rr["usage"] = usage
        actor["raw_response"] = rr

    judge = {
        "model": judge_model,
        "success": bool(ver.get("success")),
        "score": float(score),
        "raw_eval_output": fbp.get("raw_eval_output") or fbp.get("feedback") or "",
        "details": {k: ver.get(k) for k in ("evaluator_kind", "evaluator_input",
                                            "raw_output", "criteria_or_groundtruth")
                    if ver.get(k) is not None},
        # v2 recorded per-call judge accounting only for CL-bench
        # (metadata.judge_*). Elsewhere an empty list is the honest answer and
        # keeps the judge's cost out of the totals rather than inventing one.
        "calls": _judge_calls_v2(md, raw_out, judge_model),
    }
    # WHO wrote the feedback. The old runs recorded it per mode in
    # feedback_modes_metadata — CL-bench socratic/directive commonly ran with
    # `feedback_llm_model: openrouter/openai/gpt-5.2` overriding the actor — and
    # hard-coding the actor here put a model in `runs.critic_model` that never
    # wrote a word of that feedback. `critic_model` is part of the identity
    # fingerprint, so this mislabelled the experiment, not just a display field.
    fmeta = ((fbp.get("feedback_modes_metadata") or {}).get(fbp.get("selected_mode") or feedback_mode)
             or {})
    critic = {
        "model": fmeta.get("feedback_llm_model") or critic_model,
        "feedback": fbp.get("feedback_detail") or fbp.get("feedback") or "",
        # Same story as the judge: the cost and token counts of the feedback call
        # are recorded in that metadata block, so importing them keeps the
        # critic's spend in the run's totals instead of silently dropping it.
        "calls": ([{"model": fmeta.get("feedback_llm_model") or critic_model,
                    "prompt": fmeta.get("rendered_prompt") or "",
                    "output": fmeta.get("full_feedback_text") or "",
                    "input_tokens": int(fmeta.get("prompt_tokens") or 0), "cached_tokens": 0,
                    "thinking_tokens": 0, "output_tokens": int(fmeta.get("completion_tokens") or 0),
                    "finish_reason": None,
                    "raw_response": _usage_only(fmeta.get("cost_usd"), fmeta.get("prompt_tokens"),
                                                fmeta.get("completion_tokens"),
                                                fmeta.get("total_tokens"))}]
                  if fmeta.get("cost_usd") is not None or fmeta.get("prompt_tokens") else []),
    }
    out = {
        "task_id": v2.get("task_id"),
        "task_index": task_index,
        "metric": metric,
        "feedback_mode": feedback_mode,
        "attempt_index": int(v2.get("attempt_index") or 0) + 1,   # v2 is 0-based
        "actor": actor,
        "judge": judge,
        "critic": critic,
    }
    if actor_src.get("history_was_summarized"):
        out["summarizer"] = {"model": model, "summary": "", "calls": []}
    return out


def _feedback_writer(v2):
    """The model that wrote this attempt's feedback, per its own record, or None."""
    fbp = v2.get("feedback_provider") or {}
    mode = fbp.get("selected_mode") or v2.get("feedback_type")
    meta = (fbp.get("feedback_modes_metadata") or {}).get(mode) or {}
    return meta.get("feedback_llm_model")


def _judge_calls_v2(md, raw_out, judge_model):
    """The one judge call v2 accounted for (CL-bench: metadata.judge_*), or [].

    Only CL-bench recorded judge tokens and cost; everywhere else v2 is silent
    and an empty list stays the honest answer."""
    if md.get("judge_cost_usd") is None and not md.get("judge_prompt_tokens"):
        return []
    usage = md.get("judge_usage") if isinstance(md.get("judge_usage"), dict) else None
    return [{"model": judge_model, "prompt": "", "output": raw_out.get("judge_raw_output") or "",
             "input_tokens": int(md.get("judge_prompt_tokens") or 0), "cached_tokens": 0,
             "thinking_tokens": 0, "output_tokens": int(md.get("judge_completion_tokens") or 0),
             "finish_reason": None,
             "raw_response": ({"usage": usage} if usage and usage.get("cost") is not None else
                              _usage_only(md.get("judge_cost_usd"), md.get("judge_prompt_tokens"),
                                          md.get("judge_completion_tokens"),
                                          md.get("judge_total_tokens")))}]


def _usage_only(cost, prompt_tokens, completion_tokens, total_tokens):
    """A recorded usage block, placed where core/rows.py reads a call's charge.

    v2 kept no raw response for judge or critic calls, only their usage, so the
    call's `raw_response` is that usage alone. rows._cost reads
    raw_response.usage.cost; a cost stored anywhere else is ignored and the call
    is priced from the rate table instead."""
    if cost is None:
        return None
    usage = {"cost": cost}
    for k, v in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens),
                 ("total_tokens", total_tokens)):
        if v is not None:
            usage[k] = v
    return {"usage": usage}


class TaskIndexer:
    """Stable task_id -> task_index per slice.

    `tasks` is UNIQUE on both (slice_key, task_id) and (slice_key, task_index),
    so two runs must never disagree about a task's number. Existing tasks keep
    the index they already have; new ones continue from the slice's high-water
    mark in a deterministic (sorted) order.
    """

    def __init__(self, runs_root):
        self._by_slice = {}
        for _p, m in registry.iter_manifests(runs_root):
            sl = m["config"]["slice_key"]
            d = self._by_slice.setdefault(sl, {})
            for tdir in sorted(Path(_p).glob("task-*/task_meta.json")):
                try:
                    tm = json.loads(tdir.read_text(encoding="utf-8"))
                    d[tm["task_id"]] = int(tm["task_index"])
                except (ValueError, OSError, KeyError):
                    pass

    def index_for(self, slice_key, task_id):
        d = self._by_slice.setdefault(slice_key, {})
        if task_id not in d:
            d[task_id] = (max(d.values()) + 1) if d else 1
        return d[task_id]


# dataset_key in the artifacts -> (module, options). RSA is a real CL-bench
# category, so it gets the option that makes slice_name() say `clbench-rsa`.
DATASETS = {
    "advancedif":             ("benchmarks.advancedif", {}),
    "arcagi2":                ("benchmarks.arcagi2", {}),
    "clbench":                ("benchmarks.clbench", {}),
    "clbench_dkr":            ("benchmarks.clbench", {"category": "Domain Knowledge Reasoning"}),
    "clbench_rsa":            ("benchmarks.clbench", {"category": "Rule System Application"}),
    "healthbench":            ("benchmarks.healthbench", {}),
    "researchrubrics":        ("benchmarks.researchrubrics", {}),
    "terminalbench":          ("benchmarks.terminalbench", {}),
    "terminalbench_selected": ("benchmarks.terminalbench", {}),
}
_SEED = re.compile(r"/seed=(\d+)")
_HORIZON = re.compile(r"This is attempt \d+ of \d+")
# Proof that a prompt actually carries the prior-attempt history, and therefore
# that a missing horizon sentence is evidence rather than an empty stub.
_HISTORY = re.compile(r"<TRIAL_\d|<PreviousAttempt|<AttemptSummary|<Feedback",
                      re.IGNORECASE)
_METRIC_KEY = re.compile(r"^(seq|pass)@(\d+)$")
_TASKDIR = re.compile(r"/(task[_-][^/]+)/")


def task_id_of(v2, path):
    """The attempt's task, from the record or else from its `task_*` directory.

    Some records carry no top-level `task_id`. Do NOT fall back to the file path
    — it is unique per attempt, so every attempt becomes its own task (a 30-task
    run lands as 105 one-attempt "tasks"). The directory is the right grouping.
    """
    tid = v2.get("task_id")
    if tid:
        return tid
    m = _TASKDIR.search(path or "")
    return m.group(1) if m else None


def is_deprecated(v2):
    ai = v2.get("additional_info") or {}
    return bool(ai.get("deprecated") or ai.get("deprecation_label"))


def deprecation_reason(v2):
    return ((v2.get("additional_info") or {}).get("deprecation_reason")) or "unspecified"


def judge_of(v2):
    return ((v2.get("verifier") or {}).get("evaluator_input") or {}).get("judge_model")


def split_by_judge(kept):
    """[(judge, [(path, attempt), ...])] — one group per judge, or an error.

    Some runs grade different tasks with different judges (28 on gpt-5.2, 2 on
    gpt-5.4). `judge_model` is an identity field, so one run per judge is the
    honest split. Only possible because the judge is constant within each task;
    a task straddling two judges refuses.
    """
    by_task = collections.defaultdict(set)
    for path, v2 in kept:
        by_task[task_id_of(v2, path)].add(judge_of(v2))
    straddling = [t for t, js in by_task.items() if len(js) > 1]
    if straddling:
        return None, (f"{len(straddling)} task(s) graded by more than one judge, "
                      "so the run cannot be split cleanly")
    groups = collections.defaultdict(list)
    for path, v2 in kept:
        groups[judge_of(v2)].append((path, v2))
    return sorted(groups.items(), key=lambda kv: str(kv[0])), None


def k_from_metrics(metrics):
    """Highest N among seq@N / pass@N keys — the horizon the run was run to."""
    if not metrics:
        return None
    best = 0
    for key in metrics:
        m = _METRIC_KEY.match(str(key))
        if m:
            best = max(best, int(m.group(2)))
    return best or None


def config_from_attempts(prefix, attempts, metrics):
    """(identity_kwargs, extras, error) inferred from the artifacts themselves."""
    if not attempts:
        return None, None, "no readable attempts"
    a0 = attempts[0]

    notes = []
    ds = (a0.get("dataset") or {}).get("dataset_key") or (a0.get("dataset") or {}).get("dataset_family")
    if not ds:
        # A few runs record no dataset at all. The bucket lays runs out as
        # seq-k/<benchmark>/<feedback>_v0/..., so segment 1 names it. This is the
        # ONE place the path is allowed to stand in for content, it is used only
        # when the artifacts say nothing, and it is recorded as an assumption.
        seg = prefix.split("/")
        guess = seg[1] if len(seg) > 1 else None
        if guess in DATASETS:
            ds = guess
            notes.append(f"attempts record no dataset; took {guess!r} from the source path")
    entry = DATASETS.get(ds)
    if not entry:
        return None, None, f"no benchmark module here for dataset {ds!r}"
    mod_name, options = entry
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:                          # noqa: BLE001
        return None, None, f"{mod_name} failed to import ({type(exc).__name__})"
    options = dict(options)

    # CL-bench records the category per task. Trust that over the dataset key,
    # and refuse if a single run mixes categories — task numbering is per
    # category, so a mixed run has no single canonical index space.
    cats = {(a.get("task_metadata") or {}).get("context_category")
            for a in attempts} - {None, ""}
    if cats and mod_name == "benchmarks.clbench":
        if len(cats) > 1:
            return None, None, f"attempts span several CL-bench categories: {sorted(cats)}"
        options["category"] = next(iter(cats))

    metric = {"seqk": "seq@k", "passk": "pass@k"}.get(a0.get("metric_mode"))
    if not metric and metrics:
        # A few runs record no metric_mode. trajectories_metrics.json reports the
        # rate under `seq@N` or `pass@N` keys, so the family names the metric.
        fams = {m.group(1) for key in metrics
                for m in [_METRIC_KEY.match(str(key))] if m}
        if len(fams) == 1:
            metric = {"seq": "seq@k", "pass": "pass@k"}[next(iter(fams))]
            notes.append(f"attempts record no metric_mode; took {metric} from the "
                         "seq@N/pass@N keys in trajectories_metrics.json")
    if not metric:
        return None, None, f"unrecognised metric_mode {a0.get('metric_mode')!r}"

    # The v1 records predate the top-level `agent_name` / `feedback_type` fields,
    # but both facts are still in the artifact: the request records the model it
    # actually called, and the feedback provider records the mode it selected.
    # Content again, not the path.
    model = a0.get("agent_name")
    if not model:
        model = ((a0.get("additional_info") or {}).get("api_raw_request") or {}).get("model")
        if model:
            notes.append("attempts record no agent_name; took the model from "
                         "additional_info.api_raw_request.model (the call as issued)")
    if not model:
        return None, None, "no agent_name and no api_raw_request.model on the attempts"

    feedback_mode = a0.get("feedback_type")
    if not feedback_mode:
        feedback_mode = (a0.get("feedback_provider") or {}).get("selected_mode")
        if feedback_mode:
            notes.append("attempts record no feedback_type; took "
                         "feedback_provider.selected_mode")
    if not feedback_mode:
        return None, None, "no feedback_type and no feedback_provider.selected_mode"

    # The judge, as recorded. This is the whole reason these runs are importable.
    # Callers hand us attempts already grouped by judge (see split_by_judge), so
    # more than one here means the grouping failed and we must not pick a winner.
    judges = {judge_of(a) for a in attempts} - {None, ""}
    verifier = getattr(mod, "VERIFIER", "llm")
    if len(judges) > 1:
        return None, None, f"attempts disagree about judge_model: {sorted(judges)}"
    judge = next(iter(judges), None)
    if not judge and verifier == "llm":
        return None, None, "LLM-graded benchmark but no judge_model recorded on any attempt"

    summarised = any((a.get("actor") or {}).get("history_was_summarized") for a in attempts)
    context = results.context_from_legacy(metric, summarised)

    # Horizon: read it off the prompts, but ONLY where the prompt is actually in
    # the artifact. Two ways that fails, and both would fabricate a template:
    #
    #   * agentic benchmarks build the prompt inside run_attempt, so
    #     `input_text` is a stub ("Harbor will provide the authoritative task
    #     instruction separately inside the sandbox"). Absence of the horizon
    #     phrase there says nothing at all. This is the same property that makes
    #     ids.k_affects_prompt refuse to probe them.
    #   * attempt 1 and every pass@k draw carry no retry note, so they cannot
    #     testify either way.
    #
    # Conclude `-nohorizon` only on POSITIVE evidence: a later attempt whose
    # prompt demonstrably contains the prior-attempt history, yet no horizon
    # sentence. Otherwise keep the default and say why.
    prompt_variant = "legacy"
    if metric == "seq@k":
        if hasattr(mod, "run_attempt"):
            notes.append("agentic benchmark: the prompt is built inside run_attempt and is "
                         "not in the artifact, so the horizon could not be read; "
                         "prompt_variant defaulted to legacy")
        else:
            texts = [((a.get("actor") or {}).get("input_text") or "")
                     for a in attempts if int(a.get("attempt_index") or 0) > 0]
            witnesses = [t for t in texts if _HISTORY.search(t)]
            if not witnesses:
                notes.append("no attempt prompt showed prior-attempt history; the horizon "
                             "could not be read; prompt_variant defaulted to legacy")
            elif not any(_HORIZON.search(t) for t in witnesses):
                prompt_variant = "legacy-nohorizon"

    # What the FILES prove, independent of any summary: attempt_index is 0-based
    # upstream, so a run whose largest index is 4 carries five attempts.
    observed = max(int(a.get("attempt_index") or 0) for a in attempts) + 1

    k = k_from_metrics(metrics)
    if k:
        # This path used to record nothing, so a k inferred from the metrics keys
        # was indistinguishable from one read out of a config — import_notes came
        # back null and there was no way to tell which source had spoken. Every
        # branch here now says where the number came from.
        notes.append(f"k={k} taken from the highest seq@N/pass@N key in "
                     "trajectories_metrics.json")
    if not k:
        mr = {a.get("_max_rounds") for a in attempts} - {None}
        if len(mr) == 1:
            k = int(next(iter(mr)))
            notes.append("k taken from trajectories.jsonl `max_rounds` (the run's horizon)")
    if not k:
        k = observed
        notes.append(f"k={k} derived from the largest observed attempt_index "
                     "(no trajectories_metrics.json); it is a LOWER BOUND")

    if observed > k:
        # The artifacts outrank the summary — but they cannot say which number is
        # the CONFIG. The run may have been launched at k and overrun upstream, or
        # the metrics file may have been written at a smaller horizon than the run
        # actually reached. Rewriting k would assert one of those without evidence
        # and quietly restate the config the run was launched with; leaving this
        # silent is what let two ARC-AGI-2 runs import as k=3 with five attempts on
        # disk. So: keep k, record the conflict, let validate.py surface it.
        notes.append(f"CONFLICT: k={k} from the metrics summary, but the artifacts "
                     f"carry {observed} attempts for at least one task. k was NOT "
                     "rewritten — decide which number is the truth and repair by hand.")

    effort = {((a.get("additional_info") or {}).get("api_raw_request") or {}).get("reasoning_effort")
              for a in attempts} - {None}
    seed_m = _SEED.search(prefix)
    seed = int(seed_m.group(1)) if seed_m else None
    if seed is None:
        seeds = {a.get("seed") for a in attempts} - {None}
        if len(seeds) == 1:
            seed = int(next(iter(seeds)))
    temps = {a.get("temperature") for a in attempts} - {None}

    # WHO wrote the feedback, read from the attempts rather than assumed. The
    # actor was the default writer, but `--feedback-llm-model` overrode it and
    # the run recorded that per attempt; CL-bench socratic/directive runs were
    # commonly written by gpt-5.2 while the actor was something else. Assuming
    # the actor here is how `runs.critic_model` came to name a model that wrote
    # none of the feedback — and critic_model is part of the identity hash.
    writers = {w for w in (_feedback_writer(a) for a in attempts) if w}
    critic = next(iter(writers)) if len(writers) == 1 else model
    if len(writers) > 1:
        notes.append(f"attempts disagree about the feedback writer ({sorted(writers)}); "
                     f"critic_model left as the actor")
    elif writers and critic != model:
        notes.append(f"critic_model={critic} read from feedback_modes_metadata "
                     f"(the run overrode the actor as feedback writer)")

    kw = dict(benchmark_module=mod, options=options, metric=metric, k=int(k), model=model,
              judge_model=judge or verifier, critic_model=critic,
              feedback_mode=feedback_mode, context=context, prompt_variant=prompt_variant,
              temperature=float(next(iter(temps), 0.7)), seed=seed,
              reasoning_effort=next(iter(effort), None), output_budget=None,
              summarizer_model=model)
    extras = {"mod": mod, "metric": metric, "k": int(k), "model": model,
              "judge_model": judge or verifier, "feedback_mode": feedback_mode,
              "context": context, "prompt": prompt_variant, "options": options,
              "reasoning_effort": kw["reasoning_effort"], "notes": notes}
    return kw, extras, None


# --------------------------------------------------------------------------- #
def attempts_from_trajectories(traj_path, prefix):
    """[(pseudo_path, v2_attempt)] from a trajectories.jsonl.

    A handful of runs were uploaded WITHOUT per-attempt files — only this
    rollup. It carries the same records nested one level down: each line is a
    task, with `attempts[]` holding the v2 attempt shape, and the run-level
    facts (`model`, `metric_mode`, `max_rounds`, `dataset`, `run_config`) on the
    line rather than on the attempt. Merge them down so the attempts look
    exactly like the standalone files and every downstream check applies
    unchanged.
    """
    out = []
    for i, line in enumerate(Path(traj_path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        task_id = rec.get("task_id") or f"line-{i}"
        for a in rec.get("attempts") or []:
            v2 = dict(a)
            v2.setdefault("task_id", task_id)
            v2.setdefault("metric_mode", rec.get("metric_mode"))
            v2.setdefault("agent_name", rec.get("model"))
            v2.setdefault("dataset", rec.get("dataset"))
            v2.setdefault("task_metadata", rec.get("task_metadata"))
            cfg = rec.get("run_config") or {}
            v2.setdefault("temperature", (cfg.get("args") or {}).get("temperature"))
            # `max_rounds` is the run's horizon, and unlike the largest observed
            # attempt_index it does not shrink when every task succeeds early.
            v2["_max_rounds"] = rec.get("max_rounds")
            out.append((f"{prefix}/trajectories.jsonl#{task_id}#{a.get('attempt_index')}", v2))
    return out


def survey(listing_path):
    """run prefix -> {'files': [run-level names], 'attempts': [paths], 'bytes': int}"""
    runs = collections.defaultdict(
        lambda: {"files": set(), "attempts": [], "bytes": 0, "mtimes": []})
    for line in Path(listing_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.endswith("/"):
            continue
        cols = line.split()
        p = cols[-1]
        size = int(cols[0]) if cols and cols[0].isdigit() else 0
        stamp = f"{cols[1]}T{cols[2]}Z" if len(cols) >= 4 else None
        parts = p.split("/")
        ti = next((i for i, s in enumerate(parts) if s.startswith("task_") or s.startswith("task-")),
                  None)
        if ti is not None:
            run = "/".join(parts[:ti])
            runs[run]["attempts"].append(p)
            runs[run]["bytes"] += size
            if stamp:
                runs[run].setdefault("mtimes", []).append(stamp)
        else:
            # Record mtimes for RUN-LEVEL files too. Runs whose attempts live only
            # in trajectories.jsonl have no task dirs at all, so a task-only
            # capture leaves them undated and the NOT NULL created_at rejects the
            # whole load.
            run = "/".join(parts[:-1])
            runs[run]["files"].add(parts[-1])
            if stamp:
                runs[run].setdefault("mtimes", []).append(stamp)
    return runs


# `hf buckets sync` transfers a whole prefix in one parallel operation and
# measures ~4.4 MB/s (23 files/s) against this bucket, against ~470 KB/s
# (7 files/s) for the per-file `download_bucket_files` path import_hf.py uses.
# The bottleneck there is per-file round-trip latency, not bandwidth, so
# chunking and timeout tuning cannot close a 9x gap — the transfer has to be
# bulk. Budget conservatively so a slow-but-live sync is never killed as hung.
_BYTES_PER_SEC = 1_000_000


def sync_timeout(total_bytes, floor, cap=2400):
    return int(max(floor, min(cap, total_bytes / _BYTES_PER_SEC)))


def sync_run(prefix, cache, *, timeout_s, tries):
    """Pull one run's whole prefix. Idempotent: already-present files are skipped,
    so an interrupted import resumes instead of starting over."""
    dest = Path(cache) / prefix
    dest.mkdir(parents=True, exist_ok=True)
    src = f"hf://buckets/{BUCKET}/{prefix}"
    for attempt in range(tries):
        try:
            r = subprocess.run(["hf", "buckets", "sync", src, str(dest)],
                               capture_output=True, text=True, timeout=timeout_s)
            if r.returncode == 0:
                return True
            why = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["non-zero exit"]
            why = why[0][:80]
        except subprocess.TimeoutExpired:
            why = f"timed out after {timeout_s}s"
        print(f"    retry {attempt + 1}/{tries} ({why}) — {prefix[6:66]}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--listing", required=True,
                    help="output of `hf buckets ls -R namkoong-lab/seq-k/seq-k`")
    ap.add_argument("--cache", default=".hf-cache", help="where downloaded artifacts land")
    ap.add_argument("--only", help="substring filter on the run prefix")
    ap.add_argument("--all-runs", action="store_true",
                    help="also import runs that DO have events.jsonl. The config is "
                         "inferred from the attempts either way, and the pass-k/ tree "
                         "mixes both kinds, so filtering on events.jsonl there would "
                         "silently drop a third of it.")
    ap.add_argument("--skip-imported", action="store_true",
                    help="ignore source prefixes already recorded in a manifest, so a "
                         "re-run after a partial import does not collide with itself")
    ap.add_argument("--apply", action="store_true")
    # The bucket's CDN stalls connections rather than dropping them, so the
    # watchdog is what makes progress possible at all. Default it LOW: a stalled
    # chunk is far more common than a slow one, and a fresh call usually
    # succeeds immediately, so failing fast and retrying beats waiting. At the
    # inherited 900s a single stall cost 45 minutes across three tries.
    ap.add_argument("--timeout", type=int, default=180,
                    help="seconds before a download chunk is treated as hung (default 180)")
    ap.add_argument("--tries", type=int, default=6,
                    help="attempts per chunk before giving up on the run (default 6)")
    args = ap.parse_args()

    runs = survey(args.listing)
    targets = [r for r, v in runs.items()
               if (args.all_runs or "events.jsonl" not in v["files"])
               and (v["attempts"] or "trajectories.jsonl" in v["files"])
               and (not args.only or args.only in r)]
    if args.skip_imported:
        # Superseded runs do NOT count as imported. Retiring a run frees its
        # identity precisely so it can be replaced; if it also blocked re-import,
        # a bad import could never be corrected from source.
        done = {(m.get("code") or {}).get("imported_from", "").replace(f"hf://{BUCKET}/", "")
                for _p, m in registry.iter_manifests(args.runs_root)
}
        before = len(targets)
        targets = [r for r in targets if r not in done]
        print(f"skipping {before - len(targets)} already-imported run(s)")
    print(f"{len(targets)} config-less run(s) to consider\n")

    cache = Path(args.cache)
    indexer = TaskIndexer(args.runs_root) if args.apply else None
    planned, errors, skipped_total = [], [], 0

    for prefix in sorted(targets):
        att_paths = sorted(runs[prefix]["attempts"])
        metrics_path = None
        if "trajectories_metrics.json" in runs[prefix]["files"]:
            metrics_path = cache / f"{prefix}/trajectories_metrics.json"

        tmo = sync_timeout(runs[prefix]["bytes"], args.timeout)
        if not sync_run(prefix, cache, timeout_s=tmo, tries=args.tries):
            errors.append((prefix, "download failed"))
            continue

        loaded, unreadable = [], 0
        if att_paths:
            for p in att_paths:
                try:
                    loaded.append((p, json.loads((cache / p).read_text(encoding="utf-8"))))
                except (OSError, ValueError):
                    unreadable += 1
        else:
            tj = cache / prefix / "trajectories.jsonl"
            if tj.exists():
                loaded = attempts_from_trajectories(tj, prefix)

        # Track WHICH TASK each dropped attempt belonged to, so that when a run
        # is split by judge the count can follow the task into the right half.
        # Zeroing it on split (the old behaviour) made a run that lost attempts
        # report `deprecated_skipped: 0` — the exact confusion the field exists
        # to prevent.
        kept, dropped, reasons = [], 0, collections.Counter()
        dropped_by_task = collections.Counter()
        reasons_by_task = collections.defaultdict(collections.Counter)
        for p, v2 in loaded:
            if is_deprecated(v2):
                dropped += 1
                why = deprecation_reason(v2)[:60]
                reasons[why] += 1
                tid = task_id_of(v2, p)
                dropped_by_task[tid] += 1
                reasons_by_task[tid][why] += 1
            else:
                kept.append((p, v2))
        skipped_total += dropped

        metrics = None
        if metrics_path and metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metrics = None

        groups, gerr = split_by_judge(kept)
        if gerr:
            errors.append((prefix, gerr + (f" [{dropped} deprecated skipped]" if dropped else "")))
            continue

        for judge, group in groups:
            kw, extras, err = config_from_attempts(prefix, [v for _p, v in group], metrics)
            if err:
                errors.append((prefix + (f"  [judge={judge}]" if len(groups) > 1 else ""),
                               err + (f" [{dropped} deprecated skipped]" if dropped else "")))
                continue
            if len(groups) > 1:
                extras["notes"].append(
                    f"run graded by {len(groups)} judges across disjoint task sets; split "
                    f"into one run per judge. This group: {judge} "
                    f"({len({task_id_of(v, _p) for _p, v in group})} tasks)")
            ident = ids.identity(**kw)
            if len(groups) == 1:
                g_drop, g_reasons = dropped, dict(reasons)
            else:
                # Attribute each dropped attempt to the half holding its task.
                gt = {task_id_of(v, pp) for pp, v in group}
                g_drop = sum(n for t, n in dropped_by_task.items() if t in gt)
                gr = collections.Counter()
                for t in gt:
                    gr.update(reasons_by_task.get(t, {}))
                g_reasons = dict(gr)
            planned.append({"prefix": prefix, "ident": ident, "kw": kw, "extras": extras,
                            "kept": group, "dropped": g_drop, "reasons": g_reasons,
                            "unreadable": unreadable, "judge_split": len(groups) > 1})

    for p in planned:
        i, x = p["ident"], p["extras"]
        # str() every identity field before slicing. pass@k stores feedback_mode
        # and context as NULL by design (core/ids.py: it never calls feedback(),
        # so no channel shaped it), and this line subscripted them directly —
        # so the dry run, whose whole job is to show you what an import WOULD do,
        # crashed with TypeError on the first pass@k run it surveyed.
        line = (f"  {i['slice_key']:20} {i['metric']:6} k={str(i['k']):<5} {i['model']:24} "
                f"{str(i['feedback_mode'])[:16]:16} judge={str(i['judge_model'])[:22]:22} "
                f"ctx={str(i['context']):7} {str(i['prompt_variant']):16} attempts={len(p['kept'])}")
        if p["dropped"]:
            line += f"  -{p['dropped']} deprecated"
        print(line)
        for n in x["notes"]:
            print(f"        note: {n}")
        for r, c in p["reasons"].items():
            print(f"        skipped {c}x: {r}")

    if errors:
        print(f"\n{len(errors)} run(s) NOT importable:")
        for pre, e in errors:
            print(f"  {pre}\n      {e}")

    fps = collections.defaultdict(list)
    for p in planned:
        fps[ids.fingerprint(p["ident"])].append(p["prefix"])
    live = {m["fingerprint"] for _pp, m in registry.iter_manifests(args.runs_root)}
    dupes = {f: v for f, v in fps.items() if len(v) > 1}
    clash = [(f, v) for f, v in fps.items() if f in live]
    if dupes or clash:
        print("\n*** IDENTITY COLLISIONS — NOT APPLYING ***")
        for f, v in dupes.items():
            print(f"  {f[:26]}… shared by {len(v)} of these runs:")
            for x in v:
                print(f"      {x}")
        for f, v in clash:
            print(f"  {f[:26]}… already exists in {args.runs_root}: {v}")
        sys.exit(1)

    print(f"\n{len(planned)} importable · {len(errors)} not · "
          f"{skipped_total} deprecated attempt(s) skipped")
    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    root = Path(args.runs_root)
    for p in planned:
        i, x = p["ident"], p["extras"]
        stamps = sorted(runs[p["prefix"]].get("mtimes") or [])
        run_times = ((stamps[0], stamps[-1]) if stamps else (None, None))
        run_id = ids.new_run_id()
        key = ids.storage_key(i["slice_key"], run_id)
        dest = root / key
        by_task = collections.defaultdict(list)
        for path, v2 in p["kept"]:
            by_task[task_id_of(v2, path)].append(v2)

        n_tasks = 0
        for task_id, atts in sorted(by_task.items()):
            idx = indexer.index_for(i["slice_key"], task_id)
            tdir = dest / f"task-{idx}"
            tdir.mkdir(parents=True, exist_ok=True)
            for v2 in atts:
                conv = convert_attempt(
                    v2, task_index=idx, metric=x["metric"], feedback_mode=x["feedback_mode"],
                    model=x["model"], judge_model=x["judge_model"], critic_model=x["model"],
                    reasoning_effort=x["reasoning_effort"])
                (tdir / f"attempt-{conv['attempt_index']}.json").write_text(
                    json.dumps(conv, indent=1, ensure_ascii=False), encoding="utf-8")
            (tdir / "task_meta.json").write_text(json.dumps(
                {"task_id": task_id, "task_index": idx,
                 "prompt": (atts[0].get("actor") or {}).get("input_text") or ""},
                indent=1, ensure_ascii=False), encoding="utf-8")
            n_tasks += 1

        registry.write_manifest(str(dest), {
            "run_id": run_id,
            "fingerprint": ids.fingerprint(i),
            "fingerprint_version": ids.FINGERPRINT_VERSION,
            # runs.created_at is NOT NULL. The bucket records no run time, so the
            # object mtimes are the closest evidence — UPLOAD times, later than
            # the run by an unknown margin. Recorded as such in code.time_source
            # in code.time_source rather than passed off as a measured time.
            "created_at": run_times[0], "finished_at": run_times[1], "status": "complete",
            "storage_key": key,
            "config": dict(i),
            "labels": {"slice": i["slice_key"], "metric": i["metric"], "k": x["k"],
                       "agent": i["model"], "judge": i["judge_model"],
                       "fb": i["feedback_mode"], "critic": i["critic_model"],
                       "context": i["context"], "prompt": i["prompt_variant"],
                       "temp": i["temperature"], "seed": i["seed"],
                       "reason": i["reasoning_effort"]},
            "options": x["options"],
            "k_target": x["k"],
            "code": {
                "imported_from": f"hf://{BUCKET}/{p['prefix']}",
                "imported_at": ids.iso(ids.utc_now()),
                "converted_from": "seqk.attempt.v2",
                "config_source": "attempt artifacts (no events.jsonl)",
                # So a run that LOST attempts is never mistaken for one that
                # never had them.
                "deprecated_skipped": p["dropped"],
                "deprecated_reasons": p["reasons"] or None,
                "unreadable_attempts": p["unreadable"] or None,
                "import_notes": x["notes"] or None,
            },
            "rollup": {"tasks_total": n_tasks, "attempts_total": len(p["kept"])},
        })
        print(f"  imported {key}  ({n_tasks} tasks, {len(p['kept'])} attempts"
              + (f", {p['dropped']} skipped)" if p["dropped"] else ")"))

    registry.rebuild(args.runs_root, relink=True)
    print(f"\nimported {len(planned)} run(s); {skipped_total} deprecated attempt(s) skipped")


if __name__ == "__main__":
    main()
