"""Run identity: fingerprint, UUID, storage key.

`fingerprint` hashes exactly the fields that change what a run MEANS. Same
fingerprint = same experiment = same directory, which is what makes resume work.

Pure functions only — no I/O, no clock, no randomness except new_run_id().
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone

# Bump when the MEANING of a field changes. The version is part of the hash, so
# a bump changes every fingerprint; `candidates()` emits all versions newest-first
# and registry.py adopts the first already on disk, so old runs still match. That
# fallback is only sound while each bump RELAXES identity — guard it if not.
#   v1  every field always identity
#   v2  `k` is identity only when it reaches the actor's prompt
FINGERPRINT_VERSION = 2

# Every field that changes results. This is the v2 path levels PLUS the two that
# were unsafely left off the path:
#   output_budget  — was policed by a hand-written guard in results.init_run
#   summarizer_model — a summarising run with a different summariser is a
#                      different experiment
IDENTITY_FIELDS = (
    "benchmark",
    "slice_key",
    "metric",
    "k",
    "model",
    "judge_model",
    "critic_model",
    "feedback_mode",
    "context",
    "prompt_variant",
    "temperature",
    "seed",
    "reasoning_effort",
    "output_budget",
    "summarizer_model",
)


def k_affects_prompt(benchmark_module, *, context, metric="seq@k",
                     prompt_variant=None, output_budget=None, probe_k=(5, 10)):
    """Does `k` reach the actor? Renders the real prompt at two k and compares.

    Conservative: a false True costs one duplicated run, a false False MERGES
    two experiments into one directory. Anything unprobeable is assumed to
    depend on k — agentic benchmarks build prompts inside `run_attempt`, and can
    opt in with `prompt_probe(k=..., context=...) -> str`.
    """
    from core import results                       # local: avoids an import cycle
    from core import prompts
    variant = prompt_variant or prompts.DEFAULT_VARIANT
    if getattr(benchmark_module, "prompt_probe", None):
        render = lambda kk: benchmark_module.prompt_probe(
            k=kk, context=context, metric=metric, prompt_variant=variant)
    elif hasattr(benchmark_module, "run_attempt"):
        return True
    else:
        from core import harness
        from core.types import Task
        seq = metric == "seq@k"
        task = Task(id="probe", canonical_index=0, prompt="TASK", grading={})
        # Probe attempt 1 (no history) AND a later attempt (history present):
        # the retry note is worded differently in each, so testing only one could
        # miss a k reference in the other. pass@k needs no special case — with
        # seq=False no note is emitted at all, so it falls out as False.
        hist = [_probe_history(results.summarizes(context))]
        render = lambda kk: "\n".join(
            harness.build_prompt(task, h, i, kk, seq=seq, prompt_variant=variant,
                                 remaining_budget=output_budget)
            for i, h in ((0, []), (1, hist)))
    lo, hi = probe_k
    return render(lo) != render(hi)


def _probe_history(summarize):
    """One synthetic prior attempt, for rendering a probe prompt.

    `summarize` decides whether the entry carries a summary, because
    summarizer.render_history keys off exactly that: with one it emits
    <AttemptSummary>, without it the verbatim <PreviousAttempt> + <Feedback>.
    Getting this wrong would make context=full render as if it were summarised.
    """
    from core.types import Attempt
    entry = {"attempt": Attempt(index=1, output="a1"), "feedback": "f1"}
    if summarize:
        entry["summary"] = "s1"
    return entry


_ROUTE_PREFIXES = ("openrouter/",)
_VENDORS = ("anthropic/", "openai/", "google/", "deepseek/", "qwen/", "meta-llama/",
            "mistralai/", "x-ai/")


def canonical_model(model):
    """Route and vendor stripped, version separators unified:
    `openrouter/anthropic/claude-opus-4.7` and `anthropic/claude-opus-4-7`
    both become `claude-opus-4.7`.

    The route is not identity — price, the thing it affects, is recorded per
    call in `llm_calls`, and the original string is kept in `options.model_raw`.
    Trailing date stamps and unknown vendors pass through untouched.
    """
    if model is None or model == "":
        return None                    # never let a missing model become "None"
    s = str(model)
    for p in _ROUTE_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    for v in _VENDORS:
        if s.startswith(v):
            s = s[len(v):]
            break
    # claude-opus-4-7 -> claude-opus-4.7, but leave gpt-5.3-codex alone. The
    # date guard matters: without it `o3-mini-2025-01-31` would become
    # `o3-mini-2025-01.31`, inventing a version out of a release date.
    if not re.search(r"\d{4}-\d{2}-\d{2}$", s):
        s = re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", s)
    return s


def identity(*, benchmark_module, options, metric, k, model, judge_model, critic_model,
             feedback_mode, context, prompt_variant="v1", temperature=0.7, seed=None,
             reasoning_effort=None, output_budget=None, summarizer_model=None,
             version=FINGERPRINT_VERSION):
    """The canonical identity dict for a run config.

    Two normalisations:
      * `judge_model` collapses to the VERIFIER name for non-LLM graders
        ("harbor", "deterministic") — the model is never consulted.
      * `critic_model` is None unless the feedback mode invokes an LLM critic.

    `slice_key`, not the whole options dict, carries the dataset variant. That is
    what the label path keyed on, and widening it now would change the
    fingerprint of every existing run for cosmetic option keys (`pass_env`
    ordering, say) and silently start them over. Full options are still recorded
    in the manifest.

    A third, from v2: `k` is identity only when it REACHES THE PROMPT. Under a
    horizon-free context ("This is attempt 3 of 10" suppressed) a k=5 and a k=10
    run generate byte-identical prompts, so they are the same experiment — and
    seq@5 is literally the 5-attempt prefix of seq@10, readable off one
    trajectory with results.cumulative_best_by_attempt. Splitting them would pay
    for attempts 1-5 twice and turn a paired comparison into two independent
    samples.
    """
    mod = benchmark_module
    if isinstance(mod, str):
        import importlib
        mod = importlib.import_module(mod)
    verifier = getattr(mod, "VERIFIER", "llm")
    llm_critic_modes = getattr(mod, "LLM_CRITIC_MODES", set())
    # One mechanism, no per-metric special case: pass@k emits no retry note at
    # all, so the probe reports False for it too — pass@5 is the 5-draw prefix
    # of pass@10 for the same reason seq@5 is, when the horizon is not shown.
    k_is_identity = True
    if version >= 2:
        k_is_identity = k_affects_prompt(mod, context=context, metric=metric,
                                         prompt_variant=prompt_variant,
                                         output_budget=output_budget)
    return {
        "benchmark": mod.__name__,
        "slice_key": mod.slice_name(options or {}),
        "metric": metric,
        "k": int(k) if k_is_identity else None,
        "model": canonical_model(model),
        "judge_model": canonical_model(judge_model) if verifier == "llm" else verifier,
        "critic_model": canonical_model(critic_model) if feedback_mode in llm_critic_modes else None,
        "feedback_mode": str(feedback_mode),
        "context": context,
        "prompt_variant": prompt_variant,
        "temperature": _norm_float(temperature),
        "seed": None if seed is None else int(seed),
        "reasoning_effort": reasoning_effort,
        "output_budget": None if output_budget is None else int(output_budget),
        # Only identifying when summarisation is actually on: context tells us
        # that, and a non-summarising run must not be split by a stale default.
        "summarizer_model": (canonical_model(summarizer_model)
                             if str(context).startswith("summary") else None),
    }


# Fields that shape the ACTOR CALL, as opposed to the whole experiment. Two
# generations sharing this hash are interchangeable: reusing one in place of the
# other samples from the same distribution.
ACTOR_FIELDS = (
    "benchmark", "slice_key", "metric", "k", "model", "temperature", "seed",
    "context", "prompt_variant", "reasoning_effort", "output_budget",
    "summarizer_model",
    # Grading config — included ONLY for seq@k attempts after the first, where
    # the prompt carried judge-derived feedback. See actor_fingerprint.
    "judge_model", "critic_model", "feedback_mode",
)


def actor_fingerprint(ident, attempt_index, *, version=FINGERPRINT_VERSION):
    """What makes two actor generations interchangeable.

    The point of this hash is reuse: re-judging a run should not re-pay for
    answers the actor already wrote. So it covers what shaped the ACTOR CALL and
    deliberately omits what only shaped the GRADING.

    The exception is the whole subtlety of the design. In seq@k, attempt t>1 is
    prompted with feedback derived from the judge's verdict on attempts 1..t-1.
    Swap the judge and that attempt is no longer a sample from the same
    distribution — reusing it would measure the OLD judge's influence while
    claiming to measure the new one's. So for those attempts the grading config
    IS part of the generation, and they simply fail to match across a judge
    change. Nothing has to remember the rule; the hash enforces it.

        pass@k, any attempt   -> judge excluded (no feedback exists)
        seq@k, attempt 1      -> judge excluded (nothing graded yet)
        seq@k, attempt 2+     -> judge, critic and feedback_mode INCLUDED
    """
    graded_prefix = ident["metric"] == "seq@k" and int(attempt_index) > 1
    payload = {f: ident.get(f) for f in ACTOR_FIELDS}
    if not graded_prefix:
        payload["judge_model"] = payload["critic_model"] = payload["feedback_mode"] = None
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(f"actor-v{version}\n{blob}".encode("utf-8")).hexdigest()


def candidates(**identity_kwargs):
    """[(version, fingerprint, identity), ...] newest version first.

    The registry tries these in order and adopts the first that already exists,
    so bumping FINGERPRINT_VERSION does not orphan finished runs and make a
    re-run redo completed work. Sound only because each bump so far strictly
    RELAXES identity — see the note on FINGERPRINT_VERSION.
    """
    out = []
    for v in range(FINGERPRINT_VERSION, 0, -1):
        ident = identity(version=v, **identity_kwargs)
        out.append((v, fingerprint(ident, version=v), ident))
    return out


def fingerprint(ident, *, version=FINGERPRINT_VERSION):
    """Stable hash of an identity dict. Key order and float formatting are
    normalised so the same experiment hashes the same everywhere."""
    missing = set(IDENTITY_FIELDS) - set(ident)
    extra = set(ident) - set(IDENTITY_FIELDS)
    if missing or extra:
        raise ValueError(
            f"identity must have exactly {len(IDENTITY_FIELDS)} fields; "
            f"missing={sorted(missing)} unexpected={sorted(extra)}"
        )
    payload = {k: ident[k] for k in IDENTITY_FIELDS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(f"v{version}\n{blob}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def new_run_id():
    return str(uuid.uuid4())


def utc_now():
    return datetime.now(timezone.utc)


def storage_key(slice_key, run_id):
    """`<benchmark>/<run_id>` — identical locally and in S3.

    Two levels, and no more. The benchmark folder is the one thing worth seeing
    when you list the bucket; everything else about a run — when it started,
    what model, what metric, what feedback channel — is a queryable column in
    `runs`, so encoding any of it in the key would just be a second, weaker copy
    of the database that can fall out of step with it.

    (It used to be date-partitioned, `YYYY/MM/DD/<ts>-<uuid8>`. The dates bought
    a prefix query for "everything from July" that `WHERE created_at` already
    answers better, at the cost of four levels of nesting to walk.)
    """
    return f"{_safe_segment(slice_key)}/{run_id}"


def _safe_segment(name):
    """One path segment: no slashes, no leading dots."""
    seg = str(name).replace("/", "__").strip().lstrip(".")
    if not seg:
        raise ValueError(f"cannot build a storage key from {name!r}")
    return seg


def iso(dt):
    """UTC ISO-8601 with a trailing Z, seconds precision."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_float(x):
    """Round to 6dp so 0.7 and 0.7000000001 are the same experiment, and return
    an int when the value is integral so 1 and 1.0 hash alike."""
    f = round(float(x), 6)
    return int(f) if f == int(f) else f
