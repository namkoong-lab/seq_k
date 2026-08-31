"""Offline self-test for run identity, storage, and harness wiring. No API calls,
no database.

Stubs litellm so a full harness run executes end to end against a fake benchmark.
Covers the things that are easy to break and expensive to discover later: the
fingerprint, resume, the label view, and — most importantly — that a missing or
broken database NEVER affects a run.

    python scripts/selftest.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Guarantee the DB layer is inert regardless of the developer's environment:
# this test asserts the harness works WITHOUT a database.
os.environ["SEQK_DB"] = "0"

import litellm  # noqa: E402

from core import db, harness, ids, registry, results, rows  # noqa: E402
from core.types import Task, VerifierResult  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# ---- stub the provider ----------------------------------------------------- #
class _Msg:
    def __init__(s, c): s.content = c; s.thinking_blocks = None; s.reasoning_content = None
class _Choice:
    def __init__(s, c): s.message = _Msg(c); s.finish_reason = "stop"
class _Resp:
    def __init__(s, c, out=20):
        s.choices = [_Choice(c)]
        s.usage = {"prompt_tokens": 50, "completion_tokens": out,
                   "prompt_tokens_details": {"cached_tokens": 0}}
    def model_dump(s): return {"stub": True}

CALLS = []
def fake_completion(**kw):
    p = kw["messages"][0]["content"]
    CALLS.append(kw["model"])
    if p.startswith("You are reviewing one of your own failed attempts"):
        return _Resp("SUMMARY: what to fix next time.")
    if p.startswith("JUDGE:"):
        return _Resp("no")
    return _Resp("an answer " * 10)
litellm.completion = fake_completion

# ---- fake benchmark -------------------------------------------------------- #
bench = types.ModuleType("benchmarks.fake")
bench.VERIFIER = "llm"
bench.LLM_CRITIC_MODES = {"critic"}
bench.slice_name = lambda o: "fakebench"
bench.load_tasks = lambda **o: [Task(id=f"t{i}", canonical_index=i, prompt="TASK", grading={})
                                for i in range(1, 3)]
def verify(task, attempt, *, judge_model):
    from core import llm
    llm.complete(judge_model, "JUDGE: grade", 0.0)
    return VerifierResult(success=False, score=0.0, raw_eval_output="FEEDBACK: missing X.", details={})
bench.verify = verify
bench.feedback = lambda t, a, r, m, *, critic_model: r.raw_eval_output
sys.modules["benchmarks.fake"] = bench

ROOT = Path(tempfile.mkdtemp(prefix="seqk-selftest-"))


def ident(**over):
    # summarizer defaults to the actor in the harness, so mirror that here.
    # seed=1 mirrors the harness's AUTO-SEED: a new run at a non-zero temperature
    # with no seed given is stamped seed=1.
    base = dict(benchmark_module=bench, options={}, metric="seq@k", k=3, model="a/b",
                judge_model="a/b", critic_model="a/b", feedback_mode="raw",
                context="full", prompt_variant="v1", temperature=0.7, seed=1,
                summarizer_model="a/b")
    base.update(over)
    return ids.identity(**base)


def run_dir(**over):
    """Where a config RESOLVES today — the run, or None.

    `seed=1` by default because the harness AUTO-SEEDS: at a non-zero temperature
    a new run with no seed given is stamped seed=1, so that — not seed=None — is
    the fingerprint a fresh config lands on. Pass seed=None explicitly to look up
    a run made before auto-seeding existed."""
    fps = {f for _v, f, _i in ids.candidates(**{**dict(
        benchmark_module=bench, options={}, metric="seq@k", k=3, model="a/b",
        judge_model="a/b", critic_model="a/b", feedback_mode="raw", context="full",
        prompt_variant="v1", temperature=0.7, seed=1, summarizer_model="a/b"), **over})}
    for d, m in registry.iter_manifests(ROOT):
        if m["fingerprint"] in fps:
            return d
    return None


print("\n--- label shape (build_run_path is now a label formatter)")
p = results.build_run_path(runs_root="", benchmark_module=bench, options={}, metric="seq@k",
                           k=5, model="a/b", judge_model="c/d", critic_model="e/f",
                           feedback_mode="raw", context="summary", temperature=0.7)
parts = p.split("/")
check("10 label levels", len(parts) == 10, f"got {len(parts)}: {p}")
check("every level labelled", all("=" in x for x in parts[1:]), p)
check("k zero-padded", "k=05" in parts, p)
check("agent/judge distinct labels", "agent=a__b" in parts and "judge=c__d" in parts, p)
pc = results.build_run_path(runs_root="", benchmark_module=bench, options={}, metric="seq@k",
                            k=5, model="a/b", judge_model="c/d", critic_model="e/f",
                            feedback_mode="critic", context="full", temperature=0.7)
check("llm-critic keeps mode AND model", "fb=critic@e__f" in pc.split("/"), pc)

print("\n--- fingerprint")
check("stable across calls", ids.fingerprint(ident()) == ids.fingerprint(ident()))
check("insensitive to float noise",
      ids.fingerprint(ident(temperature=0.7)) == ids.fingerprint(ident(temperature=0.7000000001)))
for field, val in [("k", 10), ("temperature", 0.9), ("seed", 2), ("context", "summary"),
                   ("prompt_variant", "legacy"), ("model", "x/y"), ("feedback_mode", "binary"),
                   ("output_budget", 5000)]:
    check(f"{field} changes identity",
          ids.fingerprint(ident(**{field: val})) != ids.fingerprint(ident()))
check("judge_model ignored for non-llm verifier",
      ids.identity(**{**dict(benchmark_module=types.SimpleNamespace(
          __name__="b", VERIFIER="harbor", LLM_CRITIC_MODES=set(), slice_name=lambda o: "s"),
          options={}, metric="seq@k", k=3, model="m", judge_model="ANYTHING",
          critic_model="c", feedback_mode="raw", context="full")})["judge_model"] == "harbor")
check("critic_model dropped when fb has no llm critic", ident(feedback_mode="raw")["critic_model"] is None)
check("critic_model kept when fb has one", ident(feedback_mode="critic")["critic_model"] == "a/b")
check("summarizer_model only counts under summary context",
      ident(summarizer_model="s/m")["summarizer_model"] is None
      and ident(context="summary", summarizer_model="s/m")["summarizer_model"] == "s/m")
try:
    ids.fingerprint({"k": 3}); check("rejects a partial identity", False)
except ValueError:
    check("rejects a partial identity", True)

print("\n--- k is identity only when it reaches the prompt")
check("v1: k matters", ids.k_affects_prompt(bench, context="full", metric="seq@k",
                                            prompt_variant="v1"))
check("v1-nohorizon: k does NOT", not ids.k_affects_prompt(
    bench, context="full", metric="seq@k", prompt_variant="v1-nohorizon"))
check("v1-noframe: k does NOT", not ids.k_affects_prompt(
    bench, context="full", metric="seq@k", prompt_variant="v1-noframe"))
check("k-identity follows the VARIANT, not the context",
      ids.k_affects_prompt(bench, context="summary", metric="seq@k", prompt_variant="v1")
      and not ids.k_affects_prompt(bench, context="summary", metric="seq@k",
                                   prompt_variant="v1-nohorizon"))
check("pass@k: k never reaches the prompt",
      not ids.k_affects_prompt(bench, context="na", metric="pass@k"))
_agentic = types.SimpleNamespace(__name__="b", VERIFIER="harbor", LLM_CRITIC_MODES=set(),
                                 slice_name=lambda o: "s", run_attempt=lambda *a, **k: None)
check("agentic benchmark assumed k-dependent (conservative)",
      ids.k_affects_prompt(_agentic, context="full", metric="seq@k",
                           prompt_variant="v1-nohorizon"))
check("k=5/k=10 share identity under a horizon-free variant",
      ids.fingerprint(ident(k=5, prompt_variant="v1-nohorizon"))
      == ids.fingerprint(ident(k=10, prompt_variant="v1-nohorizon")))
check("k=5/k=10 STAY split when the horizon is shown",
      ids.fingerprint(ident(k=5)) != ids.fingerprint(ident(k=10)))
print("\n--- fingerprint version fallback")
cands = ids.candidates(benchmark_module=bench, options={}, metric="seq@k", k=3, model="a/b",
                       judge_model="a/b", critic_model="a/b", feedback_mode="raw",
                       context="full", prompt_variant="v1", temperature=0.7,
                       summarizer_model="a/b")
check("candidates newest-first", [v for v, _f, _i in cands] == list(range(ids.FINGERPRINT_VERSION, 0, -1)))
check("v1 always keeps k in identity", cands[-1][2]["k"] == 3)

print("\n--- storage key")
key = ids.storage_key("researchrubrics", "8f3c1a2b-0000-0000-0000-000000000000")
check("<benchmark>/<run_id>", key == "researchrubrics/8f3c1a2b-0000-0000-0000-000000000000", key)
check("nothing about the experiment is in the key",
      not any(x in key for x in ("metric", "k=", "temp", "seq", "pass")), key)
check("slice slashes are made safe", "/" not in ids.storage_key("a/b", "r").split("/")[0])

print("\n--- canonical model names (route and vendor are not identity)")
cm = ids.canonical_model
check("route prefix stripped", cm("openrouter/openai/gpt-5.2") == "gpt-5.2")
check("vendor prefix stripped", cm("anthropic/claude-sonnet-4-6") == "claude-sonnet-4.6")
check("the two spellings of one model converge",
      cm("anthropic/claude-opus-4-7") == cm("openrouter/anthropic/claude-opus-4.7")
      == "claude-opus-4.7")
check("bare and routed converge",
      cm("google/gemini-3-flash-preview") == cm("openrouter/google/gemini-3-flash-preview"))
# The date guard: without it `-01-31` reads as a version and becomes `-01.31`,
# inventing a version number out of a release date.
check("release date is not mangled into a version",
      cm("openai/o3-mini-2025-01-31") == "o3-mini-2025-01-31")
check("a name that is already canonical is untouched", cm("gpt-5.3-codex") == "gpt-5.3-codex")
check("non-LLM verifier sentinels pass through",
      cm("deterministic") == "deterministic" and cm("harbor") == "harbor")
check("idempotent", cm(cm("openrouter/qwen/qwen3.6-max-preview"))
      == cm("openrouter/qwen/qwen3.6-max-preview"))
check("distinct models stay distinct", cm("openrouter/qwen/qwen3-max")
      != cm("openrouter/qwen/qwen3.6-max-preview"))
check("identity uses the canonical name",
      ident(model="openrouter/openai/gpt-5.2")["model"] == "gpt-5.2")

print("\n--- context mapping")
check("pass@k -> na", results.context_from_legacy("pass@k", False, "full") == "na")
check("summarize -> summary", results.context_from_legacy("seq@k", True) == "summary")
check("no summarize -> full", results.context_from_legacy("seq@k", False) == "full")
check("only three contexts exist", results.CONTEXT_MODES == ("na", "full", "summary"))
check("summarizes()", results.summarizes("summary") and not results.summarizes("full"))
for bad in ("bogus", "full-nohorizon", "summarised", ""):
    try:
        results.summarizes(bad); check(f"rejects {bad!r}", False)
    except ValueError:
        check(f"rejects {bad!r}", True)

print("\n--- prompt wording lives on prompt_variant, not context")
_v1 = harness.build_prompt(Task(id="p", canonical_index=0, prompt="T", grading={}), [], 0, 7,
                           seq=True, prompt_variant="v1")
_nh = harness.build_prompt(Task(id="p", canonical_index=0, prompt="T", grading={}), [], 0, 7,
                           seq=True, prompt_variant="v1-nohorizon")
_nf = harness.build_prompt(Task(id="p", canonical_index=0, prompt="T", grading={}), [], 0, 7,
                           seq=True, prompt_variant="v1-noframe")
check("v1 names the horizon", "attempt 1 of 7" in _v1)
check("v1-nohorizon drops the count, keeps the frame",
      "of 7" not in _nh and "receive feedback" in _nh)
check("v1-noframe drops the note entirely", _nf.strip() == "T")
try:
    harness.build_prompt(Task(id="p", canonical_index=0, prompt="T", grading={}), [], 0, 7,
                         seq=True, prompt_variant="v1-typo")
    check("unknown variant rejected", False)
except ValueError:
    check("unknown variant rejected", True)

print("\n--- end-to-end harness (no database)")
check("db disabled in this process", not db.enabled())

def run(**kw):
    CALLS.clear()
    harness.run(bench, metric="seq@k", k=3, feedback_mode="raw", model="a/b",
                runs_root=str(ROOT), s3_sync=False, console_char_limit=0, **kw)

run(context="full")
base = run_dir(context="full")
check("run landed under a uuid key", base is not None and Path(base, "task-1", "attempt-1.json").exists(), str(base))
check("storage key is <benchmark>/<run_id>",
      base and len(Path(base).relative_to(ROOT).parts) == 2, str(base))
check("benchmark folder is the slice",
      base and Path(base).relative_to(ROOT).parts[0] == "fakebench", str(base))
a1 = json.loads(Path(base, "task-1", "attempt-1.json").read_text())
check("no summarizer key when context=full", "summarizer" not in a1)
cfg = json.loads(Path(base, "config.json").read_text())
check("config.json still written", cfg["context"] == "full" and "run_id" in cfg)

m = registry.read_manifest(base)
check("manifest has identity", m["fingerprint"] == ids.fingerprint(ident(context="full")))
check("manifest records label_path", m["label_path"].startswith("fakebench/metric=seqk/k=03/"))
check("manifest finalized", m["status"] in ("complete", "partial") and m["finished_at"])
check("manifest rollup counts tasks", m["rollup"]["tasks_total"] == 2)
check("manifest carries provenance", "litellm" in (m.get("code") or {}))

link = ROOT / registry.BY_LABEL / m["label_path"]
check("by-label symlink created", link.is_symlink(), str(link))
check("by-label symlink resolves to the run",
      link.is_symlink() and os.path.realpath(link) == os.path.realpath(base))

print("\n--- summarization")
run(context="summary")
sm = run_dir(context="summary")
check("summary run is a DIFFERENT directory", sm != base)
s1 = json.loads(Path(sm, "task-1", "attempt-1.json").read_text())
check("summarizer present when context=summary", s1.get("summarizer", {}).get("summary"))
s2 = json.loads(Path(sm, "task-1", "attempt-2.json").read_text())
check("summary replaces verbatim history", "<AttemptSummary 1>" in s2["actor"]["prompt"]
      and "<PreviousAttempt" not in s2["actor"]["prompt"])

run(context="full", prompt_variant="v1-nohorizon")
nh = run_dir(context="full", prompt_variant="v1-nohorizon")
n2 = json.loads(Path(nh, "task-1", "attempt-2.json").read_text())
check("nohorizon variant drops the counter", "This is attempt" not in n2["actor"]["prompt"])
check("nohorizon variant KEEPS history", "<PreviousAttempt 1>" in n2["actor"]["prompt"])

print("\n--- legacy summarize kwarg still accepted")
run(summarize=True)
check("summarize=True maps to context=summary", run_dir(context="summary") is not None)

print("\n--- resume")
CALLS.clear()
harness.run(bench, metric="seq@k", k=3, feedback_mode="raw", model="a/b", runs_root=str(ROOT),
            s3_sync=False, console_char_limit=0, context="full")
check("resume makes no calls", CALLS == [], f"{len(CALLS)} calls")
check("resume reuses the same directory", run_dir(context="full") == base)

print("\n--- horizon-free runs extend instead of forking")
# `summary-noframe` is horizon-free and not used by any earlier case here, so
# the counts below measure only what this block does.
CTX, PV = "summary", "v1-noframe"
harness.run(bench, metric="seq@k", k=3, feedback_mode="raw", model="a/b", runs_root=str(ROOT),
            s3_sync=False, console_char_limit=0, context=CTX, prompt_variant=PV)
n0 = len(list(registry.iter_manifests(ROOT)))
hf = run_dir(context=CTX, k=3, prompt_variant=PV)
a_before = len(list(Path(hf).rglob("attempt-*.json")))
harness.run(bench, metric="seq@k", k=5, feedback_mode="raw", model="a/b", runs_root=str(ROOT),
            s3_sync=False, console_char_limit=0, context=CTX, prompt_variant=PV)
check("k=5 reused the k=3 directory", run_dir(context=CTX, k=5, prompt_variant=PV) == hf, str(hf))
check("no second run was created", len(list(registry.iter_manifests(ROOT))) == n0)
check("more attempts were added", len(list(Path(hf).rglob("attempt-*.json"))) > a_before)
check("manifest k_target grew to 5", registry.read_manifest(hf).get("k_target") == 5)
_lbl = lambda kk: (ROOT / registry.BY_LABEL / f"fakebench/metric=seqk/k={kk}/agent=a__b/"
                   f"judge=a__b/fb=raw/context={CTX}/prompt={PV}/temp=0.7/seed=1")
check("both k labels link to the one run",
      _lbl("03").is_symlink() and _lbl("05").is_symlink()
      and os.path.realpath(_lbl("03")) == os.path.realpath(_lbl("05")) == os.path.realpath(hf))

print("\n--- output_budget is identity now, not a guard")
n_before = len(list(registry.iter_manifests(ROOT)))
harness.run(bench, metric="seq@k", k=3, feedback_mode="raw", model="a/b", runs_root=str(ROOT),
            s3_sync=False, console_char_limit=0, context="full", output_budget=5000)
budg = run_dir(context="full", output_budget=5000)
check("budgeted run gets its own directory", budg is not None and budg != base, str(budg))
check("it is a new run, not a mutation", len(list(registry.iter_manifests(ROOT))) == n_before + 1)

print("\n--- row extraction (what the DB would receive)")
_ident = ident(context="full")
task, arts, claims, calls = rows.task_rows(base, 1, ident=_ident, k=3, seq=True,
                                           storage_key="KEY", run_id="RID")
flat = [c for per in calls for c in per]
check("task dimension row", task["slice_key"] == "fakebench" and task["task_index"] == 1)
check("one artifact per attempt", len(arts) == len(claims) == 3)
check("artifacts carry a reuse key", all(a["actor_fingerprint"].startswith("sha256:") for a in arts))
check("artifacts record who generated them", all(a["generated_by_run"] == "RID" for a in arts))
check("artifacts locate their file",
      arts[0]["output_key"] == "KEY/task-1/attempt-1.json", arts[0]["output_key"])
check("verdicts live on the CLAIM, not the artifact",
      "solved" in claims[0] and "solved" not in arts[0])
check("one call row per llm call", len(flat) == sum(
    1 + len((json.loads(Path(base, "task-1", f"attempt-{i}.json").read_text())
             .get("judge") or {}).get("calls") or []) for i in (1, 2, 3)))
check("phases are readable text", {c["phase"] for c in flat} <= set(db.PHASES),
      str({c["phase"] for c in flat}))
check("cost_source is readable text", {c["cost_source"] for c in flat} <= set(db.COST_SOURCES))
check("call_index, not seq", "call_index" in flat[0] and "seq" not in flat[0])

print("\n--- actor fingerprint: what may be reused")
_pk = ident(metric="pass@k", context="na")
_sq = ident(metric="seq@k", context="full")
check("pass@k ignores the judge",
      ids.actor_fingerprint(_pk, 3) == ids.actor_fingerprint(
          ids.identity(**{**dict(benchmark_module=bench, options={}, metric="pass@k", k=3,
                                 model="a/b", judge_model="OTHER", critic_model="a/b",
                                 feedback_mode="raw", context="na", prompt_variant="v1",
                                 temperature=0.7, seed=1, summarizer_model="a/b")}), 3))
check("seq@k attempt 1 ignores the judge",
      ids.actor_fingerprint(_sq, 1) == ids.actor_fingerprint(
          ident(judge_model="OTHER", context="full"), 1))
check("seq@k attempt 2+ does NOT ignore the judge",
      ids.actor_fingerprint(_sq, 2) != ids.actor_fingerprint(
          ident(judge_model="OTHER", context="full"), 2))
check("changing the ACTOR always invalidates",
      ids.actor_fingerprint(_sq, 1) != ids.actor_fingerprint(ident(model="x/y", context="full"), 1))
roll = rows.run_rollup(base, k=3, seq=True)
check("rollup matches manifest", roll["attempts_total"] == m["rollup"]["attempts_total"])

print("\n--- registry is a rebuildable cache")
n_runs = len(list(registry.iter_manifests(ROOT)))
(ROOT / registry.INDEX_NAME).unlink()
idx, dupes = registry.rebuild(ROOT)
check("rebuilt from manifests alone", len(idx) == n_runs, f"{len(idx)} vs {n_runs}")
check("no duplicate fingerprints", not dupes, str(dupes))
CALLS.clear()
harness.run(bench, metric="seq@k", k=3, feedback_mode="raw", model="a/b", runs_root=str(ROOT),
            s3_sync=False, console_char_limit=0, context="full")
check("resume still works after rebuild", CALLS == [] and run_dir(context="full") == base)

print("\n--- upsert covers every mutable column")
import re as _re
_sql = open(Path(__file__).resolve().parent.parent / "core" / "db.py").read()
_ins = _sql[_sql.index("INSERT INTO runs"):_sql.index("ON CONFLICT (run_id)")]
_upd = _sql[_sql.index("ON CONFLICT (run_id)"):_sql.index('"""', _sql.index("ON CONFLICT (run_id)"))]
_inserted = set(_re.findall(r"\b(\w+)\b(?=\s*[,)])", _ins.split("(", 1)[1].split(")")[0]))
# Columns a tool can legitimately change after first insert. If you add one to
# the INSERT, add it here and to the DO UPDATE, or a rekey will silently no-op.
_mutable = {"storage_key", "status", "finished_at", "k", "code"}
_missing = {c for c in _mutable if c in _inserted and f"{c} = EXCLUDED.{c}" not in _upd
            and f"{c} = COALESCE(EXCLUDED.{c}" not in _upd and f"{c} = GREATEST(" not in _upd}
check("upsert_run updates every mutable column", not _missing, f"not updated: {sorted(_missing)}")

print("\n--- guards")
for kw, why in [(dict(metric="pass@k", context="full"), "pass@k rejects non-na context"),
                (dict(metric="seq@k", context="nonsense"), "invalid context rejected")]:
    try:
        harness.run(bench, k=2, feedback_mode="raw", model="a/b", runs_root=str(ROOT),
                    s3_sync=False, **kw)
        check(why, False)
    except ValueError:
        check(why, True)

print("\n--- no data is ever destroyed")
attempts_now = len(list(ROOT.rglob("attempt-*.json")))
check("every attempt file from every run still present", attempts_now >= 3 * 5,
      f"{attempts_now} attempt files")

shutil.rmtree(ROOT, ignore_errors=True)
print(f"\n{'ALL PASSED' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
