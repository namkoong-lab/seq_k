"""Query runs. The DB-backed replacement for walking runs/ by hand.

Works with OR without Postgres. With DATABASE_URL set it queries SQL; without
it, it reads the manifests on disk and answers the same questions more slowly.
That is not a fallback bolted on — it is the same guarantee as the harness: the
database is an accelerator, never a dependency.

    python scripts/runs.py ls                          # everything, newest first
    python scripts/runs.py ls --slice researchrubrics --metric seq@k --k 10
    python scripts/runs.py show <run_id|storage_key|label substring>
    python scripts/runs.py cost --by slice,metric      # cost rollup
    python scripts/runs.py phases <run_id>             # agent/judge/critic/summarizer split
    python scripts/runs.py grid experiments/rr-feedback-channels.grid.yaml
    python scripts/runs.py relink                      # regenerate by-label/
    python scripts/runs.py doctor                      # integrity checks
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, ids, registry, results, rows  # noqa: E402

FILTERS = ["slice", "metric", "k", "agent", "judge", "fb", "critic", "context",
           "prompt", "temp", "seed", "reason", "status"]
_COL = {"slice": "slice_key", "agent": "model", "judge": "judge_model",
        "fb": "feedback_mode", "critic": "critic_model", "prompt": "prompt_variant",
        "temp": "temperature", "reason": "reasoning_effort"}


# --------------------------------------------------------------------------- #
# Loading — SQL when available, manifests otherwise
# --------------------------------------------------------------------------- #
def load(runs_root, filters=None):
    filters = {k: v for k, v in (filters or {}).items() if v is not None}
    if db.enabled():
        where, params = [], {}
        for key, val in filters.items():
            col = _COL.get(key, key)
            where.append(f"{col} = %({key})s")
            params[key] = val
        # run_summary, not runs: the totals are a view over llm_calls, so they
        # are always current — including for runs still in flight.
        sql = "SELECT * FROM run_summary" + (" WHERE " + " AND ".join(where) if where else "")
        recs = db.query(sql + " ORDER BY created_at DESC", params, required=False)
        if recs:
            return [_from_sql(r) for r in recs], "postgres"
    out = []
    for path, m in registry.iter_manifests(runs_root):
        rec = _from_manifest(path, m)
        if all(str(rec.get(k)) == str(v) for k, v in filters.items()):
            out.append(rec)
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return out, "manifests"


def _from_manifest(path, m):
    cfg, roll = m.get("config", {}), m.get("rollup") or {}
    label = results.label_from_fields(
        slice_key=cfg.get("slice_key"), metric=cfg.get("metric"),
        k=m.get("k_target") or cfg.get("k"), model=cfg.get("model"),
        judge_model=cfg.get("judge_model"), critic_model=cfg.get("critic_model"),
        feedback_mode=cfg.get("feedback_mode"), context=cfg.get("context"),
        prompt_variant=cfg.get("prompt_variant"), temperature=cfg.get("temperature"),
        seed=cfg.get("seed"), reasoning_effort=cfg.get("reasoning_effort"))
    return {"run_id": m["run_id"], "created_at": m["created_at"], "status": m.get("status"),
            "storage_key": m["storage_key"], "path": path, "label": label,
            "slice": cfg.get("slice_key"), "metric": cfg.get("metric"), "k": cfg.get("k"),
            "agent": cfg.get("model"), "judge": cfg.get("judge_model"),
            "fb": cfg.get("feedback_mode"), "critic": cfg.get("critic_model"),
            "context": cfg.get("context"), "prompt": cfg.get("prompt_variant"),
            "temp": cfg.get("temperature"), "seed": _na(cfg.get("seed")),
            "reason": _na(cfg.get("reasoning_effort")),
            "tasks_done": roll.get("tasks_done", 0), "tasks_total": roll.get("tasks_total", 0),
            "tasks_success": roll.get("tasks_success", 0),
            "attempts": roll.get("attempts_total", 0), "cost_usd": roll.get("cost_usd", 0.0),
            "fingerprint": m.get("fingerprint")}


def _from_sql(r):
    # The label is COMPUTED from the columns, never stored: see the note at the
    # bottom of db/schema.sql. One implementation, in core.results.
    return {"run_id": str(r["run_id"]), "created_at": r["created_at"].isoformat(),
            "status": r["status"], "storage_key": r["storage_key"], "path": r["storage_key"],
            "label": results.label_from_fields(
                slice_key=r["slice_key"], metric=r["metric"], k=r["k"], model=r["model"],
                judge_model=r["judge_model"], critic_model=r["critic_model"],
                feedback_mode=r["feedback_mode"], context=r["context"],
                prompt_variant=r["prompt_variant"], temperature=r["temperature"],
                seed=r["seed"], reasoning_effort=r["reasoning_effort"]),
            "slice": r["slice_key"], "metric": r["metric"],
            "k": r["k"], "agent": r["model"], "judge": r["judge_model"], "fb": r["feedback_mode"],
            "critic": r["critic_model"], "context": r["context"], "prompt": r["prompt_variant"],
            "temp": r["temperature"], "seed": _na(r["seed"]), "reason": _na(r["reasoning_effort"]),
            "tasks_done": r["n_tasks_done"], "tasks_total": r["n_tasks"],
            "tasks_success": r["n_solved"], "attempts": r["n_attempts"],
            "cost_usd": float(r["cost_usd"]), "fingerprint": r["fingerprint"]}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_ls(args):
    recs, src = load(args.runs_root, {k: getattr(args, k) for k in FILTERS})
    if args.csv:
        import csv
        w = csv.DictWriter(sys.stdout, fieldnames=list(recs[0]) if recs else ["run_id"])
        w.writeheader(); w.writerows(recs)
        return
    hdr = f"{'created':16} {'slice':20} {'metric':7} {'k':>3} {'agent':38} {'fb':8} " \
          f"{'context':16} {'seed':>4} {'done':>9} {'solved':>6} {'cost':>9}"
    print(hdr); print("-" * len(hdr))
    for r in recs:
        print(f"{r['created_at'][:16]:16} {_t(r['slice'],20):20} "
              f"{_t(r['metric'],7):7} {str(r['k']):>3} {_t(r['agent'],38):38} "
              f"{_t(str(r['fb']),8):8} {_t(str(r['context']),16):16} {str(r['seed']):>4} "
              f"{str(r['tasks_done'])+'/'+str(r['tasks_total']):>9} "
              f"{r['tasks_success']:>6} {r['cost_usd']:>9.2f}")
    print(f"\n{len(recs)} runs   ${sum(r['cost_usd'] for r in recs):.2f}   [source: {src}]")


def cmd_show(args):
    recs, src = load(args.runs_root)
    hit = [r for r in recs if _matches(r, args.target)]
    if not hit:
        sys.exit(f"no run matching {args.target!r}")
    for r in hit[: args.limit]:
        print(f"\nrun_id      {r['run_id']}")
        print(f"created     {r['created_at']}    status {r['status']}")
        print(f"storage     {r['storage_key']}")
        print(f"label       {r['label']}")
        print(f"fingerprint {r['fingerprint']}")
        for f in ("slice", "metric", "k", "agent", "judge", "fb", "critic", "context",
                  "prompt", "temp", "seed", "reason"):
            print(f"  {f:<10} {r[f]}")
        print(f"  {'progress':<10} {r['tasks_done']}/{r['tasks_total']} done, "
              f"{r['tasks_success']} solved, {r['attempts']} attempts, ${r['cost_usd']:.4f}")
    print(f"\n[source: {src}]")


def cmd_phases(args):
    """agent / judge / critic / summarizer cost split for one run."""
    recs, _src = load(args.runs_root)
    hit = next((r for r in recs if _matches(r, args.target)), None)
    if not hit:
        sys.exit(f"no run matching {args.target!r}")
    if db.enabled():
        got = db.query("SELECT phase, n_calls, input_tokens, output_tokens, cost_usd "
                       "FROM run_phase_costs WHERE run_id = %s",
                       (hit["run_id"],), required=False)
        table = [(g["phase"], g["n_calls"], g["input_tokens"], g["output_tokens"],
                  float(g["cost_usd"] or 0)) for g in got]
    else:
        agg = {}
        seq = hit["metric"] == "seq@k"
        from core import results as _res
        for idx in rows.iter_task_indices(hit["path"]):
            for a in _res.load_task_attempts(hit["path"], idx):
                for c in rows.call_rows_for_attempt(a):
                    e = agg.setdefault(c["phase"], [0, 0, 0, 0.0])
                    e[0] += 1; e[1] += c["input_tokens"]; e[2] += c["output_tokens"]
                    e[3] += c["cost_usd"] or 0.0
        table = [(k, *v) for k, v in agg.items()]
    # Display order comes from db.PHASES, not from SQL: the phase vocabulary is
    # data, not schema, so an unknown phase sorts last rather than erroring.
    _ord = {p: i for i, p in enumerate(db.PHASES)}
    table.sort(key=lambda r: (_ord.get(r[0], 99), r[0]))
    total = sum(r[4] for r in table) or 1.0
    print(f"{'phase':12} {'calls':>8} {'in_tok':>12} {'out_tok':>10} {'cost':>10} {'share':>7}")
    for name, n, inp, outp, cost in table:
        print(f"{name:12} {n:>8,} {inp or 0:>12,} {outp or 0:>10,} {cost:>10.4f} "
              f"{100*cost/total:>6.1f}%")
    print(f"{'TOTAL':12} {sum(r[1] for r in table):>8,} {'':>12} {'':>10} {total:>10.4f}")


def cmd_cost(args):
    recs, src = load(args.runs_root, {k: getattr(args, k, None) for k in FILTERS})
    keys = [k.strip() for k in args.by.split(",")]
    agg = {}
    for r in recs:
        g = tuple(str(r[k]) for k in keys)
        a = agg.setdefault(g, [0, 0.0, 0])
        a[0] += 1; a[1] += r["cost_usd"]; a[2] += r["attempts"]
    print("  ".join(f"{k:<22}" for k in keys) + f"{'runs':>6} {'attempts':>10} {'cost':>11}")
    for g, (n, cost, att) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print("  ".join(f"{_t(x,22):<22}" for x in g) + f"{n:>6} {att:>10,} {cost:>11.2f}")
    print(f"\n{len(recs)} runs   ${sum(r['cost_usd'] for r in recs):.2f}   [source: {src}]")


def cmd_grid(args):
    """Which declared ablation cells are complete / partial / never run."""
    import yaml
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    axes = {k: (v if isinstance(v, list) else [v]) for k, v in spec.items() if k in FILTERS}
    keys = list(axes)
    recs, src = load(args.runs_root)
    have = {}
    for r in recs:
        have.setdefault(tuple(str(r[k]) for k in keys), []).append(r)
    complete = partial = missing = 0
    print(f"grid: {spec.get('name', args.spec)}")
    print(f"axes: {', '.join(keys)}\n")
    for combo in itertools.product(*(axes[k] for k in keys)):
        cell = tuple(str(c) for c in combo)
        label = "  ".join(f"{k}={v}" for k, v in zip(keys, cell))
        got = have.get(cell)
        if not got:
            missing += 1
            print(f"  MISSING   {label}")
        elif any(r["status"] == "complete" for r in got):
            complete += 1
            if args.verbose:
                print(f"  complete  {label}")
        else:
            partial += 1
            r = got[0]
            print(f"  partial   {label}   ({r['tasks_done']}/{r['tasks_total']} tasks)")
    print(f"\n  complete {complete} | partial {partial} | MISSING {missing}   [source: {src}]")
    return 1 if missing else 0


def cmd_doctor(args):
    """Integrity checks that no single command owns: is the registry consistent
    with the manifests, and does the database agree with disk."""
    # Cost completeness. A run whose calls recorded no token usage cannot be
    # priced, so cost_usd is NULL — honestly "unknown", not zero. That is
    # correct storage but it makes every total a LOWER BOUND, which should never
    # be silent: 23 imported advancedif runs report $0 known cost this way.
    if db.enabled():
        inc = db.query("""SELECT r.slice_key, count(DISTINCT r.run_id) runs,
                 count(*) FILTER (WHERE l.cost_usd IS NULL) nullc, count(*) total
              FROM llm_calls l JOIN runs r USING (run_id)
              GROUP BY 1 HAVING count(*) FILTER (WHERE l.cost_usd IS NULL) > 0""",
              required=False)
        if inc:
            tot = sum(x["nullc"] for x in inc)
            print(f"cost: {tot:,} call(s) have no token data -> totals are a LOWER BOUND")
            for x in inc:
                print(f"  {x['slice_key']:20} {x['nullc']:>6,}/{x['total']:<8,} calls unpriced")
        else:
            print("cost: every call is priced")

    idx, dupes = registry.rebuild(args.runs_root)
    print(f"registry: {len(idx)} runs indexed, {len(dupes)} duplicate fingerprints")
    for key, a, b in dupes:
        print(f"  ! {key}\n      {a}\n      {b}")
    if db.enabled():
        n = db.query("SELECT count(*) n FROM runs", required=False)
        disk = len(list(registry.iter_manifests(args.runs_root)))
        got = n[0]["n"] if n else 0
        print(f"database: {got} runs vs {disk} on disk"
              + ("" if got == disk else "   -> run `db_sync.py --rebuild`"))
    return 1 if dupes else 0


def cmd_relink(args):
    n = registry.relink_all(args.runs_root)
    print(f"{n} new symlinks under {args.runs_root}/{registry.BY_LABEL}/")


def cmd_rebuild(args):
    idx, dupes = registry.rebuild(args.runs_root, relink=True)
    print(f"registry rebuilt from manifests: {len(idx)} runs")
    for key, a, b in dupes:
        print(f"  ! duplicate fingerprint {key}\n      {a}\n      {b}")



# --------------------------------------------------------------------------- #
# Finishing someone else's partial run
# --------------------------------------------------------------------------- #
# Field -> YAML key. `k` deliberately comes from k_target, not config["k"]:
# config["k"] is None whenever k is NOT part of identity (a horizon-free
# variant), and the harness needs the number it was actually run to.
_RESUME_FIELDS = ("metric", "model", "judge_model", "critic_model", "feedback_mode",
                  "context", "prompt_variant", "temperature", "seed",
                  "reasoning_effort", "output_budget", "summarizer_model")


def _early_stopped():
    """run_id -> genuinely-short task count, for pass@k runs that stopped early.

    A pass@k run whose short tasks are short because they SUCCEEDED did not die:
    the harness stopped drawing at the first success. It is finished, and
    re-running adds nothing — the defect is the estimator (pass@k needs k
    INDEPENDENT draws), not missing work.

    Requiring that of EVERY short task was too strict. One 300-task run has 242
    of its 248 short tasks solved and a 6-task tail that genuinely died; calling
    that "unfinished" put it at the top of the pick-up list, inviting someone to
    re-run 248 tasks to recover 6. Treat >=90% as early-stopped and report the
    tail, so the number a contributor sees is the work actually left.
    """
    try:
        out = {}
        for r in db.query("""
            SELECT rs.run_id::text run_id,
                   count(*) FILTER (WHERE NOT ts.has_all_attempts) short,
                   count(*) FILTER (WHERE NOT ts.has_all_attempts AND ts.solved) solved_short
            FROM run_summary rs JOIN task_summary ts ON ts.run_id = rs.run_id
            WHERE rs.metric='pass@k'
            GROUP BY 1 HAVING count(*) FILTER (WHERE NOT ts.has_all_attempts) > 0"""):
            if r["solved_short"] >= 0.9 * r["short"]:
                out[r["run_id"]] = r["short"] - r["solved_short"]
        return out
    except Exception:
        return {}


def cmd_todo(args):
    """Runs that are not finished — the pick-up list for contributors."""
    recs, src = load(args.runs_root, {k: getattr(args, k, None) for k in FILTERS})
    early = _early_stopped()
    open_ = [r for r in recs if r["tasks_done"] < r["tasks_total"] or r["tasks_total"] == 0]
    stopped = [r for r in open_ if r["run_id"] in early]
    open_ = [r for r in open_ if r["run_id"] not in early]
    for r in stopped:
        r["_tail"] = early[r["run_id"]]
    open_.sort(key=lambda r: (r["tasks_total"] - r["tasks_done"]), reverse=True)
    if not open_:
        print(f"nothing unfinished [source: {src}]")
        return 0
    hdr = (f"{'remaining':>9} {'done':>9}  {'slice':20} {'metric':7} {'k':>3} "
           f"{'agent':26} {'fb':16} run_id")

    def rows(items):
        print(hdr)
        print("-" * len(hdr))
        for r in items:
            left = r["tasks_total"] - r["tasks_done"]
            print(f"{left:>9} {str(r['tasks_done'])+'/'+str(r['tasks_total']):>9}  "
                  f"{_t(r['slice'],20):20} {_t(r['metric'],7):7} {str(r['k']):>3} "
                  f"{_t(r['agent'],26):26} {_t(str(r['fb']),16):16} {r['run_id']}")

    print("UNFINISHED — these can be picked up and completed")
    rows(open_)
    print(f"\n{len(open_)} unfinished run(s)   [source: {src}]")
    if stopped:
        stopped.sort(key=lambda r: r.get("_tail", 0), reverse=True)
        print(f"\n\nEARLY-STOPPED pass@k — FINISHED, do not pick these up ({len(stopped)})")
        hdr2 = (f"{'genuinely':>9} {'done':>9}  {'slice':20} {'metric':7} {'k':>3} "
                f"{'agent':26} {'fb':16} run_id")
        print(hdr2)
        print(f"{'short':>9}")
        print("-" * len(hdr2))
        for r in stopped:
            print(f"{r.get('_tail',0):>9} {str(r['tasks_done'])+'/'+str(r['tasks_total']):>9}  "
                  f"{_t(r['slice'],20):20} {_t(r['metric'],7):7} {str(r['k']):>3} "
                  f"{_t(r['agent'],26):26} {_t(str(r['fb']),16):16} {r['run_id']}")
        print("  The gap between `done` and the task total is tasks that stopped because they")
        print("  SUCCEEDED — the harness quit at the first success, so the run is complete.")
        print("  `genuinely short` is the only real remainder. What is wrong here is the")
        print("  estimator (pass@k needs k INDEPENDENT draws), which makes these unusable as")
        print("  a pass@k baseline — not unfinished work.")
    print("\nTo finish one:  python scripts/runs.py resume <run_id> > resume.yaml")
    return 0


def _routed_models(storage_key, runs_root):
    """{field: routed model string} as recorded in the run's own attempt files."""
    import glob
    out = {}
    for path in (os.path.join(runs_root, storage_key), storage_key):
        files = sorted(glob.glob(os.path.join(path, "task-*", "attempt-*.json")))[:1]
        if not files:
            continue
        try:
            a = json.loads(Path(files[0]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # An OpenRouter response carries a `provider` field and a usage.cost;
        # a direct provider response carries neither. The recorded model name
        # does NOT include the route, so without this the config sends an
        # OpenRouter run straight to OpenAI — which rejects the run's
        # temperature and reports no cost, silently nulling the accounting.
        rr = ((a.get("actor") or {}).get("raw_response")) or {}
        via_openrouter = "provider" in rr
        for field, section in (("model", "actor"), ("judge_model", "judge"),
                               ("critic_model", "critic"), ("summarizer_model", "summarizer")):
            v = (a.get(section) or {}).get("model")
            if not v:
                continue
            if via_openrouter and not v.startswith("openrouter/"):
                v = "openrouter/" + v
            out[field] = v
        break
    return out


def cmd_resume(args):
    """Emit the exact variant YAML that RESUMES a run instead of forking it.

    Identity is a hash over every result-affecting field. Hand-writing a config
    to continue someone's run means reproducing that hash exactly — miss the
    temperature or the judge and the harness cheerfully starts a SECOND run
    under a different fingerprint, and you find out when the grid has two
    half-finished cells instead of one whole one. So derive it from the manifest
    rather than retyping it.
    """
    import yaml
    recs, _src = load(args.runs_root)
    hits = [r for r in recs if r["run_id"].startswith(args.target)
            or r["storage_key"].endswith(args.target)]
    if len(hits) != 1:
        print(f"{args.target!r} matched {len(hits)} runs; be more specific", file=sys.stderr)
        return 1
    # `path` from load() is the storage key, which is relative to runs_root when
    # the records came from the database rather than a disk walk. Try it as given,
    # then joined, so this works from either source.
    raw = hits[0]["path"]
    head_extra = []
    m = registry.read_manifest(raw) or registry.read_manifest(os.path.join(args.runs_root, raw))
    if m is None:
        print(f"no manifest.json for {raw}", file=sys.stderr)
        return 1
    cfg = m["config"]
    # ROUTED model names, not canonical ones. `config.model` is canonicalised
    # (route and vendor stripped) because identity must not split on how a model
    # was reached — but that same string is handed to litellm at call time, where
    # the prefix decides the provider and which parameter rules apply. Emitting
    # the bare name sends an OpenRouter run to OpenAI. The run's own attempt
    # files record what it actually called, so prefer that; canonicalising it
    # back gives the same fingerprint either way, so identity is unaffected.
    routed = _routed_models(raw, args.runs_root)

    doc = {"benchmark": cfg["benchmark"].replace("benchmarks.", "", 1),
           "k": m.get("k_target") or cfg.get("k")}
    for f in _RESUME_FIELDS:
        v = cfg.get(f)
        if v is None:
            continue
        if f in routed and ids.canonical_model(routed[f]) == ids.canonical_model(v):
            if routed[f] != v:
                head_extra.append(f"# {f}: {v!r} -> {routed[f]!r} (the route the run actually used)")
            v = routed[f]
        doc[f] = v
    # `options` is passed straight to benchmark.load_tasks(**options), but the
    # manifest's copy also carries PROVENANCE that importers and restamps added
    # (`model_raw`, `source_dataset`, `source_revision`). Emitting those verbatim
    # produces a config that dies with TypeError on an unexpected keyword before
    # a single task loads. Keep only what load_tasks actually accepts.
    opts = dict(m.get("options") or {})
    if opts:
        try:
            import importlib, inspect
            mod = importlib.import_module(cfg["benchmark"])
            sig = inspect.signature(mod.load_tasks)
            takes_kwargs = any(pp.kind is pp.VAR_KEYWORD for pp in sig.parameters.values())
            if not takes_kwargs:
                dropped = sorted(k for k in opts if k not in sig.parameters)
                opts = {k: v for k, v in opts.items() if k in sig.parameters}
                for k in dropped:
                    head_extra.append(f"# dropped option {k!r}: provenance, not a load_tasks argument")
        except Exception:
            pass
    if opts:
        doc["options"] = opts

    # WHICH TASKS. Without this the harness calls load_tasks() and offers the
    # WHOLE slice: resuming a 30-task HealthBench run silently proposed 316
    # fresh tasks, turning a finished cell into a 346-task run under the SAME
    # fingerprint. `max_tasks` when the run covered a contiguous 1..N prefix,
    # an explicit index list otherwise — 182 of 440 runs cover a sparse set.
    idx = []
    try:
        rows = db.query("""SELECT t.task_index FROM run_tasks rt
                           JOIN tasks t ON t.task_uid = rt.task_uid
                           WHERE rt.run_id = %s ORDER BY t.task_index""",
                        (m["run_id"],))
        idx = [r["task_index"] for r in rows]
    except Exception:
        pass
    if idx:
        if idx == list(range(1, len(idx) + 1)):
            doc["max_tasks"] = len(idx)
        else:
            doc["task_indices"] = idx
        head_extra.append(f"# covers {len(idx)} task(s) — pinned so a resume cannot "
                          f"silently widen to the whole slice")
    head = (f"# RESUME of {m['storage_key']}\n"
            f"# fingerprint {m['fingerprint']}\n"
            f"# {hits[0]['tasks_done']}/{hits[0]['tasks_total']} tasks done at time of writing.\n"
            f"#\n"
            f"# Do not edit the fields below. They reproduce this run's identity\n"
            f"# fingerprint; change one and you start a NEW run instead of\n"
            f"# continuing this one.\n"
            f"#\n"
            f"#   python -m core run resume.yaml\n"
            + ("".join(x + "\n" for x in head_extra) if head_extra else ""))
    sys.stdout.write(head + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
    return 0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ls"); p.set_defaults(fn=cmd_ls)
    for f in FILTERS:
        p.add_argument(f"--{f}")
    p.add_argument("--csv", action="store_true")

    p = sub.add_parser("show"); p.set_defaults(fn=cmd_show)
    p.add_argument("target"); p.add_argument("--limit", type=int, default=5)

    p = sub.add_parser("phases"); p.set_defaults(fn=cmd_phases)
    p.add_argument("target")

    p = sub.add_parser("cost"); p.set_defaults(fn=cmd_cost)
    p.add_argument("--by", default="slice,metric")
    for f in FILTERS:
        p.add_argument(f"--{f}")

    p = sub.add_parser("grid"); p.set_defaults(fn=cmd_grid)
    p.add_argument("spec"); p.add_argument("-v", "--verbose", action="store_true")


    p = sub.add_parser("todo", help="unfinished runs — the pick-up list")
    p.set_defaults(fn=cmd_todo)
    for f in FILTERS:
        p.add_argument(f"--{f}")

    p = sub.add_parser("resume", help="print the YAML that continues a run")
    p.set_defaults(fn=cmd_resume)
    p.add_argument("target", help="run_id (or a unique prefix)")

    sub.add_parser("relink").set_defaults(fn=cmd_relink)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("rebuild").set_defaults(fn=cmd_rebuild)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


def _matches(rec, target):
    """Identify a run the way a human would type it: a uuid prefix, any part of
    the storage key (the timestamp alone is usually enough), or any part of the
    readable label."""
    t = str(target)
    return (rec["run_id"].startswith(t)
            or t in (rec["storage_key"] or "")
            or t in (rec["label"] or ""))


def _t(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _na(v):
    return "na" if v is None else v


if __name__ == "__main__":
    main()
