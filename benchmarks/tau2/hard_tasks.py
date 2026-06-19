"""Rank tau2 tasks by a STRUCTURAL difficulty heuristic, to pick a hard subset.

Motivation: a uniform slice of retail passes ~easily on a strong model, so
pass@k vs seq@k shows little. "Hard" tasks (where the agent fails sometimes) are
where seq@k's feedback can actually help — but there is no difficulty field in
the data, and true difficulty is only known after running. This estimates it
*before* running, from the task structure.

What makes a task hard (validated against task 18, which our pass@k run failed
2/3 times — this heuristic ranks it #4 of 114):
  - Conditional / mind-changing user scenarios ("if not... instead... rethink"):
    the customer probes fallbacks, so the agent must navigate branches.
  - Multiple write actions: more DB mutations = more chances to get the
    end-state wrong.
  - Many items in those writes: bigger exchange/return sets.

This is a HEURISTIC (a guess from structure), not measured difficulty. Use it to
pick a candidate hard set; the run itself tells you the truth.

Usage:
    python -m benchmarks.tau2.hard_tasks                 # top 10 retail, prints ids
    python -m benchmarks.tau2.hard_tasks --n 15
    python -m benchmarks.tau2.hard_tasks --domain airline --n 10
    python -m benchmarks.tau2.hard_tasks --ids-only      # just a comma list (for YAML)
"""

from __future__ import annotations

import argparse
import json
import os
import re

os.environ.setdefault("LOGURU_LEVEL", "ERROR")

_WRITE_PREFIXES = ("cancel", "exchange", "modify", "return", "update", "book")
_COND_RE = re.compile(
    r"\bif\b|instead|but if|change your mind|rethink|otherwise|"
    r"unavailable|not available|unless|prefer|if not|if so",
    re.I,
)


def _tasks_path(domain):
    from tau2.utils.utils import DATA_DIR
    return f"{DATA_DIR}/tau2/domains/{domain}/tasks.json"


def _difficulty(task):
    ec = task.get("evaluation_criteria") or {}
    actions = ec.get("actions") or []
    writes = [a for a in actions if a["name"].split("_")[0] in _WRITE_PREFIXES]
    n_items = sum(
        len(a["arguments"].get("item_ids", [])) + len(a["arguments"].get("new_item_ids", []))
        for a in writes if isinstance(a.get("arguments"), dict)
    )
    instr = (task.get("user_scenario") or {}).get("instructions") or {}
    reason = (instr.get("reason_for_call") or "") + " " + (instr.get("task_instructions") or "")
    cond = len(_COND_RE.findall(reason))
    # Conditionals dominate (that's what tripped task 18); writes & items add.
    score = cond * 2 + len(writes) * 2 + n_items
    return score, {
        "id": str(task["id"]),
        "score": score,
        "conditionals": cond,
        "writes": len(writes),
        "items": n_items,
    }


def rank(domain="retail"):
    tasks = json.load(open(_tasks_path(domain)))
    tasks = tasks if isinstance(tasks, list) else list(tasks.values())
    return sorted((_difficulty(t) for t in tasks), key=lambda x: -x[0])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", default="retail")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--ids-only", action="store_true", help="print just a comma-separated id list")
    args = p.parse_args(argv)

    ranked = rank(args.domain)[: args.n]
    ids = [r["id"] for _, r in ranked]

    if args.ids_only:
        print(",".join(ids))
        return

    print(f"Top {args.n} hardest {args.domain} tasks (structural heuristic):\n")
    print(f"  {'id':<6}{'score':<7}{'conditionals':<14}{'writes':<8}{'items'}")
    for _, r in ranked:
        print(f"  {r['id']:<6}{r['score']:<7}{r['conditionals']:<14}{r['writes']:<8}{r['items']}")
    print(f"\nids for a variant's options.tasks:  {','.join(ids)}")


if __name__ == "__main__":
    main()
