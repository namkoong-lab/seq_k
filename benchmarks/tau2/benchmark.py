"""tau2 (tau-bench / tau2-bench) as a seq_k agentic benchmark.

A tau2 "attempt" is not a single LLM completion: it's a full agent<->user
simulation of a customer-service task in a domain (we use `retail`), scored by
the resulting environment (DB) end-state and required communications. So this
benchmark implements the harness's `run_attempt` hook instead of verify():
it builds a tau2 orchestrator, runs one simulation, and parses the SimulationRun
into a VerifierResult.

We change nothing in tau2: we call its public layered runner APIs
(`get_tasks`, `build_text_orchestrator`, `run_simulation`). For seq@k retries we
append distilled prior-attempt feedback to the agent's `domain_policy` (a plain
instance attribute read lazily by the agent's system_prompt), so each retry is a
*fresh* conversation primed with what went wrong last time. pass@k never gets a
preamble, and seq@1 differs from pass@1 only via the harness's "attempt t of k"
machinery — here, every attempt is an independent sim.

PREREQS (external, fail-loud if missing): the `tau2` package importable
(`pip install -e ../tau2-bench`) and provider auth for both the agent model and
the user-simulator model (`user_llm` option). Configure via a variant's
`options:`.

Score = tau2 reward (1.0 pass / <1.0 fail). Leak-safety: only the public
per-check failure summary (`raw_eval_output`) is ever fed back to the next
attempt; gold DB state stays in `judge_details`, which the harness never shows
the actor.
"""

from __future__ import annotations

from core.types import VerifierResult
from core.types import Task as SeqTask

from . import prompts

DEFAULT_DOMAIN = "retail"
DEFAULT_SPLIT = "base"
DEFAULT_USER_LLM = "anthropic/claude-sonnet-4-6"
DEFAULT_MAX_STEPS = 100

# id -> tau2 Task, populated by load_tasks and read by run_attempt. We keep the
# raw pydantic task here (not on the seq_k Task) so nothing un-serializable lands
# in the stored attempt JSON.
_TAU2_TASKS: dict = {}


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def load_tasks(tasks=None, domain=DEFAULT_DOMAIN, split=DEFAULT_SPLIT,
               num_tasks=None, **_options):
    """Load tau2 tasks and wrap them as seq_k Tasks.

    `tasks` may be a comma-separated string or list of task ids to select a
    subset; otherwise the first `num_tasks` of the split are used.
    """
    from tau2.runner import get_tasks

    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    task_ids = list(tasks) if tasks else None

    t2_tasks = get_tasks(domain, task_split_name=split, task_ids=task_ids,
                         num_tasks=num_tasks)

    seq_tasks = []
    for t in t2_tasks:
        _TAU2_TASKS[t.id] = t
        seq_tasks.append(SeqTask(id=t.id, prompt=prompts.BASE_NOTE, grading={}))
    return seq_tasks


# --------------------------------------------------------------------------- #
# Attempt = one full tau2 simulation (the run_attempt hook)
# --------------------------------------------------------------------------- #
def run_attempt(task, history, t, k, *, seq, model, judge_model, temperature,
                options, out):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import build_text_orchestrator, run_simulation

    t2_task = _lookup_task(task.id, options)

    domain = options.get("domain", DEFAULT_DOMAIN)
    user_llm = options.get("user_llm", DEFAULT_USER_LLM)
    max_steps = int(options.get("max_steps", DEFAULT_MAX_STEPS))
    base_seed = int(options.get("seed", 0))

    # tau2's scoring uses an LLM to judge "natural-language assertions" on some
    # tasks. By default that judge is an OpenAI model (gpt-4.1), which would need
    # OPENAI_API_KEY. If the variant sets `nl_judge_model`, point it elsewhere
    # (e.g. a Claude model) so the whole run can use one provider's key.
    _maybe_override_nl_judge(options.get("nl_judge_model"))

    config = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        llm_agent=model,
        llm_args_agent={"temperature": temperature},
        user="user_simulator",
        llm_user=user_llm,
        llm_args_user={"temperature": float(options.get("user_temperature", 0.0))},
        max_steps=max_steps,
        num_trials=1,
    )

    # Distinct seed per attempt so fresh sims aren't identical replays.
    seed = base_seed + t

    # Build via tau2's Layer-2 helper (constructs env + agent + user). For seq@k
    # retries we inject distilled feedback into the agent's domain_policy AFTER
    # building — system_prompt reads domain_policy lazily, so this is the minimal
    # no-core-change seam.
    orchestrator = build_text_orchestrator(config, t2_task, seed=seed)

    retry_context = _retry_context(history) if seq else ""
    preamble = prompts.build_retry_preamble(retry_context)
    if preamble:
        orchestrator.agent.domain_policy = (
            orchestrator.agent.domain_policy + "\n\n" + preamble
        )

    # The prompt we store is the agent's effective system prompt (policy +/-
    # retry preamble) — that's what actually steers this attempt and what the
    # inspect view should show the delta of.
    stored_prompt = _agent_system_prompt(orchestrator.agent)

    sim = run_simulation(orchestrator)

    reward = float(sim.reward_info.reward) if sim.reward_info else 0.0
    success = reward >= 1.0

    transcript = _render_transcript(sim.messages or [])
    public_summary = _public_failure_summary(sim.reward_info, success)
    details = _judge_details(task.id, sim, seed)

    result = VerifierResult(
        success=success,
        score=reward,
        raw_eval_output=("" if success else public_summary),
        judge_details=details,
    )
    return stored_prompt, transcript, result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_NL_JUDGE_PATCHED = None  # remembers the model we patched in, to patch only once


def _maybe_override_nl_judge(nl_judge_model):
    """Point tau2's NL-assertion judge at `nl_judge_model` if requested.

    The evaluator does `from tau2.config import DEFAULT_LLM_NL_ASSERTIONS` and
    then uses the bare name, so it captured the value into its OWN module
    namespace at import time. To override it we must rebind the name on the
    evaluator module (rebinding tau2.config alone would be too late). No-op if
    `nl_judge_model` is falsy or already applied.

    We also harden the judge's JSON parsing (see _patch_nl_json_parsing): tau2
    does a raw json.loads() on the judge reply, which assumes an OpenAI-style
    JSON-only response. Claude models wrap JSON in markdown fences / prose, which
    crashes json.loads. Needed whenever a Claude model is the judge."""
    global _NL_JUDGE_PATCHED
    if not nl_judge_model or _NL_JUDGE_PATCHED == nl_judge_model:
        return
    from tau2.evaluator import evaluator_nl_assertions as nl
    nl.DEFAULT_LLM_NL_ASSERTIONS = nl_judge_model
    _patch_nl_json_parsing()
    _NL_JUDGE_PATCHED = nl_judge_model


_NL_GENERATE_WRAPPED = False


def _patch_nl_json_parsing():
    """Make tau2's NL judge tolerant of non-OpenAI JSON formatting.

    The evaluator calls `generate(...)` (imported into its module namespace) and
    immediately does `json.loads(assistant_message.content)`. Claude often
    returns ```json ...``` fences or a prose preamble, breaking that parse. We
    wrap the module's `generate` so the NL-judge reply's content is sanitized to
    bare JSON before tau2 parses it. tau2 source is untouched."""
    global _NL_GENERATE_WRAPPED
    if _NL_GENERATE_WRAPPED:
        return
    from tau2.evaluator import evaluator_nl_assertions as nl

    original_generate = nl.generate

    def generate_with_clean_json(*args, **kwargs):
        msg = original_generate(*args, **kwargs)
        if kwargs.get("call_name") == "nl_assertions_eval":
            cleaned = _extract_json(getattr(msg, "content", None))
            if cleaned is not None:
                msg.content = cleaned
        return msg

    nl.generate = generate_with_clean_json
    _NL_GENERATE_WRAPPED = True


def _extract_json(text):
    """Return a JSON string parsed out of `text`, or None if none found.

    Handles: already-clean JSON, ```json fenced blocks, and JSON embedded in
    surrounding prose (by slicing from the first '{' to the last '}')."""
    import json as _json

    if not text or not str(text).strip():
        return None
    s = str(text).strip()

    # Fast path: already valid JSON.
    try:
        _json.loads(s)
        return s
    except ValueError:
        pass

    # Strip a ```json ... ``` (or plain ```) fence if present.
    if "```" in s:
        fenced = s.split("```")
        for chunk in fenced:
            chunk = chunk.strip()
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            try:
                _json.loads(chunk)
                return chunk
            except ValueError:
                continue

    # Last resort: slice from first '{' to last '}'.
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end + 1]
        try:
            _json.loads(candidate)
            return candidate
        except ValueError:
            return None
    return None


def _lookup_task(task_id, options):
    """Return the tau2 Task for `task_id`, repopulating the cache if needed.

    The cache is process-local; on a fresh crash-resume process run_attempt may
    fire before load_tasks for this id, so we reload defensively.
    """
    if task_id not in _TAU2_TASKS:
        load_tasks(
            domain=options.get("domain", DEFAULT_DOMAIN),
            split=options.get("split", DEFAULT_SPLIT),
        )
    if task_id not in _TAU2_TASKS:
        raise KeyError(f"tau2 task not found: {task_id!r}")
    return _TAU2_TASKS[task_id]


def _agent_system_prompt(agent):
    try:
        return agent.system_prompt
    except Exception:
        return getattr(agent, "domain_policy", "")


def _retry_context(history):
    parts = []
    for i, (_attempt, fb) in enumerate(history, 1):
        if fb:
            parts.append(f"[Attempt {i} feedback]\n{fb}")
    return "\n\n".join(parts)


def _render_transcript(messages):
    """Human-readable transcript of the conversation (the actor's 'output')."""
    lines = []
    for m in messages:
        role = getattr(m, "role", "?")
        content = (getattr(m, "content", None) or "").strip()
        tool_calls = getattr(m, "tool_calls", None) or []
        if content:
            lines.append(f"[{role}] {content}")
        for tc in tool_calls:
            name = getattr(tc, "name", None) or getattr(tc, "function_name", "tool")
            args = getattr(tc, "arguments", None)
            lines.append(f"[{role} -> tool] {name}({args})")
    return "\n".join(lines).strip() or "(no messages)"


def _public_failure_summary(reward_info, success):
    """Short, leak-safe summary of which checks failed.

    Uses only check *outcomes* (met/match flags + the public `info` strings that
    describe what the agent was supposed to communicate), never gold DB values.
    """
    if reward_info is None:
        return "No reward information was produced for this attempt."
    if success:
        return f"reward={reward_info.reward:.2f}"

    parts = [f"reward={reward_info.reward:.2f}"]

    db_check = getattr(reward_info, "db_check", None)
    if db_check is not None and not getattr(db_check, "db_match", True):
        parts.append(
            "- Environment end-state did not match what the task required "
            "(some required write action was missing, wrong, or extra)."
        )

    comm = getattr(reward_info, "communicate_checks", None) or []
    missed = [c for c in comm if not getattr(c, "met", True)]
    if missed:
        parts.append("- You did not communicate required information:")
        for c in missed:
            info = (getattr(c, "info", "") or "").strip()
            parts.append(f"    • {info}" if info else "    • (a required statement)")

    nl = getattr(reward_info, "nl_assertions", None) or []
    nl_missed = [a for a in nl if not getattr(a, "met", True)]
    if nl_missed:
        parts.append("- Some natural-language requirements were not met:")
        for a in nl_missed:
            assertion = (getattr(a, "nl_assertion", "") or "").strip()
            if assertion:
                parts.append(f"    • {assertion}")

    if len(parts) == 1:  # only the reward line
        parts.append("- The task requirements were not fully satisfied.")
    return "\n".join(parts)


def _judge_details(task_id, sim, seed):
    """Internal scratch — full reward breakdown + transcript. Never shown to the
    actor (the harness only feeds back raw_eval_output)."""
    reward_info = sim.reward_info
    breakdown = None
    if reward_info is not None:
        try:
            breakdown = reward_info.model_dump(mode="json")
        except Exception:
            breakdown = {"reward": getattr(reward_info, "reward", None)}
    agent_cost, user_cost = _attempt_cost(sim)
    return {
        "task_id": task_id,
        "seed": seed,
        "reward": float(reward_info.reward) if reward_info else 0.0,
        "reward_info": breakdown,
        # Per-attempt USD cost (LiteLLM's per-message cost, summed by role).
        # agent = the model under test; user = the user simulator. None if
        # LiteLLM couldn't price the model.
        "agent_cost": agent_cost,
        "user_cost": user_cost,
        "termination_reason": str(getattr(sim, "termination_reason", "") or ""),
        "duration": getattr(sim, "duration", None),
        "num_messages": len(sim.messages or []),
    }


def _attempt_cost(sim):
    """(agent_cost, user_cost) in USD for this simulation, via tau2's helper.

    Returns (None, None) if costs aren't available (e.g. LiteLLM has no pricing
    for the model). tau2 already attaches a per-message `cost` during the run;
    get_cost just sums it by role."""
    try:
        from tau2.utils.llm_utils import get_cost
        costs = get_cost(sim.messages or [])
        if costs is None:
            return None, None
        return costs  # (agent_cost, user_cost)
    except Exception:
        return None, None
