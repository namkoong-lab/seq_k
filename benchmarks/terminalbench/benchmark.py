"""TerminalBench via Harbor — the one agentic benchmark.

A TerminalBench "attempt" is not a single LLM completion: it's a full agent run
inside a Docker container, orchestrated by Harbor. So this benchmark implements
the harness's `run_attempt` hook instead of verify(): it shells out to
`harbor run --environment docker ...`, lets Harbor drive the agent + verifier in
the container, then parses Harbor's trial artifacts (result.json,
agent/trajectory.json, verifier/test-stdout.txt) into a VerifierResult.

We write no Docker code ourselves — Harbor manages containers/agent/verifier. We
just build the command, subprocess, and read the artifacts.

PREREQS (external, fail-loud if missing): a running Docker daemon, Harbor on PATH
(`uvx harbor`), and the agent's model auth (a provider key for terminus-2, or a
local Codex auth profile). Configure via a variant's `options:`.

Score is binary (Harbor verifier reward >= 1.0). Secret-looking values are redacted
from anything that becomes retry feedback.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from core.types import Task, VerifierResult

from . import prompts

# Curated 24-task Terminal-Bench 2 subset (systems/build, infra/webops, security,
# scientific/data, ml-systems).
SELECTED_TASKS = (
    "build-cython-ext", "cancel-async-tasks", "cobol-modernization", "kv-store-grpc",
    "merge-diff-arc-agi-task", "schemelike-metacircular-eval", "torch-tensor-parallelism",
    "configure-git-webserver", "mailman", "nginx-request-logging", "openssl-selfsigned-cert",
    "filter-js-from-html", "fix-code-vulnerability", "sanitize-git-repo",
    "dna-insert", "largest-eigenval", "modernize-scientific-stack", "multi-source-data-merger",
    "query-optimize", "raman-fitting", "reshard-c4-data", "sparql-university",
    "llm-inference-batching-scheduler", "pytorch-model-recovery",
)
DEFAULT_AGENT = "terminus-2"
DEFAULT_ENVIRONMENT = "docker"
DEFAULT_HARBOR_EXECUTABLE = "uvx harbor"
DEFAULT_DATASET = "terminal-bench/terminal-bench-2"
TASK_PREFIX = "terminal-bench/"
# Harbor caches each task package under this root; we read instruction.md from there
# to surface the raw dataset instruction in stored results.
HARBOR_PACKAGE_ROOT = Path("~/.cache/harbor/tasks/packages").expanduser()

_ERROR_PATTERNS = ("traceback", "error:", "exception", "command not found",
                   "no such file or directory", "segmentation fault", "assertionerror",
                   "failed", "failure", "permission denied", "timed out")
_SECRET_PATTERNS = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9_-]{8,}", r"or-[A-Za-z0-9_-]{8,}", r"ghp_[A-Za-z0-9_]{8,}",
    r"github_pat_[A-Za-z0-9_]{8,}", r"hf_[A-Za-z0-9_]{8,}", r"AKIA[0-9A-Z]{16}", r"ASIA[0-9A-Z]{16}")]


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def load_tasks(tasks=None, **_options):   # extra run-time options (agent, env, ...) are read by run_attempt
    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    ids = list(tasks) if tasks else list(SELECTED_TASKS)
    unknown = sorted(set(ids) - set(SELECTED_TASKS))
    if unknown:
        raise ValueError(f"unknown Terminal-Bench task(s): {', '.join(unknown)}")
    return [Task(id=tid, prompt=prompts.BASE_NOTE, grading={}) for tid in ids]


# --------------------------------------------------------------------------- #
# Attempt = a full Harbor agent run in Docker (the run_attempt hook)
# --------------------------------------------------------------------------- #
def run_attempt(task, history, t, k, *, seq, model, judge_model, temperature, options, out):
    agent = options.get("agent", DEFAULT_AGENT)
    environment = options.get("environment", DEFAULT_ENVIRONMENT)
    harbor_executable = options.get("harbor_executable", DEFAULT_HARBOR_EXECUTABLE)
    dataset = options.get("dataset", DEFAULT_DATASET)

    sidecar = _jobs_root(options, out) / _slug(task.id) / f"attempt_{t}"
    if sidecar.exists():
        shutil.rmtree(sidecar)
    sidecar.mkdir(parents=True, exist_ok=True)
    jobs_dir = sidecar / "jobs"
    job_name = f"{_slug(task.id)}-attempt-{t + 1}"
    job_dir = jobs_dir / job_name

    retry_context = _retry_context(history) if seq else ""
    prompt_text = prompts.build_prompt_template(retry_context)
    template_path = sidecar / "prompt_template.j2"
    template_path.write_text(prompt_text, encoding="utf-8")

    harbor_env = os.environ.copy()
    command = _build_command(harbor_executable, dataset, task.id, agent, model, environment,
                             jobs_dir, job_name, template_path, temperature, options, harbor_env)

    completed = subprocess.run(command, capture_output=True, text=True, env=harbor_env)

    parsed = None
    try:
        parsed = _parse_artifact(job_dir)
    except Exception as exc:
        if completed.returncode != 0:
            raise RuntimeError(
                "Harbor attempt failed before producing a trial artifact "
                f"(returncode={completed.returncode}).\nstderr:\n{completed.stderr[-1500:]}"
            ) from exc
        raise
    if parsed is None:
        raise RuntimeError("Harbor did not produce a readable trial artifact")

    # The actor's natural-language output is the agent's final message.
    actor_output = parsed.pop("final_agent_message", "") or parsed.get("trajectory_full") or parsed.get("verifier_summary", "")

    # The actor's stored prompt should be what the agent REALLY saw — Harbor renders
    # our template with the dataset's task instruction. Fall back to the template if
    # we can't recover the rendered version.
    rendered_prompt = parsed.pop("rendered_prompt", "") or prompt_text
    # Surface the dataset's raw task instruction (instruction.md) for analysis,
    # without leaking it into the actor prompt — it's already inside `rendered_prompt`
    # via Harbor's substitution.
    task_instruction = _read_task_instruction(options.get("dataset", DEFAULT_DATASET), task.id)

    success = parsed["success"]
    # `verifier_summary` is the same string we expose as the standard public diagnostic
    # (`raw_eval_output`); pop it so we don't store it twice.
    verifier_summary = parsed.pop("verifier_summary", "")
    result = VerifierResult(
        success=success,
        score=1.0 if success else 0.0,
        raw_eval_output=("" if success else verifier_summary),
        judge_details={
            **parsed,
            "task_id": task.id,
            "task_instruction": task_instruction,
            "prompt_template": prompt_text,
            "harbor_returncode": completed.returncode,
        },
    )
    return rendered_prompt, actor_output, result


def _read_task_instruction(dataset, task_id):
    """Return the dataset's raw task instruction (instruction.md), or "" if unavailable.

    Harbor caches each task package under
        HARBOR_PACKAGE_ROOT/<namespace>/<task_id>/<hash>/instruction.md
    Namespace comes from the part of `dataset` before the slash (e.g. "terminal-bench").
    """
    namespace = dataset.split("/", 1)[0] if "/" in dataset else "terminal-bench"
    task_dir = HARBOR_PACKAGE_ROOT / namespace / task_id
    if not task_dir.is_dir():
        return ""
    for hash_dir in sorted(task_dir.iterdir()):
        instruction = hash_dir / "instruction.md"
        if instruction.is_file():
            return _redact(instruction.read_text(encoding="utf-8", errors="replace"))
    return ""


def _build_command(harbor_executable, dataset, task_id, agent, model, environment,
                   jobs_dir, job_name, template_path, temperature, options, harbor_env):
    command = shlex.split(harbor_executable) + [
        "run", "-d", dataset, "-i", f"{TASK_PREFIX}{task_id}", "-l", "1",
        "-a", agent, "-m", str(model), "-e", environment, "-k", "1", "-n", "1",
        "-o", str(jobs_dir), "--job-name", job_name, "-q",
        "--ak", f"prompt_template_path={template_path}",
    ]
    if agent == "terminus-2" and temperature is not None:
        command += ["--ak", f"temperature={temperature}"]

    # Forward the agent's auth to Harbor by name (value stays in the subprocess env,
    # not on the command line).
    if agent == "codex":
        auth = Path(options.get("codex_auth_json") or "~/.codex/auth.json").expanduser()
        harbor_env["CODEX_AUTH_JSON_PATH"] = str(auth)
        harbor_env["CODEX_FORCE_AUTH_JSON"] = "1"
        command += ["--ae", "CODEX_AUTH_JSON_PATH=${CODEX_AUTH_JSON_PATH}",
                    "--ae", "CODEX_FORCE_AUTH_JSON=${CODEX_FORCE_AUTH_JSON}"]
    else:
        for key in (options.get("pass_env") or ["OPENAI_API_KEY"]):
            if os.environ.get(key):
                command += ["--ae", f"{key}=${{{key}}}"]
    return command


def _jobs_root(options, out):
    root = options.get("jobs_root")
    if root:
        return Path(root).expanduser()
    return (Path(out).parent / "_harbor_jobs") if out else Path("runs/_harbor_jobs")


def _retry_context(history):
    parts = []
    for i, (_attempt, fb) in enumerate(history, 1):
        if fb:
            parts.append(f"[Attempt {i} feedback]\n{fb}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Harbor artifact parsing
# --------------------------------------------------------------------------- #
def _parse_artifact(job_dir):
    results = sorted(p for p in job_dir.glob("*/result.json") if p.is_file() and p.parent != job_dir)
    if not results:
        raise FileNotFoundError(f"no Harbor trial result under {job_dir}")
    trial_dir = results[0].parent
    trial_result = _read_json(results[0])
    trajectory = _read_json(trial_dir / "agent" / "trajectory.json")
    verifier_output = _read_text(trial_dir / "verifier" / "test-stdout.txt")

    reward = _parse_reward(trial_result)
    success = reward is not None and reward >= 1.0
    exc = trial_result.get("exception_info") if isinstance(trial_result.get("exception_info"), dict) else {}
    exc_type = str(exc.get("exception_type") or "").strip()
    exc_msg = str(exc.get("exception_message") or "").strip()

    # All of these are stored verbatim (no truncation). The retry-feedback builder
    # in feedback.py trims for prompt size when needed.
    last_output_full = _redact(_last_output(trajectory))
    verifier_output_full = _redact(verifier_output)
    error_signals = _error_signals(verifier_output_full, last_output_full)
    verifier_summary = _redact(_verifier_summary(reward, success, exc_type, exc_msg, error_signals))
    return {
        "reward": reward,
        "success": success,
        "trial_name": str(trial_result.get("trial_name") or trial_dir.name),
        "rendered_prompt": _redact(_first_user_message(trajectory)),  # what Harbor actually sent the agent
        "last_command": _last_command(trajectory),
        "last_output": last_output_full,
        "error_signals": error_signals,
        "verifier_summary": verifier_summary,
        "verifier_output": verifier_output_full,
        "trajectory_full": _redact(_format_trace(trajectory)),
        "final_agent_message": _redact(_last_agent_message(trajectory)),
        "exception_type": exc_type,
        "exception_message": exc_msg,
    }


def _parse_reward(trial_result):
    verifier_result = trial_result.get("verifier_result")
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    if not isinstance(rewards, dict):
        return None
    try:
        return float(rewards.get("reward"))
    except (TypeError, ValueError):
        return None


def _verifier_summary(reward, success, exc_type, exc_msg, error_signals):
    if success:
        return f"reward={reward:.1f}" if reward is not None else "reward=1.0"
    parts = []
    if reward is not None:
        parts.append(f"reward={reward:.1f}")
    if error_signals:
        parts.append("remaining_issues:\n- " + "\n- ".join(error_signals))
    elif exc_type and exc_type != "AssertionError":
        parts.append(f"{exc_type}: {exc_msg}" if exc_msg else exc_type)
    return "\n\n".join(parts) if parts else "reward=0.0"


# --- trajectory.json extractors --------------------------------------------- #
def _steps(trajectory):
    steps = trajectory.get("steps")
    return steps if isinstance(steps, list) else []


def _tool_call_text(tool_call):
    args = tool_call.get("arguments")
    if isinstance(args, dict):
        for key in ("keystrokes", "command", "cmd", "input"):
            val = str(args.get(key) or "").strip()
            if val:
                return val
    return str(tool_call.get("function_name") or "").strip()


def _last_command(trajectory):
    for step in reversed(_steps(trajectory)):
        for tc in reversed(step.get("tool_calls") or []):
            if isinstance(tc, dict):
                cmd = _tool_call_text(tc)
                if cmd:
                    return cmd
    return ""


def _last_output(trajectory):
    """Full content of the last observation in the trajectory (no truncation)."""
    for step in reversed(_steps(trajectory)):
        obs = step.get("observation")
        results = obs.get("results") if isinstance(obs, dict) else None
        if isinstance(results, list):
            contents = [str(r.get("content") or "").strip() for r in results
                        if isinstance(r, dict) and str(r.get("content") or "").strip()]
            if contents:
                return "\n\n".join(contents)
    return ""


def _first_user_message(trajectory):
    """Harbor renders our prompt template with `{{ instruction }}` substituted and
    sends it to the agent as the first user message. That message IS the rendered
    actor prompt — what the agent actually saw."""
    for step in _steps(trajectory):
        if step.get("source") == "user":
            msg = str(step.get("message") or "").strip()
            if msg:
                return msg
    return ""


def _last_agent_message(trajectory):
    for step in reversed(_steps(trajectory)):
        if step.get("source") == "agent":
            msg = str(step.get("message") or "").strip()
            if msg:
                return msg
    return ""


def _format_trace(trajectory):
    """Full trajectory rendered as a human-readable transcript (no truncation)."""
    parts = []
    for step in _steps(trajectory):
        if not isinstance(step, dict) or step.get("source") in ("system", "user"):
            continue
        if step.get("source") == "agent":
            msg = str(step.get("message") or "").strip()
            if msg:
                parts.append(f"--- Agent Step {step.get('step_id', '?')} ---\n{msg}")
        for tc in step.get("tool_calls") or []:
            if isinstance(tc, dict):
                cmd = _tool_call_text(tc)
                if cmd:
                    parts.append(f"[Command] {cmd.rstrip()}")
        obs = step.get("observation")
        for r in (obs.get("results") if isinstance(obs, dict) else []) or []:
            if isinstance(r, dict) and str(r.get("content") or "").strip():
                parts.append(f"[Output]\n{str(r['content']).strip()}")
    return "\n".join(parts).strip() or "(no trajectory data)"


def _error_signals(*texts):
    """All error-flavored lines from the verifier output and terminal output (no cap)."""
    seen = []
    for text in texts:
        for line in str(text or "").splitlines():
            line = line.strip()
            if line and any(p in line.lower() for p in _ERROR_PATTERNS) and line not in seen:
                seen.append(line)
    return seen


# --- small utilities -------------------------------------------------------- #
def _redact(text):
    out = str(text or "")
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[redacted_secret]", out)
    return out


def _slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()) or "terminalbench"


def _read_json(path):
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_text(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
