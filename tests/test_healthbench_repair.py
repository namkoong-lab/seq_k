from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import healthbench
from benchmarks.healthbench import feedback as healthbench_feedback
from benchmarks.healthbench import prompts as healthbench_prompts
from core import harness, ids, results
from core import llm as core_llm
from core.types import Attempt, Task, VerifierResult


class HealthBenchRepairTests(unittest.TestCase):
    def setUp(self):
        self.task = Task(
            id="TASK_SECRET_SENTINEL",
            canonical_index=1,
            prompt="PROMPT_SECRET_SENTINEL",
            grading={
                "protocol": "healthbench-repair-v1",
                "rubrics": [{"criterion": "RUBRIC_SECRET_SENTINEL", "points": 1}],
                "prompt_messages": [{"role": "user", "content": "QUESTION_SECRET_SENTINEL"}],
                "secrets": [],
            },
        )
        self.attempt = Attempt(1, "ACTOR_RESPONSE_SENTINEL")
        self.result = VerifierResult(
            success=False,
            score=0.123456789,
            raw_eval_output="VERIFIER_SECRET_SENTINEL",
            details={"private": "DETAIL_SECRET_SENTINEL"},
        )

    def test_self_blind_critic_is_always_actor(self):
        got = harness.resolve_critic_model(
            healthbench,
            "self_blind",
            actor_model="openrouter/actor/model",
            critic_model="openrouter/stale/critic",
        )
        self.assertEqual(got, "openrouter/actor/model")
        judge = harness.resolve_critic_model(
            healthbench,
            "judge",
            actor_model="openrouter/actor/model",
            critic_model="openrouter/critic/model",
        )
        self.assertEqual(judge, "openrouter/critic/model")

    def test_self_blind_prompt_contains_only_actor_output(self):
        seen = {}

        def complete(model, prompt, temperature, *, reject_truncated=False):
            seen.update(
                model=model,
                prompt=prompt,
                temperature=temperature,
                reject_truncated=reject_truncated,
            )
            return "  improve clarity  "

        with patch("benchmarks.healthbench.feedback.llm.complete", side_effect=complete):
            output = healthbench_feedback(
                self.task,
                self.attempt,
                self.result,
                "self_blind",
                critic_model="openrouter/actor/model",
            )

        self.assertEqual(output, "  improve clarity  ")
        self.assertEqual(seen["model"], "openrouter/actor/model")
        self.assertTrue(seen["reject_truncated"])
        self.assertIn("ACTOR_RESPONSE_SENTINEL", seen["prompt"])
        for forbidden in (
            "TASK_SECRET_SENTINEL",
            "PROMPT_SECRET_SENTINEL",
            "QUESTION_SECRET_SENTINEL",
            "RUBRIC_SECRET_SENTINEL",
            "VERIFIER_SECRET_SENTINEL",
            "DETAIL_SECRET_SENTINEL",
            "0.123456789",
        ):
            self.assertNotIn(forbidden, seen["prompt"])
        for grading_claim in ("pass", "fail", "score", "rubric", "grader", "verdict"):
            self.assertNotIn(grading_claim, seen["prompt"].lower())

    def test_empty_llm_feedback_fails_loudly(self):
        for mode in ("judge", "self_blind"):
            with self.subTest(mode=mode), patch(
                "benchmarks.healthbench.feedback.llm.complete", return_value="  "
            ):
                with self.assertRaisesRegex(RuntimeError, "empty output"):
                    healthbench_feedback(
                        self.task,
                        self.attempt,
                        self.result,
                        mode,
                        critic_model="openrouter/actor/model",
                    )

    def test_truncated_llm_feedback_is_retried_then_fails_loudly(self):
        def response(content, finish_reason):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )]
            )

        with patch.object(core_llm, "_EMPTY_RETRIES", 1), patch(
            "core.llm.litellm.completion",
            side_effect=[response("partial", "length"), response("complete", "stop")],
        ) as completion:
            output = core_llm.complete(
                "openrouter/example/model", "prompt", 0.7, reject_truncated=True
            )
        self.assertEqual(output, "complete")
        self.assertEqual(completion.call_count, 2)

        with patch.object(core_llm, "_EMPTY_RETRIES", 1), patch(
            "core.llm.litellm.completion",
            side_effect=[response("partial 1", "length"), response("partial 2", "length")],
        ):
            with self.assertRaisesRegex(RuntimeError, "truncated completion"):
                core_llm.complete(
                    "openrouter/example/model", "prompt", 0.7, reject_truncated=True
                )

    def test_protocol_is_validated_and_changes_identity(self):
        with self.assertRaisesRegex(ValueError, "unknown HealthBench protocol"):
            healthbench_prompts.protocol("typo")

        common = dict(
            benchmark_module=healthbench,
            metric="pass@k",
            k=5,
            model="openrouter/example/actor",
            judge_model="openrouter/openai/gpt-5.4",
            critic_model="openrouter/example/actor",
            feedback_mode="binary",
            context="na",
            prompt_variant="v1",
            temperature=0.7,
            seed=42,
            summarizer_model="openrouter/example/actor",
        )
        unversioned = ids.identity(options={}, **common)
        repair_v1 = ids.identity(
            options={"protocol": "healthbench-repair-v1"}, **common
        )
        self.assertNotEqual(ids.fingerprint(unversioned), ids.fingerprint(repair_v1))
        self.assertIn("healthbench-repair-v1", repair_v1["slice_key"])

    def test_qwen_reasoning_effort_uses_openrouter_native_shape(self):
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="answer"), finish_reason="stop"
            )]
        )
        with patch("core.llm.litellm.completion", return_value=response) as completion:
            self.assertEqual(
                core_llm.complete(
                    "openrouter/qwen/qwen3.6-max-preview",
                    "prompt",
                    0.7,
                    reasoning_effort="low",
                ),
                "answer",
            )
        kwargs = completion.call_args.kwargs
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(kwargs["extra_body"]["reasoning"], {"effort": "low"})

    def test_pass_runs_exactly_k_while_seq_stops_on_success(self):
        benchmark = types.ModuleType("benchmarks.healthbench_test_double")
        benchmark.VERIFIER = "llm"
        benchmark.LLM_CRITIC_MODES = set()
        benchmark.verify = lambda task, attempt, *, judge_model: VerifierResult(
            success=True, score=1.0, raw_eval_output="", details={}
        )
        benchmark.feedback = lambda *args, **kwargs: self.fail(
            "feedback must not run after success"
        )
        task = Task(id="t", canonical_index=1, prompt="question", grading={})

        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.harness.llm.complete", return_value="answer"
        ) as complete:
            pass_dir = Path(tmp) / "pass"
            pass_traj = harness.run_task(
                benchmark,
                task,
                prior=[],
                metric="pass@k",
                k=5,
                feedback_mode="binary",
                model="actor",
                judge_model="judge",
                critic_model="actor",
                temperature=0.7,
                console_char_limit=0,
                out=str(pass_dir),
            )
            self.assertEqual(len(pass_traj.steps), 5)
            self.assertEqual(complete.call_count, 5)
            self.assertEqual(
                [step.attempt_index for step in pass_traj.steps], [1, 2, 3, 4, 5]
            )
            self.assertEqual(
                [step.actor["prompt"] for step in pass_traj.steps], ["question"] * 5
            )
            self.assertTrue(all(
                step.critic["feedback"] is None and not step.critic["calls"]
                for step in pass_traj.steps
            ))

            prior = results.load_task_attempts(str(pass_dir), task.canonical_index)
            complete.reset_mock()
            resumed = harness.run_task(
                benchmark,
                task,
                prior=prior,
                metric="pass@k",
                k=5,
                feedback_mode="binary",
                model="actor",
                judge_model="judge",
                critic_model="actor",
                temperature=0.7,
                console_char_limit=0,
                out=str(pass_dir),
            )
            self.assertEqual(len(resumed.steps), 5)
            complete.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp, patch(
            "core.harness.llm.complete", return_value="answer"
        ) as complete:
            seq_traj = harness.run_task(
                benchmark,
                task,
                prior=[],
                metric="seq@k",
                k=5,
                feedback_mode="binary",
                model="actor",
                judge_model="judge",
                critic_model="actor",
                temperature=0.7,
                console_char_limit=0,
                out=str(Path(tmp) / "seq"),
            )
            self.assertEqual(len(seq_traj.steps), 1)
            self.assertEqual(complete.call_count, 1)


if __name__ == "__main__":
    unittest.main()
