"""Pure unit tests for the shared MCQ core — no network, no model calls."""
from core.types import Attempt
from benchmarks import _mcq

OPTS = ["aspirin", "penicillin", "insulin", "warfarin"]


def _task(gold=1):
    return _mcq.make_task("t1", "Which drug?", OPTS, gold_index=gold)


def test_parse_answer_colon_last_wins():
    assert _mcq.parse_letter("Maybe B, but Answer: C", OPTS) == "C"


def test_parse_standalone_parenthesised_letter():
    assert _mcq.parse_letter("After reasoning, (D).", OPTS) == "D"


def test_parse_option_text_fallback():
    assert _mcq.parse_letter("It should be penicillin", OPTS) == "B"


def test_parse_garbage_returns_none():
    assert _mcq.parse_letter("I am not sure about this one", OPTS) is None


def test_parse_rejects_out_of_range_letter():
    # E is not a valid option (only A-D); fall through, no option text -> None
    assert _mcq.parse_letter("Answer: E", OPTS) is None


def test_verify_correct():
    r = _mcq.verify(_task(gold=1), Attempt(0, "Answer: B"))
    assert r.success is True and r.score == 1.0
    assert r.raw_eval_output == ""
    assert r.judge_details["parse_ok"] is True


def test_verify_incorrect_does_not_leak_gold():
    r = _mcq.verify(_task(gold=1), Attempt(0, "Answer: A"))
    assert r.success is False and r.score == 0.0
    assert "You answered A" in r.raw_eval_output
    assert "B" not in r.raw_eval_output            # gold letter never shown
    assert r.judge_details["gold_letter"] == "B"   # kept internal only


def test_verify_unparseable_is_failure_with_hint():
    r = _mcq.verify(_task(gold=1), Attempt(0, "no idea honestly"))
    assert r.success is False and r.score == 0.0
    assert r.judge_details["parse_ok"] is False
    assert "Answer:" in r.raw_eval_output


def test_feedback_binary_and_raw_no_llm():
    task = _task(gold=1)
    r = _mcq.verify(task, Attempt(0, "Answer: A"))
    assert "incorrect" in _mcq.feedback(task, Attempt(0, "Answer: A"), r, "binary",
                                        judge_model="x").lower()
    assert _mcq.feedback(task, Attempt(0, "Answer: A"), r, "raw",
                         judge_model="x") == r.raw_eval_output


def test_feedback_unknown_mode_raises():
    task = _task(gold=1)
    r = _mcq.verify(task, Attempt(0, "Answer: A"))
    try:
        _mcq.feedback(task, Attempt(0, "Answer: A"), r, "nope", judge_model="x")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")
