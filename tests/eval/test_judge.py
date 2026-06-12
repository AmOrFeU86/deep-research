"""Tests for #27: Eval suite — judge_answer().

The judge is an LLM call that takes (question, reference, criteria, answer)
and returns a (score, reason) tuple. The judge's output is parsed by
parse_judge_json(), so this layer is mostly about prompt construction
and the wiring of ask() + parser.
"""
from unittest.mock import patch

import dr


def _stub_ask_response(text: str) -> tuple:
    """Build the same (text, usage) tuple that dr.ask() returns."""
    return (text, {
        "model": "stub", "prompt_tokens": 10, "completion_tokens": 5,
        "total_tokens": 15, "cost_usd": 0.00001,
    })


def test_judge_answer_calls_ask_with_judge_model():
    """judge_answer uses the judge_model arg (default flash), not DEFAULT_MODEL."""
    with patch("dr.ask", return_value=_stub_ask_response(
        '{"score": 0.8, "reason": "ok"}'
    )) as mock_ask:
        dr.judge_answer(
            question="Q?", reference="R", answer="A", criteria=["c1"],
            model="deepseek/deepseek-v4-pro",
            judge_model="deepseek/deepseek-v4-flash",
        )
    # The judge call is the one to the cheap flash model
    assert mock_ask.call_args.kwargs["model"] == "deepseek/deepseek-v4-flash"


def test_judge_answer_prompt_includes_question():
    """The judge's prompt must contain the question (sanity check)."""
    with patch("dr.ask", return_value=_stub_ask_response(
        '{"score": 0.5, "reason": "partial"}'
    )) as mock_ask:
        dr.judge_answer(
            question="What is the capital of France?",
            reference="Paris",
            answer="Paris",
            criteria=["mentions Paris"],
            model="x", judge_model="judge-model",
        )
    user_msg = mock_ask.call_args.kwargs["prompt"]
    assert "What is the capital of France?" in user_msg


def test_judge_answer_prompt_includes_reference():
    with patch("dr.ask", return_value=_stub_ask_response(
        '{"score": 0.5, "reason": "partial"}'
    )) as mock_ask:
        dr.judge_answer(
            question="Q?", reference="Paris is the capital of France.",
            answer="Paris", criteria=["c1"], model="x", judge_model="judge-model",
        )
    user_msg = mock_ask.call_args.kwargs["prompt"]
    assert "Paris is the capital of France." in user_msg


def test_judge_answer_prompt_includes_answer_being_graded():
    with patch("dr.ask", return_value=_stub_ask_response(
        '{"score": 0.5, "reason": "partial"}'
    )) as mock_ask:
        dr.judge_answer(
            question="Q?", reference="R", answer="This is the answer to grade.",
            criteria=["c1"], model="x", judge_model="judge-model",
        )
    user_msg = mock_ask.call_args.kwargs["prompt"]
    assert "This is the answer to grade." in user_msg


def test_judge_answer_prompt_includes_criteria():
    """Each criterion is included as a checklist item the judge can verify."""
    with patch("dr.ask", return_value=_stub_ask_response(
        '{"score": 0.5, "reason": "partial"}'
    )) as mock_ask:
        dr.judge_answer(
            question="Q?", reference="R", answer="A",
            criteria=["mentions Paris", "does not invent capitals"],
            model="x", judge_model="judge-model",
        )
    user_msg = mock_ask.call_args.kwargs["prompt"]
    assert "mentions Paris" in user_msg
    assert "does not invent capitals" in user_msg


def test_judge_answer_returns_parsed_score_and_reason():
    """The returned tuple is (score, reason) parsed from the judge LLM output."""
    with patch("dr.ask", return_value=_stub_ask_response(
        '{"score": 0.85, "reason": "good answer"}'
    )):
        score, reason = dr.judge_answer(
            question="Q?", reference="R", answer="A", criteria=["c1"],
            model="x", judge_model="judge-model",
        )
    assert score == 0.85
    assert reason == "good answer"


def test_judge_answer_handles_markdown_wrapped_judge_response():
    """The judge often returns ```json ... ``` — parse_judge_json handles that."""
    with patch("dr.ask", return_value=_stub_ask_response(
        '```json\n{"score": 0.7, "reason": "ok"}\n```'
    )):
        score, reason = dr.judge_answer(
            question="Q?", reference="R", answer="A", criteria=["c1"],
            model="x", judge_model="judge-model",
        )
    assert score == 0.7
    assert reason == "ok"


def test_judge_answer_propagates_parse_error():
    """If the judge returns garbage, judge_answer raises (we don't silently 0.0)."""
    from dr import parse_judge_json
    with patch("dr.ask", return_value=_stub_ask_response("not json at all")):
        try:
            dr.judge_answer(
                question="Q?", reference="R", answer="A", criteria=["c1"],
                model="x", judge_model="judge-model",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError on unparseable judge output")
