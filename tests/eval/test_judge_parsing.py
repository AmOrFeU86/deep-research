"""Tests for #27: Eval suite — parse_judge_json().

The LLM judge is asked to return a JSON object with 'score' (0-1) and
'reason' (text). In practice the LLM often returns:

1. Pure JSON: {"score": 0.8, "reason": "good"}
2. JSON in a markdown code block
3. JSON with surrounding prose
4. Score out of [0, 1] range
5. Garbage (no recoverable JSON at all)

The parser must handle (1)-(4) and raise on (5) — we don't want a
silent zero from a parse failure that masks a real bug.
"""
import pytest

import dr


def test_parse_judge_json_pure_json():
    score, reason = dr.parse_judge_json('{"score": 0.85, "reason": "good answer"}')
    assert score == 0.85
    assert reason == "good answer"


def test_parse_judge_json_in_markdown_code_block():
    text = '```json\n{"score": 0.7, "reason": "ok"}\n```'
    score, reason = dr.parse_judge_json(text)
    assert score == 0.7
    assert reason == "ok"


def test_parse_judge_json_in_markdown_no_language():
    text = '```\n{"score": 0.5, "reason": "meh"}\n```'
    score, reason = dr.parse_judge_json(text)
    assert score == 0.5
    assert reason == "meh"


def test_parse_judge_json_with_surrounding_prose():
    text = 'Here is my evaluation:\n\n{"score": 0.9, "reason": "excellent"}\n\nDone.'
    score, reason = dr.parse_judge_json(text)
    assert score == 0.9
    assert reason == "excellent"


def test_parse_judge_json_score_above_one_is_clamped():
    """A score > 1.0 (LLM sometimes does this) is clamped to 1.0."""
    score, _ = dr.parse_judge_json('{"score": 1.5, "reason": "loved it"}')
    assert score == 1.0


def test_parse_judge_json_score_below_zero_is_clamped():
    """A negative score is clamped to 0.0."""
    score, _ = dr.parse_judge_json('{"score": -0.3, "reason": "hated it"}')
    assert score == 0.0


def test_parse_judge_json_score_as_string_is_coerced():
    """Some models return score as a quoted string like "0.7"."""
    score, reason = dr.parse_judge_json('{"score": "0.6", "reason": "ok"}')
    assert score == 0.6
    assert reason == "ok"


def test_parse_judge_json_missing_reason_returns_empty_string():
    """A judge that omits the reason field still parses — we just use ''."""
    score, reason = dr.parse_judge_json('{"score": 0.5}')
    assert score == 0.5
    assert reason == ""


def test_parse_judge_json_missing_score_raises():
    """Missing 'score' is a malformed judge output — must raise."""
    with pytest.raises(ValueError):
        dr.parse_judge_json('{"reason": "no score given"}')


def test_parse_judge_json_garbage_raises():
    """Completely non-JSON output raises — we want to know if the judge broke."""
    with pytest.raises(ValueError):
        dr.parse_judge_json("I cannot evaluate this question.")


def test_parse_judge_json_empty_string_raises():
    with pytest.raises(ValueError):
        dr.parse_judge_json("")


def test_parse_judge_json_int_score_is_accepted():
    """Score 0 or 1 (as int) is valid."""
    score, _ = dr.parse_judge_json('{"score": 1, "reason": "perfect"}')
    assert score == 1.0
    score, _ = dr.parse_judge_json('{"score": 0, "reason": "wrong"}')
    assert score == 0.0
