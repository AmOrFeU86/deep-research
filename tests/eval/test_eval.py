"""Tests for #27: Eval suite — _run_eval() (the orchestrator).

For each entry in the gold set, _run_eval:
  1. Calls _run_research() to produce an answer
  2. Calls judge_answer() to score it
  3. Records the result (id, score, reason, cost, duration, pass/fail)
  4. Wraps everything in a treval OPERATION span for the whole run

The function returns a dict with aggregate stats + per-question details.
"""
import json
import time
from unittest.mock import patch

import dr


# ---------------------------------------------------------------------------
# Helpers: stub responses for _run_research and judge_answer
# ---------------------------------------------------------------------------

def _stub_research(answer: str = "stub answer", cost: float = 0.0001):
    """Shape that _run_research returns: (response, results, usage)."""
    usage = {
        "model": "stub", "prompt_tokens": 50, "completion_tokens": 20,
        "total_tokens": 70, "cost_usd": cost,
    }
    return (answer, [{"url": "https://x", "title": "X", "content": "x"}], usage)


def _stub_judge(score: float = 0.8, reason: str = "good"):
    """Shape that judge_answer returns: (score, reason)."""
    return (score, reason)


# ---------------------------------------------------------------------------
# Aggregate behavior
# ---------------------------------------------------------------------------

def test_run_eval_loads_default_gold_set():
    """With no gold_path arg, _run_eval uses the bundled gold.jsonl (10 entries)."""
    with patch("dr._run_research", return_value=_stub_research()), \
         patch("dr.judge_answer", return_value=_stub_judge()):
        report = dr._run_eval()
    assert report["num_questions"] == 10
    assert len(report["results"]) == 10


def test_run_eval_returns_aggregate_stats():
    """Report has mean_score, pass_rate, num_passed, num_failed, threshold."""
    with patch("dr._run_research", return_value=_stub_research()), \
         patch("dr.judge_answer", return_value=_stub_judge(0.9, "great")):
        report = dr._run_eval()
    assert "mean_score" in report
    assert "pass_rate" in report
    assert "num_passed" in report
    assert "num_failed" in report
    assert "threshold" in report
    assert report["threshold"] == dr.DEFAULT_PASS_THRESHOLD  # 0.7


def test_run_eval_per_question_result_has_required_fields():
    """Each entry in report['results'] has id, score, reason, passed, etc."""
    with patch("dr._run_research", return_value=_stub_research("a")), \
         patch("dr.judge_answer", return_value=_stub_judge(0.5, "meh")):
        report = dr._run_eval(gold_path=_make_minimal_gold())
    r = report["results"][0]
    assert "id" in r
    assert "question" in r
    assert "answer" in r
    assert "score" in r
    assert "reason" in r
    assert "passed" in r
    assert "cost_usd" in r
    assert "duration_ms" in r


def test_run_eval_uses_pass_threshold_to_mark_passed():
    """A score >= threshold is passed=True; below is passed=False."""
    with patch("dr._run_research", return_value=_stub_research()), \
         patch("dr.judge_answer", side_effect=[(0.9, "ok"), (0.5, "bad")]):
        report = dr._run_eval(gold_path=_make_minimal_gold(2), threshold=0.7)
    assert report["results"][0]["passed"] is True
    assert report["results"][1]["passed"] is False


def test_run_eval_mean_score_is_average_of_per_question_scores():
    """mean_score is the arithmetic mean of the per-question scores."""
    with patch("dr._run_research", return_value=_stub_research()), \
         patch("dr.judge_answer", side_effect=[(0.6, ""), (0.8, ""), (1.0, "")]):
        report = dr._run_eval(gold_path=_make_minimal_gold(3))
    assert abs(report["mean_score"] - 0.8) < 1e-9


def test_run_eval_pass_rate_is_fraction_passing():
    """pass_rate = num_passed / num_questions."""
    with patch("dr._run_research", return_value=_stub_research()), \
         patch("dr.judge_answer", side_effect=[(0.9, ""), (0.5, ""), (0.9, "")]):
        report = dr._run_eval(gold_path=_make_minimal_gold(3), threshold=0.7)
    assert report["num_passed"] == 2
    assert report["num_failed"] == 1
    assert abs(report["pass_rate"] - 2/3) < 1e-9


def test_run_eval_total_cost_sums_per_question_costs():
    """total_cost_usd is the sum of the research + judge cost per question."""
    def _fake_judge(*a, **kw):
        # The judge cost lives in the prompt_tokens of the judge's ask() call,
        # but at the orchestrator level we just attribute a flat 0.0001
        # per question to keep it simple (research is the bigger cost).
        return (0.5, "")
    with patch("dr._run_research", return_value=_stub_research(cost=0.0002)), \
         patch("dr.judge_answer", side_effect=_fake_judge):
        report = dr._run_eval(gold_path=_make_minimal_gold(3))
    # 3 × 0.0002 = 0.0006
    assert abs(report["total_cost_usd"] - 0.0006) < 1e-6


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_run_eval_judge_receives_answer_from_research():
    """The answer passed to the judge is the one _run_research produced."""
    with patch("dr._run_research", return_value=_stub_research("the actual answer")), \
         patch("dr.judge_answer", return_value=_stub_judge()) as mock_j:
        dr._run_eval(gold_path=_make_minimal_gold(1))
    assert mock_j.call_args.kwargs["answer"] == "the actual answer"


def test_run_eval_passes_depth_to_research():
    """The depth kwarg flows from _run_eval to _run_research."""
    with patch("dr._run_research", return_value=_stub_research()) as mock_r, \
         patch("dr.judge_answer", return_value=_stub_judge()):
        dr._run_eval(gold_path=_make_minimal_gold(1), depth=3)
    assert mock_r.call_args.kwargs.get("depth") == 3


def test_run_eval_default_depth_is_one():
    """depth=1 is the default for fast smoke runs (not production's DEFAULT_DEPTH)."""
    with patch("dr._run_research", return_value=_stub_research()) as mock_r, \
         patch("dr.judge_answer", return_value=_stub_judge()):
        dr._run_eval(gold_path=_make_minimal_gold(1))
    assert mock_r.call_args.kwargs.get("depth", 1) == 1


# ---------------------------------------------------------------------------
# Treval tracing
# ---------------------------------------------------------------------------

def test_run_eval_creates_one_operation_span_for_each_question():
    """Each question gets its own OPERATION span for clear span-tree inspection."""
    from treval.db import SpanStore
    store = SpanStore()
    store.clear()
    with patch("dr._run_research", return_value=_stub_research()), \
         patch("dr.judge_answer", return_value=_stub_judge()):
        dr._run_eval(gold_path=_make_minimal_gold(3))
    # The research + judge themselves add spans; we just check the eval spans
    # are present and count matches the number of questions.
    # Filter by name prefix to find the eval question spans specifically.
    spans = store.list_spans(limit=500)
    eval_question_spans = [
        s for s in spans if s.get("name", "").startswith("eval.question.")
    ]
    assert len(eval_question_spans) == 3


def test_run_eval_question_span_records_id_and_score_in_metadata():
    """The eval question span has metadata with id, score, passed — queryable
    from the treval dashboard."""
    from treval.db import SpanStore
    store = SpanStore()
    store.clear()
    with patch("dr._run_research", return_value=_stub_research()), \
         patch("dr.judge_answer", return_value=_stub_judge(0.8, "ok")):
        dr._run_eval(gold_path=_make_minimal_gold(1))
    spans = store.list_spans(limit=500)
    eval_q = [s for s in spans if s.get("name", "").startswith("eval.question.")][0]
    meta = json.loads(eval_q["metadata"])
    assert "id" in meta
    assert "score" in meta
    assert "passed" in meta
    assert meta["score"] == 0.8
    assert meta["passed"] is True


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------

import tempfile
from pathlib import Path


def _make_minimal_gold(n: int = 1) -> str:
    """Write a tiny gold.jsonl with N entries to a temp file and return its path."""
    tmp = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    with open(tmp, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": f"q{i}",
                "question": f"Question {i}?",
                "reference": f"Reference {i}.",
                "criteria": [f"criterion {i}.1", f"criterion {i}.2"],
            }) + "\n")
    return str(tmp)
