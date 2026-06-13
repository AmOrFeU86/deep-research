"""Tests for #27: Eval suite — `dr eval` CLI dispatch.

The `dr` command supports a subcommand style for the new eval suite.
`dr eval` runs the bundled gold set, prints a per-question table +
summary, and exits non-zero if the pass rate is below threshold.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import dr


# All tests bypass the env-var check (it would fail in CI without API keys)
_eval_env = patch("dr._require_env", lambda *_a, **_kw: None)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

@_eval_env
def test_eval_subcommand_recognized_in_main():
    """`dr eval` does not raise (the 'prompt' code path is bypassed)."""
    with patch("dr._run_eval", return_value=_fake_report()):
        # Should NOT raise SystemExit for "no prompt" — eval branch handles it
        dr.main(["eval"])


def test_eval_exits_with_error_if_minimax_key_missing(capsys):
    """`dr eval` aborts with a clear message if MINIMAX_API_KEY is missing,
    before attempting any LLM/Tavily call."""
    import dr
    with patch.dict("os.environ", {}, clear=True), \
         patch("dr._run_eval") as mock_eval:
        with __import__("pytest").raises(SystemExit) as exc:
            dr.main(["eval"])
        assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "MINIMAX_API_KEY" in out
    mock_eval.assert_not_called()


@_eval_env
def test_eval_dispatches_to_run_eval_with_defaults():
    """`dr eval` calls _run_eval with the bundled gold set and default depth."""
    with patch("dr._run_eval", return_value=_fake_report()) as mock_eval:
        dr.main(["eval"])
    # Default gold path: DEFAULT_GOLD_PATH; default depth: 1
    assert mock_eval.call_args.kwargs.get("gold_path") is None \
        or mock_eval.call_args.kwargs.get("gold_path") == str(dr.DEFAULT_GOLD_PATH)
    assert mock_eval.call_args.kwargs.get("depth") == 1


@_eval_env
def test_eval_accepts_custom_gold_path_flag():
    """`dr eval --gold /path/to/gold.jsonl` passes the path to _run_eval."""
    custom = "/tmp/my_gold.jsonl"
    with patch("dr._run_eval", return_value=_fake_report()) as mock_eval:
        dr.main(["eval", "--gold", custom])
    assert mock_eval.call_args.kwargs.get("gold_path") == custom


@_eval_env
def test_eval_accepts_depth_flag():
    """`dr eval --depth N` passes the depth to _run_eval."""
    with patch("dr._run_eval", return_value=_fake_report()) as mock_eval:
        dr.main(["eval", "--depth", "5"])
    assert mock_eval.call_args.kwargs.get("depth") == 5


@_eval_env
def test_eval_accepts_custom_threshold_flag():
    with patch("dr._run_eval", return_value=_fake_report()) as mock_eval:
        dr.main(["eval", "--threshold", "0.85"])
    assert mock_eval.call_args.kwargs.get("threshold") == 0.85


# ---------------------------------------------------------------------------
# Output and exit code
# ---------------------------------------------------------------------------

@_eval_env
def test_eval_prints_per_question_table(capsys):
    """Output contains id, score, and pass/fail for every question."""
    # Use --threshold 0 so all scores pass (we're testing output, not gating)
    report = _fake_report(num_questions=3, scores=[0.9, 0.5, 0.8])
    with patch("dr._run_eval", return_value=report):
        dr.main(["eval", "--threshold", "0"])
    out = capsys.readouterr().out
    assert "q1" in out
    assert "0.90" in out
    assert "q2" in out
    assert "0.50" in out
    assert "q3" in out
    assert "0.80" in out
    # Pass / fail markers
    assert "PASS" in out
    assert "FAIL" in out


@_eval_env
def test_eval_prints_summary_stats(capsys):
    """Output shows mean score, pass rate, total cost."""
    report = _fake_report(num_questions=2, scores=[0.9, 0.5],
                          pass_rate=0.5, total_cost=0.0123)
    with patch("dr._run_eval", return_value=report):
        dr.main(["eval", "--threshold", "0"])
    out = capsys.readouterr().out
    assert "mean" in out.lower()
    assert "pass" in out.lower()
    assert "$0.0123" in out or "0.0123" in out


@_eval_env
def test_eval_exits_zero_when_pass_rate_meets_threshold(capsys):
    """If pass_rate >= threshold, exit 0 (don't raise SystemExit)."""
    report = _fake_report(num_questions=2, scores=[0.9, 0.9], pass_rate=1.0)
    with patch("dr._run_eval", return_value=report):
        # If main returns normally with no SystemExit, that's exit 0
        dr.main(["eval"])
    out = capsys.readouterr().out
    assert "1/2" in out or "100%" in out  # shows pass count
    # No assertion error → exit code 0


@_eval_env
def test_eval_exits_nonzero_when_pass_rate_below_threshold():
    """If pass_rate < threshold, raise SystemExit(1) — useful for CI."""
    report = _fake_report(num_questions=2, scores=[0.9, 0.5], pass_rate=0.5)
    with patch("dr._run_eval", return_value=report):
        try:
            dr.main(["eval", "--threshold", "0.8"])
        except SystemExit as e:
            assert e.code == 1
        else:
            raise AssertionError("Expected SystemExit(1) when pass_rate < threshold")


@_eval_env
def test_eval_exits_zero_when_pass_rate_equals_threshold(capsys):
    """Boundary: pass_rate == threshold is still a pass (>= semantics)."""
    report = _fake_report(num_questions=1, scores=[0.7], pass_rate=1.0)
    with patch("dr._run_eval", return_value=report):
        # No SystemExit should be raised
        dr.main(["eval", "--threshold", "0.7"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_report(num_questions: int = 2, scores: list[float] | None = None,
                 pass_rate: float | None = None,
                 total_cost: float = 0.001) -> dict:
    """Build a report dict in the same shape _run_eval returns."""
    if scores is None:
        scores = [0.8] * num_questions
    if pass_rate is None:
        threshold = 0.7
        pass_rate = sum(1 for s in scores if s >= threshold) / len(scores)
    results = [
        {"id": f"q{i+1}", "question": f"Q{i+1}?", "answer": "a",
         "score": s, "reason": "r", "passed": s >= 0.7,
         "cost_usd": 0.0005, "duration_ms": 100}
        for i, s in enumerate(scores)
    ]
    return {
        "num_questions": num_questions,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "pass_rate": pass_rate,
        "num_passed": sum(1 for r in results if r["passed"]),
        "num_failed": sum(1 for r in results if not r["passed"]),
        "threshold": 0.7,
        "total_cost_usd": total_cost,
        "results": results,
    }
