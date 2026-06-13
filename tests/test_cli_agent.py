"""Tests for agent-friendly CLI flags: --help, --version, --json, help subcommand.

These are UX/output-format flags (not quality knobs), designed to make dr
discoverable and machine-parseable for sub-agents and CI tooling.
"""
import json
import pytest
from unittest.mock import patch


def _setup_mocks(mock_search, mock_ask):
    """Common mock setup: search returns 1 result, ask returns a canned answer."""
    mock_search.return_value = [
        {"url": "https://example.com/x", "title": "X", "content": "snippet x"}
    ]
    mock_ask.return_value = ("the answer", {"total_tokens": 100,
                                            "prompt_tokens": 60,
                                            "completion_tokens": 40,
                                            "cost_usd": 0.0001})


# ────────────────────────── --help / --version ──────────────────────────


def test_main_help_prints_usage(capsys):
    """`dr --help` prints USAGE and exits 0."""
    from dr import main, USAGE
    with __import__("pytest").raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Deep Research CLI" in out
    assert "dr eval" in out
    assert "QUALITY KNOBS" in out


def test_main_short_help_works(capsys):
    """`dr -h` is the short form of --help."""
    from dr import main
    with __import__("pytest").raises(SystemExit) as exc:
        main(["-h"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Deep Research CLI" in out


def test_main_help_subcommand_prints_usage(capsys):
    """`dr help` (no subcommand) prints the top-level USAGE and exits 0."""
    from dr import main
    with __import__("pytest").raises(SystemExit) as exc:
        main(["help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Deep Research CLI" in out


def test_main_help_eval_prints_eval_usage(capsys):
    """`dr help eval` prints EVAL_USAGE with --json, --depth, etc., exits 0."""
    from dr import main
    with __import__("pytest").raises(SystemExit) as exc:
        main(["help", "eval"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--gold" in out
    assert "--depth" in out
    assert "--json" in out
    assert "EXIT CODES" in out


def test_main_version_prints_version(capsys):
    """`dr --version` prints the version string and exits 0."""
    from dr import main, __version__
    with __import__("pytest").raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert out.startswith("dr ")


def test_eval_help_prints_eval_usage(capsys):
    """`dr eval --help` prints EVAL_USAGE and exits 0.

    Verifies it shows the eval-specific doc (not the top-level USAGE
    that just happens to mention eval as a subcommand).
    """
    from dr import main, USAGE
    with __import__("pytest").raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # Eval-specific signals
    assert "EXIT CODES" in out
    assert "dr eval" in out
    # Top-level doc explicitly references the eval SUBCOMMAND, not eval FLAGS
    assert "FLAGS" in out
    assert out != USAGE  # not just the top-level help


# ────────────────────────── --json (single-shot) ──────────────────────────


def test_main_json_emits_valid_json(capsys):
    """`dr --json "q"` prints a JSON object with answer, sources, usage."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.parallel_search", return_value=[
             {"url": "https://a.com", "title": "A", "content": "aaa"},
             {"url": "https://b.com", "title": "B", "content": "bbb"},
         ]), \
         patch("dr.SELF_CRITIQUE", False):
        mock_ask.return_value = ("the answer", {"total_tokens": 100,
                                                 "prompt_tokens": 60,
                                                 "completion_tokens": 40,
                                                 "cost_usd": 0.0001,
                                                 "tavily_searches": 1,
                                                 "tavily_cost_usd": 0.001})
        main(["--json", "what is 1+1?"])

    out = capsys.readouterr().out
    payload = json.loads(out)  # raises if not valid JSON
    assert payload["answer"] == "the answer"
    assert isinstance(payload["sources"], list)
    assert len(payload["sources"]) == 2
    assert payload["sources"][0]["url"] == "https://a.com"
    assert payload["sources"][0]["title"] == "A"
    assert "snippet" in payload["sources"][0]
    assert payload["usage"]["cost_usd"] == 0.0001
    assert payload["usage"]["tavily_searches"] == 1


def test_main_json_progress_goes_to_stderr(capsys):
    """With --json, the progress line goes to stderr, not stdout."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.parallel_search", return_value=[]), \
         patch("dr.SELF_CRITIQUE", False):
        _setup_mocks(mock_search, mock_ask)
        main(["--json", "q"])

    out, err = capsys.readouterr()
    # stderr has the "🔍 Searching" line
    assert "🔍 Searching" in err
    # stdout is pure JSON, no progress emoji
    assert "🔍" not in out
    json.loads(out)  # stdout is valid JSON


# ────────────────────────── --json (eval) ──────────────────────────


def test_eval_json_emits_report_dict(capsys):
    """`dr eval --json` prints the full report as JSON."""
    from dr import main
    fake = {
        "num_questions": 2, "mean_score": 0.85, "pass_rate": 1.0,
        "num_passed": 2, "num_failed": 0, "threshold": 0.7,
        "total_cost_usd": 0.002,
        "results": [
            {"id": "a", "question": "q1", "answer": "ans1", "score": 1.0,
             "reason": "ok", "passed": True, "cost_usd": 0.001,
             "duration_ms": 100.0},
        ],
    }
    with patch("dr._run_eval", return_value=fake), \
         patch.dict("os.environ", {"MINIMAX_API_KEY": "x", "TAVILY_API_KEY": "y"}):
        main(["eval", "--json"])

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["num_questions"] == 2
    assert payload["mean_score"] == 0.85
    assert payload["pass_rate"] == 1.0
    assert payload["results"][0]["id"] == "a"


def test_eval_json_handles_unicode(capsys):
    """JSON output preserves non-ASCII characters (Spanish questions)."""
    from dr import main
    fake = {
        "num_questions": 1, "mean_score": 1.0, "pass_rate": 1.0,
        "num_passed": 1, "num_failed": 0, "threshold": 0.7,
        "total_cost_usd": 0.0,
        "results": [{"id": "q1", "question": "¿Cuál es la capital?",
                    "answer": "París", "score": 1.0, "reason": "ok",
                    "passed": True, "cost_usd": 0.0, "duration_ms": 0.0}],
    }
    with patch("dr._run_eval", return_value=fake), \
         patch.dict("os.environ", {"MINIMAX_API_KEY": "x", "TAVILY_API_KEY": "y"}):
        main(["eval", "--json"])

    out = capsys.readouterr().out
    assert "¿Cuál" in out  # not escaped as \u00bf...
    assert "París" in out
    json.loads(out)


# ────────────────────────── Non-TTY detection ──────────────────────────


def test_main_no_args_non_tty_exits_with_error(capsys):
    """`dr` with no args and non-TTY (e.g. agent subprocess) exits 2 with usage hint."""
    from dr import main

    with patch("dr.sys.stdin.isatty", return_value=False), \
         patch("dr.input") as mock_input, \
         patch.dict("os.environ", {"MINIMAX_API_KEY": "x", "TAVILY_API_KEY": "y"}):
        with __import__("pytest").raises(SystemExit) as exc:
            main([])

    assert exc.value.code == 2
    mock_input.assert_not_called()
    err = capsys.readouterr().err
    assert "missing PROMPT" in err
    assert "--help" in err


def test_main_empty_prompt_non_tty_exits_1(capsys):
    """`dr ""` (whitespace only) exits 1, not silent success."""
    from dr import main

    with patch.dict("os.environ", {"MINIMAX_API_KEY": "x", "TAVILY_API_KEY": "y"}):
        with __import__("pytest").raises(SystemExit) as exc:
            main(["   "])

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Nothing to ask" in err
