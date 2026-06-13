"""Tests for the main() CLI flow: search → ask → print sources → optional report.

The whole pipeline is mocked — no network, no real LLM. We verify the
*orchestration*: order of calls, prompt plumbing, output formatting,
and the --report flag triggering treval dashboard export.

Since the default depth is 10, we patch `dr.reformulate` to return []
in most tests so they exercise the same single-query path the old
depth=1 default used. Tests that explicitly want to exercise
reformulation can leave the mock off.
"""
import subprocess
from unittest.mock import MagicMock, patch


SAMPLE_RESULTS = [
    {"url": "https://a.com", "title": "A", "content": "aaa"},
    {"url": "https://b.com", "title": "B", "content": "bbb"},
]


def _setup_mocks(mock_search, mock_ask, text="the answer"):
    mock_search.return_value = SAMPLE_RESULTS
    mock_ask.return_value = (text, {
        "model": "MiniMax-M3",
        "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        "cost_usd": 0.0001,
    })


def test_main_always_searches_before_asking(capsys):
    """main() runs search() before ask() — web research is always on."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask)
        main(["what is X?"])

    assert mock_search.called
    assert mock_ask.called
    # search must be called BEFORE ask
    assert mock_search.call_args_list[0].args[0] == "what is X?"
    # ask must receive the formatted context from search results
    ask_kwargs = mock_ask.call_args.kwargs
    assert "context" in ask_kwargs
    assert "https://a.com" in ask_kwargs["context"]


def test_main_prints_sources_after_response(capsys):
    """After the LLM answer, the sources from Tavily are printed to stdout."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask)
        main(["anything"])

    out = capsys.readouterr().out
    assert "https://a.com" in out
    assert "https://b.com" in out
    assert "Sources:" in out


def test_main_prints_llm_response(capsys):
    """The LLM's answer text is printed to stdout."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask, text="the answer is 42")
        main(["question"])

    out = capsys.readouterr().out
    assert "the answer is 42" in out


def test_main_reads_prompt_from_argv(capsys):
    """When argv has a prompt, main() uses it (no stdin read)."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.subprocess"), patch("dr.input") as mock_input:
        _setup_mocks(mock_search, mock_ask)
        main(["from argv"])

    mock_input.assert_not_called()
    # With reformulate mocked to [], search is called exactly once with the prompt
    assert mock_search.call_args.args[0] == "from argv"


def test_main_prompts_via_stdin_when_no_argv(capsys):
    """When no prompt in argv, main() reads from stdin."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.subprocess"), patch("dr.input", return_value="from stdin") as mock_input:
        _setup_mocks(mock_search, mock_ask)
        main([])

    mock_input.assert_called_once()
    assert mock_search.call_args.args[0] == "from stdin"


def test_main_runs_treval_dashboard_export_with_report_flag(capsys):
    """With --report, main() invokes `treval dashboard --export report.html`."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.subprocess.run") as mock_run:
        _setup_mocks(mock_search, mock_ask)
        main(["--report", "q"])

    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "treval"
    assert cmd[1] == "dashboard"
    assert "--export" in cmd
    assert cmd[cmd.index("--export") + 1].endswith("report.html")


def test_main_skips_dashboard_export_without_report_flag(capsys):
    """Without --report, main() does NOT invoke the dashboard export."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.subprocess.run") as mock_run:
        _setup_mocks(mock_search, mock_ask)
        main(["q"])

    mock_run.assert_not_called()


def test_main_exits_with_error_if_minimax_key_missing(capsys):
    """main() aborts with a clear message if MINIMAX_API_KEY is missing."""
    from dr import main

    with patch.dict("os.environ", {}, clear=True), \
         patch("dr.search") as mock_search:
        # Need to also remove TAVILY_API_KEY so search() doesn't try Tavily
        with __import__("pytest").raises(SystemExit) as exc:
            main(["q"])
        assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "MINIMAX_API_KEY" in out
    mock_search.assert_not_called()
