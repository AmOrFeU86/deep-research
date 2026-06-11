"""Tests for #12: Interactive REPL.

A REPL session keeps a rolling history of Q&A pairs and feeds them as
context into subsequent questions, so follow-ups like 'what about its
climate?' make sense without restating the topic.
"""
import io
from unittest.mock import MagicMock, patch


SAMPLE_RESULTS = [
    {"url": "https://a.com", "title": "A", "content": "aaa"},
]


def _mock_research_factory(answers):
    """Return a mock _run_research that pops answers in order."""
    calls = []

    def mock(prompt, **kwargs):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        answer = answers.pop(0) if answers else "default"
        return answer, SAMPLE_RESULTS, {
            "model": "x", "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15, "cost_usd": 0.0001,
        }
    mock.calls = calls
    return mock


# ────────────────────────── pure helper: build_context() ──────────────────────────


def test_build_repl_context_empty_history():
    """build_repl_context([]) returns an empty string."""
    from dr import build_repl_context
    assert build_repl_context([]) == ""


def test_build_repl_context_single_qa():
    """build_repl_context([(q, a)]) includes the question and answer."""
    from dr import build_repl_context
    out = build_repl_context([("What is X?", "X is a thing.")])
    assert "What is X?" in out
    assert "X is a thing." in out


def test_build_repl_context_multiple_qa_preserves_order():
    """build_repl_context keeps Q&A in chronological order."""
    from dr import build_repl_context
    out = build_repl_context([
        ("Q1?", "A1."),
        ("Q2?", "A2."),
    ])
    # Q1 should appear before Q2
    assert out.index("Q1?") < out.index("Q2?")
    assert out.index("A1.") < out.index("A2.")


# ────────────────────────── run_repl() ──────────────────────────


def test_repl_skips_empty_input_and_reprompts(capsys):
    """Empty input is silently ignored (REPL re-prompts, like Python's REPL)."""
    from dr import run_repl

    inputs = iter(["", "What is X?", "/exit"])
    with patch("dr._run_research") as mock_research, \
         patch("dr.format_sources", return_value=""), \
         patch("dr.input", side_effect=inputs):
        mock_research.return_value = ("X is a thing.", [], {"total_tokens": 0, "cost_usd": 0.0})
        run_repl([])

    out = capsys.readouterr().out
    assert "REPL" in out or "Welcome" in out or ">" in out  # some kind of intro
    # The empty line did NOT trigger research; only the real "What is X?" did
    mock_research.assert_called_once()


def test_repl_exits_on_exit_command(capsys):
    """'/exit' (or 'exit') ends the REPL."""
    from dr import run_repl

    inputs = iter(["/exit"])
    with patch("dr._run_research") as mock_research, \
         patch("dr.format_sources", return_value=""), \
         patch("dr.input", side_effect=inputs):
        run_repl([])

    mock_research.assert_not_called()


def test_repl_exits_on_eof():
    """EOF (no more input) ends the REPL silently."""
    from dr import run_repl

    with patch("dr._run_research") as mock_research, \
         patch("dr.format_sources", return_value=""), \
         patch("dr.input", side_effect=EOFError):
        run_repl([])

    mock_research.assert_not_called()


def test_repl_processes_a_question(capsys):
    """A normal question is passed to _run_research and the answer printed."""
    from dr import run_repl

    inputs = iter(["What is X?", "/exit"])
    with patch("dr._run_research") as mock_research, \
         patch("dr.format_sources", return_value=""), \
         patch("dr.input", side_effect=inputs):
        mock_research.return_value = ("X is a thing.", [], {"total_tokens": 0, "cost_usd": 0.0})
        run_repl([])

    mock_research.assert_called_once()
    assert mock_research.call_args.args[0] == "What is X?"
    out = capsys.readouterr().out
    assert "X is a thing." in out


def test_repl_keeps_history_between_questions():
    """The second question's context includes the first Q&A."""
    from dr import run_repl

    mock_research = _mock_research_factory(["answer 1", "answer 2"])
    inputs = iter(["First?", "Second?", "/exit"])
    with patch("dr._run_research", side_effect=mock_research), \
         patch("dr.format_sources", return_value=""), \
         patch("dr.input", side_effect=inputs):
        run_repl([])

    assert len(mock_research.calls) == 2
    second = mock_research.calls[1]
    # The second call's prompt should still be the user's actual question
    assert second["prompt"] == "Second?"
    # But its context should mention the first Q&A
    context = second["kwargs"].get("context", "")
    assert "First?" in context
    assert "answer 1" in context


def test_repl_clears_history_on_clear_command():
    """'/clear' resets the history; the next question has no prior context."""
    from dr import run_repl

    mock_research = _mock_research_factory(["a1", "a2"])
    inputs = iter(["First?", "/clear", "Fresh?", "/exit"])
    with patch("dr._run_research", side_effect=mock_research), \
         patch("dr.format_sources", return_value=""), \
         patch("dr.input", side_effect=inputs):
        run_repl([])

    # Second call (after /clear) should have empty/None context
    second_call = mock_research.calls[1]
    ctx = second_call["kwargs"].get("context", "") or ""
    assert "First?" not in ctx
    assert "a1" not in ctx


def test_repl_skips_blank_lines():
    """Blank lines in the middle of a session don't trigger research."""
    from dr import run_repl

    mock_research = _mock_research_factory(["only"])
    inputs = iter(["Q?", "", "   ", "/exit"])
    with patch("dr._run_research", side_effect=mock_research), \
         patch("dr.format_sources", return_value=""), \
         patch("dr.input", side_effect=inputs):
        run_repl([])

    # Only the real "Q?" should trigger a research call
    assert len(mock_research.calls) == 1
