"""Tests for streaming output of dr.ask() and dr._run_research().

Streaming is the big UX win: the user sees the response word-by-word
instead of waiting 5+ seconds for the full answer.
"""
import pytest
from unittest.mock import MagicMock, patch


def _mock_stream_chunks(*texts):
    """Build a stream-like object that yields MagicMock chunks with delta.content."""
    chunks = []
    for t in texts:
        chunk = MagicMock()
        choice = MagicMock()
        delta = MagicMock()
        delta.content = t
        choice.delta = delta
        chunk.choices = [choice]
        chunks.append(chunk)
    return iter(chunks)


def test_ask_stream_returns_text_iterator():
    """ask(..., stream=True) returns a generator/iterator yielding text tokens."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_stream_chunks(
            "Hello", " world", "!"
        )

        result = ask("query", stream=True)

    # It should be iterable
    tokens = list(result)
    assert "Hello" in tokens
    assert " world" in tokens
    assert "!" in tokens


def test_ask_stream_sends_stream_true_to_api():
    """When stream=True, ask() passes stream=True to the OpenAI client."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_stream_chunks("a")

        list(ask("query", stream=True))

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True


def test_ask_default_is_non_streaming():
    """Without stream=True, ask() uses the non-streaming path."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        # Non-streaming: returns a response object
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "full response"
        resp.choices = [choice]
        resp.usage = None
        instance.chat.completions.create.return_value = resp

        text, _ = ask("query")

    assert text == "full response"
    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("stream", False) is False


def test_ask_stream_no_usage_available():
    """[documented limitation] Streaming responses don't include usage in the
    last chunk in the standard way; the test verifies we don't crash."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_stream_chunks("hi")

        # Should not raise even though usage info is unavailable mid-stream
        tokens = list(ask("query", stream=True))
    assert "hi" in tokens


def test_ask_stream_yields_in_order():
    """Streaming tokens arrive in the order they appear in the response."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_stream_chunks(
            "First", " second", " third"
        )

        tokens = list(ask("query", stream=True))

    # Order should be preserved
    assert tokens == ["First", " second", " third"]


# ────────────────────────── main() integration ──────────────────────────


def test_main_with_stream_flag_prints_tokens_progressively(capsys):
    """--stream prints tokens as they arrive, not waiting for the full response."""
    from dr import main

    with patch("dr.search_cached") as mock_search, \
         patch("dr.ask") as mock_ask, \
         patch("dr.subprocess"):
        mock_search.return_value = []
        # ask(stream=True) returns an iterator
        mock_ask.return_value = iter(["Hello", " world", "!"])

        main(["--stream", "query"])

    out = capsys.readouterr().out
    assert "Hello" in out
    assert " world" in out
    assert "!" in out


def test_main_without_stream_flag_uses_blocking_ask(capsys):
    """Without --stream, main() uses the non-streaming ask() that returns (text, usage)."""
    from dr import main

    with patch("dr.search_cached") as mock_search, \
         patch("dr.ask") as mock_ask, \
         patch("dr.subprocess"):
        mock_search.return_value = []
        mock_ask.return_value = ("full response", {"total_tokens": 10, "cost_usd": 0})

        main(["query"])

    # Non-streaming: ask() was called WITHOUT stream=True
    call_kwargs = mock_ask.call_args.kwargs
    assert call_kwargs.get("stream", False) is False
