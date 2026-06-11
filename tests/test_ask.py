"""Tests for ask() with search context and source formatting.

Strategy: mock the OpenAI client (deterministic, no API cost) and verify
the messages sent to the LLM. Format helpers are pure functions tested
directly.
"""
import pytest
from unittest.mock import MagicMock, patch


# ────────────────────────── Format helpers (pure functions) ──────────────────────────


def test_format_context_includes_url_title_content():
    """format_context() embeds every result's url, title, and content."""
    from dr import format_context

    results = [
        {"url": "https://a.com", "title": "Alpha", "content": "aaa"},
        {"url": "https://b.com", "title": "Beta",  "content": "bbb"},
    ]
    out = format_context(results)

    assert "https://a.com" in out
    assert "Alpha" in out
    assert "aaa" in out
    assert "https://b.com" in out
    assert "Beta" in out
    assert "bbb" in out


def test_format_context_numbers_each_source():
    """format_context() numbers sources so the LLM can cite them ([1], [2], ...)."""
    from dr import format_context

    results = [
        {"url": "https://a.com", "title": "A", "content": "x"},
        {"url": "https://b.com", "title": "B", "content": "y"},
        {"url": "https://c.com", "title": "C", "content": "z"},
    ]
    out = format_context(results)

    assert "[1]" in out
    assert "[2]" in out
    assert "[3]" in out


def test_format_context_handles_empty_results():
    """format_context([]) returns a meaningful empty string, not None or crash."""
    from dr import format_context

    out = format_context([])
    assert isinstance(out, str)
    assert out  # non-empty, so the LLM still gets a coherent prompt


def test_format_sources_returns_url_lines():
    """format_sources() returns a printable string with one URL per source."""
    from dr import format_sources

    results = [
        {"url": "https://a.com", "title": "A", "content": "x"},
        {"url": "https://b.com", "title": "B", "content": "y"},
    ]
    out = format_sources(results)

    assert "https://a.com" in out
    assert "https://b.com" in out
    assert out.count("\n") >= 1  # multi-line


# ────────────────────────── ask() with context (mocked LLM) ──────────────────────────


def _mock_openai_response(text: str = "answer", prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a mock OpenAI chat.completions response object."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    choice = MagicMock()
    choice.message.content = text

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_ask_without_context_sends_one_user_message():
    """ask(prompt) with no context sends exactly one user message (and a system one)."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_openai_response("hi")

        ask("hello")

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    msgs = call_kwargs["messages"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "hello"


def test_ask_with_context_includes_context_in_user_message():
    """ask(prompt, context) embeds the context string in the user message (not system)."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_openai_response("answer")

        ask("what is X?", context="X is a thing. See [1].")

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    msgs = call_kwargs["messages"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert "X is a thing. See [1]." in user_msgs[0]["content"]
    assert "what is X?" in user_msgs[0]["content"]


def test_ask_returns_response_text_and_usage():
    """ask() returns (text, usage_dict) tuple as before."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_openai_response("hi", 20, 8)

        text, usage = ask("hello")

    assert text == "hi"
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == 28
