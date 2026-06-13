"""Tests for #18: Configurable timeout on the OpenAI client.

A default timeout prevents indefinite hangs when the API is slow or
unreachable. The timeout should be configurable per-call.
"""
from unittest.mock import MagicMock, patch


# ────────────────────────── ask() propagates timeout ──────────────────────────


def test_ask_uses_default_timeout_by_default():
    """ask() with no timeout arg uses dr.DEFAULT_TIMEOUT_SECONDS."""
    from dr import ask, DEFAULT_TIMEOUT_SECONDS

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15
        choice = MagicMock()
        choice.message.content = "ok"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        instance.chat.completions.create.return_value = resp

        ask("query")

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("timeout") == DEFAULT_TIMEOUT_SECONDS


def test_ask_accepts_custom_timeout():
    """ask(timeout=N) passes N through to the OpenAI client."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15
        choice = MagicMock()
        choice.message.content = "ok"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        instance.chat.completions.create.return_value = resp

        ask("query", timeout=7)

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("timeout") == 7


def test_ask_can_disable_timeout_with_none():
    """ask(timeout=None) explicitly disables the timeout (passes None to client)."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15
        choice = MagicMock()
        choice.message.content = "ok"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        instance.chat.completions.create.return_value = resp

        ask("query", timeout=None)

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("timeout") is None
