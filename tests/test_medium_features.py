"""Tests for medium-priority UX and resilience features.

Covered:
- temperature=0 (deterministic research)
- --max-results flag
- retry with exponential backoff
- model fallback (Pro → Flash)
"""
import pytest
import time
from unittest.mock import MagicMock, patch


def _mock_llm_response(text: str = "ok", prompt_tokens: int = 10, completion_tokens: int = 5):
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


# ────────────────────────── temperature=0 ──────────────────────────


def test_ask_uses_temperature_zero():
    """ask() sends temperature=0 for deterministic research output."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_response("ok")

        ask("query")

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0


# ────────────────────────── --max-results ──────────────────────────


def test_max_results_flag_passes_n_to_search():
    """--max-results N propagates N through to Tavily's max_results."""
    from dr import main
    from unittest.mock import patch

    with patch("dr.search_cached") as mock_search, \
         patch("dr.ask") as mock_ask, \
         patch("dr.subprocess"):
        mock_search.return_value = []
        mock_ask.return_value = ("ok", {"total_tokens": 0, "cost_usd": 0})
        main(["--max-results", "7", "query"])

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs["max_results"] == 7


def test_max_results_flag_default_is_three():
    """Without --max-results, search_cached is called with max_results=3."""
    from dr import main

    with patch("dr.search_cached") as mock_search, \
         patch("dr.ask") as mock_ask, \
         patch("dr.subprocess"):
        mock_search.return_value = []
        mock_ask.return_value = ("ok", {"total_tokens": 0, "cost_usd": 0})
        main(["query"])

    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs["max_results"] == 3


def test_max_results_flag_works_with_depth():
    """--max-results and --depth can be used together."""
    from dr import main

    with patch("dr.search_cached") as mock_search, \
         patch("dr.ask") as mock_ask, \
         patch("dr.subprocess"):
        mock_search.return_value = []
        mock_ask.return_value = ("ok", {"total_tokens": 0, "cost_usd": 0})
        main(["--max-results", "5", "--depth", "2", "query"])

    # depth=2 means original + 1 reformulation = 2 search_cached calls
    assert mock_search.call_count == 2
    for call in mock_search.call_args_list:
        assert call.kwargs["max_results"] == 5


# ────────────────────────── retry with backoff ──────────────────────────


def test_ask_retries_on_transient_error():
    """ask() retries on transient errors; succeeds if any attempt works."""
    from dr import ask
    import dr

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        # Fail twice, then succeed
        instance.chat.completions.create.side_effect = [
            ConnectionError("network"),
            TimeoutError("timeout"),
            _mock_llm_response("ok"),
        ]
        # Patch sleep to be instant
        with patch.object(dr.time, "sleep"):
            text, _ = ask("query")

    assert text == "ok"
    assert instance.chat.completions.create.call_count == 3


def test_ask_raises_after_max_retries():
    """ask() gives up after MAX_RETRIES; re-raises the last exception."""
    from dr import ask
    import dr

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = ConnectionError("always fails")
        with patch.object(dr.time, "sleep"):
            with pytest.raises(ConnectionError, match="always fails"):
                ask("query")

    # Default MAX_RETRIES is 3 (1 original + 2 retries), or some defined value
    assert instance.chat.completions.create.call_count >= 2


def test_ask_backoff_intervals_grow_exponentially():
    """ask() sleeps with exponentially growing intervals between retries."""
    from dr import ask
    import dr

    sleep_calls = []
    with patch("dr.OpenAI") as MockOpenAI, \
         patch.object(dr.time, "sleep", side_effect=lambda s: sleep_calls.append(s)):
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = ConnectionError("fail")
        with pytest.raises(ConnectionError):
            ask("query")

    # The first sleep should be shorter than the second (exponential growth)
    assert len(sleep_calls) >= 2
    assert sleep_calls[1] >= sleep_calls[0]


def test_ask_no_retry_on_non_transient_error():
    """ask() does not retry on non-transient errors (e.g. auth failure)."""
    from dr import ask
    import dr
    from openai import AuthenticationError

    with patch("dr.OpenAI") as MockOpenAI, \
         patch.object(dr.time, "sleep") as mock_sleep:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = AuthenticationError("bad key", response=MagicMock(), body=None)
        with pytest.raises(AuthenticationError):
            ask("query")

    # No retries on auth errors
    assert instance.chat.completions.create.call_count == 1
    mock_sleep.assert_not_called()


# ────────────────────────── model fallback ──────────────────────────


def test_ask_falls_back_to_flash_when_pro_fails():
    """If the primary model (e.g. pro) raises, ask() retries with the fallback."""
    from dr import ask
    import dr

    with patch("dr.OpenAI") as MockOpenAI, \
         patch.object(dr.time, "sleep"):
        instance = MockOpenAI.return_value
        # First call (pro) fails, second call (flash) succeeds
        instance.chat.completions.create.side_effect = [
            ConnectionError("pro down"),
            _mock_llm_response("ok from flash", 50, 20),
        ]

        text, usage = ask("query", model="deepseek/deepseek-v4-pro",
                          fallback="deepseek/deepseek-v4-flash",
                          max_retries=1)

    assert text == "ok from flash"
    assert instance.chat.completions.create.call_count == 2
    # The second call should have used the fallback model
    second_call_model = instance.chat.completions.create.call_args_list[1].kwargs["model"]
    assert second_call_model == "deepseek/deepseek-v4-flash"
    # And the usage reflects the fallback
    assert usage["model"] == "deepseek/deepseek-v4-flash"


def test_ask_uses_primary_model_when_it_succeeds():
    """If the primary model works, no fallback is attempted."""
    from dr import ask
    import dr

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_response("ok from pro")

        text, usage = ask("query", model="deepseek/deepseek-v4-pro",
                          fallback="deepseek/deepseek-v4-flash", max_retries=1)

    assert text == "ok from pro"
    assert usage["model"] == "deepseek/deepseek-v4-pro"
    assert instance.chat.completions.create.call_count == 1


def test_ask_raises_after_primary_and_fallback_both_fail():
    """If both primary and fallback fail, the original primary error is raised."""
    from dr import ask
    import dr

    with patch("dr.OpenAI") as MockOpenAI, \
         patch.object(dr.time, "sleep"):
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = [
            ConnectionError("pro down"),
            ConnectionError("flash also down"),
        ]

        with pytest.raises(ConnectionError, match="pro down"):
            ask("query", model="deepseek/deepseek-v4-pro",
                fallback="deepseek/deepseek-v4-flash", max_retries=1)
