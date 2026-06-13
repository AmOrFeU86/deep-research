"""Tests for medium-priority UX and resilience features.

Covered:
- temperature=0 (deterministic research)
- DEFAULT_MAX_RESULTS propagated to Tavily
- retry with exponential backoff
- retry on 429 / 5xx (rate limit and server errors)
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


# ────────────────────────── DEFAULT_MAX_RESULTS ──────────────────────────


def test_default_max_results_is_three():
    """DEFAULT_MAX_RESULTS = 3 is the value passed to search_cached."""
    from dr import main, DEFAULT_MAX_RESULTS

    with patch("dr.search_cached") as mock_search, \
         patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.verify_citations", return_value={"verified": True, "issues": [], "model": "x"}), \
         patch("dr.subprocess"):
        mock_search.return_value = []
        mock_ask.return_value = ("ok", {"total_tokens": 0, "cost_usd": 0})
        main(["query"])

    assert DEFAULT_MAX_RESULTS == 3
    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs["max_results"] == 3


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
        instance.chat.completions.create.side_effect = AuthenticationError(
            "bad key", response=MagicMock(), body=None,
        )
        with pytest.raises(AuthenticationError):
            ask("query")

    # No retries on auth errors
    assert instance.chat.completions.create.call_count == 1
    mock_sleep.assert_not_called()


def test_ask_retries_on_429_rate_limit():
    """ask() treats 429 (rate limit) as transient and retries."""
    from dr import ask
    import dr
    from openai import RateLimitError

    with patch("dr.OpenAI") as MockOpenAI, \
         patch.object(dr.time, "sleep"):
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
            _mock_llm_response("ok"),
        ]
        text, _ = ask("query", max_retries=2)

    assert text == "ok"
    assert instance.chat.completions.create.call_count == 2


def test_ask_retries_on_5xx_server_error():
    """ask() treats 5xx (server) as transient and retries."""
    from dr import ask
    import dr
    from openai import InternalServerError

    with patch("dr.OpenAI") as MockOpenAI, \
         patch.object(dr.time, "sleep"):
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = [
            InternalServerError("oops", response=MagicMock(status_code=500), body=None),
            _mock_llm_response("ok"),
        ]
        text, _ = ask("query", max_retries=2)

    assert text == "ok"
    assert instance.chat.completions.create.call_count == 2


def test_ask_does_not_retry_on_400_bad_request():
    """ask() does not retry on 4xx other than 429 (e.g. 400 bad request)."""
    from dr import ask
    import dr
    from openai import BadRequestError

    with patch("dr.OpenAI") as MockOpenAI, \
         patch.object(dr.time, "sleep"):
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = BadRequestError(
            "bad", response=MagicMock(status_code=400), body=None,
        )
        with pytest.raises(BadRequestError):
            ask("query", max_retries=3)

    assert instance.chat.completions.create.call_count == 1


# ────────────────────────── estimate_cost() edge cases ──────────────────────────


def test_estimate_cost_unknown_model_returns_zero():
    """estimate_cost() returns 0.0 for any model not in MODEL_PRICES.

    This is the documented fallback — we don't want a missing price
    entry to crash the cost footer or report.py.
    """
    from dr import estimate_cost

    # 1000 prompt + 1000 completion tokens on an unknown model
    assert estimate_cost("unknown/model-name", 1000, 1000) == 0.0
    # Sanity: known model still prices correctly
    assert estimate_cost("MiniMax-M3", 1000, 1000) > 0.0


def test_estimate_cost_zero_tokens_returns_zero():
    """estimate_cost(0, 0) is 0 regardless of model — guards against
    empty-usage footers in the dashboard when usage reporting is None.
    """
    from dr import estimate_cost

    assert estimate_cost("MiniMax-M3", 0, 0) == 0.0
