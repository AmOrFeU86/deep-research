"""Tests for the web search functionality powered by Tavily.

Two layers:
- Unit tests: mock TavilyClient, verify dr.search() behavior
- Integration tests (marked): real Tavily API, validate contract
"""
import pytest
from unittest.mock import MagicMock, patch


# ────────────────────────── Unit tests (no network) ──────────────────────────


def test_search_returns_list_of_result_dicts():
    """search() returns a list of dicts with url/title/content keys."""
    from dr import search

    mock_results = [
        {"url": "https://a.com", "title": "A", "content": "aaa"},
        {"url": "https://b.com", "title": "B", "content": "bbb"},
        {"url": "https://c.com", "title": "C", "content": "ccc"},
    ]
    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": mock_results}

        results = search("test query")

    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert "url" in r
        assert "title" in r
        assert "content" in r


def test_search_passes_query_to_tavily():
    """search() forwards the query string to TavilyClient.search()."""
    from dr import search

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": []}

        search("deepseek v4 release date")

    instance.search.assert_called_once()
    call_kwargs = instance.search.call_args.kwargs
    assert call_kwargs["query"] == "deepseek v4 release date"


def test_search_default_max_results_is_three():
    """search() requests 3 results by default (the project's chosen default)."""
    from dr import search

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": []}

        search("anything")

    call_kwargs = instance.search.call_args.kwargs
    assert call_kwargs.get("max_results") == 3


def test_search_respects_explicit_max_results():
    """search(query, max_results=N) forwards N to Tavily."""
    from dr import search

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": []}

        search("anything", max_results=7)

    call_kwargs = instance.search.call_args.kwargs
    assert call_kwargs["max_results"] == 7


def test_search_loads_api_key_from_env():
    """search() reads TAVILY_API_KEY from the environment."""
    from dr import search

    with patch("dr.TavilyClient") as MockClient, \
         patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-test-key"}):
        search("anything")

    MockClient.assert_called_once_with(api_key="tvly-test-key")


# ────────────────────────── Integration tests (real Tavily) ──────────────────────────


@pytest.mark.integration
def test_search_real_tavily_returns_results(tavily_key):
    """End-to-end: real Tavily call returns a list with expected shape."""
    from dr import search

    results = search("deepseek v4 model release", max_results=2)

    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 2
    first = results[0]
    assert "url" in first and first["url"].startswith("http")
    assert "title" in first and first["title"]
    assert "content" in first and first["content"]
