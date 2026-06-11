"""Tests for dr.search_cached() — local SQLite cache with TTL.

Caches Tavily search results to avoid repeat HTTP calls. Cache lives in
~/.treval/search_cache.db (per-test tmp file via the isolation fixture).
"""
import time
from unittest.mock import patch


# ────────────────────────── Cache miss / hit semantics ──────────────────────────


def test_search_cached_first_call_invokes_tavily():
    """First call for a query: Tavily is hit (cache miss)."""
    from dr import search_cached

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": [
            {"url": "https://a.com", "title": "A", "content": "x"},
        ]}

        results = search_cached("test query", max_results=3)

    instance.search.assert_called_once()
    assert len(results) == 1
    assert results[0]["url"] == "https://a.com"


def test_search_cached_second_call_with_same_query_is_cache_hit():
    """Second call with the same query: Tavily is NOT called (cache hit)."""
    from dr import search_cached

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": [
            {"url": "https://a.com", "title": "A", "content": "x"},
        ]}

        search_cached("same query")
        instance.search.reset_mock()
        results = search_cached("same query")

    instance.search.assert_not_called()
    assert results[0]["url"] == "https://a.com"


# ────────────────────────── Query normalization ──────────────────────────


def test_search_cached_normalizes_query_for_cache_key():
    """Whitespace and case differences map to the same cache entry."""
    from dr import search_cached

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": [
            {"url": "https://a.com", "title": "A", "content": "x"},
        ]}

        search_cached("Hello World")
        instance.search.reset_mock()
        search_cached("  hello world  ")  # case + whitespace differ
        search_cached("HELLO WORLD")

    # Only the first call should have hit Tavily
    assert instance.search.call_count == 0  # 2nd and 3rd were cache hits


# ────────────────────────── TTL expiry ──────────────────────────


def test_search_cached_expired_entry_triggers_refetch(monkeypatch):
    """If a cache entry is older than CACHE_TTL_SECONDS, it's re-fetched."""
    from dr import search_cached
    import dr

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": [
            {"url": "https://a.com", "title": "A", "content": "x"},
        ]}

        # First call populates the cache
        search_cached("query")
        instance.search.reset_mock()

        # Make all entries expire immediately
        monkeypatch.setattr(dr, "CACHE_TTL_SECONDS", 0)

        search_cached("query")  # Should re-fetch

    instance.search.assert_called_once()


# ────────────────────────── DB isolation ──────────────────────────


def test_search_cached_creates_cache_db_in_configured_path(_isolated_search_cache):
    """[integration-of-fixtures] The cache DB file is created at the configured path."""
    from dr import search_cached
    from pathlib import Path
    from unittest.mock import patch

    with patch("dr.TavilyClient") as MockClient:
        MockClient.return_value.search.return_value = {"results": [
            {"url": "https://a.com", "title": "A", "content": "x"},
        ]}
        search_cached("path test query")

    # The autouse _isolated_search_cache fixture redirected the path
    assert _isolated_search_cache.exists()
    assert _isolated_search_cache.stat().st_size > 0  # has the row
