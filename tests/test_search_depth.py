"""Tests for #3: Tavily search_depth="advanced" support.

Tavily supports two search modes:
- "basic" (default): cheap, fast, good for most queries
- "advanced": 3x more expensive, deeper relevance and source quality

We expose this as a `search_depth` parameter, defaulting to "basic" for
backward compatibility, with the option to opt in.
"""
from unittest.mock import MagicMock, patch


# ────────────────────────── DEFAULT_SEARCH_DEPTH constant ──────────────────────────


def test_default_search_depth_constant_exists():
    """dr.DEFAULT_SEARCH_DEPTH is "basic" (cheap mode by default)."""
    import dr

    assert hasattr(dr, "DEFAULT_SEARCH_DEPTH")
    assert dr.DEFAULT_SEARCH_DEPTH == "basic"


# ────────────────────────── search() propagates search_depth ──────────────────────────


def test_search_passes_search_depth_to_tavily_client():
    """search(depth='advanced') propagates search_depth='advanced' to Tavily."""
    from dr import search

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": []}

        search("query", search_depth="advanced")

    call_kwargs = instance.search.call_args.kwargs
    assert call_kwargs.get("search_depth") == "advanced"


def test_search_defaults_to_basic_search_depth():
    """search() with no search_depth arg uses 'basic'."""
    from dr import search

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {"results": []}

        search("query")

    call_kwargs = instance.search.call_args.kwargs
    assert call_kwargs.get("search_depth") == "basic"


# ────────────────────────── search_cached() propagates search_depth ──────────────────────────


def test_search_cached_propagates_search_depth():
    """search_cached(search_depth=...) passes it through to the underlying search()."""
    from dr import search_cached

    with patch("dr.search") as mock_search:
        mock_search.return_value = []

        # First call to populate the cache, second call would hit the cache.
        search_cached("query", search_depth="advanced")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs.get("search_depth") == "advanced"
