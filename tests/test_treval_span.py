"""Tests for the @treval.tool instrumentation on dr.search().

These tests verify that dr.search() is wrapped with @treval.tool so each
Tavily call is recorded as a TOOL span (input=query, output=results,
duration_ms>0, status=ok/error).

Note: We patch `sys.modules['treval.tool'].SpanStore` because the
`treval.tool` module's top-level name is shadowed by the `tool` function
when imported as `treval.tool`. The wrapper reads `SpanStore` from the
module's global scope on every call, so a module-attr swap works.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch


# ────────────────────────── Decorator presence ──────────────────────────


def test_search_is_decorated_with_treval_tool():
    """dr.search must have the @treval.tool marker so treval records spans."""
    import dr
    import treval

    assert getattr(dr.search, "_treval_tool", False) is True
    # _treval_name is the optional `name=` arg, default = func name
    assert getattr(dr.search, "_treval_name", None) == "tavily.search"


# ────────────────────────── Span creation (mocked SpanStore) ──────────────────────────


def test_search_creates_tool_span_with_query_in_input():
    """dr.search() records a TOOL span whose input contains the query string."""
    import dr

    mock_results = [
        {"url": "https://a.com", "title": "A", "content": "aaa"},
    ]
    with patch("dr.TavilyClient") as MockClient, \
         patch.object(sys.modules["treval.tool"], "SpanStore") as MockStore:
        instance = MockClient.return_value
        instance.search.return_value = {"results": mock_results}
        mock_store = MockStore.return_value

        dr.search("test query")

    mock_store.save.assert_called_once()
    call_kwargs = mock_store.save.call_args.kwargs
    assert call_kwargs["type"] == "TOOL"
    assert call_kwargs["name"] == "tavily.search"
    assert call_kwargs["status"] == "ok"
    assert "test query" in call_kwargs["input"]


def test_search_span_output_contains_result_urls():
    """The TOOL span's output contains the URLs returned by Tavily."""
    import dr

    mock_results = [
        {"url": "https://a.com", "title": "A", "content": "aaa"},
        {"url": "https://b.com", "title": "B", "content": "bbb"},
    ]
    with patch("dr.TavilyClient") as MockClient, \
         patch.object(sys.modules["treval.tool"], "SpanStore") as MockStore:
        instance = MockClient.return_value
        instance.search.return_value = {"results": mock_results}
        mock_store = MockStore.return_value

        dr.search("query")

    call_kwargs = mock_store.save.call_args.kwargs
    assert "https://a.com" in call_kwargs["output"]
    assert "https://b.com" in call_kwargs["output"]


def test_search_span_has_nonzero_duration():
    """The TOOL span's duration_ms is a positive number."""
    import dr

    with patch("dr.TavilyClient") as MockClient, \
         patch.object(sys.modules["treval.tool"], "SpanStore") as MockStore:
        instance = MockClient.return_value
        instance.search.return_value = {"results": []}
        mock_store = MockStore.return_value

        dr.search("query")

    call_kwargs = mock_store.save.call_args.kwargs
    assert "duration_ms" in call_kwargs
    assert call_kwargs["duration_ms"] > 0


def test_search_span_records_error_status_on_tavily_failure():
    """If Tavily raises, the TOOL span is saved with status='error'."""
    import dr

    with patch("dr.TavilyClient") as MockClient, \
         patch.object(sys.modules["treval.tool"], "SpanStore") as MockStore:
        instance = MockClient.return_value
        instance.search.side_effect = RuntimeError("Tavily 500")
        mock_store = MockStore.return_value

        with pytest.raises(RuntimeError, match="Tavily 500"):
            dr.search("query")

    call_kwargs = mock_store.save.call_args.kwargs
    assert call_kwargs["status"] == "error"
    assert "Tavily 500" in call_kwargs["output"]
