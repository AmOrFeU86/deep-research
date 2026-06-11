"""Tests for #8: Structured metadata in tavily.search TOOL spans.

The @treval.tool decorator now supports a metadata_fn that attaches a
JSON-encoded dict to the span (query, max_results, num_results). The
dashboard of treval already parses this and renders it in the detail
panel, so each Tavily span is now self-describing.
"""
import json
from unittest.mock import patch

import dr


def test_search_cached_span_has_query_metadata():
    """The tavily.search span records the query string in its metadata."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    fake_results = [{"url": "https://a.com", "title": "A", "content": "a"}]
    with patch("dr.search", return_value=fake_results):
        dr.search_cached("quantum entanglement", max_results=3)

    spans = store.list_spans(type="TOOL")
    assert len(spans) == 1
    assert spans[0]["name"] == "tavily.search"
    meta = json.loads(spans[0]["metadata"])
    assert meta["query"] == "quantum entanglement"


def test_search_cached_metadata_records_max_results():
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    with patch("dr.search", return_value=[]):
        dr.search_cached("test", max_results=7)

    meta = json.loads(store.list_spans(type="TOOL")[0]["metadata"])
    assert meta["max_results"] == 7


def test_search_cached_metadata_records_num_results():
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    fake_results = [{"url": "https://a.com", "title": "A", "content": "a"},
                    {"url": "https://b.com", "title": "B", "content": "b"},
                    {"url": "https://c.com", "title": "C", "content": "c"}]
    with patch("dr.search", return_value=fake_results):
        dr.search_cached("test", max_results=5)

    meta = json.loads(store.list_spans(type="TOOL")[0]["metadata"])
    assert meta["num_results"] == 3


def test_search_cached_metadata_uses_default_max_results():
    """When max_results is omitted, metadata records the default value."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    with patch("dr.search", return_value=[]):
        dr.search_cached("test")  # no max_results arg

    meta = json.loads(store.list_spans(type="TOOL")[0]["metadata"])
    assert meta["max_results"] == dr.DEFAULT_MAX_RESULTS


def test_search_cached_metadata_records_all_three_keys():
    """Sanity: query, max_results, num_results are all present."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    fake_results = [{"url": f"https://{i}.com", "title": "", "content": ""}
                    for i in range(2)]
    with patch("dr.search", return_value=fake_results):
        dr.search_cached("hello world", max_results=5)

    meta = json.loads(store.list_spans(type="TOOL")[0]["metadata"])
    assert set(meta.keys()) >= {"query", "max_results", "num_results"}
    assert meta["query"] == "hello world"
    assert meta["max_results"] == 5
    assert meta["num_results"] == 2


def test_search_cached_metadata_zero_results_when_empty():
    """Empty result list still produces a span with num_results=0."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    with patch("dr.search", return_value=[]):
        dr.search_cached("nothing here", max_results=3)

    meta = json.loads(store.list_spans(type="TOOL")[0]["metadata"])
    assert meta["num_results"] == 0
