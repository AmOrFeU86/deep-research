"""Tests for #9: Source report as structured span input/output.

The parent OPERATION span (research / research_stream / research_agentic)
stores the full source list as structured metadata, not just prints
them to stdout. This makes the sources queryable and inspectable from
the treval dashboard without re-running the search.

Metadata shape:
    {
        "num_sources": int,
        "sources": [{"url": str, "title": str, "content": str}, ...],
        "tavily_searches": int,
        "tavily_cost_usd": float,
    }
"""
import json
from unittest.mock import patch

import dr


def test_research_span_metadata_has_num_sources():
    """The parent OPERATION span records how many unique sources were used."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    fake_results = [
        {"url": "https://a.com", "title": "A", "content": "a"},
        {"url": "https://b.com", "title": "B", "content": "b"},
        {"url": "https://c.com", "title": "C", "content": "c"},
    ]
    with patch("dr.search", return_value=fake_results), \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.SELF_CRITIQUE", False), \
         patch("dr.ask", return_value=("answer", {
             "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
             "cost_usd": 0.0,
         })):
        dr._run_research("test", depth=1)

    spans = store.list_spans(type="OPERATION")
    assert len(spans) == 1
    meta = json.loads(spans[0]["metadata"])
    assert meta["num_sources"] == 3


def test_research_span_metadata_has_full_source_list():
    """Every deduped source (url, title, content) is stored in the metadata."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    fake_results = [
        {"url": "https://a.com", "title": "Title A", "content": "content A"},
        {"url": "https://b.com", "title": "Title B", "content": "content B"},
    ]
    with patch("dr.search", return_value=fake_results), \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.SELF_CRITIQUE", False), \
         patch("dr.ask", return_value=("answer", {
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "cost_usd": 0.0,
         })):
        dr._run_research("test", depth=1)

    meta = json.loads(store.list_spans(type="OPERATION")[0]["metadata"])
    assert len(meta["sources"]) == 2
    assert meta["sources"][0] == fake_results[0]
    assert meta["sources"][1] == fake_results[1]
    # Make sure no field is dropped
    assert "url" in meta["sources"][0]
    assert "title" in meta["sources"][0]
    assert "content" in meta["sources"][0]


def test_research_span_metadata_has_tavily_counts():
    """tavily_searches and tavily_cost_usd from the research run are stored."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    # reformulate returns 2 extra queries, so depth=3 means 1+2=3 total
    fake_results = [
        {"url": "https://a.com", "title": "A", "content": "a"},
    ]
    with patch("dr.search", return_value=fake_results), \
         patch("dr.reformulate", return_value=["q2", "q3"]), \
         patch("dr.SELF_CRITIQUE", False), \
         patch("dr.ask", return_value=("answer", {
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "cost_usd": 0.0,
         })):
        dr._run_research("test", depth=3)

    meta = json.loads(store.list_spans(type="OPERATION")[0]["metadata"])
    # depth=3 means 1 original query + 2 reformulated = 3 searches
    assert meta["tavily_searches"] == 3
    assert "tavily_cost_usd" in meta
    assert isinstance(meta["tavily_cost_usd"], (int, float))


def test_research_span_metadata_handles_empty_results():
    """When Tavily returns nothing, the span has num_sources=0 and sources=[]."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    with patch("dr.search", return_value=[]), \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.SELF_CRITIQUE", False), \
         patch("dr.ask", return_value=("no info", {
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "cost_usd": 0.0,
         })):
        dr._run_research("test", depth=1)

    meta = json.loads(store.list_spans(type="OPERATION")[0]["metadata"])
    assert meta["num_sources"] == 0
    assert meta["sources"] == []


def test_research_span_metadata_reflects_dedup():
    """After URL dedup, the metadata source count matches the deduped list."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    # Tavily returns the same URL twice; the run dedupes by URL.
    fake_results = [
        {"url": "https://a.com", "title": "A", "content": "a"},
        {"url": "https://a.com", "title": "A2", "content": "a2"},  # dup
        {"url": "https://b.com", "title": "B", "content": "b"},
    ]
    with patch("dr.search", return_value=fake_results), \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.SELF_CRITIQUE", False), \
         patch("dr.ask", return_value=("answer", {
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "cost_usd": 0.0,
         })):
        dr._run_research("test", depth=1)

    meta = json.loads(store.list_spans(type="OPERATION")[0]["metadata"])
    assert meta["num_sources"] == 2
    assert len(meta["sources"]) == 2


def test_agentic_research_span_metadata_has_sources():
    """The agentic (ReAct) loop variant also stores sources in the OPERATION span."""
    from treval.db import SpanStore

    store = SpanStore()
    store.clear()

    fake_results = [
        {"url": "https://a.com", "title": "A", "content": "a"},
    ]
    # First LLM call: search action; second LLM call: final answer.
    search_action = '{"action": "search", "query": "follow-up"}'
    final_answer = "The answer based on [1]."
    with patch("dr.search_cached", return_value=fake_results), \
         patch("dr.ask", side_effect=[
             (search_action, {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2, "cost_usd": 0.0}),
             (final_answer, {"prompt_tokens": 1, "completion_tokens": 1,
                             "total_tokens": 2, "cost_usd": 0.0}),
         ]):
        dr.run_research_agentic("test", max_iterations=3)

    spans = store.list_spans(type="OPERATION")
    assert len(spans) == 1
    meta = json.loads(spans[0]["metadata"])
    assert meta["num_sources"] >= 1
    assert any(s["url"] == "https://a.com" for s in meta["sources"])
