"""Tests for the parent-child span hierarchy of dr._run_research().

When main() runs a research task, the tavily.search TOOL span and the
LLM span should both have a common OPERATION parent named "research".
That way the dashboard can show a tree:

    research (OPERATION)
    ├── tavily.search (TOOL)
    └── llm.deepseek... (LLM)
"""
import pytest
from unittest.mock import MagicMock, patch


def _mock_llm_response(text: str = "the answer", prompt_tokens: int = 50, completion_tokens: int = 10):
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


def test_run_research_creates_root_operation_span():
    """_run_research() saves an OPERATION span named 'research' with the query as input."""
    from dr import _run_research
    from treval.db import SpanStore

    with patch("dr.TavilyClient") as MockClient, \
         patch("dr.OpenAI") as MockOpenAI:
        MockClient.return_value.search.return_value = {"results": []}
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_llm_response("ok")

        _run_research("what is X?")

    store = SpanStore()
    roots = store.list_spans(limit=10, type="OPERATION")
    assert len(roots) == 1
    assert roots[0]["name"] == "research"
    assert roots[0]["status"] == "ok"
    assert "what is X?" in (roots[0]["input"] or "")


def test_tavily_search_span_is_child_of_research():
    """In cache-miss mode, both search_cached and search create TOOL spans under research."""
    from dr import _run_research
    from treval.db import SpanStore

    with patch("dr.TavilyClient") as MockClient, \
         patch("dr.OpenAI") as MockOpenAI:
        MockClient.return_value.search.return_value = {"results": [
            {"url": "https://a.com", "title": "A", "content": "aaa"},
        ]}
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_llm_response("ok")

        _run_research("query")

    store = SpanStore()
    roots = store.list_spans(limit=10, type="OPERATION")
    research_id = roots[0]["id"]

    # Cache miss → 2 TOOL spans: the cache wrapper + the underlying search()
    tools = store.list_spans(limit=10, type="TOOL")
    assert len(tools) == 2
    for t in tools:
        assert t["name"] == "tavily.search"
        assert t["parent_id"] == research_id


def test_llm_span_is_child_of_research(openrouter_key, tavily_key):
    """[integration] Real OpenAI + treval.instrument: LLM span has research as parent."""
    from dr import _run_research
    from treval.db import SpanStore

    _run_research("what is 1+1?")

    store = SpanStore()
    roots = store.list_spans(limit=10, type="OPERATION")
    research_id = roots[0]["id"]

    llms = store.list_spans(limit=10, type="LLM")
    assert len(llms) == 1
    assert llms[0]["name"].startswith("llm.")
    assert llms[0]["parent_id"] == research_id


test_llm_span_is_child_of_research = pytest.mark.integration(test_llm_span_is_child_of_research)


def test_research_span_records_response_as_output():
    """When the research finishes, the OPERATION span's output contains the LLM's answer."""
    from dr import _run_research
    from treval.db import SpanStore

    with patch("dr.TavilyClient") as MockClient, \
         patch("dr.OpenAI") as MockOpenAI:
        MockClient.return_value.search.return_value = {"results": []}
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_llm_response("the answer is 42")

        _run_research("query")

    store = SpanStore()
    roots = store.list_spans(limit=10, type="OPERATION")
    assert "the answer is 42" in (roots[0]["output"] or "")


def test_run_research_returns_response_results_and_usage():
    """_run_research() returns (response, results, usage) for main() to print."""
    from dr import _run_research

    results = [{"url": "https://a.com", "title": "A", "content": "x"}]
    with patch("dr.TavilyClient") as MockClient, \
         patch("dr.OpenAI") as MockOpenAI:
        MockClient.return_value.search.return_value = {"results": results}
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_llm_response("ok", 100, 50)

        response, returned_results, usage = _run_research("query")

    assert response == "ok"
    assert returned_results == results
    assert usage["total_tokens"] == 150
