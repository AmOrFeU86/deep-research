"""Tests for query reformulation and multi-query depth search."""
import re
import pytest
from unittest.mock import MagicMock, patch


def _mock_llm_response(text: str = "ok", prompt_tokens: int = 50, completion_tokens: int = 10):
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


# ────────────────────────── reformulate() ──────────────────────────


def test_reformulate_returns_n_variants():
    """reformulate(prompt, n=3) returns exactly 3 query variants."""
    from dr import reformulate

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_response(
            "1. variant A\n2. variant B\n3. variant C"
        )

        variants = reformulate("original question", n=3)

    assert len(variants) == 3
    assert variants[0] == "variant A"
    assert variants[1] == "variant B"
    assert variants[2] == "variant C"


def test_reformulate_does_not_include_original_prompt():
    """reformulate() returns only the new variants, not the original prompt."""
    from dr import reformulate

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_response(
            "1. new angle on topic\n2. related but distinct question"
        )

        variants = reformulate("original question", n=2)

    assert "original question" not in variants
    assert all(v != "original question" for v in variants)


def test_reformulate_handles_numbered_list_parsing():
    """reformulate() parses common list formats: '1.', '1)', '-', '•'."""
    from dr import reformulate

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_response(
            "1) first\n- second\n• third"
        )

        variants = reformulate("q", n=3)

    assert variants == ["first", "second", "third"]


def test_reformulate_handles_empty_response():
    """If the LLM returns nothing, reformulate() returns an empty list (caller handles)."""
    from dr import reformulate

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_response("")

        variants = reformulate("q", n=3)

    assert variants == []


def test_reformulate_uses_specific_system_prompt():
    """reformulate() sends a system prompt that asks for variants, not citations."""
    from dr import reformulate

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_response("1. v")

        reformulate("q", n=1)

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    msgs = call_kwargs["messages"]
    system_msg = next((m for m in msgs if m["role"] == "system"), None)
    assert system_msg is not None
    # Should NOT be the citation-enforcement system prompt
    assert "EXCLUSIVAMENTE" not in system_msg["content"]
    # Should mention reformulation/variants
    assert any(kw in system_msg["content"].lower()
               for kw in ["reformula", "variante", "rephrase", "variant"])


# ────────────────────────── _run_research(depth=N) ──────────────────────────


def test_run_research_depth_1_calls_search_once():
    """_run_research(prompt, depth=1) calls search exactly once (no reformulation)."""
    from dr import _run_research

    with patch("dr.TavilyClient") as MockClient, \
         patch("dr.OpenAI") as MockOpenAI:
        MockClient.return_value.search.return_value = {"results": [
            {"url": "https://a.com", "title": "A", "content": "x"},
        ]}
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_llm_response("ok")

        _run_research("query", depth=1)

    assert MockClient.return_value.search.call_count == 1


def test_run_research_depth_3_calls_search_three_times():
    """_run_research(prompt, depth=3) calls search 3 times (1 original + 2 reformulated)."""
    from dr import _run_research

    with patch("dr.TavilyClient") as MockClient, \
         patch("dr.OpenAI") as MockOpenAI:
        MockClient.return_value.search.return_value = {"results": [
            {"url": "https://a.com", "title": "A", "content": "x"},
        ]}
        # First call is reformulation, second is the real LLM
        MockOpenAI.return_value.chat.completions.create.side_effect = [
            _mock_llm_response("1. variant one\n2. variant two"),
            _mock_llm_response("answer"),
        ]

        _run_research("query", depth=3)

    assert MockClient.return_value.search.call_count == 3


def test_run_research_dedupes_results_by_url():
    """If two searches return the same URL, it appears only once in the merged results."""
    from dr import _run_research

    with patch("dr.TavilyClient") as MockClient, \
         patch("dr.OpenAI") as MockOpenAI:
        # Both searches return the same URL
        MockClient.return_value.search.return_value = {"results": [
            {"url": "https://dup.com", "title": "Dup", "content": "x"},
        ]}
        MockOpenAI.return_value.chat.completions.create.side_effect = [
            _mock_llm_response("1. variant"),
            _mock_llm_response("answer"),
        ]

        _, results, _ = _run_research("query", depth=2)

    urls = [r["url"] for r in results]
    assert urls.count("https://dup.com") == 1  # deduped


def test_run_research_combines_unique_results_from_all_queries():
    """Results from all depth queries are merged, preserving uniqueness."""
    from dr import _run_research

    with patch("dr.TavilyClient") as MockClient, \
         patch("dr.OpenAI") as MockOpenAI:
        # Each search returns a different URL
        MockClient.return_value.search.side_effect = [
            {"results": [{"url": "https://1.com", "title": "1", "content": "a"}]},
            {"results": [{"url": "https://2.com", "title": "2", "content": "b"}]},
        ]
        MockOpenAI.return_value.chat.completions.create.side_effect = [
            _mock_llm_response("1. variant"),
            _mock_llm_response("answer"),
        ]

        _, results, _ = _run_research("query", depth=2)

    urls = sorted(r["url"] for r in results)
    assert urls == ["https://1.com", "https://2.com"]


# ────────────────────────── CLI flag ──────────────────────────


def test_main_with_depth_flag_passes_depth_to_research(capsys):
    """main(['--depth', '2', 'q']) calls _run_research(prompt, depth=2)."""
    from dr import main

    with patch("dr._run_research") as mock_run_research, \
         patch("dr.subprocess"):
        mock_run_research.return_value = (
            "answer",
            [{"url": "https://a.com", "title": "A", "content": "x"}],
            {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        )

        main(["--depth", "2", "query"])

    mock_run_research.assert_called_once()
    args, kwargs = mock_run_research.call_args
    assert args[0] == "query"
    assert kwargs.get("depth") == 2 or (
        len(args) > 1 and args[1] == 2
    )


def test_main_without_depth_flag_uses_default_depth_1(capsys):
    """main(['q']) calls _run_research(prompt, depth=1)."""
    from dr import main

    with patch("dr._run_research") as mock_run_research, \
         patch("dr.subprocess"):
        mock_run_research.return_value = (
            "answer", [], {"total_tokens": 0, "prompt_tokens": 0,
                           "completion_tokens": 0, "cost_usd": 0.0}
        )

        main(["query"])

    mock_run_research.assert_called_once()
    args, kwargs = mock_run_research.call_args
    depth = kwargs.get("depth") or (args[1] if len(args) > 1 else None)
    assert depth == 1
