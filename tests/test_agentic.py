"""Tests for #5: Agentic ReAct loop.

The LLM is queried iteratively. On each turn, it either:
- responds with a 'search' action (JSON with a query) → we run a Tavily search
  and feed the result back as observation.
- responds with an 'answer' action (JSON with the final answer) → loop ends.

This contrasts with the deterministic depth=N approach (#1+#2): the model
decides autonomously when it has enough information.
"""
import json
from unittest.mock import MagicMock, patch


def _llm_json_response(payload: dict, prompt_tokens: int = 50, completion_tokens: int = 10):
    """Build a mock OpenAI response whose .choices[0].message.content is JSON."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    choice = MagicMock()
    choice.message.content = json.dumps(payload)

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


SAMPLE_RESULTS = [
    {"url": "https://a.com", "title": "A", "content": "aaa"},
]


# ────────────────────────── pure helper: parse_action() ──────────────────────────


def test_parse_action_handles_pure_json():
    """parse_action() returns the dict when given valid JSON."""
    from dr import parse_action

    out = parse_action('{"action": "answer", "answer": "42"}')
    assert out == {"action": "answer", "answer": "42"}


def test_parse_action_handles_json_embedded_in_text():
    """parse_action() finds a JSON object even when surrounded by text."""
    from dr import parse_action

    raw = 'Sure! Here is my decision:\n{"action": "search", "query": "X"}\nDone.'
    out = parse_action(raw)
    assert out == {"action": "search", "query": "X"}


def test_parse_action_raises_on_invalid_json():
    """parse_action() raises ValueError on garbage input (caller decides fallback)."""
    import pytest
    from dr import parse_action

    with pytest.raises(ValueError):
        parse_action("this is not json at all")


# ────────────────────────── run_research_agentic() ──────────────────────────


def test_agentic_returns_immediately_when_llm_says_answer():
    """When the LLM's first action is 'answer', loop ends with no searches."""
    from dr import run_research_agentic

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _llm_json_response(
            {"action": "answer", "answer": "Paris is the capital."}
        )
        with patch("dr.search_cached") as mock_search:
            answer, results, usage = run_research_agentic("Capital of France?")

    assert answer == "Paris is the capital."
    assert mock_search.call_count == 0
    # Only one LLM call (no iterations after)
    assert instance.chat.completions.create.call_count == 1
    assert results == []


def test_agentic_searches_then_answers():
    """LLM asks for one search, then answers → 1 search + 2 LLM calls."""
    from dr import run_research_agentic

    # Turn 1: ask for search. Turn 2: answer.
    responses = iter([
        _llm_json_response({"action": "search", "query": "France capital"}),
        _llm_json_response({"action": "answer", "answer": "Paris"}),
    ])

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = lambda **kwargs: next(responses)

        with patch("dr.search_cached") as mock_search:
            mock_search.return_value = SAMPLE_RESULTS
            answer, results, usage = run_research_agentic("Capital of France?")

    assert answer == "Paris"
    assert mock_search.call_count == 1
    assert mock_search.call_args.args[0] == "France capital"
    assert instance.chat.completions.create.call_count == 2
    # Results should include the search results
    assert len(results) == 1
    assert results[0]["url"] == "https://a.com"


def test_agentic_aggregates_context_across_iterations():
    """Observations from earlier searches are included in later LLM prompts."""
    from dr import run_research_agentic

    # Two searches, then answer.
    responses = iter([
        _llm_json_response({"action": "search", "query": "Q1"}),
        _llm_json_response({"action": "search", "query": "Q2"}),
        _llm_json_response({"action": "answer", "answer": "Final"}),
    ])

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = lambda **kwargs: next(responses)
        with patch("dr.search_cached") as mock_search:
            mock_search.return_value = SAMPLE_RESULTS
            run_research_agentic("X")

    # Third call (the "answer" one) should contain BOTH observations in context
    final_call = instance.chat.completions.create.call_args_list[2]
    final_messages = final_call.kwargs["messages"]
    user_content = final_messages[-1]["content"]
    # The first result "aaa" from Q1 should be there
    assert "aaa" in user_content
    # And from Q2 as well
    assert "https://a.com" in user_content


def test_agentic_stops_at_max_iterations():
    """If the LLM keeps asking to search, the loop stops at max_iterations."""
    from dr import run_research_agentic

    # LLM always asks to search — never answers. Loop must stop on its own.
    def always_search(**kwargs):
        return _llm_json_response({"action": "search", "query": "more"})

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = always_search
        with patch("dr.search_cached") as mock_search:
            mock_search.return_value = SAMPLE_RESULTS
            answer, results, usage = run_research_agentic("X", max_iterations=3)

    # At most max_iterations LLM calls (the 3rd is still a "search" call;
    # we don't get an explicit "answer" so the loop terminates by iteration count)
    assert instance.chat.completions.create.call_count == 3
    assert mock_search.call_count == 3


def test_agentic_returns_search_results_in_final_return():
    """The 'results' field of the return value contains all search results seen."""
    from dr import run_research_agentic

    responses = iter([
        _llm_json_response({"action": "search", "query": "Q1"}),
        _llm_json_response({"action": "answer", "answer": "Final"}),
    ])

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = lambda **kwargs: next(responses)
        with patch("dr.search_cached") as mock_search:
            mock_search.return_value = SAMPLE_RESULTS
            _, results, _ = run_research_agentic("X")

    assert len(results) == 1
    assert results[0]["url"] == "https://a.com"
