"""Tests for the agentic loop fix: when the LLM refuses on the first
iteration ("my knowledge cutoff", "I cannot", etc.), the loop must
still do a real search and re-prompt — not just return the refusal.

Bug: a time-sensitive question like "what open-source LLMs were
released in 2026" with --agentic returned the LLM's refusal verbatim
because parse_action() saw action=answer and the loop exited. A
non-agentic run with --depth 2 of the same question found Gemma 4 12B
in two Tavily calls — proving the search was the missing piece.

Fix: if the first iteration of the agentic loop returns an answer
that looks like a refusal (knowledge cutoff / cannot / don't have
info / etc.), do a Tavily search with the original prompt first,
then re-prompt the LLM with the search observations.
"""
from unittest.mock import patch

import dr


def _stub_usage(prompt=10, completion=5, cost=0.0001):
    return {
        "model": "stub", "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": prompt + completion, "cost_usd": cost,
    }


_FAKE_RESULTS = [
    {"url": "https://a.com", "title": "A", "content": "Gemma 4 12B released June 2026."},
    {"url": "https://b.com", "title": "B", "content": "Qwen3 released May 2026."},
]


# ---------------------------------------------------------------------------
# Bug reproduction
# ---------------------------------------------------------------------------

def test_agentic_refusal_on_first_iteration_triggers_search():
    """When the LLM refuses on the first iteration, the agentic loop must
    do a Tavily search and use those results in the final answer.

    Regression: this used to return the LLM's refusal verbatim (0 searches).
    """
    refusal = '{"action": "answer", "answer": "I cannot provide information about events after my knowledge cutoff."}'
    final = '{"action": "answer", "answer": "Based on search results, Gemma 4 12B and Qwen3 were released in 2026 [1]."}'

    with patch("dr.ask", side_effect=[
        (refusal, _stub_usage()),       # 1st iter: LLM refuses
        (final, _stub_usage()),         # 2nd iter: LLM has search context
    ]) as mock_ask, \
         patch("dr.search_cached", return_value=_FAKE_RESULTS) as mock_search:
        answer, results, usage = dr.run_research_agentic(
            "What open-source LLMs were released in 2026?", max_iterations=3,
        )

    # The fix: search_cached was called at least once
    assert mock_search.called, "agentic loop must search when LLM refuses on iter 1"
    # The fix: the final answer is NOT the refusal — it uses the search results
    assert "knowledge cutoff" not in answer.lower()
    assert "Gemma" in answer or "Qwen" in answer
    # The fix: results are populated
    assert len(results) >= 1


def test_agentic_refusal_pattern_recognized_various_phrasings():
    """The fix recognizes several common refusal phrasings, not just one."""
    refusals = [
        "As of my knowledge cutoff in 2025, I cannot provide information about 2026 events.",
        "I don't have information about events after my training cutoff.",
        "I cannot answer this question as it relates to future events.",
        "My training data does not include information about 2026.",
        "I am unable to provide a response about recent events.",
    ]
    final = '{"action": "answer", "answer": "OK answer [1]."}'

    for refusal_text in refusals:
        refusal = '{"action": "answer", "answer": "' + refusal_text + '"}'
        with patch("dr.ask", side_effect=[(refusal, _stub_usage()),
                                          (final, _stub_usage())]), \
             patch("dr.search_cached", return_value=_FAKE_RESULTS) as mock_search:
            answer, _, _ = dr.run_research_agentic("Q?", max_iterations=3)
        assert mock_search.called, f"Failed to trigger search for: {refusal_text!r}"


def test_agentic_does_not_force_search_when_first_answer_is_real():
    """If the first LLM answer is genuine (not a refusal), don't force a
    redundant search — respect the model's confidence and just return it.
    """
    real_answer = '{"action": "answer", "answer": "Python was created by Guido van Rossum in 1991."}'

    with patch("dr.ask", return_value=(real_answer, _stub_usage())) as mock_ask, \
         patch("dr.search_cached", return_value=_FAKE_RESULTS) as mock_search:
        answer, results, _ = dr.run_research_agentic(
            "Who created Python?", max_iterations=3,
        )

    # No search triggered — the LLM gave a real answer
    assert not mock_search.called
    assert "Guido" in answer
    assert results == []


def test_agentic_force_search_increments_tavily_count_in_usage():
    """The forced search must show up in the usage dict's tavily_searches."""
    refusal = '{"action": "answer", "answer": "I cannot provide information about 2026."}'
    final = '{"action": "answer", "answer": "Found it [1]."}'

    with patch("dr.ask", side_effect=[(refusal, _stub_usage()),
                                      (final, _stub_usage())]), \
         patch("dr.search_cached", return_value=_FAKE_RESULTS):
        _, _, usage = dr.run_research_agentic("Q?", max_iterations=3)

    assert usage.get("tavily_searches", 0) >= 1


def test_agentic_refusal_observation_includes_search_results():
    """The forced search feeds the results back to the LLM on iter 2 — the
    LLM's prompt on iter 2 should contain the search result content."""
    refusal = '{"action": "answer", "answer": "I cannot provide information about 2026."}'
    final = '{"action": "answer", "answer": "Found it [1]."}'

    with patch("dr.ask", side_effect=[(refusal, _stub_usage()),
                                      (final, _stub_usage())]) as mock_ask, \
         patch("dr.search_cached", return_value=_FAKE_RESULTS):
        dr.run_research_agentic("Q about 2026 LLM releases?", max_iterations=3)

    # ask() takes `prompt` positionally — check args, not kwargs
    second_call = mock_ask.call_args_list[1]
    second_call_prompt = second_call.args[0] if second_call.args else ""
    second_call_context = second_call.kwargs.get("context", "") or ""
    combined = second_call_prompt + " " + second_call_context
    assert "Gemma 4 12B" in combined or "Qwen" in combined
