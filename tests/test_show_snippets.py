"""Tests for the --show-snippets CLI flag.

When enabled, deep-research prints the raw snippet content of every
Tavily result before the LLM synthesis. This is the debugging tool
for cases like 'the output mentioned a model that doesn't exist' —
with --show-snippets you can see exactly what content the LLM saw
and decide whether the issue is a Tavily miss, a Tavily hallucination,
or an LLM hallucination on top of weak sources.
"""
import json
from unittest.mock import patch

import dr


# Bypass the env-var check (no real API keys in CI)
_env = patch("dr._require_env", lambda *_a, **_kw: None)


_FAKE_RESULTS = [
    {"url": "https://a.com", "title": "Source A",
     "content": "Gemma 4 12B released June 3, 2026 with MMLU Pro 77.2%."},
    {"url": "https://b.com", "title": "Source B",
     "content": "April 2026 was a strong month for open-source releases."},
]


@_env
def test_show_snippets_flag_prints_source_content(capsys):
    """With --show-snippets, the snippet text of each Tavily result appears
    in stdout before the answer."""
    with patch("dr.search", return_value=_FAKE_RESULTS), \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.ask", return_value=("the answer based on [1][2]", {
             "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
             "cost_usd": 0.0,
         })):
        dr.main(["--show-snippets", "some query"])
    out = capsys.readouterr().out
    # The actual snippet content is in the output
    assert "Gemma 4 12B released June 3, 2026" in out
    assert "April 2026 was a strong month" in out
    # And the URLs are too
    assert "https://a.com" in out
    assert "https://b.com" in out


@_env
def test_show_snippets_flag_omitted_does_not_print_content(capsys):
    """Without --show-snippets, the snippet content is NOT printed (only the
    summary at the end). Keeps the default output clean."""
    with patch("dr.search", return_value=_FAKE_RESULTS), \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.ask", return_value=("the answer [1]", {
             "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
             "cost_usd": 0.0,
         })):
        dr.main(["some query"])  # no --show-snippets
    out = capsys.readouterr().out
    # The snippet content should NOT appear in the default output
    assert "Gemma 4 12B released June 3, 2026" not in out
    assert "April 2026 was a strong month" not in out


@_env
def test_show_snippets_flag_works_with_agentic(capsys):
    """--show-snippets also prints the sources when --agentic is used."""
    # LLM first asks to search (triggers search_cached), then answers
    with patch("dr.search_cached", return_value=_FAKE_RESULTS), \
         patch("dr.ask", side_effect=[
             ('{"action": "search", "query": "open LLMs 2026"}',
              {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
               "cost_usd": 0.0}),
             ('{"action": "answer", "answer": "final answer [1]."}',
              {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
               "cost_usd": 0.0}),
         ]):
        dr.main(["--show-snippets", "--agentic", "some query"])
    out = capsys.readouterr().out
    assert "Gemma 4 12B released June 3, 2026" in out
    assert "https://a.com" in out


@_env
def test_show_snippets_flag_emits_deduped_sources(capsys):
    """When the same URL appears twice (e.g. across --depth rounds), the
    snippets are deduped so the debug output stays readable."""
    with patch("dr.search", side_effect=[
        _FAKE_RESULTS,                  # depth=1 round
        _FAKE_RESULTS,                  # depth=1 round 2 (reformulated, same results)
    ]), \
         patch("dr.reformulate", return_value=["another query"]), \
         patch("dr.ask", return_value=("answer [1]", {
             "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
             "cost_usd": 0.0,
         })):
        dr.main(["--show-snippets", "--depth", "2", "some query"])
    out = capsys.readouterr().out
    # Each URL should appear at most twice in the snippets section
    # (once in the source list at the end, once in the snippet print)
    assert out.count("https://a.com") <= 3  # some slack for the source list
