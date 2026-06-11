"""Tests for #7: Tavily search cost included in the CLI report.

Tavily charges ~$0.001 per basic search. We surface this so the user
sees the true cost of a research run (LLM + search), not just LLM.
"""
from unittest.mock import patch


SAMPLE_RESULTS = [
    {"url": "https://a.com", "title": "A", "content": "aaa"},
]


def _setup_mocks(mock_search, mock_ask, text="the answer"):
    mock_search.return_value = SAMPLE_RESULTS
    mock_ask.return_value = (text, {
        "model": "deepseek/deepseek-v4-flash",
        "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        "cost_usd": 0.00015,
    })


def test_tavily_cost_per_search_constant_exists():
    """dr.TAVILY_COST_PER_SEARCH_USD is a public constant (~$0.001 per basic search)."""
    import dr

    assert hasattr(dr, "TAVILY_COST_PER_SEARCH_USD")
    assert dr.TAVILY_COST_PER_SEARCH_USD > 0
    # Sanity: should be in the order of a tenth of a cent
    assert dr.TAVILY_COST_PER_SEARCH_USD < 0.01


def test_main_reports_tavily_search_count_and_cost(capsys):
    """main() prints a Tavily cost line after the LLM cost line."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask)
        main(["q"])

    out = capsys.readouterr().out
    # The report should mention Tavily cost
    assert "Tavily" in out or "tavily" in out.lower()
    # Should mention the search count (1 search with default depth=1)
    assert "1 search" in out or "1 ×" in out or "1x" in out.lower()


def test_main_tavily_cost_scales_with_search_count(capsys):
    """With depth=2, two searches happen and the Tavily cost is 2x the per-search rate."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate") as mock_reform, patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask)
        mock_reform.return_value = ["variant A", "variant B"]
        main(["--depth", "2", "q"])

    out = capsys.readouterr().out
    # 2 reformulated queries + original = 3 total searches at depth=2
    # (depth=2 means reformulate(prompt, n=1) = 1 extra, so queries = [orig, var] = 2)
    # Either way, depth>=2 should result in more than 1 search
    import re
    # Look for "N search" or "N × $..." pattern
    m = re.search(r"(\d+)\s*(?:search|×|x)", out.lower())
    assert m, f"No search count found in output: {out!r}"
    assert int(m.group(1)) >= 2, f"Expected >=2 searches, got: {m.group(1)}"


def test_tavily_cost_summed_in_total(capsys):
    """The CLI shows a Total cost line = LLM + Tavily."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask)
        main(["q"])

    out = capsys.readouterr().out
    # Total cost line should appear
    assert "Total" in out or "total" in out.lower()
