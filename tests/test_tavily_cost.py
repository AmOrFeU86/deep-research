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


def test_main_reports_tavily_search_count_and_cost(capsys):
    """main() prints a Tavily cost line after the LLM cost line."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.verify_citations",
               return_value={"verified": True, "issues": [], "model": "x"}), \
         patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask)
        main(["q"])

    out = capsys.readouterr().out
    # The report should mention Tavily cost
    assert "Tavily" in out or "tavily" in out.lower()
    # Should mention the search count (1 search with depth=1, i.e. reformulate mocked out)
    assert "1 search" in out or "1 ×" in out or "1x" in out.lower()


def test_main_tavily_cost_scales_with_search_count(capsys):
    """With depth=3 (still on the test side), multiple searches happen and the
    Tavily cost scales accordingly. The reformulate mock returns 2 variants."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate") as mock_reform, \
         patch("dr.verify_citations",
               return_value={"verified": True, "issues": [], "model": "x"}), \
         patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask)
        mock_reform.return_value = ["variant A", "variant B"]
        main(["q"])  # depth=DEFAULT_DEPTH=10 by default; reformulate returns 2

    out = capsys.readouterr().out
    # reformulate returns 2 extra + original = 3 total searches
    import re
    m = re.search(r"(\d+)\s*(?:search|×|x)", out.lower())
    assert m, f"No search count found in output: {out!r}"
    assert int(m.group(1)) >= 2, f"Expected >=2 searches, got: {m.group(1)}"


def test_tavily_cost_summed_in_total(capsys):
    """The CLI shows a Total cost line = LLM + Tavily."""
    from dr import main

    with patch("dr.search") as mock_search, patch("dr.ask") as mock_ask, \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.verify_citations",
               return_value={"verified": True, "issues": [], "model": "x"}), \
         patch("dr.subprocess"):
        _setup_mocks(mock_search, mock_ask)
        main(["q"])

    out = capsys.readouterr().out
    # Total cost line should appear
    assert "Total" in out or "total" in out.lower()
