"""Tests for #22: Citation enforcement + post-processing (URL regex).

The LLM is told to cite [1]-[N] where N is the number of sources, but in
practice it sometimes invents citations like [7] when only 3 sources
exist. enforce_citations() detects these and either strips them or
returns the list of invalid numbers so the CLI can warn the user.

Detection: regex finds every [N] (single integer inside square brackets).
A citation is "valid" iff 1 <= N <= num_sources.
"""
import dr


def test_enforce_citations_all_valid_returns_empty_list():
    text = "According to [1] and [2], the answer is [3]."
    cleaned, invalid = dr.enforce_citations(text, num_sources=3)
    assert invalid == []
    assert cleaned == text


def test_enforce_citations_detects_oversized_citation():
    text = "Source [5] says it rains a lot."  # only 3 sources exist
    cleaned, invalid = dr.enforce_citations(text, num_sources=3)
    assert invalid == [5]
    assert "[5]" in cleaned  # we report, not strip, by default


def test_enforce_citations_detects_multiple_invalid():
    text = "[1] is good. [4] is fake. [2] is fine. [9] is also fake."
    cleaned, invalid = dr.enforce_citations(text, num_sources=3)
    assert sorted(invalid) == [4, 9]


def test_enforce_citations_detects_zero_and_negative():
    """[0] and [-1] are not valid 1-indexed citations."""
    text = "Bad: [0] and [-1] and [3]."
    cleaned, invalid = dr.enforce_citations(text, num_sources=3)
    assert sorted(invalid) == [-1, 0]


def test_enforce_citations_strips_when_strip_true():
    text = "Real [1] vs fake [5] vs real [2]."
    cleaned, invalid = dr.enforce_citations(text, num_sources=3, strip=True)
    assert invalid == [5]
    assert "[5]" not in cleaned
    assert "[1]" in cleaned
    assert "[2]" in cleaned


def test_enforce_citations_preserves_text_around_stripped():
    """strip=True should leave the surrounding words intact, not collapse them."""
    text = "Per [1] and [4], it's [3]."
    cleaned, invalid = dr.enforce_citations(text, num_sources=3, strip=True)
    assert invalid == [4]
    # The space and the word 'and' are still there
    assert "Per [1] and" in cleaned
    assert "it's [3]" in cleaned


def test_enforce_citations_dedupes_invalid_list():
    """The same invalid citation appearing twice is reported once."""
    text = "[5] says X. [5] also says Y."
    cleaned, invalid = dr.enforce_citations(text, num_sources=3, strip=True)
    assert invalid == [5]


def test_enforce_citations_handles_no_citations():
    text = "No citations here at all."
    cleaned, invalid = dr.enforce_citations(text, num_sources=3)
    assert invalid == []
    assert cleaned == text


def test_enforce_citations_handles_zero_sources():
    """When num_sources=0, every [N] is invalid."""
    text = "Pointless [1]."
    cleaned, invalid = dr.enforce_citations(text, num_sources=0, strip=True)
    assert invalid == [1]
    assert "[1]" not in cleaned


def test_enforce_citations_ignores_non_numeric_brackets():
    """Brackets that don't contain a single integer are not citations."""
    text = "See [v1.0] and [ref] and [nope] but also [2]."
    cleaned, invalid = dr.enforce_citations(text, num_sources=5)
    assert invalid == []
    assert cleaned == text


def test_enforce_citations_realistic_oversized_at_end():
    text = ("The capital of France is Paris [1]. Its population is 2.1M [2]. "
            "It is known for the Eiffel Tower [3] and the Louvre [99].")
    cleaned, invalid = dr.enforce_citations(text, num_sources=3)
    assert invalid == [99]


def test_run_research_warns_on_invalid_citations(capsys):
    """When the LLM returns a [N] that points past len(sources), the CLI
    prints a warning so the user knows the citation is broken. The
    response text is left untouched (we only warn, don't strip)."""
    from unittest.mock import patch

    fake_results = [
        {"url": "https://a.com", "title": "A", "content": "a"},
        {"url": "https://b.com", "title": "B", "content": "b"},
    ]
    fake_response = "Per [1] and a hallucinated [7] citation, the answer is [2]."

    with patch("dr.search", return_value=fake_results), \
         patch("dr.reformulate", return_value=[]), \
         patch("dr.ask", return_value=(fake_response, {
             "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
             "cost_usd": 0.0,
         })):
        response, _, _ = dr._run_research("test query", depth=1,
                                          max_results=2, model="x")

    out = capsys.readouterr().out
    assert "Invalid citations" in out
    assert "[7]" in out  # the actual citation number is named
    # Text returned to the caller is NOT modified (we only warn by default)
    assert "a hallucinated [7] citation" in response
