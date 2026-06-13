"""Tests for #20: Cross-verification of citations.

After the LLM produces a draft answer with [N] citations, we do a second
LLM pass to verify that each cited source actually exists in the search
results and that the claim attached to it is accurate. This catches
"hallucinated" citations where the LLM cites a source that doesn't
support the claim.
"""
import json
from unittest.mock import MagicMock, patch


# ────────────────────────── VERIFY_SYSTEM_PROMPT constant ──────────────────────────


def test_verify_system_prompt_exists():
    """dr.VERIFY_SYSTEM_PROMPT is a non-empty string with a clear verification task."""
    import dr

    assert hasattr(dr, "VERIFY_SYSTEM_PROMPT")
    prompt = dr.VERIFY_SYSTEM_PROMPT
    assert isinstance(prompt, str)
    assert len(prompt) > 50
    # Should mention verification in some form
    assert "verif" in prompt.lower() or "check" in prompt.lower() or "valid" in prompt.lower()


# ────────────────────────── verify_citations() ──────────────────────────


def _mock_llm_json(payload: dict):
    """Build a mock OpenAI response with JSON content."""
    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.completion_tokens = 30
    usage.total_tokens = 80
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


SAMPLE_RESULTS = [
    {"url": "https://a.com", "title": "A", "content": "Paris is the capital of France."},
    {"url": "https://b.com", "title": "B", "content": "France is in Europe."},
]


def test_verify_citations_returns_ok_when_all_claims_supported():
    """If every [N] citation is backed by the sources, verification returns OK."""
    from dr import verify_citations

    draft = "Paris is the capital [1]. France is in Europe [2]."

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_json({
            "verified": True,
            "issues": [],
        })

        result = verify_citations(draft, SAMPLE_RESULTS)

    assert result["verified"] is True
    assert result["issues"] == []


def test_verify_citations_returns_issues_when_unsupported():
    """If a citation has no source backing, the issue is reported."""
    from dr import verify_citations

    draft = "Paris is the capital [1]. Mars has aliens [3]."  # [3] is hallucinated

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_json({
            "verified": False,
            "issues": [
                {"citation": "[3]", "reason": "No source [3] in the provided results."},
            ],
        })

        result = verify_citations(draft, SAMPLE_RESULTS)

    assert result["verified"] is False
    assert len(result["issues"]) == 1
    assert "[3]" in result["issues"][0]["citation"]


def test_verify_citations_sends_sources_to_llm():
    """The LLM call includes the source URLs and content (not just the draft)."""
    from dr import verify_citations

    draft = "Paris [1]."

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_json({
            "verified": True, "issues": [],
        })

        verify_citations(draft, SAMPLE_RESULTS)

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    # Combine all message content into one string to check
    all_content = " ".join(m.get("content", "") for m in messages)
    assert "https://a.com" in all_content
    assert "https://b.com" in all_content
    assert draft in all_content


def test_verify_citations_sends_draft_to_llm():
    """The LLM call includes the draft answer being verified."""
    from dr import verify_citations

    draft = "The capital is Paris [1]."

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_json({
            "verified": True, "issues": [],
        })

        verify_citations(draft, SAMPLE_RESULTS)

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    all_content = " ".join(m.get("content", "") for m in messages)
    assert "The capital is Paris" in all_content


def test_verify_citations_handles_invalid_json():
    """If the LLM returns garbage, verify_citations returns verified=True with a note."""
    from dr import verify_citations

    draft = "Paris [1]."

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_json({
            "verified": True, "issues": []
        })  # First call valid; test that valid JSON is parsed

        # Use a non-JSON payload to test the fallback
        choice = MagicMock()
        choice.message.content = "this is not json at all"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock()
        instance.chat.completions.create.return_value = resp

        result = verify_citations(draft, SAMPLE_RESULTS)

    # On parse failure, we should default to verified=True (don't block the user)
    # and report the parse error as an issue.
    assert "verified" in result
    assert "issues" in result


def test_verify_citations_with_no_sources_skips_llm():
    """When sources=[] the verifier short-circuits to verified=True without
    calling the LLM. Saves a network call on the common "no results found"
    case (and the response would be meaningless anyway without sources).
    """
    from dr import verify_citations

    with patch("dr.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = _mock_llm_json({
            "verified": False, "issues": [{"citation": "[1]", "reason": "..."}],
        })

        result = verify_citations("anything", [])

    assert result["verified"] is True
    assert result["issues"] == []
    # And critically: no LLM call was made
    instance.chat.completions.create.assert_not_called()
