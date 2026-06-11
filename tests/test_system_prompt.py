"""Tests for the system prompt with citation enforcement in ask()."""
import pytest
from unittest.mock import MagicMock, patch


def _mock_resp(text="ok"):
    usage = MagicMock(); usage.prompt_tokens=10; usage.completion_tokens=5; usage.total_tokens=15
    choice = MagicMock(); choice.message.content = text
    resp = MagicMock(); resp.choices=[choice]; resp.usage=usage
    return resp


def test_ask_sends_a_system_message_first():
    """ask() always sends a system message to steer the model's behavior."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_resp()
        ask("hello")

    msgs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0]["role"] == "system"


def test_system_message_contains_citation_instructions():
    """The system prompt instructs the model to cite sources (e.g. [1], [2])."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_resp()
        ask("hello")

    system_msg = MockOpenAI.return_value.chat.completions.create.call_args.kwargs["messages"][0]
    content = system_msg["content"].lower()
    # Some form of citation or source attribution must be mentioned
    assert "cit" in content or "fuente" in content or "source" in content


def test_system_message_advises_against_inventing_facts():
    """The system prompt tells the model NOT to invent information beyond sources."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_resp()
        ask("hello")

    system_msg = MockOpenAI.return_value.chat.completions.create.call_args.kwargs["messages"][0]
    content = system_msg["content"].lower()
    # The model should be told to stick to sources / not invent / not hallucinate
    assert any(kw in content for kw in ("no invent", "no fabr", "sticking", "no men", "don't", "do not"))


def test_ask_with_context_still_sends_system_and_user_separately():
    """ask(prompt, context) sends system + user; context goes in the user message, not system."""
    from dr import ask

    with patch("dr.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_resp()
        ask("what is X?", context="X is a thing. See [1].")

    msgs = MockOpenAI.return_value.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # Context is in the user message
    assert "X is a thing. See [1]." in msgs[1]["content"]
    # The prompt is also in the user message
    assert "what is X?" in msgs[1]["content"]
    # Context is NOT in the system message (it's per-query, not per-behavior)
    assert "X is a thing" not in msgs[0]["content"]
