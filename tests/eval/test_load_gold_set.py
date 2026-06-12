"""Tests for #27: Eval suite — load_gold_set().

The eval harness reads a JSONL file with one entry per line. Each
entry has id, question, reference (the ground-truth answer), and
criteria (a checklist the LLM judge will use to score the answer).

This file tests the loader in isolation: pure function, no LLM calls.
"""
import json
from pathlib import Path

import dr


def test_load_gold_set_returns_list_of_entries(tmp_path: Path):
    """load_gold_set reads a JSONL file and returns a list of dicts."""
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        '{"id": "q1", "question": "Q?", "reference": "R", "criteria": ["a"]}\n'
        '{"id": "q2", "question": "Q2?", "reference": "R2", "criteria": ["b", "c"]}\n'
    )
    entries = dr.load_gold_set(str(gold))
    assert len(entries) == 2
    assert entries[0]["id"] == "q1"
    assert entries[1]["id"] == "q2"


def test_load_gold_set_preserves_entry_fields(tmp_path: Path):
    """Every field of a gold entry is preserved as-is."""
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({
            "id": "capitals_fr",
            "question": "What is the capital of France?",
            "reference": "Paris",
            "criteria": ["mentions Paris", "no other capitals"],
        }) + "\n"
    )
    entry = dr.load_gold_set(str(gold))[0]
    assert entry["id"] == "capitals_fr"
    assert entry["question"] == "What is the capital of France?"
    assert entry["reference"] == "Paris"
    assert entry["criteria"] == ["mentions Paris", "no other capitals"]


def test_load_gold_set_ignores_blank_lines(tmp_path: Path):
    """Blank lines (and trailing newlines) do not produce empty entries."""
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        '{"id": "q1", "question": "?", "reference": "r", "criteria": []}\n'
        "\n"
        '{"id": "q2", "question": "?", "reference": "r", "criteria": []}\n'
        "\n"
    )
    entries = dr.load_gold_set(str(gold))
    assert len(entries) == 2


def test_load_gold_set_on_empty_file_returns_empty_list(tmp_path: Path):
    """An empty file yields [] (not None, not an error)."""
    gold = tmp_path / "empty.jsonl"
    gold.write_text("")
    assert dr.load_gold_set(str(gold)) == []


def test_load_gold_set_default_path_returns_real_gold():
    """Calling load_gold_set() with no argument uses the bundled gold.jsonl
    and returns the 10 starter entries."""
    entries = dr.load_gold_set()
    assert len(entries) == 10
    ids = {e["id"] for e in entries}
    # Spot-check a few known ids from the bundled gold set
    assert "capitals_fr" in ids
    assert "python_gil" in ids


def test_load_gold_set_raises_on_malformed_json(tmp_path: Path):
    """A line that isn't valid JSON raises a clear error (not silent skip)."""
    gold = tmp_path / "bad.jsonl"
    gold.write_text(
        '{"id": "q1", "question": "?", "reference": "r", "criteria": []}\n'
        "this is not json\n"
    )
    try:
        dr.load_gold_set(str(gold))
    except json.JSONDecodeError:
        pass  # expected
    else:
        raise AssertionError("Expected JSONDecodeError on malformed line")


def test_load_gold_set_raises_on_missing_required_field(tmp_path: Path):
    """An entry missing one of the required fields raises ValueError
    with a clear message — the user must know what's wrong."""
    gold = tmp_path / "incomplete.jsonl"
    gold.write_text(
        '{"id": "q1", "question": "?"}\n'  # missing reference, criteria
    )
    try:
        dr.load_gold_set(str(gold))
    except ValueError as e:
        assert "reference" in str(e) or "criteria" in str(e)
    else:
        raise AssertionError("Expected ValueError on missing required field")
