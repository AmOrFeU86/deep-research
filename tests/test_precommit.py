"""Tests for #26: Pre-commit hook infrastructure.

The hook is a bash script at .githooks/pre-commit that runs the test
suite before allowing commits. These tests verify the file exists, is
executable, and contains the right commands.
"""
import os
import stat
from pathlib import Path

HOOK_PATH = Path(__file__).resolve().parent.parent / ".githooks" / "pre-commit"


def test_precommit_hook_exists():
    """The pre-commit hook file exists at .githooks/pre-commit."""
    assert HOOK_PATH.exists(), f"Missing: {HOOK_PATH}"


def test_precommit_hook_is_executable():
    """The pre-commit hook has the executable bit set (git needs this)."""
    mode = HOOK_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, "Hook not executable by owner"


def test_precommit_hook_runs_pytest():
    """The pre-commit hook calls pytest as part of its checks."""
    content = HOOK_PATH.read_text()
    assert "pytest" in content, "Hook does not call pytest"
    assert "tests/" in content, "Hook does not target the tests/ directory"


def test_precommit_hook_uses_existing_trevals_venv_or_local():
    """The hook falls back to the shared treval venv if no local .venv."""
    content = HOOK_PATH.read_text()
    assert "treval" in content or ".venv" in content, \
        "Hook should discover a venv (local or shared treval)"


def test_precommit_hook_skips_integration_tests():
    """Integration tests are excluded (they require real API keys)."""
    content = HOOK_PATH.read_text()
    assert "not integration" in content or "-m integration" in content, \
        "Hook should skip or target integration tests explicitly"


def test_precommit_hook_has_bypass_instructions():
    """The hook mentions --no-verify as a bypass (so devs know the escape hatch)."""
    content = HOOK_PATH.read_text()
    assert "--no-verify" in content, "Hook should document the --no-verify bypass"
