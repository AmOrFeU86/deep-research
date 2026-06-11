"""Shared fixtures for deep-research tests.

Loads OPENROUTER_API_KEY and TAVILY_API_KEY from ~/.bashrc so tests can
hit the real APIs in integration mode (the early-return in .bashrc
prevents them from being in os.environ for non-interactive shells).
"""
import os
import pytest
from pathlib import Path


def _load_bashrc_keys() -> dict:
    keys = {}
    bashrc = Path.home() / ".bashrc"
    if not bashrc.exists():
        return keys
    for line in bashrc.read_text().splitlines():
        if line.startswith("export ") and "=" in line:
            kv = line[len("export "):].split("=", 1)
            if len(kv) == 2 and kv[0].endswith("_API_KEY"):
                key = kv[0]
                val = kv[1].strip().strip('"').strip("'")
                keys[key] = val
    return keys


@pytest.fixture(scope="session", autouse=True)
def _load_api_keys():
    """Make .bashrc API keys available to all tests."""
    for key, val in _load_bashrc_keys().items():
        os.environ.setdefault(key, val)


@pytest.fixture(autouse=True)
def _isolated_treval_db(tmp_path, monkeypatch):
    """Redirect treval's SQLite DB to a per-test temp file.

    Without this, tests (especially the real-Tavily integration test)
    would write to the user's real ~/.treval/spans.db and pollute the
    dashboard with rows like "TOOL tavily.search 0.1ms".

    The SpanStore reads DB_PATH from the treval.db module globals
    inside its __init__ on every call, so a monkey-patch of the
    module attribute is enough. We also clear the thread-local
    connection cache (SpanStore._local.conn) so the new path is
    actually used — without this, the second test in the same process
    reuses the first test's cached SQLite connection.
    """
    from treval.db import SpanStore

    db_path = tmp_path / "treval_test_spans.db"
    import treval.db as _treval_db
    monkeypatch.setattr(_treval_db, "DB_PATH", db_path)
    # Drop any cached connection from a previous test in this process.
    if hasattr(SpanStore._local, "conn"):
        SpanStore._local.conn = None
    yield db_path


@pytest.fixture(autouse=True)
def _isolated_search_cache(tmp_path, monkeypatch):
    """Redirect dr.CACHE_DB_PATH to a per-test temp file.

    Same rationale as _isolated_treval_db: without this, tests that
    populate the search cache would leave stale entries in the user's
    real ~/.treval/search_cache.db, causing other tests (and the user)
    to receive unexpectedly-cached results.
    """
    import dr
    cache_path = tmp_path / "search_cache_test.db"
    monkeypatch.setattr(dr, "CACHE_DB_PATH", cache_path)
    yield cache_path


@pytest.fixture
def tavily_key():
    """Skip integration tests if TAVILY_API_KEY is missing."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        pytest.skip("TAVILY_API_KEY not available")
    return key


@pytest.fixture
def openrouter_key():
    """Skip integration tests if OPENROUTER_API_KEY is missing."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not available")
    return key
