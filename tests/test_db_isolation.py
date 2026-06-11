"""Tests that treval's SQLite DB is isolated to a per-test temp file.

Without isolation, integration tests (and any test that lets
@treval.tool write to its real SpanStore) would write to
~/.treval/spans.db, polluting the user's dashboard with rows like
"TOOL tavily.search 0.1ms".
"""
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch


def test_during_tests_db_path_lives_in_tmp_path():
    """The autouse fixture redirects treval.db.DB_PATH to tmp_path."""
    from treval import db
    db_path = db.DB_PATH
    # The parent must be somewhere inside pytest's tmp_path
    assert "pytest" in str(db_path) or str(db_path).startswith("/tmp/")


def test_search_cached_writes_to_isolated_db_not_real_one(tmp_path):
    """dr.search_cached() with the real SpanStore writes to tmp_path, not ~/.treval/spans.db."""
    import dr

    real_db = Path.home() / ".treval" / "spans.db"
    real_db_mtime_before = real_db.stat().st_mtime if real_db.exists() else None

    with patch("dr.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = {
            "results": [{"url": "https://x", "title": "X", "content": "x"}],
        }
        dr.search_cached("isolation test query")

    # 1. The isolated DB (the one we monkey-patched) gained a row
    from treval import db
    isolated_db = db.DB_PATH
    assert isolated_db.exists(), f"isolated DB not created at {isolated_db}"
    assert isolated_db.parent == tmp_path

    conn = sqlite3.connect(str(isolated_db))
    rows = conn.execute("SELECT name, type, input FROM spans").fetchall()
    conn.close()
    assert any(r[0] == "tavily.search" and r[1] == "TOOL" for r in rows), \
        f"expected TOOL tavily.search row, got {rows}"

    # 2. The real DB was NOT touched
    if real_db.exists():
        assert real_db.stat().st_mtime == real_db_mtime_before, \
            "real ~/.treval/spans.db was modified — isolation broken"
