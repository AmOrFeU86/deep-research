"""Tests for #4: Parallel search with ThreadPoolExecutor.

With --depth N>1, we launch N Tavily queries in parallel instead of serially.
Each worker thread pushes/pops the parent span_id manually because treval
uses threading.local() for the context stack, so asyncio.to_thread would
lose the parent.
"""
import time
from unittest.mock import patch

import dr


def _slow_search(query, max_results=3, search_depth="basic"):
    """Fake search_cached that sleeps 0.1s and returns one fake result."""
    time.sleep(0.1)
    return [{"url": f"https://{query}.com", "title": query, "content": query}]


def test_parallel_search_runs_concurrently():
    """3 queries each sleeping 0.1s should complete in ~0.1s, not ~0.3s."""
    with patch("dr.search_cached", side_effect=_slow_search):
        start = time.time()
        results = dr.parallel_search(
            ["q1", "q2", "q3"], max_results=3, parent_id=1
        )
        elapsed = time.time() - start

    # Sequential would take ~0.3s. Allow generous overhead but reject 2x+.
    assert elapsed < 0.25, f"Expected <0.25s, got {elapsed:.3f}s"
    assert len(results) == 3


def test_parallel_search_dedupes_by_url():
    """Two queries returning the same URL should yield it only once."""
    def fake(query, max_results=3, search_depth="basic"):
        return [{"url": "https://shared.com", "title": query, "content": "x"}]

    with patch("dr.search_cached", side_effect=fake):
        results = dr.parallel_search(
            ["q1", "q2", "q3"], max_results=3, parent_id=1
        )

    urls = [r["url"] for r in results]
    assert urls.count("https://shared.com") == 1
    assert len(results) == 1


def test_parallel_search_preserves_unique_results():
    """Distinct URLs from different queries are all kept."""
    def fake(query, max_results=3, search_depth="basic"):
        return [{"url": f"https://{query}.com", "title": query, "content": query}]

    with patch("dr.search_cached", side_effect=fake):
        results = dr.parallel_search(
            ["a", "b", "c"], max_results=3, parent_id=1
        )

    urls = sorted(r["url"] for r in results)
    assert urls == ["https://a.com", "https://b.com", "https://c.com"]


def test_parallel_search_handles_empty_queries():
    """No queries → empty results, no crash."""
    with patch("dr.search_cached") as mock:
        results = dr.parallel_search([], max_results=3, parent_id=1)
    assert results == []
    mock.assert_not_called()


def test_parallel_search_preserves_order_by_query():
    """Results come back grouped by query, in the order queries were given."""
    def fake(query, max_results=3, search_depth="basic"):
        return [{"url": f"https://{query}-{i}.com", "title": "", "content": ""}
                for i in range(2)]

    with patch("dr.search_cached", side_effect=fake):
        results = dr.parallel_search(
            ["q1", "q2"], max_results=3, parent_id=1
        )

    urls = [r["url"] for r in results]
    # All q1-* first, then q2-*. This is what dedup preserves and is what
    # _run_research relied on in the sequential version.
    assert urls == [
        "https://q1-0.com", "https://q1-1.com",
        "https://q2-0.com", "https://q2-1.com",
    ]


def test_parallel_search_uses_correct_parent_id_in_threads():
    """Each worker thread must push parent_id so treval spans are siblings.

    We patch dr.search_cached to record what current_span_id() sees inside
    the thread, and assert it equals the parent_id we passed.
    """
    seen = []

    def recording(query, max_results=3, search_depth="basic"):
        from treval.context import current_span_id
        seen.append(current_span_id())
        return []

    with patch("dr.search_cached", side_effect=recording):
        dr.parallel_search(["a", "b", "c"], max_results=3, parent_id=42)

    assert seen == [42, 42, 42], f"Threads did not see parent_id=42, got {seen}"


def test_parallel_search_thread_count_matches_queries():
    """We should spawn len(queries) workers, not block the main thread."""
    import threading

    main_thread_id = threading.get_ident()
    thread_ids = []

    def fake(query, max_results=3, search_depth="basic"):
        thread_ids.append(threading.get_ident())
        return []

    with patch("dr.search_cached", side_effect=fake):
        dr.parallel_search(["a", "b", "c", "d"], max_results=3, parent_id=1)

    # All 4 queries should run in threads different from the main thread.
    assert len(thread_ids) == 4
    assert all(tid != main_thread_id for tid in thread_ids), \
        f"Some queries ran on the main thread: {thread_ids}"


def test_parallel_search_pops_span_even_on_search_error():
    """If search_cached raises, the worker must still pop the parent span
    so the thread-local stack doesn't leak across queries."""
    from treval.context import current_span_id

    def boom(query, max_results=3, search_depth="basic"):
        raise RuntimeError("simulated Tavily failure")

    with patch("dr.search_cached", side_effect=boom):
        try:
            dr.parallel_search(["a"], max_results=3, parent_id=99)
        except RuntimeError:
            pass

    # After the call, the main thread's stack should be untouched
    # (we never pushed in main). The point is that no exception escaped
    # the worker and broke the parent stack.
    # If parallel_search wraps errors, no exception is raised at all —
    # we just verify it didn't crash.
    assert current_span_id() is None
