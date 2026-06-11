#!/usr/bin/env python3
"""Deep Research CLI — LLM queries with treval tracing, tokens & costs."""

import json
import os
import sqlite3
import subprocess
import sys
import time
import treval
from pathlib import Path

from openai import OpenAI
from tavily import TavilyClient

# Auto-instrument OpenAI to capture every call as a treval span
treval.instrument()

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_FALLBACK = "deepseek/deepseek-v4-pro"  # Used when primary model fails
API_KEY_ENV = "OPENROUTER_API_KEY"
TAVILY_KEY_ENV = "TAVILY_API_KEY"
DEFAULT_MAX_RESULTS = 3

# Resilience
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0  # Sleep = base * 2^attempt, so 1s, 2s, 4s...
DEFAULT_TEMPERATURE = 0  # Deterministic output for research

# Local SQLite cache for Tavily results (avoids repeat HTTP calls)
CACHE_DB_PATH = Path.home() / ".treval" / "search_cache.db"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# Cost per Tavily search (USD) — Tavily charges per "basic" search credit.
# Used to surface true cost of a research run (LLM + search), not just LLM.
TAVILY_COST_PER_SEARCH_USD = 0.001

# Simple pricing per 1K tokens (USD) — updated manually or via treval prices
MODEL_PRICES = {
    "deepseek/deepseek-v4-flash": {"input": 0.0001, "output": 0.0004},
    "deepseek/deepseek-v4-pro":   {"input": 0.0003, "output": 0.0012},
    "deepseek/deepseek-r1":       {"input": 0.00055, "output": 0.00219},
    "deepseek/deepseek-v3":       {"input": 0.00027, "output": 0.0011},
    # fallback for unknown models
}


@treval.tool(name="tavily.search")
def search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict]:
    """Search the web via Tavily and return a list of result dicts.

    Each dict has 'url', 'title', 'content' keys.
    """
    client = TavilyClient(api_key=os.environ.get(TAVILY_KEY_ENV))
    response = client.search(query=query, max_results=max_results)
    return response.get("results", [])


def _cache_key(query: str) -> str:
    """Normalize a query for use as a cache key."""
    return query.strip().lower()


def _ensure_cache_table(conn: sqlite3.Connection) -> None:
    """Create the search_cache table if it doesn't exist yet."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS search_cache ("
        "  query TEXT PRIMARY KEY,"
        "  results TEXT NOT NULL,"
        "  created_at REAL NOT NULL"
        ")"
    )


def _cache_get(query: str) -> list[dict] | None:
    """Return cached results for a query, or None if missing/expired."""
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    try:
        _ensure_cache_table(conn)
        row = conn.execute(
            "SELECT results, created_at FROM search_cache WHERE query = ?",
            (_cache_key(query),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    results_json, created_at = row
    if time.time() - created_at > CACHE_TTL_SECONDS:
        return None  # expired
    return json.loads(results_json)


def _cache_set(query: str, results: list[dict]) -> None:
    """Store results in the cache, overwriting any existing entry."""
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    try:
        _ensure_cache_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO search_cache (query, results, created_at) "
            "VALUES (?, ?, ?)",
            (_cache_key(query), json.dumps(results), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


@treval.tool(name="tavily.search")
def search_cached(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict]:
    """Search the web via Tavily with a local TTL cache.

    On cache hit, returns the cached results instantly (no HTTP call).
    On cache miss or expired entry, calls Tavily and stores the result.
    The @treval.tool span records either way; cache hits show as ~0.1ms
    duration in the dashboard.
    """
    cached = _cache_get(query)
    if cached is not None:
        return cached
    results = search(query, max_results=max_results)
    _cache_set(query, results)
    return results


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICES.get(model)
    if not pricing:
        return 0.0
    return (pricing["input"] * prompt_tokens + pricing["output"] * completion_tokens) / 1000


def format_context(results: list[dict]) -> str:
    """Render Tavily results as a numbered block suitable for the LLM prompt.

    Format:
        Search results:

        [1] <title>
        <url>
        <content>

        [2] ...
    """
    if not results:
        return "No search results were found for this query."

    blocks = ["Search results:\n"]
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] {r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}\n")
    return "\n".join(blocks)


def format_sources(results: list[dict]) -> str:
    """Render sources as a numbered list for the CLI footer."""
    if not results:
        return ""
    lines = ["Sources:"]
    for i, r in enumerate(results, start=1):
        lines.append(f"  [{i}] {r.get('url', '')}")
    return "\n".join(lines)


def _require_env(var: str) -> str:
    """Read an env var; abort with a clear message if missing."""
    val = os.environ.get(var)
    if not val:
        print(f"❌ Set {var} in your environment")
        sys.exit(1)
    return val


SYSTEM_PROMPT = (
    "You are a deep research assistant. "
    "Answer using EXCLUSIVELY the information from the sources provided. "
    "Cite every claim with the source number in brackets, e.g. [1], [2]. "
    "Do not invent information that is not in the sources. If the requested "
    "information is not in the sources, say 'Not found in the consulted sources'."
)

REFORMULATE_SYSTEM_PROMPT = (
    "You are an assistant that reformulates research questions to "
    "maximize web search coverage. Generate variants of the "
    "provided question, each focusing on a different aspect "
    "or using different terminology. Respond ONLY with the numbered "
    "list of variants, one per line, with no explanations or extra text."
)

AGENTIC_SYSTEM_PROMPT = (
    "You are a deep research assistant that decides whether it needs "
    "more information before answering. Your output must be EXCLUSIVELY "
    "a JSON object in one of these two formats:\n"
    '  - To request a web search: {"action": "search", "query": "your query"}\n'
    '  - To give the final answer: {"action": "answer", "answer": "your answer"}\n'
    "Respond ONLY with the JSON, no extra text or markdown."
)


def _is_transient_error(exc: Exception) -> bool:
    """True for errors worth retrying (network, timeout, server errors)."""
    try:
        from openai import AuthenticationError, BadRequestError
        if isinstance(exc, (AuthenticationError, BadRequestError)):
            return False
    except ImportError:
        pass
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def ask(prompt: str, model: str = DEFAULT_MODEL, context: str | None = None,
        system: str | None = None, fallback: str | None = None,
        max_retries: int = MAX_RETRIES, stream: bool = False):
    """Call the LLM with retry-on-transient and optional model fallback.

    Tries `model` up to `max_retries` times. On exhausted retries, if
    `fallback` is set, tries it up to `max_retries` times. If both fail,
    re-raises the original primary-model exception.

    Non-transient errors (auth, bad request) raise immediately.

    If stream=True, returns an iterator yielding text tokens.
    If stream=False, returns a (full_text, usage_dict) tuple.
    """
    import time

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get(API_KEY_ENV),
        default_headers={
            "HTTP-Referer": "https://github.com/AmOrFeU86/deep-research",
            "X-Title": "deep-research",
        },
    )
    user_content = prompt
    if context:
        user_content = f"{context}\n\n---\n\n{prompt}"
    messages = [{"role": "system", "content": system or SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_content})

    models_to_try = [model]
    if fallback:
        models_to_try.append(fallback)

    primary_exc: Exception | None = None
    last_exc: Exception | None = None

    def _stream_response(m: str):
        """Yield text tokens from a streaming response."""
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=2000,
                    stream=True,
                )
                for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                nonlocal primary_exc, last_exc
                last_exc = e
                if m == model:
                    primary_exc = e
                if not _is_transient_error(e):
                    raise
                if attempt < max_retries - 1:
                    time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))
        if primary_exc is not None:
            raise primary_exc
        if last_exc is not None:
            raise last_exc

    if stream:
        return _stream_response(model)

    for m in models_to_try:
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=2000,
                )
                text = resp.choices[0].message.content or ""
                usage = {}
                if hasattr(resp, "usage") and resp.usage:
                    prompt_tk = getattr(resp.usage, "prompt_tokens", 0)
                    completion_tk = getattr(resp.usage, "completion_tokens", 0)
                    total_tk = getattr(resp.usage, "total_tokens", 0)
                    cost = estimate_cost(m, prompt_tk, completion_tk)
                    usage = {
                        "model": m,
                        "prompt_tokens": prompt_tk,
                        "completion_tokens": completion_tk,
                        "total_tokens": total_tk,
                        "cost_usd": round(cost, 5),
                    }
                return text, usage
            except Exception as e:
                last_exc = e
                if m == model:
                    primary_exc = e
                if not _is_transient_error(e):
                    raise
                if attempt < max_retries - 1:
                    time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))

    if primary_exc is not None:
        raise primary_exc
    if last_exc is not None:
        raise last_exc
    return "", {}  # unreachable, satisfies type checkers


def reformulate(prompt: str, n: int, model: str = DEFAULT_MODEL) -> list[str]:
    """Use an LLM to rephrase `prompt` in `n` different ways.

    Returns a list of n variants. Each variant is a distinct way to ask
    the same research question, designed to surface different web results.
    """
    text, _ = ask(
        f"Pregunta original: {prompt}",
        model=model,
        system=REFORMULATE_SYSTEM_PROMPT,
    )
    variants = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip common list markers: "1.", "1)", "-", "•"
        import re
        m = re.match(r"^[\d]+[.)]\s*(.+)", line)
        if m:
            variants.append(m.group(1).strip())
        elif line.startswith(("-", "•", "*")):
            variants.append(line[1:].strip())
        else:
            variants.append(line)
    return variants[:n]


def _run_research(prompt: str, depth: int = 1,
                  max_results: int = DEFAULT_MAX_RESULTS,
                  model: str = DEFAULT_MODEL,
                  fallback: str | None = None) -> tuple[str, list[dict], dict]:
    """Run a full research task wrapped in a parent OPERATION span.

    With depth=1: 1 search of the original query.
    With depth=N>1: reformulate prompt into N-1 variants, search all N,
    merge results deduped by URL, then ask the LLM with the full context.

    Creates a 'research' OPERATION span and pushes it onto the context
    stack, so nested @treval.tool spans (Tavily) and the auto-instrumented
    OpenAI LLM span both have it as their parent_id.
    """
    from treval.context import pop_span, push_span
    from treval.db import SpanStore

    store = SpanStore()
    root_id = store.save(
        name="research",
        type="OPERATION",
        status="ok",
        input=f"{prompt} [depth={depth}]",
    )
    push_span(root_id)
    try:
        # Build the list of queries to search
        queries = [prompt]
        if depth > 1:
            extra = reformulate(prompt, n=depth - 1)
            queries.extend(extra)
            if not extra:
                # Reformulation failed; fall back to original-only
                queries = [prompt]

        # Multi-search with dedup by URL
        all_results: list[dict] = []
        seen_urls: set[str] = set()
        for q in queries:
            for r in search_cached(q, max_results=max_results):
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)

        response, usage = ask(prompt, context=format_context(all_results),
                              model=model, fallback=fallback)
        tavily_searches = len(queries)
        usage["tavily_searches"] = tavily_searches
        usage["tavily_cost_usd"] = round(tavily_searches * TAVILY_COST_PER_SEARCH_USD, 5)
        store.update(root_id, output=response)
        return response, all_results, usage
    finally:
        pop_span()


def _run_research_streaming(prompt: str, depth: int = 1,
                            max_results: int = DEFAULT_MAX_RESULTS,
                            model: str = DEFAULT_MODEL,
                            fallback: str | None = None) -> None:
    """Same as _run_research but prints the LLM response as it streams in."""
    from treval.context import pop_span, push_span
    from treval.db import SpanStore

    store = SpanStore()
    root_id = store.save(
        name="research",
        type="OPERATION",
        status="ok",
        input=f"{prompt} [depth={depth}, stream]",
    )
    push_span(root_id)
    try:
        queries = [prompt]
        if depth > 1:
            extra = reformulate(prompt, n=depth - 1)
            queries.extend(extra)
            if not extra:
                queries = [prompt]

        all_results: list[dict] = []
        seen_urls: set[str] = set()
        for q in queries:
            for r in search_cached(q, max_results=max_results):
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)

        # Stream the response, printing tokens as they arrive
        full_response = []
        for token in ask(prompt, context=format_context(all_results),
                         model=model, fallback=fallback, stream=True):
            print(token, end="", flush=True)
            full_response.append(token)
        print()  # newline after stream completes
        response = "".join(full_response)

        print(f"\n{'─' * 40}")
        sources = format_sources(all_results)
        if sources:
            print(f"\n{sources}")
        store.update(root_id, output=response)
    finally:
        pop_span()


def parse_action(text: str) -> dict:
    """Parse an LLM response as a JSON object describing the next action.

    Tolerant of surrounding text: first tries a full parse, then falls back
    to extracting the first {…} block. Raises ValueError on garbage input
    (caller decides whether to retry, fall back, or treat as final answer).
    """
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse action from LLM response: {text!r}")


def run_research_agentic(prompt: str, model: str = DEFAULT_MODEL,
                         fallback: str | None = None,
                         max_iterations: int = 3,
                         max_results: int = DEFAULT_MAX_RESULTS
                         ) -> tuple[str, list[dict], dict]:
    """ReAct-style research loop where the LLM decides when to stop searching.

    On each iteration the LLM responds with JSON:
        {"action": "search", "query": "..."}   → run a Tavily search and
                                                  feed the result back as
                                                  an observation.
        {"action": "answer", "answer": "..."}  → loop ends, return answer.

    The loop also stops after `max_iterations` rounds to prevent infinite
    loops when the model keeps asking to search.

    Each LLM call is auto-instrumented by treval (one LLM span per turn);
    each search is wrapped by the @treval.tool span on search_cached.
    """
    from treval.context import pop_span, push_span
    from treval.db import SpanStore

    store = SpanStore()
    root_id = store.save(
        name="research_agentic",
        type="OPERATION",
        status="ok",
        input=f"{prompt} [agentic, max_iter={max_iterations}]",
    )
    push_span(root_id)
    try:
        all_results: list[dict] = []
        seen_urls: set[str] = set()
        observations: list[str] = []
        answer: str = ""
        total_usage = {
            "model": model,
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0,
        }
        searches_done = 0

        for i in range(max_iterations):
            context = "\n\n".join(observations) if observations else None
            text, usage = ask(prompt, context=context, model=model,
                              fallback=fallback, system=AGENTIC_SYSTEM_PROMPT)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
                total_usage[k] += usage.get(k, 0)

            try:
                action = parse_action(text)
            except ValueError:
                # Garbage from the LLM: treat the raw text as the final answer
                answer = text
                break

            if action.get("action") == "answer":
                answer = action.get("answer", "")
                break

            if action.get("action") == "search":
                query = action.get("query", prompt)
                for r in search_cached(query, max_results=max_results):
                    if r["url"] not in seen_urls:
                        seen_urls.add(r["url"])
                        all_results.append(r)
                        observations.append(
                            f"Observation {i+1} (query: \"{query}\"):\n"
                            f"{format_context([r])}"
                        )
                searches_done += 1
                continue

            # Unknown action — treat the raw text as the final answer
            answer = text
            break

        total_usage["tavily_searches"] = searches_done
        total_usage["tavily_cost_usd"] = round(
            searches_done * TAVILY_COST_PER_SEARCH_USD, 5
        )
        store.update(root_id, output=answer)
        return answer, all_results, total_usage
    finally:
        pop_span()


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    _require_env(API_KEY_ENV)
    _require_env(TAVILY_KEY_ENV)

    gen_report = "--report" in args
    args = [a for a in args if a != "--report"]

    stream = "--stream" in args
    args = [a for a in args if a != "--stream"]

    depth = 1
    if "--depth" in args:
        idx = args.index("--depth")
        if idx + 1 < len(args):
            depth = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
    depth = max(1, min(depth, 3))  # clamp 1..3

    max_results = DEFAULT_MAX_RESULTS
    if "--max-results" in args:
        idx = args.index("--max-results")
        if idx + 1 < len(args):
            max_results = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]

    model = DEFAULT_MODEL
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            model = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    agentic = "--agentic" in args
    args = [a for a in args if a != "--agentic"]

    prompt = " ".join(args) if args else input("❓ ")
    if not prompt:
        print("Nothing to ask.")
        return

    print(f"\n🔍 Searching the web for: {prompt} (depth={depth}{', agentic' if agentic else ''})")
    if stream:
        _run_research_streaming(prompt, depth=depth, max_results=max_results,
                                 model=model, fallback=DEFAULT_FALLBACK)
    elif agentic:
        response, results, usage = run_research_agentic(prompt, model=model,
                                                         fallback=DEFAULT_FALLBACK,
                                                         max_iterations=depth + 2)
        print(response)
        print(f"\n{'─' * 40}")
        if usage.get("total_tokens"):
            llm_cost = usage.get('cost_usd', 0)
            tavily_n = usage.get('tavily_searches', 0)
            tavily_cost = usage.get('tavily_cost_usd', 0.0)
            print(f"  Tokens:    {usage.get('prompt_tokens', 0)}↑ + {usage.get('completion_tokens', 0)}↓ = {usage['total_tokens']}")
            print(f"  LLM cost:  ${llm_cost:.5f}")
            print(f"  Tavily:    {tavily_n} search × ${TAVILY_COST_PER_SEARCH_USD:.4f} = ${tavily_cost:.5f}")
            print(f"  Total:     ${llm_cost + tavily_cost:.5f}")
        sources = format_sources(results)
        if sources:
            print(f"\n{sources}")
    else:
        response, results, usage = _run_research(prompt, depth=depth,
                                                   max_results=max_results,
                                                   model=model,
                                                   fallback=DEFAULT_FALLBACK)
        print(response)
        print(f"\n{'─' * 40}")
        if usage.get("total_tokens"):
            llm_cost = usage.get('cost_usd', 0)
            tavily_n = usage.get('tavily_searches', 0)
            tavily_cost = usage.get('tavily_cost_usd', 0.0)
            print(f"  Tokens:    {usage.get('prompt_tokens', 0)}↑ + {usage.get('completion_tokens', 0)}↓ = {usage['total_tokens']}")
            print(f"  LLM cost:  ${llm_cost:.5f}")
            print(f"  Tavily:    {tavily_n} search × ${TAVILY_COST_PER_SEARCH_USD:.4f} = ${tavily_cost:.5f}")
            print(f"  Total:     ${llm_cost + tavily_cost:.5f}")
        sources = format_sources(results)
        if sources:
            print(f"\n{sources}")

    if gen_report:
        path = os.path.join(os.path.dirname(__file__), "report.html")
        subprocess.run(["treval", "dashboard", "--export", path], check=True)
    else:
        print(f"\n  📊 treval dashboard --export report.html  — generate report")
    print()


if __name__ == "__main__":
    main()