#!/usr/bin/env python3
"""Deep Research CLI — LLM queries with treval tracing, tokens & costs."""

import json
import os
import sqlite3
import subprocess
import sys
import time
import treval
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tavily import TavilyClient

# Auto-instrument OpenAI to capture every call as a treval span
treval.instrument()

# --- Provider configuration --------------------------------------------------
# Single OpenAI-compatible endpoint, fixed by design. Kept as a dict so
# adding a new provider later is a one-line change.
PROVIDERS = {
    "minimax": {
        "base_url": "https://api.minimaxi.chat/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "default_model": "MiniMax-M3",
    },
}

DEFAULT_PROVIDER = "minimax"
DEFAULT_MODEL = PROVIDERS[DEFAULT_PROVIDER]["default_model"]
TAVILY_KEY_ENV = "TAVILY_API_KEY"

# Search behavior — fixed by design (no flags). Quality knobs live here.
DEFAULT_MAX_RESULTS = 3           # Tavily results per query
DEFAULT_DEPTH = 10                # 1 original + 9 reformulations
DEFAULT_SEARCH_DEPTH = "basic"    # Tavily: "basic" (cheap) or "advanced" (deeper, ~3x)

# Post-answer quality gate
SELF_CRITIQUE = True              # Always re-prompt if verify_citations finds issues

# Resilience
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0          # Sleep = base * 2^attempt, so 1s, 2s, 4s...
DEFAULT_TEMPERATURE = 0           # Deterministic output for research
DEFAULT_TIMEOUT_SECONDS = 30      # Per-request timeout for the OpenAI client

# Local SQLite cache for Tavily results (avoids repeat HTTP calls)
CACHE_DB_PATH = Path.home() / ".treval" / "search_cache.db"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# Cost per Tavily search (USD) — Tavily charges per "basic" search credit.
# Used to surface true cost of a research run (LLM + search), not just LLM.
TAVILY_COST_PER_SEARCH_USD = 0.001

# Eval suite — bundled gold set lives next to the eval tests
DEFAULT_GOLD_PATH = Path(__file__).parent / "tests" / "eval" / "gold.jsonl"
DEFAULT_JUDGE_MODEL = PROVIDERS[DEFAULT_PROVIDER]["default_model"]  # cheap LLM-as-judge
DEFAULT_PASS_THRESHOLD = 0.7  # score >= threshold counts as a pass

# Simple pricing per 1K tokens (USD) — updated manually or via treval prices.
# MiniMax-M3 pricing is a placeholder; real rates need to be confirmed
# against the provider's published price list. Unknown models fall through
# estimate_cost() to a 0.0 cost.
MODEL_PRICES = {
    "MiniMax-M3": {"input": 0.0001, "output": 0.0004},  # TODO: verify real rate
    # fallback for unknown models
}


def search(query: str, max_results: int = DEFAULT_MAX_RESULTS,
           search_depth: str = DEFAULT_SEARCH_DEPTH) -> list[dict]:
    """Search the web via Tavily and return a list of result dicts.

    Each dict has 'url', 'title', 'content' keys.

    `search_depth` is "basic" (cheap, default) or "advanced" (deeper
    relevance, ~3x more expensive per Tavily pricing).

    This function is the undecorated raw helper kept around for tests
    and direct use. The traced, cache-fronted path is `search_cached`,
    which is the one wrapped by `@treval.tool` and given a metadata_fn
    in #8. Callers should use `search_cached`; calling `search` directly
    will still produce a span, but without the structured metadata.
    """
    client = TavilyClient(api_key=os.environ.get(TAVILY_KEY_ENV))
    response = client.search(query=query, max_results=max_results,
                             search_depth=search_depth)
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


def _tavily_metadata(args, kwargs, result):
    """Build the metadata dict for a tavily.search span.

    Stored as JSON in the span's metadata column and rendered in the
    treval dashboard detail panel. Keeps each Tavily call self-describing:
    what we asked for, the limit, and how many results came back.
    """
    return {
        "query": kwargs.get("query", args[0] if args else None),
        "max_results": kwargs.get("max_results", DEFAULT_MAX_RESULTS),
        "num_results": len(result),
    }


@treval.tool(name="tavily.search", metadata_fn=_tavily_metadata)
def search_cached(query: str, max_results: int = DEFAULT_MAX_RESULTS,
                  search_depth: str = DEFAULT_SEARCH_DEPTH) -> list[dict]:
    """Search the web via Tavily with a local TTL cache.

    On cache hit, returns the cached results instantly (no HTTP call).
    On cache miss or expired entry, calls Tavily and stores the result.
    The @treval.tool span records either way; cache hits show as ~0.1ms
    duration in the dashboard.

    `search_depth` is "basic" (default) or "advanced".
    """
    cached = _cache_get(query)
    if cached is not None:
        return cached
    results = search(query, max_results=max_results, search_depth=search_depth)
    _cache_set(query, results)
    return results


def _search_with_parent(query: str, max_results: int, parent_id: int) -> list[dict]:
    """Thread worker: push parent span, search, pop.

    Required because treval's span stack is stored in threading.local(),
    so the context pushed on the main thread is invisible inside a worker
    spawned via ThreadPoolExecutor. Each worker must push/pop its own copy
    of the parent_id to keep the resulting Tavily spans as siblings under
    the 'research' OPERATION span.
    """
    from treval.context import pop_span, push_span
    push_span(parent_id)
    try:
        return search_cached(query, max_results=max_results)
    finally:
        pop_span()


def parallel_search(queries: list[str], max_results: int,
                    parent_id: int) -> list[dict]:
    """Run N Tavily queries in parallel, merge results deduped by URL.

    Order of returned results: all results from queries[0] first (deduped),
    then all from queries[1], etc. — same ordering as the sequential loop
    used to produce, so downstream code is unchanged.

    With len(queries)==0 returns [] without spawning a pool.
    """
    if not queries:
        return []

    all_results: list[dict] = []
    seen_urls: set[str] = set()
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = [pool.submit(_search_with_parent, q, max_results, parent_id)
                   for q in queries]
        for fut in futures:
            for r in fut.result():
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
    return all_results


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


def enforce_citations(text: str, num_sources: int,
                      strip: bool = False) -> tuple[str, list[int]]:
    """Find every [N] in `text` and flag any that point past `num_sources`.

    The LLM is told in the system prompt that it must only cite [1]..[N]
    for N = len(sources), but in practice it occasionally invents
    citations like [7] when only 3 sources exist. This function is the
    static (regex) check: it returns the list of invalid citation numbers
    so the CLI can warn the user, and optionally strips them from the
    text so the output doesn't ship with broken references.

    Detection: a single integer (allowing optional leading minus) inside
    square brackets. Brackets that contain anything else (e.g. [v1.0],
    [ref]) are ignored — they are not citations.

    Args:
        text: the LLM response to scan.
        num_sources: how many real sources exist (citations must satisfy
            1 <= N <= num_sources).
        strip: if True, remove invalid [N] tokens from the returned text
            (preserving surrounding whitespace so words don't collapse).

    Returns:
        (cleaned_text, invalid_numbers) where invalid_numbers is a
        sorted, de-duplicated list of every N that was out of range.
    """
    import re
    pattern = re.compile(r"\[(-?\d+)\]")

    invalid_set: set[int] = set()
    for match in pattern.finditer(text):
        n = int(match.group(1))
        if not (1 <= n <= num_sources):
            invalid_set.add(n)

    cleaned = text
    if strip and invalid_set:
        # Build a single regex that matches any of the invalid citations.
        # Escape the negative sign — re.escape on "-7" gives "\\-7" but
        # we already accept "-?\d+" so a plain "-" needs no escape here.
        bad = "|".join(re.escape(f"[{n}]") for n in sorted(invalid_set))
        cleaned = re.sub(bad, "", text)

    return cleaned, sorted(invalid_set)


def _source_metadata(all_results: list[dict],
                     tavily_searches: int,
                     tavily_cost_usd: float) -> dict:
    """Build the metadata dict stored on a research OPERATION span.

    Pairs with the per-call metadata in #8 (`_tavily_metadata`): every
    tavily.search span has its own query/limit/result-count, and the
    parent OPERATION span has the deduped source list plus the totals
    for the whole run. The treval dashboard renders both, so a user
    clicking a research span can see exactly which sources fed the
    answer and which queries produced them.

    `all_results` is copied so the caller can keep mutating its own
    list without surprising the dashboard on a later read.
    """
    return {
        "num_sources": len(all_results),
        "sources": list(all_results),
        "tavily_searches": tavily_searches,
        "tavily_cost_usd": tavily_cost_usd,
    }


def _require_env(var: str) -> str:
    """Read an env var; abort with a clear message if missing."""
    val = os.environ.get(var)
    if not val:
        print(f"❌ Set {var} in your environment")
        sys.exit(1)
    return val


def _get_provider_config(name: str) -> dict:
    """Look up a provider entry from PROVIDERS; abort with a clear message
    on unknown names. Centralises the 'is this provider registered?' check
    so ask() / verify_citations() / main() all behave consistently.
    """
    if name not in PROVIDERS:
        valid = ", ".join(sorted(PROVIDERS))
        print(f"❌ Unknown provider '{name}'. Available: {valid}")
        sys.exit(2)
    return PROVIDERS[name]


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
    "Respond ONLY with the JSON, no extra text or markdown.\n\n"
    "IMPORTANT: If you are uncertain (e.g. the question is about events "
    "after your training cutoff, or asks for sources you don't remember "
    "exactly), you MUST request a search. Never answer with 'I cannot', "
    "'my knowledge cutoff', or 'I don't have information' — instead, "
    "request a search so the user gets a real, sourced answer."
)


# Refusal patterns the LLM uses to dodge time-sensitive questions. When
# detected on the first iteration of run_research_agentic, we force a
# Tavily search with the original prompt and re-prompt with the results
# — the LLM almost always returns a real answer on iter 2 once it has
# search context. This is the safety net for "I cannot" responses that
# would otherwise leave the user with no information.
_REFUSAL_PATTERNS = (
    "knowledge cutoff", "as of my", "as an ai", "as a language model",
    "i cannot", "i can't", "i don't have", "i do not have",
    "i'm unable", "i am unable", "training data", "training cutoff",
    "my knowledge", "no information", "i cannot provide",
    "i'm not able", "i am not able", "future events",
    "i don't know", "i do not know", "do not have access",
)


def _looks_like_refusal(text: str) -> bool:
    """Detect when an LLM 'answer' is actually a refusal/uncertainty
    statement, not a substantive answer to the user's question.

    Used by run_research_agentic to decide whether to force a search
    when the LLM tries to exit the loop with a non-answer.
    """
    if not text:
        return False
    lower = text.lower()
    return any(pat in lower for pat in _REFUSAL_PATTERNS)

VERIFY_SYSTEM_PROMPT = (
    "You are a citation verifier. You receive a draft answer and a list of "
    "consulted sources (numbered [1], [2], ...). Your task is to check that "
    "every [N] citation in the draft actually supports the claim attached to "
    "it, AND that no claim in the draft is missing a citation.\n\n"
    "Respond ONLY with a JSON object in this format:\n"
    '  {"verified": true|false, "issues": [{"citation": "[N]", "reason": "..."}]}\n\n'
    "If everything checks out, return verified=true and an empty issues list. "
    "If something is wrong, return verified=false and a list of issues, each "
    "pointing to a specific [N] citation and explaining the problem."
)


def _is_transient_error(exc: Exception) -> bool:
    """True for errors worth retrying (network, timeout, rate limit, server)."""
    try:
        from openai import APIStatusError, AuthenticationError, BadRequestError
        if isinstance(exc, (AuthenticationError, BadRequestError)):
            return False
        if isinstance(exc, APIStatusError):
            # 429 (rate limit) and 5xx (server) are worth retrying
            return exc.status_code in (429,) or exc.status_code >= 500
    except ImportError:
        pass
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def ask(prompt: str, context: str | None = None,
        system: str | None = None,
        max_retries: int = MAX_RETRIES,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS):
    """Call the LLM with retry-on-transient.

    Tries the configured model up to `max_retries` times on transient
    errors (network, timeout, 429, 5xx). Non-transient errors (auth,
    bad request) raise immediately.

    Returns a (full_text, usage_dict) tuple.
    """
    import time

    pcfg = _get_provider_config(DEFAULT_PROVIDER)
    client = OpenAI(
        base_url=pcfg["base_url"],
        api_key=os.environ.get(pcfg["api_key_env"]),
    )
    user_content = prompt
    if context:
        user_content = f"{context}\n\n---\n\n{prompt}"
    messages = [{"role": "system", "content": system or SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_content})

    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=2000,
                timeout=timeout,
            )
            text = resp.choices[0].message.content or ""
            usage = {}
            if hasattr(resp, "usage") and resp.usage:
                prompt_tk = getattr(resp.usage, "prompt_tokens", 0)
                completion_tk = getattr(resp.usage, "completion_tokens", 0)
                total_tk = getattr(resp.usage, "total_tokens", 0)
                cost = estimate_cost(DEFAULT_MODEL, prompt_tk, completion_tk)
                usage = {
                    "model": DEFAULT_MODEL,
                    "prompt_tokens": prompt_tk,
                    "completion_tokens": completion_tk,
                    "total_tokens": total_tk,
                    "cost_usd": round(cost, 5),
                }
            return text, usage
        except Exception as e:
            last_exc = e
            if not _is_transient_error(e):
                raise
            if attempt < max_retries - 1:
                time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))

    if last_exc is not None:
        raise last_exc
    return "", {}  # unreachable, satisfies type checkers


def reformulate(prompt: str, n: int) -> list[str]:
    """Use an LLM to rephrase `prompt` in `n` different ways.

    Returns a list of n variants. Each variant is a distinct way to ask
    the same research question, designed to surface different web results.
    """
    text, _ = ask(
        f"Pregunta original: {prompt}",
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


def _print_sources(results: list[dict]) -> None:
    """Print the raw snippet content of every Tavily result.

    Used by the --show-snippets CLI flag as a debugging tool: when an
    answer looks suspicious (e.g. mentions a model that doesn't exist),
    this lets you see exactly what the LLM had to work with — separating
    'Tavily returned weak sources' from 'LLM hallucinated on top of
    good sources'.
    """
    if not results:
        return
    print(f"\n{'─' * 78}")
    print(f"  📄 SOURCES ({len(results)} results, --show-snippets)")
    print(f"{'─' * 78}")
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        content = r.get("content", "")
        print(f"  [{i}] {title}")
        print(f"      URL: {url}")
        # Indent multi-line content; truncate very long snippets for readability
        snippet = content[:600] + ("…" if len(content) > 600 else "")
        for line in snippet.splitlines() or [snippet]:
            print(f"      │ {line}")
    print(f"{'─' * 78}\n")


def _self_critique(prompt: str, response: str, results: list[dict],
                  context: str, usage_in: dict) -> tuple[str, dict, dict]:
    """Run verify_citations and, on issues, re-prompt the LLM with the critique.

    Always runs when SELF_CRITIQUE is True. Returns the final response
    (original if already clean, rewritten if the verifier flagged issues),
    the verifier report, and the accumulated usage (including the re-prompt
    if any).
    """
    verify_report = verify_citations(response, results)
    if verify_report["verified"] or not verify_report["issues"]:
        return response, verify_report, usage_in

    # Build a critique block and re-prompt the model to fix the issues
    critique_lines = [
        f"- {i.get('citation', '?')}: {i.get('reason', '?')}"
        for i in verify_report["issues"]
    ]
    critique = "\n".join(critique_lines)
    new_response, extra_usage = ask(
        f"Your previous answer to: {prompt}\n\n"
        f"Was flagged with these citation issues:\n{critique}\n\n"
        f"Rewrite the answer correcting those issues. Use the same sources "
        f"and citation format.",
        context=context,
    )
    usage_out = dict(usage_in)
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
        usage_out[k] = usage_out.get(k, 0) + extra_usage.get(k, 0)
    return new_response, verify_report, usage_out


def _print_footer(usage: dict) -> None:
    """Print the standard tokens / cost footer."""
    if not usage.get("total_tokens"):
        return
    llm_cost = usage.get("cost_usd", 0)
    tavily_n = usage.get("tavily_searches", 0)
    tavily_cost = usage.get("tavily_cost_usd", 0.0)
    print(f"  Tokens:    {usage.get('prompt_tokens', 0)}↑ + "
          f"{usage.get('completion_tokens', 0)}↓ = {usage['total_tokens']}")
    print(f"  LLM cost:  ${llm_cost:.5f}")
    print(f"  Tavily:    {tavily_n} search × "
          f"${TAVILY_COST_PER_SEARCH_USD:.4f} = ${tavily_cost:.5f}")
    print(f"  Total:     ${llm_cost + tavily_cost:.5f}")


def _run_research(prompt: str, depth: int = DEFAULT_DEPTH) -> tuple[str, list[dict], dict]:
    """Run a full research task wrapped in a parent OPERATION span.

    With depth=1: 1 search of the original query.
    With depth=N>1: reformulate prompt into N-1 variants, search all N,
    merge results deduped by URL, then ask the LLM with the full context.

    Creates a 'research' OPERATION span and pushes it onto the context
    stack, so nested @treval.tool spans (Tavily) and the auto-instrumented
    OpenAI LLM span both have it as their parent_id.

    When SELF_CRITIQUE is True, runs verify_citations after the first
    answer and re-prompts with the critique if issues are found.
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
        queries = [prompt]
        if depth > 1:
            extra = reformulate(prompt, n=depth - 1)
            queries.extend(extra)
            if not extra:
                # Reformulation failed; fall back to original-only
                queries = [prompt]

        all_results = parallel_search(queries, max_results=DEFAULT_MAX_RESULTS,
                                      parent_id=root_id)

        # Combine search results into the context for the LLM.
        search_ctx = format_context(all_results)
        response, usage = ask(prompt, context=search_ctx)
        tavily_searches = len(queries)
        usage["tavily_searches"] = tavily_searches
        usage["tavily_cost_usd"] = round(tavily_searches * TAVILY_COST_PER_SEARCH_USD, 5)

        # Self-critique: re-prompt if verify_citations finds issues
        if SELF_CRITIQUE:
            response, verify_report, usage = _self_critique(
                prompt, response, all_results, search_ctx, usage,
            )
            if not verify_report["verified"]:
                print(f"  ⚠️  Self-critique re-prompted: "
                      f"{len(verify_report['issues'])} issue(s) flagged")
        else:
            # Static citation enforcement (regex) when self-critique is off
            _, invalid = enforce_citations(response, num_sources=len(all_results))
            if invalid:
                print(f"  ⚠️  Invalid citations: {invalid} "
                      f"(only {len(all_results)} sources available)")

        store.update(root_id, output=response,
                     metadata=_source_metadata(all_results, tavily_searches,
                                               usage["tavily_cost_usd"]))
        return response, all_results, usage
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


def verify_citations(draft: str, sources: list[dict]) -> dict:
    """Second LLM pass that checks every [N] citation in `draft` is supported.

    Returns a dict with keys:
        - "verified": True if all citations check out, False otherwise
        - "issues": list of {"citation": "[N]", "reason": "..."} for problems
        - "model": which model ran the verification

    On JSON parse failure (LLM returns garbage), defaults to verified=True
    with a single issue noting the parse error — we don't want to block the
    user on a flaky verifier.
    """
    import re

    if not sources:
        return {"verified": True, "issues": [], "model": DEFAULT_MODEL}

    # Build the verification prompt: sources list + draft
    sources_block = "\n\n".join(
        f"[{i+1}] {s.get('title', '')}\nURL: {s.get('url', '')}\n{s.get('content', '')}"
        for i, s in enumerate(sources)
    )
    user_content = (
        f"Sources:\n{sources_block}\n\n"
        f"---\n\nDraft answer to verify:\n{draft}\n\n"
        f"Return a JSON object with 'verified' (true/false) and 'issues' (list)."
    )

    try:
        pcfg = _get_provider_config(DEFAULT_PROVIDER)
        client = OpenAI(
            base_url=pcfg["base_url"],
            api_key=os.environ.get(pcfg["api_key_env"]),
        )
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=1000,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        return {"verified": True, "issues": [{"citation": "—", "reason": f"verifier error: {e}"}], "model": DEFAULT_MODEL}

    # Try to parse the JSON response (with the same tolerant logic as parse_action)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"verified": True, "issues": [{"citation": "—", "reason": "verifier returned unparseable output"}], "model": DEFAULT_MODEL}
        else:
            return {"verified": True, "issues": [{"citation": "—", "reason": "verifier returned unparseable output"}], "model": DEFAULT_MODEL}

    # Normalize the result
    if not isinstance(result, dict):
        return {"verified": True, "issues": [{"citation": "—", "reason": "verifier returned non-dict"}], "model": DEFAULT_MODEL}

    return {
        "verified": bool(result.get("verified", False)),
        "issues": result.get("issues", []),
        "model": DEFAULT_MODEL,
    }


def run_research_agentic(prompt: str, max_iterations: int = 3,
                         min_searches: int = 0) -> tuple[str, list[dict], dict]:
    """ReAct-style research loop where the LLM decides when to stop searching.

    On each iteration the LLM responds with JSON:
        {"action": "search", "query": "..."}   -> run a Tavily search and
                                                  feed the result back as
                                                  an observation.
        {"action": "answer", "answer": "..."}  -> loop ends, return answer.

    The loop also stops after `max_iterations` rounds to prevent infinite
    loops when the model keeps asking to search.

    `min_searches` (default 0) forces the loop to perform at least that
    many Tavily searches before the LLM is allowed to exit with
    `action: answer`. When the LLM tries to answer too early, the code
    silently forces another search with the original prompt and
    re-prompts — the early "answer" is discarded.

    NOTE: Not exposed in the CLI anymore — the default route (depth=10 +
    self-critique) is the recommended path. Kept here as a public
    function for tests and for #28 (deep exploration) on the roadmap.
    """
    from treval.context import pop_span, push_span
    from treval.db import SpanStore

    # Augment the system prompt to enforce min_searches when set. We
    # tell the LLM up front (it tends to comply on the first iteration,
    # which saves us from having to override its decisions later).
    system_prompt = AGENTIC_SYSTEM_PROMPT
    if min_searches > 0:
        system_prompt = system_prompt + (
            f"\n\nIMPORTANT: You MUST do at least {min_searches} web "
            f"searches before answering. The first {min_searches} "
            f"iterations MUST request a search (action: search). Only "
            f"after {min_searches} searches have been performed may you "
            f"respond with action: answer."
        )

    store = SpanStore()
    root_id = store.save(
        name="research_agentic",
        type="OPERATION",
        status="ok",
        input=f"{prompt} [agentic, max_iter={max_iterations}, min_searches={min_searches}]",
    )
    push_span(root_id)
    try:
        all_results: list[dict] = []
        seen_urls: set[str] = set()
        observations: list[str] = []
        answer: str = ""
        total_usage = {
            "model": DEFAULT_MODEL,
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0,
        }
        searches_done = 0

        for i in range(max_iterations):
            context = "\n\n".join(observations) if observations else None
            text, usage = ask(prompt, context=context, system=system_prompt)
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

                # min_searches enforcement: if the LLM tries to exit before
                # the budget is exhausted, discard the "answer" and force
                # another search with the original prompt. The LLM gets
                # another chance on the next iteration to either search
                # again or — once the budget is met — give the real answer.
                if searches_done < min_searches:
                    query = prompt
                    for r in search_cached(query, max_results=DEFAULT_MAX_RESULTS):
                        if r["url"] not in seen_urls:
                            seen_urls.add(r["url"])
                            all_results.append(r)
                            observations.append(
                                f"Observation {searches_done + 1} "
                                f"(min_searches forced, query: "
                                f"\"{query}\"):\n{format_context([r])}"
                            )
                    searches_done += 1
                    continue

                # Refusal safety net: if the LLM tries to exit the loop on
                # any iteration with a refusal-style answer (cutoff, "I cannot",
                # etc.) and no search has been done yet, force a search
                # with the original prompt and re-prompt with the
                # observations. The LLM almost always returns a real,
                # sourced answer once it has search context.
                if (searches_done == 0
                        and _looks_like_refusal(answer)):
                    for r in search_cached(prompt, max_results=DEFAULT_MAX_RESULTS):
                        if r["url"] not in seen_urls:
                            seen_urls.add(r["url"])
                            all_results.append(r)
                            observations.append(
                                f"Observation 1 (forced search for "
                                f"time-sensitive question, query: "
                                f"\"{prompt}\"):\n{format_context([r])}"
                            )
                    searches_done += 1
                    continue
                break

            if action.get("action") == "search":
                query = action.get("query", prompt)
                for r in search_cached(query, max_results=DEFAULT_MAX_RESULTS):
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
        store.update(root_id, output=answer,
                     metadata=_source_metadata(all_results, searches_done,
                                               total_usage["tavily_cost_usd"]))
        return answer, all_results, total_usage
    finally:
        pop_span()


# ---------------------------------------------------------------------------
# Eval suite (#27) — load gold set, judge answers, run the suite, CLI dispatch
# ---------------------------------------------------------------------------

_REQUIRED_GOLD_FIELDS = ("id", "question", "reference", "criteria")


def load_gold_set(path: str | Path | None = None) -> list[dict]:
    """Read a JSONL file of gold entries and return them as a list of dicts.

    Each entry must have: id, question, reference, criteria. The bundled
    gold set lives at tests/eval/gold.jsonl (DEFAULT_GOLD_PATH) and is
    loaded when `path` is None.

    Blank lines are ignored. Malformed JSON or missing fields raise a
    clear error (we want to fail loud on a broken gold set — silent
    skipping would let a typo in the gold file corrupt the eval).
    """
    p = Path(path) if path is not None else DEFAULT_GOLD_PATH
    entries: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)  # raises JSONDecodeError on garbage
            missing = [k for k in _REQUIRED_GOLD_FIELDS if k not in entry]
            if missing:
                raise ValueError(
                    f"{p}:{lineno} entry is missing required field(s): {missing}"
                )
            entries.append(entry)
    return entries


def _extract_json_block(text: str) -> str | None:
    """Return the first {...} block in `text`, or None if not found.

    Handles JSON embedded in markdown code fences or surrounding prose.
    Used as a fallback when json.loads(text) fails.
    """
    import re
    # Try to find a JSON block. Use a non-greedy match for the first {...}
    # (greedy matches would swallow across multiple top-level objects, but
    # in practice the judge returns exactly one).
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    return match.group(0)


def parse_judge_json(text: str) -> tuple[float, str]:
    """Parse the LLM judge's response into a (score, reason) tuple.

    Tolerates markdown code fences and surrounding prose. Clamps the
    score to [0.0, 1.0]. Raises ValueError on unparseable output or if
    the score field is missing — we want a loud failure if the judge
    is broken, not a silent zero that masks the bug.
    """
    if not text or not text.strip():
        raise ValueError("judge returned empty output")

    # Strategy 1: direct json.loads
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Strategy 2: extract first {...} block (handles markdown + prose)
        block = _extract_json_block(text)
        if block is not None:
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                pass

    if parsed is None or not isinstance(parsed, dict):
        raise ValueError(f"judge returned no parseable JSON object: {text!r}")

    if "score" not in parsed:
        raise ValueError(f"judge output missing 'score' field: {parsed!r}")

    try:
        score = float(parsed["score"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"judge score is not numeric: {parsed.get('score')!r}") from e

    score = max(0.0, min(1.0, score))  # clamp to [0, 1]
    reason = str(parsed.get("reason", ""))
    return score, reason


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator. You will receive a question, a "
    "reference answer, the model's answer, and a checklist of criteria. "
    "Decide how well the model's answer satisfies the criteria, using the "
    "reference answer as ground truth. Return ONLY a JSON object of the "
    "form {\"score\": <0.0-1.0>, \"reason\": \"<one-sentence rationale>\"}."
)


def judge_answer(question: str, reference: str, answer: str,
                 criteria: list[str], model: str, judge_model: str
                 ) -> tuple[float, str]:
    """Use a cheap LLM to score `answer` against `reference` and `criteria`.

    The judge LLM is `judge_model` (default: flash — cheap). `model` is
    the model that produced `answer`, recorded for traceability but not
    used by the judge.

    Returns (score, reason). Raises ValueError if the judge returns
    output that parse_judge_json cannot handle.
    """
    criteria_block = "\n".join(f"- {c}" for c in criteria)
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Model's answer to grade:\n{answer}\n\n"
        f"Criteria (checklist):\n{criteria_block}\n\n"
        f"Return {{\"score\": <0.0-1.0>, \"reason\": \"<why>\"}}."
    )
    text, _usage = ask(
        prompt=user_prompt, model=judge_model, system=JUDGE_SYSTEM_PROMPT,
    )
    return parse_judge_json(text)


def _run_eval(gold_path: str | Path | None = None,
              research_model: str = DEFAULT_MODEL,
              judge_model: str = DEFAULT_JUDGE_MODEL,
              threshold: float = DEFAULT_PASS_THRESHOLD,
              depth: int = 1) -> dict:
    """Run the eval suite: for every gold entry, research + judge + record.

    Returns a dict with aggregate stats (`mean_score`, `pass_rate`,
    `num_passed`, `num_failed`, `threshold`, `total_cost_usd`,
    `num_questions`) and per-question details in `results`. Each
    result has id, question, answer, score, reason, passed, cost_usd,
    duration_ms.

    Tracing: each question runs inside its own OPERATION span
    `eval.question.<id>` so the treval dashboard shows a clean tree
    (one span per question, with the research + judge as children).
    """
    from treval.context import pop_span, push_span
    from treval.db import SpanStore

    gold = load_gold_set(gold_path)
    results: list[dict] = []
    total_cost = 0.0
    store = SpanStore()

    for entry in gold:
        # Per-question span: makes the eval legible in the treval dashboard
        q_span_id = store.save(
            name=f"eval.question.{entry['id']}",
            type="OPERATION",
            status="ok",
            input=entry["question"],
        )
        push_span(q_span_id)
        start = time.perf_counter()
        try:
            response, _results, research_usage = _run_research(
                entry["question"], depth=depth, max_results=DEFAULT_MAX_RESULTS,
                model=research_model,
            )
            score, reason = judge_answer(
                question=entry["question"],
                reference=entry["reference"],
                answer=response,
                criteria=entry["criteria"],
                model=research_model,
                judge_model=judge_model,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            cost = research_usage.get("cost_usd", 0.0)
            total_cost += cost
            passed = score >= threshold
            result = {
                "id": entry["id"],
                "question": entry["question"],
                "answer": response,
                "score": score,
                "reason": reason,
                "passed": passed,
                "cost_usd": cost,
                "duration_ms": duration_ms,
            }
            results.append(result)
            # Tag the span with the eval outcome — queryable from the dashboard
            store.update(q_span_id, output=response,
                         metadata={"id": entry["id"], "score": score,
                                   "passed": passed, "reason": reason})
        finally:
            pop_span()

    n = len(results)
    scores = [r["score"] for r in results]
    num_passed = sum(1 for r in results if r["passed"])
    return {
        "num_questions": n,
        "mean_score": (sum(scores) / n) if n else 0.0,
        "pass_rate": (num_passed / n) if n else 0.0,
        "num_passed": num_passed,
        "num_failed": n - num_passed,
        "threshold": threshold,
        "total_cost_usd": total_cost,
        "results": results,
    }


def _print_eval_report(report: dict) -> None:
    """Print the eval report in a clean human-readable table.

    Per-question rows show id, score, pass/fail, and the judge's reason.
    The summary block shows mean score, pass rate, threshold, total cost.
    """
    print(f"\n{'─' * 78}")
    print(f"  EVAL REPORT  ({report['num_questions']} questions, "
          f"threshold {report['threshold']:.2f})")
    print(f"{'─' * 78}")
    print(f"  {'id':<25} {'score':>6}  {'result':<8}  reason")
    print(f"  {'─' * 25} {'─' * 6}  {'─' * 8}  {'─' * 30}")
    for r in report["results"]:
        marker = "✅ PASS" if r["passed"] else "❌ FAIL"
        reason = (r["reason"] or "")[:60]
        print(f"  {r['id']:<25} {r['score']:>6.2f}  {marker:<8}  {reason}")
    print(f"{'─' * 78}")
    print(f"  mean score : {report['mean_score']:.3f}")
    print(f"  pass rate  : {report['num_passed']}/{report['num_questions']} "
          f"({report['pass_rate'] * 100:.0f}%)")
    print(f"  total cost : ${report['total_cost_usd']:.5f}")
    print(f"{'─' * 78}\n")


def _run_eval_cli(args: list[str]) -> None:
    """Parse `dr eval ...` flags and dispatch to _run_eval.

    Flags:
      --gold PATH             path to gold.jsonl (default: bundled)
      --judge-model MODEL     LLM used as judge (default: flash)
      --research-model MODEL  LLM used for research (default: DEFAULT_MODEL)
      --threshold FLOAT       pass threshold in [0, 1] (default: 0.7)
    """
    gold_path: str | None = None
    judge_model = DEFAULT_JUDGE_MODEL
    research_model = DEFAULT_MODEL
    threshold = DEFAULT_PASS_THRESHOLD

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--gold" and i + 1 < len(args):
            gold_path = args[i + 1]; i += 2
        elif a == "--judge-model" and i + 1 < len(args):
            judge_model = args[i + 1]; i += 2
        elif a == "--research-model" and i + 1 < len(args):
            research_model = args[i + 1]; i += 2
        elif a == "--threshold" and i + 1 < len(args):
            threshold = float(args[i + 1]); i += 2
        else:
            print(f"Unknown flag for `dr eval`: {a}")
            sys.exit(2)

    report = _run_eval(
        gold_path=gold_path, research_model=research_model,
        judge_model=judge_model, threshold=threshold,
    )
    _print_eval_report(report)

    # CI mode: if pass rate is below threshold, exit non-zero so CI can fail.
    if report["pass_rate"] < threshold:
        print(f"  ✗ Pass rate {report['pass_rate'] * 100:.0f}% is below "
              f"threshold {threshold * 100:.0f}%. Exit 1.")
        sys.exit(1)


def main(args: list[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    # --- Subcommands ---------------------------------------------------------
    if args and args[0] == "eval":
        _run_eval_cli(args[1:])
        return

    # --- Flags (only UX ones; quality knobs are constants) ------------------
    gen_report = "--report" in args

    pcfg = _get_provider_config(DEFAULT_PROVIDER)
    _require_env(pcfg["api_key_env"])
    _require_env(TAVILY_KEY_ENV)

    prompt = " ".join(a for a in args if not a.startswith("--")) if args else input("❓ ")
    if not prompt:
        print("Nothing to ask.")
        return

    print(f"\n🔍 Searching the web for: {prompt} (depth={DEFAULT_DEPTH})")
    response, results, usage = _run_research(prompt)
    print(response)
    print(f"\n{'─' * 40}")
    _print_footer(usage)
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