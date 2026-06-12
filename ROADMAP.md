# deep-research — Roadmap

Ideas gathered across sessions of 2026-06-11. Grouped by category and prioritized
(🔴 = high, 🟡 = medium, 🟢 = low). `[x]` means done;
`[ ]` pending; `[~]` in progress.

---

## ✅ Done in this session (baseline commit + iterations)

### Session 3 — High priority closed
- [x] **#7 Tavily cost in the report** — constant `TAVILY_COST_PER_SEARCH_USD = 0.001`,
  footer with `LLM cost / Tavily (N × rate) / Total` added to `_run_research` and
  to the `--agentic` mode (4 tests)
- [x] **#23 README.md with examples** — description, features, installation,
  config (including the `.bashrc` early-return warning), usage of every flag,
  output explained, tests, architecture, roadmap
- [x] **#5 Agentic ReAct loop** — `parse_action()` (tolerant JSON parser) +
  `run_research_agentic()` (LLM decides with JSON `search`/`answer`,
  configurable max_iter, every iter instrumented by treval), CLI flag
  `--agentic` (8 tests)
- [x] **#24 PyPI — infrastructure** — `pyproject.toml` (PEP 639),
  `LICENSE` (MIT), `dr` and `deep-research` as entry points, build
  verified, `twine check` PASSED, install in clean venv OK. **Still
  pending: explicit permission for `twine upload` and post-upload
  verification.**

### Session 4 — Quick wins
- [x] **#18 Configurable timeout** — `DEFAULT_TIMEOUT_SECONDS = 30` constant,
  `ask()` accepts a `timeout` param (None disables), both streaming and
  non-streaming paths pass it through (4 tests)
- [x] **#26 Pre-commit hook** — bash script at `.githooks/pre-commit` runs
  `pytest -x -m "not integration"`, auto-discovers venv, bypass via
  `--no-verify` (6 tests + manual end-to-end verification)
- [x] **#3 Tavily `search_depth: "advanced"`** — `DEFAULT_SEARCH_DEPTH = "basic"`,
  `search()` and `search_cached()` take a `search_depth` param ("basic"
  cheap vs "advanced" deeper, ~3x cost) (4 tests)

### Session 5 — REPL, cross-verification, parallel search
- [x] **#12 Interactive REPL** — `run_repl()` with `/clear`, `/exit`, blank
  input; rolling Q&A history passed as `context` to follow-up questions;
  CLI flag `--repl` (10 tests)
- [x] **#20 Cross-verification** — `verify_citations()` runs a second LLM
  pass with `VERIFY_SYSTEM_PROMPT` to check every `[N]` matches a real
  source; tolerant of malformed JSON; CLI flag `--verify` (6 tests)
- [x] **#4 Parallel search** — `parallel_search(queries, max_results,
  parent_id)` uses `ThreadPoolExecutor`; each worker pushes the parent
  span_id manually (treval's `threading.local()` context would otherwise
  be invisible in worker threads); 4x speedup with 4 queries (8 tests)
- Tests: **119 unit + 2 integration (121 total, ~2s)**

### Session 6 — Metadata + citation enforcement
- [x] **#8 Structured metadata in TOOL span** — `_tavily_metadata(args,
  kwargs, result)` attached to the `@treval.tool` decorator via
  `metadata_fn=`. Each `tavily.search` span now records `query`,
  `max_results`, `num_results` in JSON metadata. The treval dashboard
  renders it in the detail panel — every search is self-describing
  (6 tests)
- [x] **#22 Citation enforcement** — `enforce_citations(text,
  num_sources, strip=False)` runs a regex pass over the LLM output,
  returns the list of invalid citation numbers, and the CLI prints
  a `⚠️  Invalid citations` warning. Wired into both `_run_research`
  and `_run_research_streaming` so the warning fires in non-streaming
  and streaming modes alike. Detection: single integer (with optional
  leading minus) inside square brackets; `[v1.0]` and `[ref]` are
  ignored (12 tests)
- Tests: **137 unit + 2 integration (139 total, ~12s including 2 integration)**

### Session 7 — Source report in OPERATION span
- [x] **#9 Source report as structured span metadata** —
  `_source_metadata(all_results, tavily_searches, tavily_cost_usd)`
  builds a dict with `num_sources`, the full deduped `sources`
  list (url/title/content), plus `tavily_searches` and
  `tavily_cost_usd`. Wired into all three research paths
  (`_run_research`, `_run_research_streaming`, `run_research_agentic`)
  via `store.update(root_id, metadata=...)`. Pair with #8: every
  Tavily call has per-call metadata (#8) and the parent OPERATION
  span has the full run's totals + sources (#9). Required a one-line
  fix in treval's `SpanStore.update()` to JSON-encode metadata
  (separate commit in treval repo)
- Tests: **144 unit + 2 integration (146 total)**

### Session 8 — Eval suite (#27)
- [x] **#27 Eval suite for response quality** — LLM-as-judge harness
  with a bundled gold set. New functions: `load_gold_set()`
  (JSONL loader, validates required fields, fails loud on broken
  gold), `parse_judge_json()` (tolerant parser: pure JSON, markdown
  fences, surrounding prose, score clamping, raises on garbage),
  `judge_answer()` (builds prompt with question + reference +
  criteria, calls `ask()` with the cheap flash judge model, parses
  response), `_run_eval()` (orchestrator — for each gold entry
  runs `_run_research` then `judge_answer`, records id/score/
  reason/cost/duration, wraps each question in its own OPERATION
  span `eval.question.<id>` with `{id, score, passed, reason}`
  metadata for the treval dashboard). CLI: `dr eval` subcommand
  with `--gold /path`, `--judge-model`, `--research-model`,
  `--threshold` flags; prints a per-question table + summary
  (mean score, pass rate, total cost); exit 1 when pass_rate <
  threshold (CI-friendly). Gold set: 10 questions at
  `tests/eval/gold.jsonl` — 3 factual simple, 2 multi-hecho,
  2 comparativa, 2 explicativa, 1 técnica Python (GIL).
  Stable over time, no current-events. Integration test on real
  APIs: 10/10 PASS, mean 0.947, $0.00195 total, ~30s
- Tests: **194 unit + 2 integration (196 total)**

---

### Session 1 — Baseline
- [x] `dr.search()` with Tavily (3 results default, configurable)
- [x] `format_context()` + `format_sources()` (pure functions)
- [x] `ask(prompt, context=...)` with optional context
- [x] `main(args)` refactored to take args (testable)
- [x] Always-on search (no opt-in flag)
- [x] 3 sources printed after the answer
- [x] `@treval.tool(name="tavily.search")` → TOOL span with real duration
- [x] Tests: 28/28 passing (1.5s) in `tests/`
- [x] Isolated test DB (fixture `autouse` in conftest)
- [x] Clean production DB: garbage spans removed, backup in `~/.treval/spans.db.bak.1781199802`
- [x] `_run.py` fixed: uses `sys.executable` + venv in PATH
- [x] Typical cost: ~$0.0002 / search (Tavily + LLM flash)

### Session 2 — Recommended Top-5 + medium batch
- [x] **#6 Parent-child hierarchy** — OPERATION span `research` with TOOL and LLM children
- [x] **#10 Flag `--model`** — selects between flash/pro/r1/v3 with prices from the dict
- [x] **#19 System prompt with citation** — citations [1]-[N] + "not found" instead of inventing
- [x] **#15 Local cache with 24h TTL** — SQLite in `~/.treval/search_cache.db`, 0.5ms on hit
- [x] **#1+#2 Multi-query / `--depth N`** — reformulates with LLM, dedupes by URL, depth=1/2/3
- [x] **#11 Flag `--max-results N`** — propagated to Tavily
- [x] **#16 Retry with exponential backoff** — 1s, 2s, 4s; does not retry auth errors
- [x] **#17 Model fallback** — Pro → Flash automatic if primary fails
- [x] **#21 Temperature 0** — deterministic output
- [x] **#14 Streaming** — `--stream` shows the response token by token
- [x] Tests: **69/69 unit + 2 integration** (~2s)

---

## 🔴 High priority

### Search depth
- [x] **#1 Multi-query**: reformulate the question into 2-3 variants, merge results
- [x] **#2 Flag `--depth N`**: 1 round (current) vs 2-3 rounds with follow-up
- [x] **#5 Agentic loop (ReAct)**: the LLM decides whether it needs more searches

### Observability
- [x] **#6 Parent-child hierarchy**: TOOL span as child of LLM
- [x] **#7 Tavily cost** summed in the report (~$0.001/search)

### Distribution
- [x] **#23 README.md** with usage examples
- [~] **#24 Publish on PyPI** as `deep-research` — infra ready, pending upload with your OK

### Evaluation (NEW — quality measurement)
- [x] **#27 Eval suite for response quality** — a `tests/eval/` directory of
  10–20 Q&A pairs with reference answers, runnable as `dr eval` or
  `pytest tests/eval/ -m eval`. Uses LLM-as-judge (cheap model) to score
  subjective answers. The whole eval run is traced by treval so you can
  see cost, latency, and the full span tree per question. Without this,
  changes to the system prompt, model choice, or `--depth` are blind —
  you can't tell if a change improved or regressed the answer quality.
  This is the single highest-leverage addition left: it gates
  confidence in all other quality work. Gold set is the expensive part;
  the harness is small.

---

## 🟡 Medium priority

### Depth
- [x] **#3 `search_depth: "advanced"` from Tavily** (vs "basic") — pricier, more relevant
- [x] **#4 Parallel search** of sub-questions (ThreadPoolExecutor, manual parent push per thread because treval uses threading.local() — 4x speedup with 4 queries)

### Real deep research (NEW)
- [ ] **#28 Iterative deep exploration** — the current `--depth N` is
  breadth, not depth: it reformulates the prompt into N variants and
  searches all of them. Real "deep" research would: read the top
  result, follow its citations/links, drill into specifics (e.g.
  "what is the cost?", "what about edge cases?"), and iterate until
  the LLM signals it has enough. The `--agentic` ReAct loop touches
  this but is shallow (max ~3 iterations, no persistent context
  between iterations). A proper deep mode would distinguish the
  project name from "another Tavily wrapper". Significant scope —
  probably its own design doc before implementation.

### Memory (NEW)
- [ ] **#29 Persistent knowledge base between runs** — every `dr.py`
  invocation is currently ephemeral: search history, source
  relevance signals, and follow-up context die with the process. The
  `--repl` mode has a transient version of this, but it dies when
  the session ends. A persistent local log (SQLite at
  `~/.deep-research/notes.db` or append-only markdown) would let a
  new query use prior runs as context. Changes how the tool is used:
  from "answer a question" to "build a research notebook over time".
  Also unlocks eval against historical data (#27) without re-running
  Tavily every time.

### Observability
- [x] **#8 Structured metadata in TOOL span**: query, max_results, num_results
- [x] **#9 Source report** as structured span input/output

### CLI UX
- [x] **#10 Flag `--model`**
- [x] **#11 Flag `--max-results N`**
- [x] **#12 Interactive mode / REPL**: keep context between questions
- [x] **#14 Streaming**

### Robustness
- [x] **#16 Retry with exponential backoff**
- [x] **#17 Model fallback**
- [x] **#18 Configurable timeout** (30s default, disable with `None`)

### Response quality
- [x] **#20 Cross-verification**: a second LLM pass to verify citations
- [x] **#22 Citation enforcement** + post-processing (URL regex)

### Distribution
- [ ] **#25 GitHub Actions**: pytest + ruff on every PR
- [x] **#26 Pre-commit hook**: `pytest tests/ -x` before commit

---

## 🟢 Low priority (nice-to-have)

- [ ] **#13 Markdown output** with `rich` (skip: LLM already returns markdown, `| mdcat`)
- [x] **#19 Richer system prompt** (citation + no inventing)
- [x] **#21 Temperature 0** for research

---

## 💰 Effort vs impact table (updated summary)

| # | Idea | Impact | Effort | Status |
|---|------|--------|--------|--------|
| 6 | Parent-child spans hierarchy | 🟢 high | 🟢 low | ✅ |
| 10 | Flag `--model` | 🟢 high | 🟢 low | ✅ |
| 19 | System prompt with citation enforcement | 🟢 high | 🟢 low | ✅ |
| 15 | Local search cache | 🟢 high | 🟡 medium | ✅ |
| 1+2 | Multi-query + `--depth` | 🟢 high | 🟡 medium | ✅ |
| 11 | Flag `--max-results N` | 🟡 medium | 🟢 low | ✅ |
| 16 | Retry with backoff | 🟢 high | 🟡 medium | ✅ |
| 17 | Model fallback | 🟢 high | 🟡 medium | ✅ |
| 14 | Streaming | 🟢 high | 🟡 medium | ✅ |
| 21 | Temperature 0 | 🟢 medium | 🟢 trivial | ✅ |
| 7 | Tavily cost in the report | 🟢 medium | 🟢 low | ✅ |
| 23 | README | 🟢 high | 🟢 low | ✅ |
| 5 | Agentic ReAct loop | 🟢 high | 🟡 medium | ✅ |
| 24 | Publish on PyPI | 🟢 high | 🟡 medium | 🔄 (infra ready, upload pending) |
| 18 | Configurable timeout | 🟡 medium | 🟢 low | ✅ |
| 26 | Pre-commit hook | 🟡 medium | 🟢 low | ✅ |
| 3  | Tavily `search_depth: "advanced"` | 🟡 medium | 🟢 low | ✅ |
| 4  | Parallel search (asyncio/threads) | 🟡 medium | 🟡 medium | ✅ |
| 12 | Interactive REPL | 🟡 medium | 🟡 medium | ✅ |
| 20 | Cross-verification (2nd LLM pass) | 🟡 medium | 🟡 medium | ✅ |
| 8  | Structured metadata in TOOL span | 🟡 medium | 🟢 low | ✅ |
| 22 | Citation enforcement (regex) | 🟢 high | 🟢 low | ✅ |
| 9  | Source report in OPERATION span | 🟡 medium | 🟢 low | ✅ |
| 27 | Eval suite (response quality) | 🟢 high | 🟡 medium | ✅ |
| 28 | Iterative deep exploration | 🟢 high | 🔴 high | ❌ |
| 29 | Persistent knowledge base | 🟡 high | 🟡 medium | ❌ |

---

## 🎯 Recommended Top-5 — ALL DONE ✅

1. ✅ **#6** Parent-child hierarchy
2. ✅ **#10** Flag `--model`
3. ✅ **#19** System prompt with citation
4. ✅ **#15** Local cache with TTL
5. ✅ **#1+#2** Multi-query / `--depth`

---

## 🚀 Suggested next steps (next session)

**The natural next moves** (in priority order):

1. **#24 PyPI upload** — infra is ready, only needs your explicit OK
   and a `twine upload` + post-upload verification.
2. **#25 GitHub Actions** — pytest + ruff on every PR. ~20 min of YAML
   if you have any PRs in flight; skip if not.
3. **#29 Persistent knowledge base** — depends on what shape the
   eval takes. With #27 done, you can now run `dr eval` as a
   regression detector before/after changes to #29.
4. **#28 Iterative deep exploration** — biggest scope, deserves its
   own design doc. The existing `--agentic` is the seed.

**Low effort, high impact:**
- ~~#24b Finish PyPI (`twine upload` + post-upload verification)~~ pending explicit permission
- ~~#27 Eval suite~~ ✅ done (10 Q&A starter, harness ready, dr eval CLI, traces)

**Advanced robustness:**
- ~~#18 Configurable timeout in OpenAI client~~ ✅ done
- ~~#8 Structured metadata in spans (for post-mortem queries)~~ ✅ done
- ~~#20 Cross-verification (a second LLM pass)~~ ✅ done
- ~~#22 Citation enforcement (regex)~~ ✅ done
- ~~#9 Source report as structured span metadata~~ ✅ done

**Product features:**
- ~~#12 Interactive REPL (keeps context between questions)~~ ✅ done

**Distribution:**
- **#25** GitHub Actions with pytest + ruff

---

## 📊 Project metrics (at session close)

- **Tests**: 194 unit + 2 integration (196 total)
- **Suite time**: ~20 seconds (includes 2 integration tests with real API)
- **CLI flags**: 9 (`--report`, `--depth`, `--max-results`, `--model`, `--stream`, `--agentic`, `--repl`, `--verify`, prompt)
- **Subcommands**: 1 (`dr eval` with `--gold`, `--judge-model`, `--research-model`, `--threshold`)
- **Public functions**: `search`, `search_cached`, `_tavily_metadata`, `_search_with_parent`, `parallel_search`, `ask`, `format_context`, `format_sources`, `enforce_citations`, `_source_metadata`, `reformulate`, `estimate_cost`, `parse_action`, `verify_citations`, `run_research_agentic`, `run_repl`, `load_gold_set`, `parse_judge_json`, `judge_answer`, `_run_eval`, `_print_eval_report`, `main`
- **Public constants**: `DEFAULT_MODEL`, `DEFAULT_FALLBACK`, `DEFAULT_MAX_RESULTS`, `DEFAULT_TEMPERATURE`, `DEFAULT_TIMEOUT_SECONDS`, `DEFAULT_SEARCH_DEPTH`, `DEFAULT_GOLD_PATH`, `DEFAULT_JUDGE_MODEL`, `DEFAULT_PASS_THRESHOLD`, `TAVILY_COST_PER_SEARCH_USD`, `MAX_RETRIES`, `RETRY_BASE_SECONDS`, `CACHE_TTL_SECONDS`
- **Integrated APIs**: Tavily (search), OpenRouter (chat completions)
- **Spans per research run**: 3-5 (1 OPERATION research + 1-2 TOOL tavily.search + 1 LLM + optional TOOL reformulate)
- **Spans per research run (--agentic)**: 2-7 (1 OPERATION research_agentic + N×TOOL tavily.search + N×LLM, where N ≤ max_iter)
- **Spans per eval run (10 Q)**: ~30-50 (10 OPERATION eval.question + 10 OPERATION research + ~20 LLM + ~10 TOOL)
- **Typical cost per query**: $0.0002 (depth=1, basic) to $0.0006 (depth=3, basic)
- **Typical cost --agentic**: $0.0001 to $0.0010 depending on iterations
- **Typical cost --eval (10 Q)**: ~$0.002 with flash judge, ~$0.02 with pro judge
- **Tavily cost (advanced depth)**: ~3x basic — $0.003 per search instead of $0.001
- **OpenAI client timeout**: 30s default, configurable per `ask()` call
- **Eval baseline (real APIs, 10 Q, flash)**: mean 0.947, 10/10 PASS, $0.00195
