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
- Tests: **81 unit + 2 integration (83 total)** (~6s)

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

---

## 🟡 Medium priority

### Depth
- [ ] **#3 `search_depth: "advanced"` from Tavily** (vs "basic") — pricier, more relevant
- [ ] **#4 Parallel search** of sub-questions (asyncio)

### Observability
- [ ] **#8 Structured metadata in TOOL span**: query, max_results, num_results
- [ ] **#9 Source report** as structured span input/output

### CLI UX
- [x] **#10 Flag `--model`**
- [x] **#11 Flag `--max-results N`**
- [ ] **#12 Interactive mode / REPL**: keep context between questions
- [x] **#14 Streaming**

### Robustness
- [x] **#16 Retry with exponential backoff**
- [x] **#17 Model fallback**
- [ ] **#18 Configurable timeout**

### Response quality
- [ ] **#20 Cross-verification**: a second LLM pass to verify citations
- [ ] **#22 Citation enforcement** + post-processing (URL regex)

### Distribution
- [ ] **#25 GitHub Actions**: pytest + ruff on every PR
- [ ] **#26 Pre-commit hook**: `pytest tests/ -x` before commit

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

---

## 🎯 Recommended Top-5 — ALL DONE ✅

1. ✅ **#6** Parent-child hierarchy
2. ✅ **#10** Flag `--model`
3. ✅ **#19** System prompt with citation
4. ✅ **#15** Local cache with TTL
5. ✅ **#1+#2** Multi-query / `--depth`

---

## 🚀 Suggested next steps (next session)

**Low effort, high impact:**
- **#24b** Finish PyPI (`twine upload` + post-upload verification)

**Advanced robustness:**
- **#18** Configurable timeout in OpenAI client
- **#8** Structured metadata in spans (for post-mortem queries)

**Product features:**
- **#12** Interactive REPL (keeps context between questions)
- **#20** Cross-verification (a second LLM pass)

**Distribution:**
- **#25** GitHub Actions with pytest + ruff

---

## 📊 Project metrics (at session close)

- **Tests**: 81 unit + 2 integration (83 total)
- **Suite time**: ~6 seconds (includes 2 integration tests with real API)
- **CLI flags**: 7 (`--report`, `--depth`, `--max-results`, `--model`, `--stream`, `--agentic`, prompt)
- **Public functions**: `search`, `search_cached`, `ask`, `format_context`, `format_sources`, `reformulate`, `estimate_cost`, `parse_action`, `run_research_agentic`, `main`
- **Integrated APIs**: Tavily (search), OpenRouter (chat completions)
- **Spans per research run**: 3-5 (1 OPERATION research + 1-2 TOOL tavily.search + 1 LLM + optional TOOL reformulate)
- **Spans per research run (--agentic)**: 2-7 (1 OPERATION research_agentic + N×TOOL tavily.search + N×LLM, where N ≤ max_iter)
- **Typical cost per query**: $0.0002 (depth=1) to $0.0006 (depth=3)
- **Typical cost --agentic**: $0.0001 to $0.0010 depending on iterations
