# deep-research

CLI that answers research questions with web search context, auto-instrumented with **[treval](https://github.com/AmOrFeU86/treval)** for observability and cost tracking.

```
$ python dr.py "What is the capital of France?"
🔍 Searching the web for: What is the capital of France? (depth=1)
The capital of France is Paris [1].

────────────────────────────────────────
  Tokens:    100↑ + 50↓ = 150
  LLM cost:  $0.00015
  Tavily:    1 search × $0.0010 = $0.00100
  Total:     $0.00115

Sources:
  [1] https://en.wikipedia.org/wiki/Paris
```

## Features

- 🔍 **Always-on search** — every question is preceded by a web search (Tavily).
- 🧠 **Multi-query with `--depth N`** — reformulates the question with the LLM and runs all N queries in parallel via `ThreadPoolExecutor`, then merges deduplicated results by URL.
- 🎯 **Forced citation** — the system prompt requires `[1]`, `[2]`...; if info is not in the sources, it says "Not found".
- 🔁 **Retry with backoff + model fallback** — resilient to transient failures.
- 💸 **Real cost report** — LLM tokens + Tavily searches summed in the footer.
- 📺 **Optional streaming** — `--stream` shows the response token by token.
- 🤖 **Agentic ReAct loop** — `--agentic` lets the LLM decide whether it needs more searches before answering.
- 💾 **Local cache with 24h TTL** — repeated searches cost nothing.
- 📊 **Auto-instrumented with treval** — every run creates 3-7 spans (OPERATION → TOOL → LLM) visible in the HTML dashboard.

## Installation

```bash
git clone https://github.com/AmOrFeU86/deep-research.git
cd deep-research
python3 -m venv .venv
source .venv/bin/activate
pip install treval openai tavily-python pytest
```

## Configuration

API keys are read from environment variables. Set them in `~/.bashrc`:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export TAVILY_API_KEY="tvly-..."
```

> ⚠️ The early-return in `.bashrc` for non-interactive shells makes the keys NOT available to scripts. Use the included wrapper `_run.py` (it loads them explicitly), or `bash -i -c "..."`.

## Usage

### Basic — one question

```bash
python dr.py "What is the capital of France?"
```

### Search depth

```bash
# depth=1: 1 search of the original question
python dr.py --depth 1 "What is quantum entanglement?"

# depth=2: reformulates into 1 variant → 2 searches → merged
python dr.py --depth 2 "What is quantum entanglement?"

# depth=3: reformulates into 2 variants → 3 searches → merged
python dr.py --depth 3 "What is quantum entanglement?"
```

### Change model

```bash
python dr.py --model deepseek/deepseek-v4-pro "..."
python dr.py --model deepseek/deepseek-v4-flash "..."  # default
python dr.py --model deepseek/deepseek-r1 "..."  # reasoning
```

### More results per search

```bash
python dr.py --max-results 5 "What is the latest in AI?"
```

### Streaming

```bash
python dr.py --stream "Explain the CAP theorem"
```

### Agentic mode (ReAct)

```bash
# The LLM decides autonomously whether it needs more searches.
# max_iter = depth + 2, so --depth 3 caps the loop at 5 iterations.
python dr.py --agentic --depth 2 "Compare the educational systems of Finland and Singapore"
```

### Generate HTML dashboard

```bash
python dr.py --report "What is the meaning of life?"
# → writes report.html with the span tree + cost metrics
```

### REPL (coming soon)

Pending — `--repl` to keep context between questions.

## Output

Each run prints:

1. **The LLM's answer** (with `[N]` citations).
2. **The metrics footer**:
   - `Tokens:` tokens consumed
   - `LLM cost:` LLM cost
   - `Tavily:` number of searches × unit cost
   - `Total:` sum
3. **Sources:** URLs of the consulted sources.

## Tests

```bash
# Full suite (93 tests, ~6s, no API keys needed for unit tests)
pytest tests/ -q

# Integration only (requires OPENROUTER_API_KEY + TAVILY_API_KEY)
pytest -m integration -v
```

### Pre-commit hook (optional but recommended)

Run the test suite automatically before each commit:

```bash
git config core.hooksPath .githooks
```

Bypass for a single commit (use sparingly): `git commit --no-verify`.

## Architecture

Each `python dr.py "question"` generates a span tree in treval:

```
# Non-agentic (--depth 2):
OPERATION  research  [question] [depth=2]
  └─ TOOL    tavily.search  [question]  ~500ms  $0.001
  └─ TOOL    tavily.search  [variant 1] ~400ms  $0.001
  └─ LLM     ask             [reformulate]  ~800ms  $0.0001
  └─ LLM     ask             [question + context] ~1500ms $0.0002

# Agentic (--agentic, 2 iterations):
OPERATION  research_agentic  [question] [agentic, max_iter=4]
  └─ LLM     ask             [parse action: search]  ~600ms  $0.0001
  └─ TOOL    tavily.search  [follow-up query]  ~400ms  $0.001
  └─ LLM     ask             [parse action: answer]  ~1500ms $0.0002
```

Visualize with `treval dashboard --export report.html`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for details. Current status:

- ✅ Top-5 recommended (parent-child spans, --model, citation, cache, multi-query)
- ✅ Robustness (retry, fallback, streaming, temperature 0)
- ✅ High priority: agentic ReAct loop, Tavily cost, README, PyPI infra
- 🟡 Medium: advanced search_depth, asyncio parallel, REPL, cross-verification
- 🟢 Low: rich markdown output, GitHub Actions

## License

MIT
