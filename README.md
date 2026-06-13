# deep-research

CLI that answers research questions with web search context, auto-instrumented with **[treval](https://github.com/AmOrFeU86/treval)** for observability and cost tracking. Always-on self-critique pass catches citation hallucinations before they reach you.

```
$ python dr.py "What is the capital of France?"
🔍 Searching the web for: What is the capital of France? (depth=10)
The capital of France is Paris [1].

────────────────────────────────────────
  Tokens:    100↑ + 50↓ = 150
  LLM cost:  $0.00015
  Tavily:    10 search × $0.0010 = $0.01000
  Total:     $0.01015

Sources:
  [1] https://en.wikipedia.org/wiki/Paris
```

## Features

- 🔍 **Always-on search** — every question is preceded by a web search (Tavily).
- 🧠 **Deep by default (depth=10)** — reformulates the original question into 9 variants with the LLM, runs all 10 in parallel via `ThreadPoolExecutor`, then merges deduplicated results by URL.
- 🎯 **Forced citation** — the system prompt requires `[1]`, `[2]`...; if info is not in the sources, it says "Not found".
- 🔁 **Self-critique pass** — after the first answer, a second LLM pass verifies every citation; if issues are found, the model is re-prompted with the critique and the rewritten answer replaces the original. No `−−verify` flag needed — always on.
- 🔁 **Retry with backoff + 429/5xx handling** — resilient to network, rate limits, and server errors.
- 💸 **Real cost report** — LLM tokens + Tavily searches summed in the footer.
- 📺 **Optional streaming** — `−−stream` shows the response token by token.
- 💾 **Local cache with 24h TTL** — repeated searches cost nothing.
- 📊 **Auto-instrumented with treval** — every run creates 3-7 spans (OPERATION → TOOL → LLM) visible in the HTML dashboard.
- 🧪 **`dr eval`** — bundled gold set of 10 Q&A + LLM-as-judge for response-quality regression testing.

## Installation

```bash
git clone https://github.com/AmOrFeU86/deep-research.git
cd deep-research
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # editable install + pytest
pip install -e ~/proyectos/treval   # local treval with metadata_fn support
```

## Configuration

API keys are read from environment variables. Set them in `~/.bashrc`:

```bash
export MINIMAX_API_KEY="..."
export TAVILY_API_KEY="tvly-..."
```

The default LLM provider is **minimax** (model `MiniMax-M3`), hardcoded in `dr.py` (see `PROVIDERS` at the top).

> ⚠️ The early-return in `.bashrc` for non-interactive shells makes the keys NOT available to scripts. Use the included wrapper `_run.py` (it loads them explicitly), or `bash -i -c "..."`.

## Usage

### Basic — one question

```bash
python dr.py "What is the capital of France?"
```

### Streaming

```bash
python dr.py --stream "Explain the CAP theorem"
```

### Interactive REPL

```bash
python dr.py --repl
```

The REPL keeps a rolling Q&A history — each new question sees prior turns as context, so follow-ups like *"what about its climate?"* make sense. Commands: `/clear` (reset history), `/exit` or `/quit` (end session).

### Generate HTML dashboard

```bash
python dr.py "What is the meaning of life?"
# writes report.html with the span tree + cost metrics
```

### Eval suite

```bash
python dr.py eval                       # run the bundled gold set
python dr.py eval --threshold 0.8       # stricter pass threshold
python dr.py eval --gold path/to.jsonl  # custom gold set
```

Each gold entry is scored 0.0-1.0 by a cheap LLM judge against a reference answer and a criteria checklist. Exits 1 if pass rate is below threshold (CI-friendly).

## Output

Each run prints:

1. **The LLM's answer** (with `[N]` citations).
2. **The metrics footer**:
   - `Tokens:` tokens consumed
   - `LLM cost:` LLM cost
   - `Tavily:` number of searches × unit cost
   - `Total:` sum
3. **Sources:** URLs of the consulted sources.

Streaming prints the answer as it arrives, then a reduced footer (Tavily cost only; LLM token counts aren't available mid-stream).

## Quality knobs (constants, not flags)

The following are now hardcoded at the top of `dr.py` to keep the eval signal clean. Tweak them and re-run `dr eval` to measure the delta:

| Constant | Default | What it does |
|---|---|---|
| `DEFAULT_DEPTH` | 10 | 1 original + 9 reformulated queries |
| `DEFAULT_MAX_RESULTS` | 3 | Tavily results per query |
| `DEFAULT_SEARCH_DEPTH` | `"basic"` | Tavily `basic` (cheap) vs `advanced` (3x, deeper) |
| `SELF_CRITIQUE` | `True` | Re-prompt if verifier finds citation issues |

## Tests

```bash
# Full suite (no API keys needed for unit tests)
pytest tests/ -q

# Integration only (requires MINIMAX_API_KEY + TAVILY_API_KEY)
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
# Default (depth=10):
OPERATION  research  [question] [depth=10]
  ├─ TOOL    tavily.search  [question]  ~500ms  $0.001
  ├─ TOOL    tavily.search  [variant 1] ~400ms  $0.001
  ├─ TOOL    tavily.search  [variant 2] ~380ms  $0.001
  ├─ ...                          (8 more parallel)
  ├─ LLM     ask             [reformulate]  ~800ms  $0.0001
  ├─ LLM     ask             [question + context] ~1500ms $0.0002
  └─ LLM     ask             [self-critique, if issues] ~1500ms $0.0002
```

Visualize with `treval dashboard --export report.html`.

## License

MIT
