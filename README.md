# Health agents — a LangGraph router demo

A console app where a **router** reads your message and dispatches it to one of three
agents:

| Agent | Handles | How it works |
|---|---|---|
| **Summarizer** | "summarize `samples/sample_report.txt`", "explain these results" | Reads a `.txt` path from your message, or pauses and asks you to paste the report |
| **Report builder** | "create a health report for me" | Interviews you one field at a time, then writes a mock report |
| **Q/A bot** | "what causes anemia?" | Answers general questions, remembering the conversation |

Reports it generates are **mock documents for demonstration**, not medical records or
medical advice.

## Setup

```bash
cp .env.example .env      # then add your GROQ_API_KEY
uv sync
uv run main.py
```

Get a key at <https://console.groq.com/keys>.

## Choosing models

`GROQ_MODEL` sets the model for every agent. Each agent can then be overridden
independently, falling back to `GROQ_MODEL` when unset:

| Agent | Override | What the job actually demands |
|---|---|---|
| `router` | `GROQ_ROUTER_MODEL` | One 3-way classification, on **every turn**. Highest-volume call and the easiest task — where a smaller model pays off most. |
| `summarizer` | `GROQ_SUMMARIZER_MODEL` | Reading a dense numeric report without inventing values. The hardest job; most rewards a larger model. |
| `report_writer` | `GROQ_REPORT_MODEL` | A formatter. Every fact comes from the interview, so it needs restraint, not knowledge. |
| `qa` | `GROQ_QA_MODEL` | Open-ended questions. Wants breadth and readable prose. |

Run **`/models`** inside the app to see what is in effect and which var overrides it.

Model ids are **checked at startup** against what your key can actually reach, so a typo
or a retired id fails immediately with a suggestion rather than surfacing later when that
one agent happens to run:

```
Configured models are not usable:
  • report_writer: GROQ_REPORT_MODEL=gpt-oss-20b is not available to this key.
    Did you mean openai/gpt-oss-20b?
```

If the model list cannot be fetched (offline), startup warns and continues.

Whatever you pick **must support tool calling** — that is what the router's structured
output is built on. Groq retires model ids periodically, so list what your key reaches:

```bash
uv run python -c "import groq,dotenv;dotenv.load_dotenv();print([m.id for m in groq.Groq().models.list().data])"
```

Measured on this project's sample report (one run each, so treat as rough):

| Model | Router accuracy | Summarize latency |
|---|---|---|
| `qwen/qwen3.8-27b` | 8/8 | 1.8s |
| `openai/gpt-oss-120b` | 8/8 | 2.4s |
| `openai/gpt-oss-20b` | not measured | 1.5s |

## Using it

```
you › what causes anemia?                    → Q/A bot
you › summarize samples/sample_report.txt    → summarizer
you › summarize this report                  → summarizer, asks you to paste
you › create a health report for me          → report builder, asks 8 questions
```

Commands: `/help`, `/models`, `/new` (forget the conversation), `/quit`.

When asked to paste a report, finish with a line containing only `EOF`. A single line
naming a real file is accepted as a path instead.

## How it fits together

```
                    ┌────────────────────────────► summarizer ────────────┐
                    │                              (interrupt if no text) │
START ──► router ───┼──────► qa ──────────────────────────────────────────┼──► END
                    │        (reads + appends messages)                   │
                    └──► report_builder ◄──┐                              │
                              │  asks ONE  │ fields still missing         │
                              │  field     └──────────────────────────────┘
                              └──► report_writer ──► END   (all fields present)
```

Five nodes for three agents. The report builder is split in two on purpose:

`report_builder` asks for exactly **one** field per execution and makes **no** LLM call.
`report_writer` makes a single prose call once every field is in.

That split is forced by how `interrupt()` works. When you resume with
`Command(resume=...)`, LangGraph **re-runs the interrupted node from its first line**,
with `interrupt()` returning your answer instead of pausing. So everything above the
`interrupt()` call runs again for every field. Keeping that region pure
(`next_missing_field`) makes the repeat harmless; moving the generative call into its own
node means it happens once instead of once per question.

### Design notes

- **`next_missing_field` is the only thing driving the interview.** The node calls it to
  pick a question and the graph's loop edge calls it to decide whether to ask again. One
  pure function, so the interview provably terminates and is testable with no LLM.
- **The router returns a single enum field** via `with_structured_output`. That is the
  smallest surface an open-weight model can get wrong, and any failure falls back to the
  Q/A agent — always a sane thing to do with an arbitrary message.
- **Interrupt payloads are data, not strings**: `{"prompt": ..., "multiline": True}`. The
  graph says *what* input it needs; `cli.py` decides *how* to collect it. That is why no
  agent contains a call to `input()`.
- **The router clears the chosen agent's carried-over state**, so a second report starts
  from scratch and "summarize this report" cannot silently reuse an earlier one.
- **`messages` is Q/A-only**, keeping chat history a clean dialogue rather than a log of
  every node that ran.
- **A mid-turn API failure is recoverable, not fatal.** The checkpointer still holds
  everything answered so far, so a rate limit on the final call of an 8-question
  interview offers `Retry from there?` and resumes with `stream(None, config)` — the
  answers are never re-asked. This is a benefit that only exists *because* the interview
  uses `interrupt()` and a checkpointer; the plain `input()` loop we rejected early on
  would have had nothing to resume.
- **Each agent resolves its own model** via `config.get_llm(role)`, and clients are cached
  by resolved model id — so roles sharing a model share one connection. `build_graph(llm=...)`
  still accepts a single model for every role, which is how the tests inject a fake.

## Tests

```bash
uv run pytest
```

All tests run offline with **no API key and no network**. A scripted `FakeChatModel`
(`tests/conftest.py`) stands in for Groq, which is what lets `tests/test_report_flow.py`
drive the entire 8-question interview through `stream` / `Command(resume=...)` and assert
one interrupt per field, in order.

## Layout

```
main.py                     entrypoint
src/health_agents/
  config.py                 env loading, the one place ChatGroq is built
  state.py                  AppState
  schema.py                 field list + next_missing_field (pure)
  io_utils.py               path-vs-pasted-text detection
  prompts.py                every system prompt
  graph.py                  node wiring
  cli.py                    console loop, the only module touching stdin
  nodes/                    router, summarizer, report_builder, qa
tests/                      offline suite
samples/sample_report.txt   fixture for the summarizer
```

## Not included (deliberate)

PDF input, saving reports to disk, resuming an interview after restart (needs a SQLite
checkpointer instead of `InMemorySaver`), adaptive LLM-chosen follow-up questions, and
token streaming. Each is additive — disk saving is a few lines in `report_writer`.
