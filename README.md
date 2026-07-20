# Property Insights Assistant

[![CI](https://github.com/aghababaeiali/property-insights-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/aghababaeiali/property-insights-assistant/actions/workflows/ci.yml)

An agent that answers questions about a portfolio of short-stay property
listings and their bookings — combining a SQL layer over booking data,
keyword retrieval over listing descriptions, and a cancellation-risk model,
orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph).

```
START -> router -> { sql_node | rag_node | risk_node }
       -> validate_node -> judge_node -> log_node -> END
```

- **router** classifies the question (`sql` / `rag` / `risk`) via Groq.
- **sql_node** answers aggregate questions (counts, rates, averages) by
  querying Postgres directly.
- **rag_node** answers descriptive questions from listing
  descriptions/host notes via keyword retrieval + Groq synthesis.
- **risk_node** ranks listings by predicted cancellation risk, combining
  the DB, a scikit-learn model, and retrieval for the qualitative "why."
- **validate_node** deterministically checks every answer only cites
  listings that were actually retrieved/scored before it ships.
- **judge_node** samples a fraction of `rag`/`risk` answers for an
  LLM-as-judge groundedness check.
- **log_node** persists every request to Postgres for review (see
  `agent/review.py`) and regression testing (see `agent/eval.py`).

See [docs/design.md](docs/design.md) for the fuller design discussion —
observability, scaling, failure modes, and what's deliberately out of scope.

## Setup

Requires [uv](https://docs.astral.sh/uv/), Docker, and a
[Groq API key](https://console.groq.com).

```bash
cp .env.example .env       # fill in GROQ_API_KEY
docker compose up -d --wait   # starts Postgres on localhost:5432
uv sync                       # installs dependencies into .venv
uv run python -m ml.model     # trains the cancellation-risk model
```

`.env` is loaded automatically (via `python-dotenv`) by anything that
imports the `agent` package — no need to `export` variables manually.

## Usage

```bash
uv run python -m agent.run "what is the average price in Lisbon?"
uv run python -m agent.run "why might some Lisbon listings be problematic?"
uv run python -m agent.run "which Lisbon listings are most at risk of cancellation?"
```

Prints the classified intent, the answer, and any validation issues found.

## Testing

```bash
uv run pytest              # full suite — needs Postgres running (docker compose up -d)
uv run python -m agent.eval   # same golden-set checks, as a standalone CLI report
uv run ruff check .           # lint
```

Tests run against `LLM_PROVIDER=offline` (a deterministic, no-network path —
see `agent/llm.py`), so no API key is required to run them; this is also
what CI uses. `LLM_PROVIDER=groq` (the runtime default) is never used in
tests.

## Reviewing flagged answers

```bash
uv run python -m agent.review list                        # pending review items
uv run python -m agent.review promote <id> --intent rag    # add a reviewed case to the eval golden set
uv run python -m agent.review resolve <id>                 # dismiss without promoting
```

## Layout

```
agent/
  graph.py           the LangGraph pipeline (router + nodes)
  db.py              builds/queries the Postgres DB
  retriever.py       keyword retrieval over listing text
  llm.py             Groq (+ offline test path)
  run.py             CLI entry point
  eval.py            offline regression harness (golden set in eval_cases.json)
  review.py          human review queue CLI
ml/
  model.py           cancellation-risk model: features, training, tuning
data/
  listings.json      listings: attributes + free-text description & host_notes
  bookings.csv       initial bookings load
  bookings_update.csv  a later batch: new bookings + corrections
tests/               pytest suite (mirrors agent/ and ml/)
notebooks/           exploratory/verification notebooks
docs/                design notes
```

## The domain, briefly

- A **booking** has a status: `confirmed`, `completed`, or `cancelled`.
- Some listings are riskier than others; sometimes the reason is only
  written down in a host's free-text notes, not in any structured column.
