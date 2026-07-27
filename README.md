# Property Insights Assistant

[![CI](https://github.com/aghababaeiali/property-insights-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/aghababaeiali/property-insights-assistant/actions/workflows/ci.yml)

An agent that answers questions about a portfolio of short-stay property
listings and their bookings — combining a SQL layer over booking data,
retrieval over listing descriptions, and a cancellation-risk model,
orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph).

```
START -> router -> { sql_node | rag_node | risk_node }
       -> validate_node -> judge_node -> log_node -> END
```

- **router** classifies the question (`sql` / `rag` / `risk`).
- **sql_node** answers aggregate questions (counts, rates, averages) by
  querying Postgres directly.
- **rag_node** answers descriptive questions from listing
  descriptions/host notes via retrieval + LLM synthesis.
- **risk_node** ranks listings by predicted cancellation risk, combining
  the DB, a scikit-learn model, and retrieval for the qualitative "why."
- **validate_node** deterministically checks every answer only cites
  listings that were actually retrieved/scored before it ships.
- **judge_node** samples a fraction of `rag`/`risk` answers for an
  LLM-as-judge groundedness check.
- **log_node** persists every request to Postgres for review (see
  `agent/review.py`) and regression testing (see `agent/eval.py`).

Every one of the LLM/retrieval/risk-scoring backends above comes in two
implementations, selected by a single `LLM_PROVIDER` flag — see
[Running locally](#running-locally-no-azure-account-needed) vs
[Running against Azure](#running-against-azure) below. See
[ARCHITECTURE.md](ARCHITECTURE.md) for how the Azure deployment was built,
the decisions behind it, and real bugs hit and fixed along the way.

See [docs/design.md](docs/design.md) for the fuller design discussion —
observability, scaling, failure modes, and what's deliberately out of scope.

## Running locally (no Azure account needed)

This is the original, unchanged setup — `LLM_PROVIDER=groq` (or `offline`
for tests), a local Postgres via `docker compose`, keyword retrieval, and
the local joblib-backed risk model. None of the Azure additions below are a
requirement for this path; it's still exactly what CI runs.

Requires [uv](https://docs.astral.sh/uv/), Docker, and a
[Groq API key](https://console.groq.com).

```bash
cp .env.example .env          # fill in GROQ_API_KEY
docker compose up -d --wait   # starts Postgres on localhost:5432
uv sync                       # installs dependencies into .venv
uv run python -m agent.db seed   # loads data/*.csv,json into Postgres
uv run python -m ml.model        # trains the cancellation-risk model
```

`.env` is loaded automatically (via `python-dotenv`) by anything that
imports the `agent` package — no need to `export` variables manually.

Seeding is idempotent and explicit — it's a no-op if the DB already has
data, so it's safe to run repeatedly (that's also how CI does it, as its
own step, rather than seeding implicitly as a side effect of the first
query). If you edit `data/*.csv`/`data/*.json`, re-run with `--force` to
reload:

```bash
uv run python -m agent.db seed --force
```

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

## Running against Azure

Everything below is opt-in — set `LLM_PROVIDER=azure` and the app switches,
node by node, to the Azure-backed implementation of each piece. Nothing
here is required for local dev/testing (see above); CI never sets these.

| Local (`groq`/`offline`) | Azure (`azure`) |
|---|---|
| Groq / offline heuristics | Azure OpenAI (chat) |
| In-memory keyword retrieval | Azure AI Search hybrid (keyword + vector) search |
| Local joblib model, trained in-process | Azure ML managed online endpoint, MLflow-tracked training |
| Local docker-compose Postgres | Any reachable Postgres (e.g. Azure Database for PostgreSQL) |

```bash
cp .env.example .env   # uncomment and fill in the "Azure mode" section
```

Required env vars (all read lazily inside the relevant `_azure_*` function
or client-builder — importing the app never requires these unless
`LLM_PROVIDER=azure` actually selects that path):

- `LLM_PROVIDER=azure`
- `DATABASE_URL` — any reachable Postgres (optional — omit to keep using
  the local docker-compose one even while every other piece runs on Azure)
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`,
  `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, `AZURE_SEARCH_INDEX` — and
  an index actually populated via `uv run python -m scripts.build_search_index`
- `AZURE_ML_RISK_ENDPOINT_URL`, `AZURE_ML_RISK_ENDPOINT_KEY` — see
  `ml/azure/` for the job/environment/endpoint/deployment specs that produce
  this endpoint from `ml/model.py`

```bash
uv run python -m agent.run "which Lisbon listings are most at risk of cancellation?"
uv run uvicorn agent.api:app --port 8000   # same FastAPI wrapper used by the container image
```

Container deployment (what's actually running in Azure Container Apps):

```bash
docker build --platform linux/amd64 -t property-insights-agent .
# push to your registry, deploy to Azure Container Apps with the env vars
# above set as secrets/env vars on the container app
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture of what's
Azure-backed, why it's built this way, and real bugs hit along the way.

## Layout

```
agent/
  graph.py           the LangGraph pipeline (router + nodes) — also where the
                      LLM_PROVIDER dispatch for risk-scoring lives
  db.py              builds/queries the Postgres DB
  retriever.py       retrieval over listing text — local keyword search
                      (groq/offline) or Azure AI Search hybrid search (azure)
  llm.py             Groq / Azure OpenAI (+ offline test path)
  risk_client.py     HTTP client for the deployed Azure ML risk endpoint
  api.py             FastAPI wrapper (GET /health, POST /ask) for container deployment
  run.py             CLI entry point
  eval.py            offline regression harness (golden set in eval_cases.json)
  review.py          human review queue CLI
ml/
  model.py           cancellation-risk model: features, training, tuning,
                      MLflow logging/registration
  azure/             Azure ML job/environment/endpoint/deployment specs +
                      the online endpoint's scoring script
scripts/
  build_search_index.py  builds/refreshes the Azure AI Search index
data/
  listings.json      listings: attributes + free-text description & host_notes
  bookings.csv       initial bookings load
  bookings_update.csv  a later batch: new bookings + corrections
tests/               pytest suite (mirrors agent/ and ml/) — runs entirely
                     against LLM_PROVIDER=offline, no Azure credentials needed
notebooks/           exploratory/verification notebooks
docs/                design notes
Dockerfile           container image for the FastAPI app (agent/api.py)
```

## The domain, briefly

- A **booking** has a status: `confirmed`, `completed`, or `cancelled`.
- Some listings are riskier than others; sometimes the reason is only
  written down in a host's free-text notes, not in any structured column.
