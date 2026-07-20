# Design notes: productionizing this

Working notes on what it would take to run this for real, not just locally —
observability, scaling, failure modes, and what's deliberately left out of
scope for now. Written to be revisited, not a one-time writeup.

## Observability

Already built, not hypothetical:

- **Per-request tracing**: every graph run is logged to `agent_logs`
  (`log_node` in `agent/graph.py`) — question, intent, answer, latency,
  provider, and validation/judge outcomes. LangSmith tracing is also wired
  in (`LANGCHAIN_TRACING_V2=true`) for per-node input/output/latency with
  zero extra code, since LangGraph auto-instruments when the env vars are
  set.
- **Answer quality signal**: `validate_node` (deterministic, every request)
  checks that an answer only cites listings that were actually
  retrieved/scored — catches hallucinated citations with no added latency.
  `judge_node` (sampled, `JUDGE_SAMPLE_RATE`) asks the model itself whether
  an answer is grounded in its context — catches subtler issues the
  deterministic check can't (a claim that overstates real, cited data).
  Anything either flags lands in `review_queue` for human review, and
  reviewed cases can be promoted into the regression suite (`agent/review.py
  promote`).
- **Regression gate**: `agent/eval.py` / `tests/test_eval.py` — a small
  golden set, checked in CI on every push.

What's still missing, not yet built:

- **Dashboards/alerting** on what's already being collected —
  `validation_issues` rate, `judge_grounded=false` rate, p50/p95 latency per
  intent, cost per day per provider. The data exists in `agent_logs`;
  nothing currently watches it or pages anyone.
- **Model drift monitoring**. Cancellation base rate isn't stationary in
  this data (41.9% early in the training window vs 38.5% later) — a real
  deployment needs a scheduled job comparing live prediction distribution
  against training-time assumptions, not a one-time AUC number from
  `ml/metrics.json`.

## Scaling

The agent logic itself isn't the bottleneck — two specific design choices
are:

1. `agent/db.py`'s `_build()` drops and reloads `listings`/`bookings`/
   `bookings_initial` from the CSV/JSON source files on a process's first DB
   connection. Fine for one local instance; breaks with more than one
   process running (each would race to rebuild the same tables), and
   doesn't reflect how real transactional booking data would actually
   arrive. Production needs a real ETL/migration pipeline, with the agent
   connecting to an already-live analytics DB it never rebuilds itself.
2. `agent/retriever.py` is an O(n) linear scan over every listing on every
   query. Entirely fine at 80 listings; would need real vector search
   (embeddings + an index) past a few thousand.

On the other hand: the compiled `AGENT` graph is stateless per request — no
server-side session, all state lives in Postgres and the request payload —
so horizontally scaling the agent process itself is trivial once the two
issues above are addressed. `JUDGE_SAMPLE_RATE` is the existing cost/coverage
lever for the LLM-as-judge at higher traffic.

## Failure modes

The honest gap: `sql_node`, `rag_node`, and `risk_node` have **no error
handling**. A Postgres blip or a malformed query throws an unhandled
exception straight out of `AGENT.invoke(...)` — no graceful degradation, no
user-facing fallback message. `log_node` is the one node that *is* wrapped
in try/except, deliberately, since logging must never break the answer
path; the same discipline needs to extend to the answer-producing nodes
before this is customer-facing (catch, log, return a clear "something went
wrong" answer instead of propagating).

Also not yet handled: no retry/circuit-breaker around the Groq calls
(`_groq_classify` / `_groq_synth` / `_groq_judge` in `agent/llm.py`) — a
provider outage or rate-limit currently propagates as a raw exception rather
than a retry or a fallback path.

## Deliberately out of scope (for now)

Named explicitly rather than silently skipped:

- **PII / data retention.** `agent_logs` stores raw questions and answers
  indefinitely, no redaction, no retention window. Fine for internal/dev
  use; a deployment touching real guest/host data needs a retention policy
  and a PII review before shipping.
- **Auth / multi-tenancy.** No concept of "who is asking" anywhere in the
  current design.
- **Model versioning / rollback.** `ml/cancellation_model.joblib` is
  overwritten in place on every `train()` run — no registry, no ability to
  roll back a bad retrain, no A/B comparison between model versions.
- **Automatic retraining.** `python -m ml.model` is a manual step. Given the
  base-rate drift noted above, a real system needs a scheduled retrain plus
  the drift-monitoring mentioned there to know *when* to trigger one.

## What was actually tried and rejected (ml/model.py)

Kept here instead of only in code comments, since it's easy to re-litigate
a decision without this context:

- **GradientBoostingClassifier vs LogisticRegression**: 4 model families
  compared via 5-fold CV: GBM (0.701), RandomForest (0.678),
  HistGradientBoosting (0.671), LogisticRegression (0.714, lowest variance
  of the four). LR won — with ~4,000 rows and 14 features, tree ensembles
  don't have enough data to find real nonlinear interactions beyond what a
  linear boundary already captures.
- **Extra features** (`city`, `guest_country`, `property_type`,
  `instant_book`, and later a leakage-safe `listing_cancel_rate_prior`):
  each tested properly (multiple model families, L1 regularization, 100
  repeated-CV folds for the listing-rate feature specifically) and each
  came back as a genuine null result. Reverted rather than carry unused
  complexity — notably an extra live DB query per `risk_node` call, for the
  listing-rate feature.
- **Time-based vs random train/test split**: verified the random-split
  metrics weren't optimistic due to hidden temporal leakage — a strict
  earliest-75%/most-recent-25% split gives essentially the same AUC (0.707
  vs 0.718), even after re-tuning hyperparameters specifically for the
  time-respecting CV.
- **Ceiling**: after exhausting model family, hyperparameters, and every
  reasonably-derivable feature, performance sits at ROC-AUC ~0.70–0.72. The
  one untried, more expensive lever: deriving structured signal from
  `host_notes` (currently pure display text, never reaches the model) —
  e.g. keyword flags or embeddings.
