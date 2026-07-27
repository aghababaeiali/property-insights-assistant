# Architecture: the Azure migration

A record of what was built to run this app on Azure end-to-end, the
decisions behind it, and real bugs hit and fixed along the way — written
while the resources were still up and the details were verifiable, not
reconstructed later from memory. The Azure resources themselves were built
on a free-tier subscription and won't outlive it; this document is the
durable record.

## What was built

Every backend the app depends on has a local implementation and an
Azure-backed one, selected by a single `LLM_PROVIDER` flag (see the README's
"Running against Azure" section for the env vars). Concretely:

| Piece | Local | Azure |
|---|---|---|
| LLM (intent classification, synthesis, judge) | Groq / offline heuristics | Azure OpenAI (`gpt-5-mini`) |
| Retrieval | in-memory keyword scoring over `data/listings.json` | Azure AI Search, hybrid keyword + vector search over an index built from the same data, embeddings from Azure OpenAI (`text-embedding-3-small`) |
| Cancellation-risk model | local joblib file, trained in-process | trained as an Azure ML job with MLflow tracking/registration, served from a managed online endpoint |
| Postgres | local docker-compose | Azure Database for PostgreSQL |
| App runtime | `uv run` on a laptop | Docker image on Azure Container Apps |

Resource group `property-insights-rg`, workspace `property-insights-ml`,
container registry `propertyinsightsacr`, container app
`property-insights-app`, compute cluster `cpu-cluster`.

## Key decisions

**One flag drives everything, not three.** The original ask was narrower —
just wire the deployed risk endpoint into `risk_node`. But `LLM_PROVIDER`
already existed as the toggle for the LLM backend, and the retriever/risk
backends needed the exact same two-mode split for the same reason: CI runs
`LLM_PROVIDER=offline` with zero Azure credentials, and the retriever and
risk-scoring modules are imported unconditionally by `agent/graph.py`
regardless of which node actually runs. Reusing `LLM_PROVIDER` rather than
adding `RETRIEVER_PROVIDER`/`RISK_PROVIDER` flags keeps "local mode" and
"Azure mode" as exactly two coherent, testable configurations instead of
2×2×2 combinations, most of which nobody would ever run.

**Azure credentials are read lazily, never at import time.** Both
`agent/retriever.py` and `agent/llm.py` build their Azure SDK clients
(`SearchClient`, `AzureOpenAI`) inside the function that needs them on
first call, not as module-level globals. The first version of the Azure
Search retriever read `os.environ["AZURE_SEARCH_ENDPOINT"]` at module scope
— harmless in isolation, except `agent/graph.py` imports `retriever`
unconditionally, so it made Azure Search credentials a hard requirement
just to import the agent at all, in any mode. Local/offline runs and CI
would have broken immediately.

**A custom scoring script, not Azure ML's no-code MLflow deployment.**
Azure ML can deploy a registered MLflow model with zero scoring code — but
that path always calls the pyfunc flavor's `predict()` method, which for
the sklearn flavor returns class labels (`model.predict()`), not
`predict_proba()`. This app needs the probability, not the class, so
`ml/azure/score.py` loads the model directly via `mlflow.sklearn.load_model`
and calls `predict_proba` explicitly. (It also has to patch `sys.path`
before that import — Azure ML's inference server only puts the scoring
script's own directory on `sys.path`, not the uploaded code root two levels
up, so `from ml.model import FEATURES` doesn't resolve without it.)

**MLflow tracking is a no-op locally, real inside Azure ML.**
`ml.model.train()` now wraps training in `mlflow.start_run()` and logs
params/metrics/the model unconditionally. With no `MLFLOW_TRACKING_URI` set
(the local/laptop case), MLflow defaults to a local `./mlruns` file store —
no network call, nothing Azure-specific. Inside an Azure ML job, the
tracking URI is auto-injected by the platform, so the exact same training
code registers the model into the Azure ML workspace's registry. One
code path, two environments, no branching needed in `ml/model.py` itself.

## Real bugs found and fixed

### 1. ACR pull permission failure, obscured by the Azure CLI

Submitting the first Azure ML training job (`az ml job create`) failed with
a generic image-pull error at the top level — nothing in the headline
message pointed at permissions. The real cause only showed up by pulling
the job's actual logs (`azureml-logs/`) rather than trusting the CLI's
summary: the compute cluster's system-assigned managed identity had no
`AcrPull` role on the container registry (`propertyinsightsacr`), so it
could authenticate to Azure but was denied at the registry-permission
layer when trying to pull the training image. Fixed by granting the
`AcrPull` role to the compute cluster's managed identity against the ACR
resource. The general lesson that held for the rest of this migration:
Azure ML's and the Azure CLI's top-level error messages are frequently a
generic wrapper (`BadArgument`, `SubscriptionNotRegistered`,
`ResourceNotReady`) — the real cause is almost always one layer down, in
the job's `azureml-logs/`/`user_logs/` or in `--debug` output, never in the
headline text.

### 2. MLflow's `/logged-models` 404 — the version pairing was outdated advice

`mlflow.sklearn.log_model()` inside the training job failed with:

```
mlflow.exceptions.MlflowException: API request to endpoint
/api/2.0/mlflow/logged-models failed with error code 404 != 200
```

The commonly-cited fix (`mlflow<2.8`) didn't hold — pinning it made no
difference. The actual, current constraint (per Microsoft's own docs, which
had moved on from the old advice): Azure ML's MLflow-compatible tracking
backend supports **MLflow 2.16.x and earlier**; 2.17+ introduced
LoggedModels API changes the `azureml-mlflow` plugin doesn't implement, and
that's what 404s. Two things were needed together, not just the version
pin: `mlflow==2.16.2` and `azureml-mlflow==1.60.0.post1` pinned explicitly
(both — leaving one unpinned risks pip resolving an incompatible pairing),
and the environment's version bumped and re-registered
(`az ml environment create`) so the job actually ran against a rebuilt
image rather than a stale cached one with the old pin still in effect —
which is the more likely reason the first, correct-looking pin "didn't
work": the environment was never actually rebuilt.

### 3. pyarrow had no prebuilt wheel for Python 3.12

Also inside the training environment build: `pyarrow<14` (an indirect
constraint from another pinned package at the time) has no prebuilt wheel
for `cp312`, so pip fell back to building it from source inside the image
build — which then failed outright (pyarrow's C++ build has its own
toolchain requirements not present in the base image). Fixed by pinning
`python=3.11` in `ml/azure/conda.yml` instead of chasing the pyarrow
constraint itself — 3.11 has prebuilt wheels for every package in the
environment, so the whole problem disappears rather than needing a
version-matrix fix.

### Also worth a line each

- **Env var substitution silently not happening**: `environment_variables:`
  in `job.yml` does not reliably get `${{inputs.x}}` template-substituted;
  moving the substitution into the `command:` string itself
  (`export VAR=${{inputs.x}} && python -m ml.model`) fixed it.
- **`SubscriptionNotRegistered` on the first online-endpoint create**: another
  generic top-level error; `--debug` plus `az provider list` showed
  `Microsoft.Network` and `Microsoft.PolicyInsights` were both unregistered
  on the subscription — registering both (and cleaning up the endpoint
  object Azure ML had already half-created via a stale async operation) fixed it.
- **Double-JSON-encoded response body**: the online endpoint's response to a
  scoring request is a JSON string *containing* JSON — `score.py`'s `run()`
  returns `json.dumps(...)`, and Azure ML's inference server serializes
  whatever `run()` returns, so a string return value gets encoded twice.
  `agent/risk_client.py` accounts for this explicitly rather than assuming
  the obvious shape.