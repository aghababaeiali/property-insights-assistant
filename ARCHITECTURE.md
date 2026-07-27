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
| Deployment | manual `docker build` + `az containerapp update` | GitHub Actions: a CI workflow (test) gates a separate Deploy workflow (build, push, `az containerapp update`), authenticated via OIDC federation — no stored cloud secret |
| Observability | print statements / local-only logs | Application Insights — request/dependency tracing via `azure-monitor-opentelemetry`, covering Postgres, Azure AI Search, Azure OpenAI, and the Azure ML risk endpoint |
| Infra-as-code | none — every resource above created by hand via `az` | Partial Terraform (`terraform/*.tf`): resource group, Postgres, ACR, Container Apps Environment+App, Azure OpenAI + its two model deployments. Azure AI Search and Azure ML are deliberately not covered — see the Terraform section below |

Resource group `property-insights-rg`, workspace `property-insights-ml`,
container registry `propertyinsightsacr`, container app
`property-insights-app`, compute cluster `cpu-cluster`, Application
Insights resource `property-insights-appinsights`, Azure AD app
registration `github-actions-property-insights` (GitHub Actions OIDC).

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

### 4. CI/CD: OIDC federation, and two federated-credential bugs

The Deploy workflow (`.github/workflows/deploy.yml`) authenticates to Azure
via `azure/login@v2` using OIDC — a GitHub Actions run exchanges its own
OIDC token for a short-lived Azure access token, scoped to a specific Azure
AD app registration (`github-actions-property-insights`) and a specific
federated-credential subject. No `AZURE_CLIENT_SECRET` (or any long-lived
credential) exists anywhere: `AZURE_CLIENT_ID`/`AZURE_TENANT_ID`/
`AZURE_SUBSCRIPTION_ID` in GitHub's repo secrets are all non-secret
identifiers, not credentials — the actual proof of identity is the
short-lived GitHub-issued OIDC token itself, which nothing outside GitHub
can forge and which nothing needs to be rotated or leaked to be safe.
Standard "workload identity federation" over the older
client-secret-in-a-GitHub-secret approach: no expiring secret to rotate, no
credential sitting in GitHub's secret store for someone to exfiltrate.

Two real bugs surfaced getting this working:

- **AADSTS700213 — the federated credential subject didn't match the token
  GitHub actually sent.** The obvious subject format,
  `repo:aghababaeiali/property-insights-assistant:ref:refs/heads/main`,
  authenticated with `AADSTS700213: No matching federated identity record
  found for presented assertion`. GitHub changed the OIDC token's subject
  claim to an "immutable" format that embeds the numeric owner and
  repository IDs alongside the names — `repo:OWNER@OWNER_ID/REPO@REPO_ID:
  ref:refs/heads/main` — specifically so a repo rename or ownership
  transfer can't silently let a different repo assume an old trust
  relationship. The federated credential actually in place today has
  subject
  `repo:aghababaeiali@195648191/property-insights-assistant@1307004771:ref:refs/heads/main`
  — the numeric IDs aren't guessable ahead of time; they have to be read
  back off the token (or off `https://api.github.com/repos/OWNER/REPO`,
  which reports both IDs) after the first `AADSTS700213` failure, not
  assumed from the plain `owner/repo` form the older docs and examples
  still show.
- **`az ad app federated-credential update` silently not applying.** After
  the first credential was created with the wrong (pre-immutable-format)
  subject, updating it in place via `az ad app federated-credential update`
  returned success with no error, but the deploy kept failing with the
  exact same `AADSTS700213`. The update call reported success without the
  new subject actually taking effect. Deleting the credential
  (`az ad app federated-credential delete`) and creating a fresh one
  (`az ad app federated-credential create`) with the correct subject
  fixed it immediately — the general lesson: for federated credentials
  specifically, treat `update` as unreliable and delete+recreate instead
  of trusting an in-place edit actually landed.

### 5. Application Insights: the httpx auto-instrumentation gap for the openai SDK

Once `configure_azure_monitor()` was wired into `agent/api.py`, Postgres,
Azure AI Search, and the Azure ML risk endpoint all showed up on the
Application Map immediately with real call counts and latencies. Azure
OpenAI calls did not show up at all — not merged into another node, not
delayed, genuinely absent, confirmed by querying the raw `AppDependencies`
table directly rather than trusting the visual map (which can lag or
merge nodes): sending a fresh `rag`-intent request and querying the exact
timestamp window showed Postgres and Azure AI Search dependency rows, and
zero rows for `openai.azure.com`, anywhere.

The cause: `azure-monitor-opentelemetry`'s auto-instrumentation only covers
a fixed allowlist (`azure.monitor.opentelemetry._constants
._FULLY_SUPPORTED_INSTRUMENTED_LIBRARIES`) —
`django`/`fastapi`/`flask`/`psycopg2`/`requests`/`urllib`/`urllib3`. Azure
AI Search (via `azure-core`'s `RequestsTransport`) and the risk endpoint
(`agent/risk_client.py` calls `requests` directly) both happen to ride on
`requests`, so they were auto-instrumented for free. The `openai` Python
SDK builds its transport on **httpx** exclusively (confirmed by reading
`openai/_base_client.py` — `import httpx`, `SyncHttpxClientWrapper`), and
httpx isn't on that allowlist in any azure-monitor-opentelemetry version.
Fixed by adding `opentelemetry-instrumentation-httpx` as a dependency and
calling `HTTPXClientInstrumentor().instrument()` explicitly in
`agent/api.py`, right alongside `configure_azure_monitor()`. Confirmed
fixed the same way the gap was confirmed — the raw `AppDependencies` table,
not the visual map — showing `property-insights-openai.openai.azure.com`
as its own HTTP dependency target with real call/latency data after
redeploying.

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

## Terraform

`terraform/*.tf` is a hand-written, partial Terraform configuration —
written after the fact, describing the resources as they actually stand,
not used to originally provision this environment (everything here was
built via `az` CLI, incrementally, across the sections above).

**What it covers**: resource group, Azure Database for PostgreSQL (flexible
server + firewall rule + database), Azure Container Registry, the
Container Apps Environment + Container App (with its full set of
env vars/secrets — `LLM_PROVIDER`, `DATABASE_URL`, all `AZURE_OPENAI_*`,
`AZURE_SEARCH_*`, and `AZURE_ML_RISK_ENDPOINT_*`), the Log Analytics
workspace backing the Container Apps Environment, and Azure OpenAI (the
cognitive account plus both model deployments — `gpt-5-mini` and
`text-embedding-3-small`).

**What it deliberately does not cover**: Azure AI Search and Azure ML
(the training job, compute cluster, MLflow-registered model, and the
managed online endpoint) remain entirely `az`-CLI/job-yaml-provisioned,
same as `ml/azure/` already describes. This is a scope call, not an
oversight: Azure ML in particular is not "one resource" the way Postgres
or ACR are — a real Terraform representation of it means modeling a
training job, an environment, a compute cluster, a model registration, an
online endpoint, and a deployment, several of which (the job run itself,
the model registration it produces) are one-shot/imperative actions that
don't map cleanly onto Terraform's declarative "describe the end state"
model. For a portfolio piece, hand-rolling that mapping is real effort
with no real payoff — it would demonstrate fighting the tool more than
using it. Azure AI Search was left out for a smaller version of the same
reason: the index population step (`scripts/build_search_index.py`) is
itself an imperative data-loading action, not infrastructure state.

`AZURE_OPENAI_*` env vars on the Container App are wired directly off the
`azurerm_cognitive_account`/`azurerm_cognitive_deployment` resources this
same config creates (`container_apps.tf`), rather than duplicated into
their own input variables — Terraform already owns those values here, so a
second, independently maintained copy of them would just be one more place
for the two to drift. `AZURE_SEARCH_*` and `AZURE_ML_RISK_ENDPOINT_*` don't
have that option (no local resource to read them from, since those
resources live outside this config) and are plain input variables instead,
the same pattern already established by `database_url`.

**Known limitation, named rather than glossed over: state is local, not a
remote backend.** There's no `backend` block in `main.tf` — running
`terraform apply` from a second machine (or losing the local
`terraform.tfstate`) has no shared source of truth to reconcile against,
and two people applying concurrently would race. Acceptable for a
single-operator portfolio project; not acceptable as-is for anything with
more than one person able to run `apply` — that needs an
`azurerm`/blob-backed remote backend with state locking before real
collaboration on this config would be safe.

**Second known limitation: Terraform and the Deploy workflow both claim
ownership of the container image tag, and don't agree.**
`container_apps.tf` sets the container's `image` to a `:latest` tag, but
`.github/workflows/deploy.yml` actually manages the running image day to
day via `az containerapp update --image ...:${{ github.sha }}` — a
per-commit tag, not `:latest`. Running `terraform apply` after any CI/CD
deploy would see that as drift and try to revert the running revision back
to whatever `:latest` last pointed at. Not fixed here — reconciling it
means either having Terraform not manage the `image` field at all
(`lifecycle { ignore_changes = [template[0].container[0].image] }`) or
having the Deploy workflow update Terraform state instead of calling
`az containerapp update` directly, and this config doesn't yet do either.
Named here so it doesn't surprise whoever runs `apply` next.

Verified with `terraform fmt` and `terraform validate` (against the
`azurerm ~> 3.100` provider actually pinned in `main.tf`) — no `apply` was
run against the live resources, since they already exist and were not
created by this config. Two real schema bugs in the original hand-written
files surfaced this way and were fixed: `azurerm_cognitive_deployment`'s
capacity block is called `scale` in the azurerm 3.x provider line (not
`sku` — that rename landed in the azurerm 4.x major version), and
`azurerm_postgresql_flexible_server`'s `high_availability.mode` only
accepts `ZoneRedundant`/`SameZone` in this provider version — there's no
`"Disabled"` value; omitting the block entirely is how "no HA" is
expressed.

## Measured performance

Real numbers, not estimates — pulled directly from this deployment on
2026-07-27, sourced and captioned individually since each came from a
different tool.

**Dependency latency** (Application Insights, queried via
`az monitor log-analytics query` against the `AppDependencies` table in
the backing Log Analytics workspace — the classic
`az monitor app-insights query` path against the same resource was
returning empty results by the time these final numbers were pulled, so
the underlying workspace tables were queried directly instead; ground
truth, not the visual Application Map). 163 total dependency records
across the full day's testing, `2026-07-27T01:04Z`–`2026-07-27T14:25Z`:

| Dependency | Calls | Avg | p50 | p95 | Min | Max |
|---|---|---|---|---|---|---|
| Azure ML risk endpoint | 47 | 29.2ms | 28ms | 40ms | 23ms | 44ms |
| Postgres | 44 | 3.6ms | 2ms | 10ms | 0ms | 52ms |
| Azure OpenAI (chat + embeddings) | 25 | 1386ms | 1280ms | 3087ms | 301ms | 4674ms |
| Azure AI Search | 20 | 90.7ms | 79ms | 201ms | 34ms | 219ms |

Azure OpenAI dominates end-to-end latency by a wide margin, as expected —
it's the one dependency doing generative inference rather than a lookup;
the others are all sub-100ms at p95. One related, honest gap found while
pulling these numbers: the `AppRequests` table (inbound `/ask` request
spans) was empty for the entire window — only outbound dependency spans
are being captured. Not chased further here (this is the final pass, no
new instrumentation work), but it means the request-level latency numbers
below came from direct `curl` timing, not from Application Insights.

**Cold start vs. warm request** (direct `curl` timing against the live
Container Apps URL, not Application Insights): the app's
`scale.minReplicas` is unset (defaults to 0) with a 300s scale-to-zero
cooldown, and it had received no traffic for roughly 13 hours before this
measurement — a real cold start, not a simulated one.

| | Latency |
|---|---|
| Cold (first request after ~13h idle) | 40.08s |
| Warm (immediately after, 3 consecutive requests) | 3.01s / 2.83s / 3.27s |

**Cost accrued so far** (Azure Cost Management, `Microsoft.CostManagement/query`
REST API, month-to-date, scoped to `property-insights-rg`): **≈2.19 EUR
total**. Breakdown by service:

| Service | Cost (EUR) |
|---|---|
| Virtual Machines (Azure ML compute cluster) | 1.535 |
| Container Registry | 0.353 |
| Storage | 0.215 |
| Virtual Network | 0.073 |
| Foundry Models (Azure OpenAI usage) | 0.013 |
| Key Vault | 0.0004 |
| Bandwidth | ~0 |
| Azure Cognitive Search, Postgres, Load Balancer, Log Analytics | 0.00 |

The Azure ML compute cluster (`cpu-cluster`) is the largest line item by
far — training-job compute, not anything request-driven. Azure AI Search,
Postgres, Load Balancer, and Log Analytics show exactly 0.00, consistent
with this being low-volume, largely free-tier/free-grant-eligible usage
rather than a rounding artifact of the query.