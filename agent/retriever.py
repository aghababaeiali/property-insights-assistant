"""
Retrieval over listing free-text (description + host_notes).

Two backends, selected by the same LLM_PROVIDER flag agent.llm uses — the
Azure additions are opt-in, not a hard dependency for local dev:

- "groq" / "offline" (default): local, in-memory keyword scoring over
  data/listings.json. Zero network calls, zero Azure credentials required —
  this is what keeps `docker compose up` + no Azure account working end to
  end, and what the offline test suite / CI exercise.
- "azure": Azure AI Search hybrid (keyword + vector) search against a
  prebuilt index (see scripts/build_search_index.py), with embeddings from
  Azure OpenAI. Requires AZURE_SEARCH_ENDPOINT/AZURE_SEARCH_API_KEY/
  AZURE_OPENAI_* to be set — only read (and the Azure SDK clients only
  built) when actually selected, so importing this module never requires
  Azure credentials.

Both backends guarantee the same thing: a question that names a listing_id
explicitly (e.g. "what is the wifi password for L0001?") always returns that
listing via direct lookup, rather than depending on it also winning the
relevance ranking — keyword scoring alone can't do that (an ID's digits
don't survive tokenization, and it's never part of the indexed blob text to
begin with), and there's no reason vector similarity would reliably surface
it either.
"""
import json
import os
import re
from collections import Counter

PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # "groq" | "azure" | "offline"

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

_ID_PATTERN = re.compile(r"l\d{4}", re.IGNORECASE)
_STOP = set("the a an of in on at to for and or is are with up next month why which".split())


def _tokens(text: str):
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOP]


# --- local (groq/offline) — keyword scoring, zero Azure dependency ---------
_LOCAL_DOCS = None
_LOCAL_DOCS_BY_ID = None


def _load_local_docs():
    global _LOCAL_DOCS, _LOCAL_DOCS_BY_ID
    if _LOCAL_DOCS is not None:
        return
    with open(os.path.join(DATA, "listings.json")) as f:
        listings = json.load(f)
    docs = []
    for listing in listings:
        amenities = " ".join(listing["amenities"])
        blob = (f"{listing['title']} {listing['city']} {listing['neighborhood']} "
                f"{listing['description']} {listing['host_notes']} {amenities}")
        docs.append({"listing_id": listing["listing_id"], "city": listing["city"],
                      "text": blob, "counts": Counter(_tokens(blob))})
    _LOCAL_DOCS = docs
    _LOCAL_DOCS_BY_ID = {d["listing_id"]: d for d in docs}


def _local_retrieve(query: str, k: int) -> list[dict]:
    _load_local_docs()
    requested_ids = dict.fromkeys(m.upper() for m in _ID_PATTERN.findall(query))
    exact = [_LOCAL_DOCS_BY_ID[lid] for lid in requested_ids if lid in _LOCAL_DOCS_BY_ID]
    exact_ids = {d["listing_id"] for d in exact}

    q = _tokens(query)
    scored = []
    for d in _LOCAL_DOCS:
        if d["listing_id"] in exact_ids:
            continue
        score = sum(d["counts"].get(t, 0) for t in q)
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    remaining = max(0, k - len(exact))
    results = exact + [d for _, d in scored[:remaining]]
    return [{"listing_id": d["listing_id"], "city": d["city"], "text": d["text"]}
            for d in results[:k]]


# --- azure — hybrid (keyword + vector) search ------------------------------
_search_client = None
_openai_client = None


def _azure_clients():
    """Built lazily on first use, not at import time — so importing this
    module (which agent.graph always does) never requires Azure credentials
    unless LLM_PROVIDER=azure actually selects this path.
    """
    global _search_client, _openai_client
    if _search_client is None:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from openai import AzureOpenAI

        _search_client = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=os.environ.get("AZURE_SEARCH_INDEX", "listings"),
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
        )
        _openai_client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    return _search_client, _openai_client


def _embed(text: str) -> list[float]:
    _, openai_client = _azure_clients()
    resp = openai_client.embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"], input=text)
    return resp.data[0].embedding


def _get_by_id(search_client, listing_id: str) -> dict | None:
    from azure.core.exceptions import ResourceNotFoundError

    try:
        return search_client.get_document(key=listing_id)
    except ResourceNotFoundError:
        return None


def _azure_retrieve(query: str, k: int) -> list[dict]:
    from azure.search.documents.models import VectorizedQuery

    search_client, _ = _azure_clients()

    # exact ID lookup first — same guarantee as the local path, now via a
    # direct document fetch instead of a dict lookup keyed on regex-parsed IDs.
    requested_ids = dict.fromkeys(m.upper() for m in _ID_PATTERN.findall(query))
    exact = [d for lid in requested_ids if (d := _get_by_id(search_client, lid)) is not None]
    exact_ids = {d["listing_id"] for d in exact}

    remaining = max(0, k - len(exact))
    results = list(exact)
    if remaining:
        vector_query = VectorizedQuery(
            vector=_embed(query), k_nearest_neighbors=remaining, fields="text_vector")
        hits = search_client.search(
            search_text=query, vector_queries=[vector_query], top=remaining + len(exact_ids))
        for hit in hits:
            if hit["listing_id"] in exact_ids:
                continue
            results.append(hit)
            if len(results) >= k:
                break

    return [{"listing_id": d["listing_id"], "city": d["city"], "text": d["text"]}
            for d in results[:k]]


def retrieve(query: str, k: int = 4) -> list[dict]:
    if PROVIDER == "azure":
        return _azure_retrieve(query, k)
    return _local_retrieve(query, k)
