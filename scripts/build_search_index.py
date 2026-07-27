"""One-time (or re-run-on-demand) script: builds/refreshes the Azure AI
Search index from data/listings.json. Not part of the live app — run this
manually whenever listings.json changes.

Usage: uv run python -m scripts.build_search_index
"""
import json
import os
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "listings")
EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
VECTOR_DIMENSIONS = 1536  # text-embedding-3-small's output size

openai_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)


def embed(text: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=text)
    return resp.data[0].embedding


def build_blob(listing: dict) -> str:
    amenities = " ".join(listing["amenities"])
    return (f"{listing['title']} {listing['city']} {listing['neighborhood']} "
            f"{listing['description']} {listing['host_notes']} {amenities}")


def create_index(index_client: SearchIndexClient):
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
        profiles=[VectorSearchProfile(
            name="vector-profile", algorithm_configuration_name="hnsw-config")],
    )
    fields = [
        SimpleField(name="listing_id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="city", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchField(
            name="text_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True, vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]
    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"Index '{INDEX_NAME}' created/updated.")


def upload_documents(search_client: SearchClient):
    with open(os.path.join(DATA, "listings.json")) as f:
        listings = json.load(f)

    docs = []
    for listing in listings:
        blob = build_blob(listing)
        docs.append({
            "listing_id": listing["listing_id"],
            "city": listing["city"],
            "text": blob,
            "text_vector": embed(blob),
        })
        print(f"Embedded {listing['listing_id']}")

    result = search_client.upload_documents(documents=docs)
    failed = [r for r in result if not r.succeeded]
    print(f"Uploaded {len(docs) - len(failed)}/{len(docs)} documents.")
    if failed:
        print("Failures:", failed)


if __name__ == "__main__":
    credential = AzureKeyCredential(SEARCH_KEY)
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    create_index(index_client)

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)
    upload_documents(search_client)