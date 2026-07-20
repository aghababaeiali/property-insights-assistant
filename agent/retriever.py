"""
Retrieval over listing free-text (description + host_notes).

Deliberately simple keyword scoring so the repo runs offline with no embedding
API. The 'why is X risky' signal lives in host_notes, not in any column.

Keyword scoring alone can never find a listing by its ID: `_tokens()`'s regex
is letters-only, so "L0001" tokenizes down to just the single letter "l" (the
digits vanish), and `listing_id` was never part of the indexed blob text to
begin with. So a question like "what is the wifi password for L0001?" scores
purely on "wifi" — matching whichever *other* listings happen to mention wifi
in their amenities, with zero relevance to L0001 or even its city. retrieve()
special-cases this: if the query names a listing_id explicitly, that listing
is included via direct lookup, not keyword luck.
"""
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

with open(os.path.join(DATA, "listings.json")) as f:
    _LISTINGS = json.load(f)

_STOP = set("the a an of in on at to for and or is are with up next month why which".split())
_ID_PATTERN = re.compile(r"l\d{4}", re.IGNORECASE)


def _tokens(text: str):
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOP]


_DOCS = []
for listing in _LISTINGS:
    amenities = " ".join(listing["amenities"])
    blob = (f"{listing['title']} {listing['city']} {listing['neighborhood']} "
            f"{listing['description']} {listing['host_notes']} {amenities}")
    _DOCS.append({"listing_id": listing["listing_id"], "city": listing["city"],
                  "text": blob, "counts": Counter(_tokens(blob))})
_DOCS_BY_ID = {d["listing_id"]: d for d in _DOCS}


def retrieve(query: str, k: int = 4):
    # exact ID lookup first: guarantees a listing the question explicitly
    # names is actually returned, rather than depending on it also winning
    # the keyword-overlap contest (which it structurally can't, since its
    # own ID is never part of what gets keyword-matched).
    requested_ids = dict.fromkeys(m.upper() for m in _ID_PATTERN.findall(query))
    exact = [_DOCS_BY_ID[lid] for lid in requested_ids if lid in _DOCS_BY_ID]
    exact_ids = {d["listing_id"] for d in exact}

    q = _tokens(query)
    scored = []
    for d in _DOCS:
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
