"""
Retrieval over listing free-text (description + host_notes).

Deliberately simple keyword scoring so the repo runs offline with no embedding
API. The 'why is X risky' signal lives in host_notes, not in any column.
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


def _tokens(text: str):
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOP]


_DOCS = []
for listing in _LISTINGS:
    amenities = " ".join(listing["amenities"])
    blob = (f"{listing['title']} {listing['city']} {listing['neighborhood']} "
            f"{listing['description']} {listing['host_notes']} {amenities}")
    _DOCS.append({"listing_id": listing["listing_id"], "city": listing["city"],
                  "text": blob, "counts": Counter(_tokens(blob))})


def retrieve(query: str, k: int = 4):
    q = _tokens(query)
    scored = []
    for d in _DOCS:
        score = sum(d["counts"].get(t, 0) for t in q)
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"listing_id": d["listing_id"], "city": d["city"], "text": d["text"]}
            for _, d in scored[:k]]
