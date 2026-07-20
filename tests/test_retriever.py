"""Tests for agent/retriever.py's keyword search + exact-ID lookup."""
from agent import retriever


def test_finds_listing_by_id_even_with_no_keyword_overlap():
    """L0001's amenities don't include "wifi" — a pure keyword scorer would
    never surface it for a wifi question (see docs/design.md). The exact-ID
    lookup path must return it anyway, since the question names it directly.
    """
    docs = retriever.retrieve("what is the wifi password for L0001?", k=4)
    assert "L0001" in [d["listing_id"] for d in docs]


def test_id_lookup_is_case_insensitive():
    docs = retriever.retrieve("tell me about l0001", k=4)
    assert docs[0]["listing_id"] == "L0001"


def test_id_lookup_finds_multiple_ids():
    docs = retriever.retrieve("compare L0001 and L0011", k=4)
    ids = {d["listing_id"] for d in docs}
    assert {"L0001", "L0011"} <= ids


def test_unknown_id_in_query_does_not_crash():
    docs = retriever.retrieve("what about L9999?", k=4)
    assert all(d["listing_id"] != "L9999" for d in docs)


def test_keyword_search_still_works_without_an_id():
    docs = retriever.retrieve("balcony", k=5)
    assert len(docs) > 0
    assert all("listing_id" in d for d in docs)


def test_respects_k_limit_even_with_exact_matches():
    docs = retriever.retrieve("compare L0001 and L0011 and L0029", k=2)
    assert len(docs) == 2
