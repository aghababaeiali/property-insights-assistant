"""Tests for the offline LLM provider path (deterministic, no network)."""
from agent import llm


def test_provider_is_offline():
    assert llm.PROVIDER == "offline"


def test_classify_sql():
    assert llm.classify_intent("what is the average price in Lisbon?") == "sql"
    assert llm.classify_intent("how many confirmed bookings do we have?") == "sql"


def test_classify_rag():
    assert llm.classify_intent("why might some listings be problematic?") == "rag"
    assert llm.classify_intent("what makes this listing special?") == "rag"


def test_classify_risk():
    assert llm.classify_intent("which listings are most at risk of cancellation?") == "risk"
    assert llm.classify_intent("what is most likely to cancel?") == "risk"


def test_synthesize_answer_includes_context():
    context = "L0001 (Lisbon): a lovely apartment"
    answer = llm.synthesize_answer("tell me about L0001", context)
    assert context in answer


def test_judge_groundedness_is_noop_offline():
    assert llm.judge_groundedness("q", "context", "answer") is None
