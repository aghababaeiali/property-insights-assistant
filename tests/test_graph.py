"""Tests for the LangGraph pipeline: routing, validation, and full runs."""
from agent.graph import AGENT, State, resolve_city, route, validate_node


def test_route_maps_all_known_intents():
    assert route({"intent": "sql"}) == "sql"
    assert route({"intent": "rag"}) == "rag"
    assert route({"intent": "risk"}) == "risk"


def test_route_falls_back_to_sql_for_unknown_intent():
    assert route({"intent": "bogus"}) == "sql"
    assert route({}) == "sql"


def test_validate_node_passes_grounded_answer():
    state: State = {
        "intent": "rag",
        "rag_context": "L0001 (Lisbon): a lovely apartment",
        "answer": "L0001 is a lovely apartment.",
    }
    out = validate_node(state)
    assert out["validation_issues"] == []
    assert "[UNVERIFIED]" not in out["answer"]


def test_validate_node_catches_ungrounded_citation():
    state: State = {
        "intent": "rag",
        "rag_context": "L0001 (Lisbon): a lovely apartment",
        "answer": "L0001 is fine, but L9999 has serious issues.",
    }
    out = validate_node(state)
    assert out["validation_issues"]
    assert "[UNVERIFIED]" in out["answer"]


def test_validate_node_flags_empty_answer():
    out = validate_node({"answer": ""})
    assert "empty answer" in out["validation_issues"]


def test_agent_sql_end_to_end():
    out = AGENT.invoke({"question": "how many confirmed bookings do we have?"})
    assert out["intent"] == "sql"
    assert out["answer"].startswith("[SQL]")
    assert out["validation_issues"] == []


def test_agent_rag_end_to_end():
    out = AGENT.invoke({"question": "why might some Lisbon listings be problematic?"})
    assert out["intent"] == "rag"
    assert out["validation_issues"] == []


def test_agent_risk_end_to_end():
    out = AGENT.invoke({"question": "which Lisbon listings are most at risk of cancellation?"})
    assert out["intent"] == "risk"
    assert out["answer"].startswith("[RISK]")


def test_resolve_city_known_city():
    city, error = resolve_city("what is the average price in Lisbon?")
    assert city == "Lisbon"
    assert error is None


def test_resolve_city_no_city_intended():
    city, error = resolve_city("how many confirmed bookings do we have?")
    assert city is None
    assert error is None


def test_resolve_city_unrecognized_city_returns_error_not_silent_aggregate():
    """Regression test: a city we have no data for must not silently fall
    through to the all-cities aggregate as if it matched (see docs — this
    was a real bug: "cancellation rate in Tehran?" answered with the
    portfolio-wide rate as if it were Tehran-specific).
    """
    city, error = resolve_city("what is the cancellation rate in Tehran?")
    assert city is None
    assert error is not None
    assert "Tehran" in error


def test_agent_sql_rejects_unrecognized_city():
    out = AGENT.invoke({"question": "what is the cancellation rate in Tehran?"})
    assert out["intent"] == "sql"
    assert "Tehran" in out["answer"]
    assert "%" not in out["answer"]  # must not have computed a rate at all


def test_agent_risk_rejects_unrecognized_city():
    out = AGENT.invoke({"question": "which listings in Tehran are most at risk of cancellation?"})
    assert out["intent"] == "risk"
    assert "Tehran" in out["answer"]
    assert "Highest-risk listings" not in out["answer"]
