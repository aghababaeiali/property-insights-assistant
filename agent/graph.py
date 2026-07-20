"""
Property Insights Assistant — LangGraph agent.

Graph (current):

    START -> router -> { sql_node | rag_node | risk_node }
           -> validate_node -> judge_node -> log_node -> END

The agent answers descriptive questions (RAG), simple aggregate questions
(SQL), and cancellation-risk questions (risk). Every branch passes through:
  - validate_node: deterministic check that the answer only cites
    listings that were actually retrieved/scored.
  - judge_node: sampled LLM-as-judge groundedness check (rag/risk only,
    real providers only) — see JUDGE_SAMPLE_RATE.
  - log_node: writes every request to the `agent_logs` table, and
    flagged ones to `review_queue`, for offline review
    (see agent/review.py and agent/eval.py).

Run:  python -m agent.run "what is the average price in Lisbon?"
"""
import os
import random
import re
import sys
import time
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent import db, llm, retriever
from ml.model import CITIES, engineer_features, predict_cancellation_risk

JUDGE_SAMPLE_RATE = float(os.environ.get("JUDGE_SAMPLE_RATE", "0.3"))


class State(TypedDict, total=False):
    question: str
    intent: str
    sql_result: str
    rag_context: str
    risk_result: str
    answer: str
    validation_issues: list
    judge_result: dict
    _started_at: float


# --- nodes ------------------------------------------------------------------
def router(state: State) -> State:
    state["_started_at"] = time.time()
    state["intent"] = llm.classify_intent(state["question"])
    return state


def sql_node(state: State) -> State:
    """Answer simple aggregate questions from the DB.

    NOTE: this node currently classifies, queries, and formats the final answer
    all in one place.
    """
    q = state["question"].lower()
    city = next((c for c in CITIES if c.lower() in q), None)
    where = f"WHERE l.city = '{city}'" if city else ""

    if "cancel" in q:
        sql = f"""SELECT round(100.0*sum((status='cancelled')::int)/count(*),1)
                  FROM bookings b JOIN listings l USING(listing_id) {where}"""
        rows = db.query(sql)
        val = rows[0][0]
        state["sql_result"] = f"cancellation rate: {val}%"
    elif "price" in q or "revenue" in q:
        sql = f"""SELECT round(avg(total_price),0)
                  FROM bookings b JOIN listings l USING(listing_id) {where}"""
        rows = db.query(sql)
        state["sql_result"] = f"average booking value: {rows[0][0]}"
    else:
        sql = f"""SELECT count(*)
                  FROM bookings b JOIN listings l USING(listing_id)
                  {where or 'WHERE 1=1'} AND status='confirmed'"""
        rows = db.query(sql)
        state["sql_result"] = f"confirmed bookings: {rows[0][0]}"

    # formats the final answer directly
    state["answer"] = f"[SQL] {state['sql_result']}"
    return state


def rag_node(state: State) -> State:
    docs = retriever.retrieve(state["question"], k=4)
    context = "\n\n".join(f"{d['listing_id']} ({d['city']}): {d['text']}" for d in docs)
    state["rag_context"] = context
    state["answer"] = llm.synthesize_answer(state["question"], context)
    return state


def risk_node(state: State) -> State:
    """Rank listings by predicted cancellation risk, with a retrieved 'why'.

    Composes the DB (candidate bookings), the ML model (risk score), and
    retrieval (the qualitative reason).
    """
    q = state["question"].lower()
    city = next((c for c in CITIES if c.lower() in q), None)
    where = f"AND l.city = '{city}'" if city else ""
    rows = db.query(f"""
        SELECT b.listing_id, b.lead_time_days, b.nights, b.num_guests,
               b.total_price, b.is_repeat_guest, b.deposit_taken,
               b.channel, b.check_in_date,
               l.review_score, l.cancellation_policy, l.review_count
        FROM bookings b JOIN listings l USING(listing_id)
        WHERE b.status='confirmed' {where}
    """)
    cols = ["listing_id", "lead_time_days", "nights", "num_guests", "total_price",
            "is_repeat_guest", "deposit_taken", "channel", "check_in_date",
            "review_score", "cancellation_policy", "review_count"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        state["risk_result"] = "no upcoming bookings found"
        state["answer"] = "No confirmed upcoming bookings to score."
        return state
    df = engineer_features(df)
    df["risk"] = df.apply(lambda r: predict_cancellation_risk(r.to_dict()), axis=1)
    top = (df.groupby("listing_id")["risk"].mean()
             .sort_values(ascending=False).head(5))
    ids = "','".join(top.index)
    notes = dict(db.query(
        f"SELECT listing_id, host_notes FROM listings WHERE listing_id IN ('{ids}')"))
    lines = [f"{lid}: risk={risk:.0%} — {notes.get(lid, '(no notes)')}"
             for lid, risk in top.items()]
    state["risk_result"] = "\n".join(lines)
    state["answer"] = "[RISK] Highest-risk listings:\n" + state["risk_result"]
    return state


def validate_node(state: State) -> State:
    """Sanity-check an answer before it goes out.

    Deterministic, no extra LLM call: catches the failure mode that matters
    most for a citation-heavy agent like this one — the answer naming a
    listing_id that was never actually retrieved/scored (i.e. the LLM
    inventing or misattributing a listing rather than grounding in context).
    """
    answer = state.get("answer", "")
    issues = []

    if not answer.strip():
        issues.append("empty answer")

    if state.get("intent") == "rag":
        context_ids = set(re.findall(r"L\d{4}", state.get("rag_context", "")))
        answer_ids = set(re.findall(r"L\d{4}", answer))
        ungrounded = answer_ids - context_ids
        if ungrounded:
            issues.append(f"cites listing(s) not in retrieved context: {sorted(ungrounded)}")

    if state.get("intent") == "risk":
        scored_ids = set(re.findall(r"L\d{4}", state.get("risk_result", "")))
        answer_ids = set(re.findall(r"L\d{4}", answer))
        ungrounded = answer_ids - scored_ids
        if ungrounded:
            issues.append(f"cites listing(s) not in scored results: {sorted(ungrounded)}")

    state["validation_issues"] = issues
    if issues:
        state["answer"] = f"{answer}\n\n[UNVERIFIED] {'; '.join(issues)}"
    return state


def judge_node(state: State) -> State:
    """Sampled LLM-as-judge groundedness check.

    Only runs for LLM-synthesized answers (rag/risk — sql answers are
    deterministic SQL results, nothing to judge), only when a real provider
    is configured (LLM_PROVIDER=offline, used by tests, has no model to judge
    with), and only for a sample of traffic (JUDGE_SAMPLE_RATE) — mirrors how
    this would run in production: too slow/expensive to run on every request.
    """
    state["judge_result"] = None
    if llm.PROVIDER == "offline" or state.get("intent") not in ("rag", "risk"):
        return state
    if random.random() >= JUDGE_SAMPLE_RATE:
        return state

    context = state.get("rag_context") if state["intent"] == "rag" else state.get("risk_result", "")
    try:
        state["judge_result"] = llm.judge_groundedness(
            state["question"], context, state.get("answer", ""))
    except Exception as e:
        state["judge_result"] = {"grounded": None, "reason": f"judge call failed: {e}"}
    return state


def log_node(state: State) -> State:
    """Persist every request for offline review (agent/review.py, agent/eval.py)."""
    started = state.get("_started_at")
    latency_ms = (time.time() - started) * 1000 if started else None
    judge = state.get("judge_result")
    issues = state.get("validation_issues") or []
    needs_review = bool(issues) or bool(judge and judge.get("grounded") is False)
    try:
        db.log_agent_request(
            question=state.get("question", ""),
            intent=state.get("intent", ""),
            answer=state.get("answer", ""),
            validation_issues=issues,
            latency_ms=latency_ms,
            provider=llm.PROVIDER,
            judge_grounded=judge.get("grounded") if judge else None,
            judge_reason=judge.get("reason") if judge else None,
            needs_review=needs_review,
        )
    except Exception as e:
        # logging must never break the answer path
        print(f"[log_node] failed to write agent_logs: {e}", file=sys.stderr)
    return state


# --- graph ------------------------------------------------------------------
def route(state: State) -> str:
    intent = state.get("intent", "sql")
    return intent if intent in ("sql", "rag", "risk") else "sql"


def build_agent():
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("sql_node", sql_node)
    g.add_node("rag_node", rag_node)
    g.add_node("risk_node", risk_node)
    g.add_node("validate_node", validate_node)
    g.add_node("judge_node", judge_node)
    g.add_node("log_node", log_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route,
                            {"sql": "sql_node", "rag": "rag_node", "risk": "risk_node"})
    g.add_edge("sql_node", "validate_node")
    g.add_edge("rag_node", "validate_node")
    g.add_edge("risk_node", "validate_node")
    g.add_edge("validate_node", "judge_node")
    g.add_edge("judge_node", "log_node")
    g.add_edge("log_node", END)
    return g.compile()


AGENT = build_agent()
