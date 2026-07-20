"""
LLM layer — Groq by default. Set LLM_PROVIDER=offline for a deterministic,
no-network/no-key path used by tests and CI (see tests/conftest.py) — never
the runtime default, so a misconfigured deployment fails loudly (missing
GROQ_API_KEY) instead of silently degrading to canned answers.
"""
import json
import os
import re

PROVIDER = os.environ.get("LLM_PROVIDER", "groq")


def classify_intent(question: str) -> str:
    """Return one of: 'sql', 'rag', 'risk'."""
    if PROVIDER == "offline":
        return _offline_classify(question.lower())
    return _groq_classify(question.lower())


def synthesize_answer(question: str, context: str) -> str:
    if PROVIDER == "offline":
        return f"[offline] Based on the data:\n{context}"
    return _groq_synth(question, context)


def judge_groundedness(question: str, context: str, answer: str) -> dict | None:
    """Sampled LLM-as-judge: does the answer only make claims supported by context?

    Returns None in offline mode — no real model to judge with, and sampling
    this check only makes sense once real inference is already being spent
    on synthesis in the first place.
    """
    if PROVIDER == "offline":
        return None
    return _groq_judge(question, context, answer)


def extract_city(question: str) -> str | None:
    """Return the literal city name the question mentions/implies, or None if
    no city is mentioned at all.

    Does NOT check whether it's a city this system actually has data for —
    that's the caller's job (see agent.graph.resolve_city). This function's
    only job is extraction, same separation as classify_intent/synthesize.
    """
    if PROVIDER == "offline":
        return _offline_extract_city(question)
    return _groq_extract_city(question)


# --- offline (tests/CI only — never the runtime default) --------------------
def _offline_classify(q: str) -> str:
    if any(p in q for p in ["at risk", "risk of cancel", "cancellation risk",
                             "most likely to cancel", "likely to cancel"]):
        return "risk"
    if any(p in q for p in ["why", "describe", "tell me about", "what makes",
                             "host notes", "notes", "special"]):
        return "rag"
    return "sql"


def _offline_extract_city(question: str) -> str | None:
    match = re.search(r"\bin ([A-Z][a-zA-Z]+)\b", question)
    return match.group(1) if match else None


# --- groq ---------------------------------------------------------------
_JUDGE_PROMPT = """You are checking an AI assistant's answer for hallucination.

Context:
{context}

Question: {question}

Answer: {answer}

Does the answer make any claim that is NOT supported by the context above?
Reply with strict JSON only, no other text: {{"grounded": true or false, "reason": "<one short sentence>"}}"""

_CLASSIFY_PROMPT = """Classify the question into exactly one category. Reply with exactly one word: sql, rag, or risk.

Categories:
- sql: quantitative questions answerable by aggregating the bookings database (counts, rates, averages, totals) — e.g. "how many confirmed bookings do we have?", "what is the average price in Lisbon?", "what's the cancellation rate for Porto?"
- rag: qualitative/descriptive questions about listings, answered from listing descriptions or host notes — e.g. "why might some listings be problematic?", "what makes this listing special?", "describe the amenities in Rome"
- risk: questions specifically asking to rank or identify listings by cancellation-risk score — e.g. "which listings are most at risk of cancellation?", "what's most likely to cancel?"

If the question is a general "why is X bad/problematic" question without an explicit mention of cancellation risk or likelihood, prefer rag over risk.

Q: {q}"""

_CITY_EXTRACT_PROMPT = """Extract the city name this question mentions or implies, if any.
Reply with EXACTLY the city name, normally capitalized (e.g. "Lisbon"), and nothing else.
If no city is mentioned or implied at all, reply with exactly: NONE

Examples:
Q: what is the average price in Lisbon?
A: Lisbon

Q: how many confirmed bookings do we have?
A: NONE

Q: what is the cancellation rate in Tehran?
A: Tehran

Q: {q}
A:"""


def _parse_judge_output(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"grounded": None, "reason": "judge returned unparseable output"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"grounded": None, "reason": "judge returned invalid JSON"}


def _groq_classify(q: str) -> str:
    from groq import Groq

    client = Groq()
    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile", max_tokens=20,
        messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(q=q)}])
    text = msg.choices[0].message.content
    match = re.search(r"\b(sql|rag|risk)\b", text.lower())
    return match.group(1) if match else "sql"


def _groq_synth(question: str, context: str) -> str:
    from groq import Groq

    client = Groq()
    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile", max_tokens=500,
        messages=[{"role": "user", "content":
            f"Answer using only this context.\nContext:\n{context}\n\nQ: {question}"}])
    return msg.choices[0].message.content


def _groq_judge(question: str, context: str, answer: str) -> dict:
    from groq import Groq

    client = Groq()
    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile", max_tokens=100,
        messages=[{"role": "user", "content":
            _JUDGE_PROMPT.format(context=context, question=question, answer=answer)}])
    return _parse_judge_output(msg.choices[0].message.content)


def _groq_extract_city(question: str) -> str | None:
    from groq import Groq

    client = Groq()
    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile", max_tokens=10,
        messages=[{"role": "user", "content": _CITY_EXTRACT_PROMPT.format(q=question)}])
    text = msg.choices[0].message.content.strip()
    return None if text.upper() == "NONE" else text
