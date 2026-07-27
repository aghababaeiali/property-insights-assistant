"""FastAPI wrapper around the LangGraph agent — POST /ask {"question": "..."}"""

import os

from azure.monitor.opentelemetry import configure_azure_monitor
from fastapi import FastAPI
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from pydantic import BaseModel

from agent.graph import AGENT

if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()
    # azure-monitor-opentelemetry only auto-instruments django/fastapi/flask/
    # psycopg2/requests/urllib/urllib3 (see azure.monitor.opentelemetry._constants
    # ._FULLY_SUPPORTED_INSTRUMENTED_LIBRARIES) — httpx isn't in that list, so the
    # openai SDK's calls (it uses httpx exclusively, not requests) are invisible
    # to App Insights without this.
    HTTPXClientInstrumentor().instrument()



app = FastAPI(title="Property Insights Assistant")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    question: str
    intent: str | None
    answer: str
    validation_issues: list | None = None


@app.get("/health")
def health():
    """Basic liveness check — Container Apps can use this to confirm the app is up."""
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    out = AGENT.invoke({"question": q.question})
    return Answer(
        question=q.question,
        intent=out.get("intent"),
        answer=out.get("answer", "(no answer)"),
        validation_issues=out.get("validation_issues"),
    )
