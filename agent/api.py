"""FastAPI wrapper around the LangGraph agent — POST /ask {"question": "..."}"""
from fastapi import FastAPI
from pydantic import BaseModel

from agent.graph import AGENT

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
