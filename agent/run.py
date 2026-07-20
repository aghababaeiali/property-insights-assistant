"""CLI: python -m agent.run "your question here" """
import sys

from agent.graph import AGENT

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "what is the average price in Lisbon?"
    out = AGENT.invoke({"question": q})
    print(f"Q: {q}\n")
    print(f"[intent: {out.get('intent')}]")
    print(out.get("answer", "(no answer)"))
    issues = out.get("validation_issues")
    if issues:
        print(f"\n[validation issues: {issues}]")
