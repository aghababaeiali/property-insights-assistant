"""
Human review queue CLI.

Every agent request is logged (agent_logs, via graph.log_node). Requests
where validate_node found an ungrounded citation, or the sampled LLM-judge
scored the answer as not grounded (judge_node), also land in review_queue
for a human to look at.

Reviewing an item lets you promote it into the eval harness's golden set
(agent/eval_cases.json) so a real failure, once seen, gets a permanent
regression check — the golden set grows from production behavior instead
of just what was thought to test upfront.

Run:
  python -m agent.review list
  python -m agent.review promote <id> --intent rag [--contains "TEXT"]
  python -m agent.review resolve <id>
"""
import argparse
import json
import os

from agent import db

HERE = os.path.dirname(__file__)
CASES_PATH = os.path.join(HERE, "eval_cases.json")


def list_pending() -> None:
    rows = db.query("""
        SELECT id, question, answer, intent, reason, created_at
        FROM review_queue WHERE status='pending' ORDER BY created_at
    """)
    if not rows:
        print("No pending reviews.")
        return
    for review_id, question, answer, intent, reason, created_at in rows:
        print(f"[{review_id}] ({intent}) {question}")
        print(f"    answer: {answer[:200]}")
        print(f"    reason: {reason}")
        print(f"    logged: {created_at}")
        print()


def resolve(review_id: int) -> None:
    db.query(
        "UPDATE review_queue SET status='resolved', reviewed_at=now() WHERE id=:id",
        {"id": review_id},
    )
    print(f"Marked review {review_id} resolved.")


def promote(review_id: int, intent: str, contains: str | None) -> None:
    rows = db.query("SELECT question FROM review_queue WHERE id=:id", {"id": review_id})
    if not rows:
        print(f"No review item with id {review_id}")
        return
    question = rows[0][0]

    with open(CASES_PATH) as f:
        cases = json.load(f)
    case = {"question": question, "expect_intent": intent}
    if contains:
        case["answer_contains"] = contains
    cases.append(case)
    with open(CASES_PATH, "w") as f:
        json.dump(cases, f, indent=2)
        f.write("\n")

    resolve(review_id)
    print(f"Promoted to {CASES_PATH}: {case}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    p_resolve = sub.add_parser("resolve")
    p_resolve.add_argument("id", type=int)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("id", type=int)
    p_promote.add_argument("--intent", required=True, choices=["sql", "rag", "risk"])
    p_promote.add_argument("--contains", default=None)

    args = parser.parse_args()
    if args.cmd == "list":
        list_pending()
    elif args.cmd == "resolve":
        resolve(args.id)
    elif args.cmd == "promote":
        promote(args.id, args.intent, args.contains)
