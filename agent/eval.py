"""
Offline regression eval for the agent.

Runs a small golden set of questions through the graph and checks:
  - the router picked the expected intent
  - the answer wasn't flagged by validate_node (no ungrounded citations)
  - any extra per-question assertions (e.g. answer contains an expected marker)

This is the fast, deterministic layer of "evaluating the agent" — cheap
enough to run on every change (CI gate). It does not replace production
tracing/monitoring (see docs/design.md / agent/review.py): this only tells
you "did this known set of behaviors regress," not "how is the agent doing
on real traffic right now."

Cases live in agent/eval_cases.json (not hardcoded here) so that
agent/review.py can append newly-reviewed real-world cases to the golden
set programmatically.

Run:  python -m agent.eval
"""
import json
import os
import time

from agent.graph import AGENT

HERE = os.path.dirname(__file__)
CASES_PATH = os.path.join(HERE, "eval_cases.json")

with open(CASES_PATH) as f:
    CASES = json.load(f)


def check_case(case: dict) -> tuple[dict, list[str]]:
    """Run one golden-set case through the graph, return (raw state, failures).

    Shared by run() (the CLI table below) and tests/test_eval.py (pytest
    parametrizes over CASES and asserts failures == []) — one source of
    truth for what "passing" means, instead of two copies that could drift.
    """
    out = AGENT.invoke({"question": case["question"]})
    answer = out.get("answer", "")
    intent = out.get("intent", "")
    issues = out.get("validation_issues", [])

    failures = []
    if intent != case["expect_intent"]:
        failures.append(f"expected intent={case['expect_intent']!r}, got {intent!r}")
    if issues:
        failures.append(f"validation flagged: {issues}")
    needle = case.get("answer_contains")
    if needle and needle not in answer:
        failures.append(f"answer missing expected marker {needle!r}")
    if not answer.strip():
        failures.append("empty answer")
    return out, failures


def run() -> bool:
    all_passed = True
    print(f"{'result':6s} {'intent':6s} {'ms':>6s}  question")
    print("-" * 70)

    for case in CASES:
        start = time.time()
        out, failures = check_case(case)
        elapsed_ms = (time.time() - start) * 1000

        passed = not failures
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"{status:6s} {out.get('intent', ''):6s} {elapsed_ms:6.0f}  {case['question']}")
        for f in failures:
            print(f"       - {f}")

    print("-" * 70)
    print("ALL PASSED" if all_passed else "SOME FAILED")
    return all_passed


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
