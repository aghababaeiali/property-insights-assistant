"""Parametrized pytest wrapper around agent/eval.py's golden set.

Kept as a thin wrapper (not a re-implementation) around agent.eval.check_case
so "what counts as passing" has one definition, reused by both the
standalone `python -m agent.eval` CLI and this CI-facing pytest suite.
"""
import pytest

from agent.eval import CASES, check_case


@pytest.mark.parametrize("case", CASES, ids=[c["question"] for c in CASES])
def test_golden_case(case):
    _, failures = check_case(case)
    assert failures == []
