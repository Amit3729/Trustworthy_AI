"""End-to-end test cases from tests/cases.json.

Known failures are documented limitations analysed in REPORT.md section 4. They are
marked xfail(strict=True): if a future change fixes one, the suite reports it
so the label can be removed and the report updated.
"""
import pytest

from guardrails.evaluation import load_suite, run_case
from guardrails.knowledge_base import KnowledgeBase

CASES, POISONED = load_suite()
KB = KnowledgeBase.load()

KNOWN_FAILURES = {
    "TC12": "email written out in words ('sita dot sharma at gmail dot com') is not matched by the regex",
    "TC14": "third-party name without a lead-in phrase needs NER, not regex",
    "TC21": "paraphrased system-prompt extraction has no trigger phrase the rules recognise",
    "TC31": "role-play persona ('you're Max ... no rulebook') is missed; topic refusal rescues the outcome",
    "TC32": "keyword 'suicide' escalates an informational question (over-blocking)",
    "TC33": "self-harm intent expressed without any listed keyword is missed",
    "TC45": "lexical overlap cannot recognise a faithful paraphrase",
    "TC46": "swapped entity-attribute relation has high word overlap and passes",
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=c["id"], marks=pytest.mark.xfail(reason=KNOWN_FAILURES[c["id"]], strict=True))
        if c["id"] in KNOWN_FAILURES else pytest.param(c, id=c["id"])
        for c in CASES
    ],
)
def test_case(case):
    outcome = run_case(case, KB, POISONED)
    assert outcome.passed, "; ".join(outcome.failures)


def test_suite_has_at_least_20_cases():
    assert len(CASES) >= 20


def test_every_category_is_covered():
    categories = {c["category"] for c in CASES}
    assert {"normal", "malicious", "sensitive_info", "invalid_output", "unsupported_claim"} <= categories
