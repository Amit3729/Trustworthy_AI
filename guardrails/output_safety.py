"""Sensitive-information detection on model output (beyond PII).

Three leak channels are checked on the raw model output:
  * the canary token planted in the system prompt,
  * verbatim fragments (word n-grams) of the system prompt,
  * content of internal, staff-only documents and generic secret patterns.
Any of these blocks the response outright: a leak is a security event, so the
system does not retry and does not try to "clean" the text.
PII in the answer is handled separately in the pipeline by redaction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import normalize

SECRET_PATTERNS = {
    "api_key": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b|\bAKIA[0-9A-Z]{16}\b"),
    "credential_disclosure": re.compile(
        r"\b(?:password|passcode|pass code|door code|override code|access code)\s*(?:is|:|=)\s*\S+", re.I),
}


@dataclass
class LeakResult:
    blocked: bool
    reasons: list[str] = field(default_factory=list)


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9#'-]+", normalize(text))
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class LeakDetector:
    def __init__(self, system_prompt: str, canary: str, internal_docs: list[str], n: int = 6):
        self.canary = canary.lower()
        self.n = n
        self.prompt_shingles = _shingles(system_prompt, n)
        self.internal_shingles = set().union(*(_shingles(d, 4) for d in internal_docs)) if internal_docs else set()

    def check(self, text: str) -> LeakResult:
        reasons = []
        if self.canary in text.lower():
            reasons.append("system-prompt canary token present in output")
        overlap = _shingles(text, self.n) & self.prompt_shingles
        if overlap:
            reasons.append(f"verbatim system-prompt fragment: {' '.join(next(iter(overlap)))!r}")
        internal = _shingles(text, 4) & self.internal_shingles
        if internal:
            reasons.append(f"internal document content: {' '.join(next(iter(internal)))!r}")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                reasons.append(f"secret pattern: {name}")
        return LeakResult(bool(reasons), reasons)
