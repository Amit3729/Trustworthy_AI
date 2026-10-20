"""PII detection and redaction, shared by the input and output guardrails.

Approach: regular expressions for structured identifiers, each followed by a
validator that removes look-alikes (Luhn checksum for card numbers, SSA rules
for SSNs, octet ranges for IPs, digit counts for phone numbers). Spans are
claimed in priority order so a card number is not also reported as a phone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class PIIMatch:
    entity: str
    value: str
    start: int
    end: int


def _luhn_ok(value: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def _phone_ok(value: str) -> bool:
    return 10 <= len(re.sub(r"\D", "", value)) <= 15


_DATE = (
    r"(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{4}|"
    r"\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]* \d{4})"
)

# (entity, compiled pattern, optional validator, capture group holding the PII)
_RULES: list[tuple[str, re.Pattern, Callable[[str], bool] | None, int]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None, 0),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), _luhn_ok, 0),
    ("SSN", re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)"), None, 0),
    ("MRN", re.compile(r"\b(?:MRN|medical record (?:number|no\.?|#))[:#\s-]*(\d{6,10})\b", re.I), None, 1),
    ("DATE_OF_BIRTH", re.compile(
        r"\b(?:dob|d\.o\.b\.?|date of birth|born on|born|birthday)\b[:\s-]*(?:is\s+)?(" + _DATE + r")", re.I), None, 1),
    ("IP_ADDRESS", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None, 0),
    ("PHONE", re.compile(
        r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)\s?|\d{2,5}[\s.-]?)\d{3,4}[\s.-]?\d{3,5}(?![\w-])"),
        _phone_ok, 0),
    ("PERSON_NAME", re.compile(
        r"\b(?i:my name is|i am|i'm|this is|patient name:?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"), None, 1),
]


def detect_pii(text: str, allowlist: Iterable[str] = ()) -> list[PIIMatch]:
    """Find PII spans. Values in `allowlist` (e.g. the clinic's public phone) are ignored."""
    allowed = {_canonical(a) for a in allowlist}
    claimed: list[tuple[int, int]] = []
    found: list[PIIMatch] = []
    for entity, pattern, validator, group in _RULES:
        for m in pattern.finditer(text):
            value, start, end = m.group(group), m.start(group), m.end(group)
            if validator and not validator(value):
                continue
            if any(start < e and s < end for s, e in claimed):
                continue
            if _canonical(value) in allowed:
                claimed.append((start, end))
                continue
            claimed.append((start, end))
            found.append(PIIMatch(entity, value, start, end))
    return sorted(found, key=lambda p: p.start)


def redact(text: str, matches: list[PIIMatch]) -> str:
    """Replace each PII span with a typed placeholder such as [EMAIL]."""
    for m in sorted(matches, key=lambda p: p.start, reverse=True):
        text = text[: m.start] + f"[{m.entity}]" + text[m.end:]
    return text


def collect_public_values(texts: Iterable[str]) -> set[str]:
    """PII-shaped values that appear in public reference documents (safe to repeat)."""
    values: set[str] = set()
    for t in texts:
        values.update(m.value for m in detect_pii(t))
    return values


def _canonical(value: str) -> str:
    v = value.strip().lower()
    digits = re.sub(r"\D", "", v)
    return digits if "@" not in v and len(digits) >= 7 else v
