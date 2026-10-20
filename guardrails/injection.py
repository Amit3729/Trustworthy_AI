"""Prompt-injection detection.

Approach: weighted rules grouped by attack family. Each rule that matches
contributes its weight, and weights combine with a noisy-OR
(score = 1 - prod(1 - w)), so several weak signals can add up to a block while
one weak signal cannot. Rules run against three views of the text: the
normalized text, a de-obfuscated version (leetspeak, s-p-a-c-e-d letters) and
any base64 payloads decoded from it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import decode_base64_segments, deobfuscate, normalize

_TARGET = r"(?:instructions?|rules|guidelines|prompts?|directives?|policies|polic(?:y)|constraints|programming|guardrails)"
_PRIOR = r"(?:all|any|every|previous|prior|above|earlier|preceding|original|initial|your|system|the system|developer)"

# (attack family, weight, pattern)
RULES: list[tuple[str, float, re.Pattern]] = [
    # 1. Overriding the operator's instructions
    ("instruction_override", 0.90, re.compile(
        rf"\b(?:ignore|disregard|forget|override|bypass|skip|drop|abandon)\b(?:\s+\w+){{0,3}}?\s+{_PRIOR}\s+(?:\w+\s+){{0,2}}{_TARGET}")),
    ("instruction_override", 0.60, re.compile(
        rf"\b(?:new|updated|real|actual)\s+{_TARGET}\s*[:\-]")),
    ("instruction_override", 0.50, re.compile(
        r"\bfrom now on\b.{0,40}\b(?:you (?:will|must|are)|only|always|never)\b")),
    # 2. Persona / role hijacking
    ("role_hijack", 0.85, re.compile(
        r"\b(?:you are now|you're now|act as|pretend (?:to be|you are)|roleplay as|become)\b.{0,40}"
        r"\b(?:unrestricted|unfiltered|uncensored|jailbroken|evil|no (?:rules|limits|restrictions|filters)|without (?:any )?(?:rules|limits|restrictions|filters))")),
    ("role_hijack", 0.85, re.compile(
        r"\b(?:dan|developer|god|admin|debug) mode\b|\bjailbreak(?:ed)?\b|\bdo anything now\b|\byou are (?:now )?dan\b")),
    ("role_hijack", 0.40, re.compile(r"\byou are (?:now )?(?:no longer|not) (?:an? )?(?:assistant|bound|restricted)")),
    # 3. Extracting the system prompt or hidden configuration
    ("prompt_extraction", 0.90, re.compile(
        r"\b(?:reveal|show|print|display|output|repeat|leak|tell me|give me|what (?:is|are|were)|share|dump|recite)\b.{0,30}"
        r"\b(?:system prompt|system message|initial (?:prompt|instructions)|hidden (?:prompt|instructions|rules)|"
        r"your (?:instructions|prompt|configuration|programming))")),
    ("prompt_extraction", 0.40, re.compile(r"\b(?:what|which) (?:rules|guidelines) (?:are you|were you|do you)\b")),
    ("prompt_extraction", 0.80, re.compile(r"\b(?:repeat|print|output) (?:everything|all|the text|the words) (?:above|before)")),
    # 4. Extracting confidential data about others or the organization
    ("data_exfiltration", 0.75, re.compile(
        r"\b(?:list|show|give|send|export|dump|share|tell)\b.{0,25}\b(?:all|every|other|another)\b.{0,15}"
        r"\b(?:patients?|users?|customers?|staff|employees?)'?s?\b.{0,30}\b(?:records?|data|details|information|numbers?|emails?|addresses|histor(?:y|ies))")),
    ("data_exfiltration", 0.75, re.compile(
        r"\b(?:internal|confidential|staff[- ]only|private)\s+(?:documents?|files?|notes?|codes?|passwords?|credentials?|data)")),
    ("data_exfiltration", 0.45, re.compile(r"\b(?:password|passcode|door code|api key|access token|credentials)\b")),
    # 5. Explicit bypass of safety controls
    ("restriction_bypass", 0.90, re.compile(
        r"\b(?:bypass|disable|turn off|deactivate|circumvent|evade|get around)\b.{0,20}"
        r"\b(?:safety|filters?|guardrails?|restrictions?|content policy|moderation|security)")),
    ("restriction_bypass", 0.45, re.compile(r"\bwithout (?:any )?(?:restrictions|filters|limitations|censorship)\b")),
    # 6. Fake conversation structure / delimiter injection
    ("delimiter_injection", 0.70, re.compile(
        r"</?\s*(?:system|assistant|instructions?|admin)\s*>|\[/?(?:system|inst)\]|<\|im_start\|>|"
        r"#{2,}\s*(?:system|instructions?|admin)|\b(?:begin|end) (?:system|admin) (?:prompt|message)")),
    ("delimiter_injection", 0.50, re.compile(r"^\s*(?:system|admin|developer)\s*:", re.M)),
]


@dataclass
class InjectionResult:
    score: float
    matches: list[dict] = field(default_factory=list)
    decoded_payloads: list[str] = field(default_factory=list)

    @property
    def families(self) -> list[str]:
        return sorted({m["family"] for m in self.matches})


def _score_view(view: str, label: str) -> list[dict]:
    hits = []
    for family, weight, pattern in RULES:
        m = pattern.search(view)
        if m:
            hits.append({"family": family, "weight": weight, "view": label, "evidence": m.group()[:80]})
    return hits


def detect_injection(text: str) -> InjectionResult:
    views = {"normalized": normalize(text), "deobfuscated": deobfuscate(text)}
    decoded = decode_base64_segments(text)
    for i, payload in enumerate(decoded):
        views[f"base64[{i}]"] = normalize(payload)

    # A rule counts once even if it matches several views of the same text.
    best: dict[tuple[str, float], dict] = {}
    for label, view in views.items():
        for hit in _score_view(view, label):
            best.setdefault((hit["family"], hit["weight"]), hit)

    matches = list(best.values())
    remaining = 1.0
    for hit in matches:
        remaining *= 1.0 - hit["weight"]
    if decoded and matches and any(h["view"].startswith("base64") for h in matches):
        remaining *= 0.7  # hiding instructions in an encoding is itself a strong signal
    return InjectionResult(score=round(1.0 - remaining, 3), matches=matches, decoded_payloads=decoded)
