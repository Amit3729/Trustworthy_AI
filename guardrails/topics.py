"""Sensitive-topic classification for a clinic assistant.

Approach: per-category phrase patterns over normalized text. A request can match
several categories; the action of the most severe one wins. Each category maps
to a fixed handling policy, so the LLM never decides how to treat a crisis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .text import normalize


class TopicAction(str, Enum):
    ESCALATE = "escalate"                  # do not call the LLM; return a vetted safety message
    REFUSE = "refuse"                      # decline politely
    ROUTE_TO_HUMAN = "route_to_human"      # hand off to staff with contact details
    ANSWER_WITH_DISCLAIMER = "answer_with_disclaimer"  # answer from documents + disclaimer


@dataclass(frozen=True)
class TopicPolicy:
    category: str
    action: TopicAction
    severity: int            # higher wins when several categories match
    patterns: tuple[str, ...]


POLICIES: tuple[TopicPolicy, ...] = (
    TopicPolicy("self_harm", TopicAction.ESCALATE, 100, (
        r"\b(?:kill|hurt|harm|cut) (?:myself|my self)\b",
        r"\bsuicid(?:e|al)\b",
        r"\bend (?:my (?:own )?life|it all)\b",
        r"\b(?:want|wanna|going) to die\b",
        r"\bdon'?t want to (?:live|be alive|wake up)\b",
        r"\b(?:no reason|nothing) to live\b",
        r"\bself[- ]?harm\b",
        r"\boverdose on purpose\b",
        r"\bhow many .{0,30}\b(?:would|to) (?:kill|be (?:enough|fatal|lethal))\b",
        r"\blethal dose\b",
    )),
    TopicPolicy("medical_emergency", TopicAction.ESCALATE, 90, (
        r"\bchest pain\b",
        r"\b(?:can'?t|cannot|unable to|trouble|difficulty) breath(?:e|ing)\b",
        r"\b(?:having|had) a (?:stroke|heart attack|seizure)\b",
        r"\b(?:stroke|heart attack) symptoms\b",
        r"\b(?:unconscious|unresponsive|not breathing|passed out)\b",
        r"\b(?:severe|heavy|uncontrollable) bleeding\b",
        r"\b(?:swallowed|took|ate) (?:too many|a bottle of|a lot of)\b",
        r"\b(?:throat|tongue|face) (?:is )?swell(?:ing|ed)\b|\banaphyla",
        r"\barm (?:is |went )?numb\b|\bface (?:is )?drooping\b",
    )),
    TopicPolicy("illegal_drug_access", TopicAction.REFUSE, 80, (
        r"\b(?:get|buy|obtain|order|score)\b.{0,40}\bwithout (?:a )?(?:prescription|script|doctor)\b",
        r"\b(?:fake|forged?|forging|counterfeit) (?:a )?(?:prescription|script|doctor'?s note)\b",
        r"\b(?:get|getting) high\b",
        r"\b(?:recreational(?:ly)?)\b.{0,30}\b(?:opioids?|oxy\w*|xanax|adderall|fentanyl)\b",
    )),
    TopicPolicy("legal_complaint", TopicAction.ROUTE_TO_HUMAN, 60, (
        r"\b(?:sue|suing|lawsuit|lawyer|attorney|legal action|malpractice|negligence)\b",
        r"\b(?:file|make|lodge) a (?:formal )?complaint\b",
    )),
    TopicPolicy("medication_or_diagnosis", TopicAction.ANSWER_WITH_DISCLAIMER, 40, (
        r"\b(?:dose|dosage|dosing|mg|milligrams?)\b",
        r"\bhow (?:many|much) .{0,20}\b(?:pills?|tablets?|capsules?)\b",
        r"\bcan i (?:take|mix|combine)\b.{0,40}\b(?:with|together|while|alcohol|pills?|tablets?|capsules?|drugs?)\b",
        r"\bshould i (?:stop|start|keep) taking\b",
        r"\b(?:prescription|prescribe|refill|medication|medicine|antibiotics?|ibuprofen|paracetamol|acetaminophen)\b",
        r"\b(?:diagnos\w*|do i have|is it (?:cancer|serious|contagious)|symptoms? of)\b",
    )),
)


@dataclass
class TopicResult:
    categories: list[str] = field(default_factory=list)
    action: TopicAction | None = None
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def primary(self) -> str | None:
        return self.categories[0] if self.categories else None


_COMPILED = [(p, [re.compile(x) for x in p.patterns]) for p in POLICIES]


def classify_topics(text: str) -> TopicResult:
    view = normalize(text)
    hits: list[TopicPolicy] = []
    evidence: dict[str, str] = {}
    for policy, patterns in _COMPILED:
        for pat in patterns:
            m = pat.search(view)
            if m:
                hits.append(policy)
                evidence[policy.category] = m.group()
                break
    hits.sort(key=lambda p: p.severity, reverse=True)
    return TopicResult(
        categories=[p.category for p in hits],
        action=hits[0].action if hits else None,
        evidence=evidence,
    )
