"""Groundedness validation: is every claim in the answer supported by the references?

Approach (claim-level, lexical):
  1. Split the answer into sentences; each sentence is one claim. Apologies and
     "I could not find..." abstentions make no factual claim and are exempt.
  2. Find the best-supporting window (one sentence, or two adjacent sentences)
     across the provided documents by content-word overlap.
  3. The claim is supported only if
       - overlap >= threshold (most of its content words appear in the window),
       - every number in the claim appears in that window (catches invented
         times, prices, phone numbers), and
       - the claim and its best-matching source sentence agree on negation
         (catches "walk-ins are NOT accepted" vs "walk-ins are accepted").
This is deliberately transparent and dependency-free; REPORT.md discusses what
it cannot catch (paraphrase, swapped relations) and the NLI-based upgrade.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import GuardrailConfig
from .text import content_tokens, has_negation, numbers_in, split_sentences

_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
_ABSTAIN = re.compile(
    r"\b(?:i(?:'m| am) sorry|i (?:do not|don't|could not|couldn't|cannot|can't) (?:have|find|answer|help)|"
    r"(?:no|don't have) (?:information|details)|not (?:covered|mentioned) in)\b",
    re.I,
)


@dataclass
class ClaimCheck:
    sentence: str
    supported: bool
    overlap: float
    source: str | None
    reasons: list[str] = field(default_factory=list)
    exempt: bool = False


@dataclass
class GroundednessResult:
    grounded: bool
    claims: list[ClaimCheck] = field(default_factory=list)
    citation_warnings: list[str] = field(default_factory=list)

    @property
    def unsupported(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.supported and not c.exempt]

    @property
    def support_ratio(self) -> float:
        checked = [c for c in self.claims if not c.exempt]
        return 1.0 if not checked else sum(c.supported for c in checked) / len(checked)


def _windows(doc_id: str, text: str) -> list[tuple[str, str, bool]]:
    """(doc_id, window_text, is_single_sentence) for sentences and adjacent pairs."""
    sents = split_sentences(text)
    out = [(doc_id, s, True) for s in sents]
    out += [(doc_id, f"{a} {b}", False) for a, b in zip(sents, sents[1:])]
    return out


def check_groundedness(
    answer: str,
    sources: dict[str, str],
    cited: list[str],
    config: GuardrailConfig,
) -> GroundednessResult:
    windows = [w for doc_id, text in sources.items() for w in _windows(doc_id, text)]
    window_tokens = [content_tokens(w[1]) for w in windows]
    claims: list[ClaimCheck] = []
    citation_warnings: list[str] = []

    for sentence in split_sentences(answer):
        clean = _PLACEHOLDER.sub(" ", sentence)
        tokens = content_tokens(clean)
        if _ABSTAIN.search(sentence) or len(tokens) < 2:
            claims.append(ClaimCheck(sentence, True, 1.0, None, ["no factual claim"], exempt=True))
            continue
        if not windows:
            claims.append(ClaimCheck(sentence, False, 0.0, None, ["no reference documents available"]))
            continue

        def score(i: int) -> tuple[float, bool]:
            overlap = len(tokens & window_tokens[i]) / len(tokens)
            return overlap, windows[i][0] in cited  # prefer cited docs on ties

        best = max(range(len(windows)), key=score)
        overlap = score(best)[0]
        doc_id, window_text, _ = windows[best]
        reasons = []
        if overlap < config.grounding_min_overlap:
            reasons.append(f"low lexical support ({overlap:.2f} < {config.grounding_min_overlap})")
        if config.grounding_check_numbers:
            missing = numbers_in(clean) - numbers_in(window_text)
            if missing:
                reasons.append(f"numbers not found in source: {sorted(missing)}")
        if config.grounding_check_negation:
            singles = [i for i, w in enumerate(windows) if w[2] and w[0] == doc_id]
            best_single = max(singles, key=lambda i: len(tokens & window_tokens[i]))
            if has_negation(clean) != has_negation(windows[best_single][1]):
                reasons.append("negation mismatch with source sentence")
        supported = not reasons
        if supported and cited and doc_id not in cited:
            citation_warnings.append(f"claim supported by {doc_id}, which was not cited: {sentence!r}")
        claims.append(ClaimCheck(sentence, supported, round(overlap, 2), doc_id, reasons))

    grounded = all(c.supported for c in claims)
    return GroundednessResult(grounded, claims, citation_warnings)
