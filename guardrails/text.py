"""Text utilities shared by the guardrails: normalization, tokenization, sentences."""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SPACED_LETTERS = re.compile(r"\b(?:[a-z][\s.\-_*]){3,}[a-z]\b")

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be been before being below
between both but by can could did do does doing down during each few for from further had
has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves may might must shall us
please let lets get got also there's it's we're you're i'm s t d ll re ve m
""".split())

NEGATION_RE = re.compile(r"\b(?:not|no|never|cannot|none|nor|nobody|nothing)\b|n't\b", re.I)


def normalize(text: str) -> str:
    """Unicode-normalize, drop invisible characters, lowercase, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    return re.sub(r"\s+", " ", text).strip().lower()


def deobfuscate(text: str) -> str:
    """Undo common filter-evasion tricks: leetspeak and s-p-a-c-e-d letters.

    Spaced letters are rejoined before whitespace is collapsed, so a wider gap
    ("i g n o r e   y o u r") still separates the hidden words.
    """
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).lower().translate(_LEET)
    text = _SPACED_LETTERS.sub(lambda m: re.sub(r"[\s.\-_*]", "", m.group()), text)
    return re.sub(r"\s+", " ", text).strip()


def decode_base64_segments(text: str) -> list[str]:
    """Return printable decodings of base64-looking substrings (hidden payloads)."""
    decoded = []
    for token in re.findall(r"[A-Za-z0-9+/]{16,}={0,2}", text):
        try:
            raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
            candidate = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if candidate.isprintable() and sum(c.isalpha() for c in candidate) > len(candidate) * 0.6:
            decoded.append(candidate)
    return decoded


def stem(word: str) -> str:
    """Tiny suffix stripper: doses/dose, booked/booking, rescheduled/reschedule share a stem."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies"):
        word = word[:-3] + "y"
    elif word.endswith(("sses", "xes", "ches", "shes", "zes")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def content_token_list(text: str) -> list[str]:
    """Lowercased, stemmed, stopword-free word tokens (numbers kept), in order."""
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return [stem(w) for w in words if w not in STOPWORDS and not (len(w) == 1 and w.isalpha())]


def content_tokens(text: str) -> set[str]:
    return set(content_token_list(text))


def numbers_in(text: str) -> set[str]:
    """Numbers in canonical form: '8:00' -> '8', '$25.00' -> '25', '24' -> '24'."""
    out = set()
    for n in re.findall(r"\d+(?:[.:]\d+)?", text):
        n = re.sub(r"[.:]0+$", "", n)
        out.add(n.lstrip("0") or "0")
    return out


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def has_negation(text: str) -> bool:
    return bool(NEGATION_RE.search(text))
