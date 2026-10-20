"""Reference documents and a small TF-IDF retriever (the RAG part of the app)."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .text import content_token_list

DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    topic: str
    visibility: str
    text: str


class KnowledgeBase:
    def __init__(self, documents: list[Document]):
        self.documents = {d.id: d for d in documents}
        self.public = [d for d in documents if d.visibility == "public"]
        self.internal = [d for d in documents if d.visibility != "public"]
        self._tf = {d.id: Counter(content_token_list(f"{d.title} {d.text}")) for d in self.public}
        df = Counter(t for tf in self._tf.values() for t in tf)
        n = len(self.public)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._norm = {i: self._vec_norm(tf) for i, tf in self._tf.items()}

    @classmethod
    def load(cls, path: Path | str = DEFAULT_KB_PATH) -> "KnowledgeBase":
        raw = json.loads(Path(path).read_text())
        return cls([Document(**d) for d in raw["documents"]])

    def _vec_norm(self, tf: Counter) -> float:
        return math.sqrt(sum((c * self._idf.get(t, 0)) ** 2 for t, c in tf.items())) or 1.0

    def search(self, query: str, top_k: int = 3, min_score: float = 0.1) -> list[tuple[Document, float]]:
        """Cosine similarity over TF-IDF vectors. Internal documents are never searchable."""
        q = Counter(content_token_list(query))
        q_norm = self._vec_norm(q)
        scored = []
        for doc in self.public:
            tf = self._tf[doc.id]
            dot = sum(c * tf.get(t, 0) * self._idf.get(t, 0) ** 2 for t, c in q.items())
            score = dot / (q_norm * self._norm[doc.id])
            if score >= min_score:
                scored.append((doc, round(score, 3)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get(self, doc_ids: list[str]) -> list[Document]:
        return [self.documents[i] for i in doc_ids if i in self.documents]
