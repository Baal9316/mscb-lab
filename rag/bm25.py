"""Pure-Python BM25 (Okapi BM25) over tokenized text.

Implemented directly (no third-party dependency) so the retrieval stack stays
lightweight and reproducible. A best-match keyword ranker: given a query, each
document in the corpus is scored by term-frequency saturation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenization (letters/digits), whitespace/punct split."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class BM25Result:
    index: int
    score: float
    tokens: tuple[str, ...]

    def __repr__(self) -> str:
        return f"BM25Result(index={self.index}, score={self.score:.4f}, tokens={self.tokens})"


class BM25:
    """Okapi BM25 ranker.

    Args:
        corpus: list of documents (each is pre-tokenized text or raw text).
        k1: term-frequency saturation parameter.
        b: document-length normalization (0..1).
        delta: smoothing added to TF for short/empty docs.
    """

    def __init__(
        self,
        corpus: Iterable[Iterable[str]] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.0,
        tokenize_docs: bool = False,
    ):
        self.k1 = k1
        self.b = b
        self.delta = delta
        if corpus is None:
            self._docs: list[tuple[str, ...]] = []
        else:
            self._docs = [
                tuple(tokenize(" ".join(d)) if tokenize_docs else tuple(d))
                for d in corpus
            ]
        # IDF per term, computed over this corpus
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._recompute_stats()

    # ------------------------------------------------------------------ #
    def _recompute_stats(self) -> None:
        doc_freq: dict[str, int] = {}
        total_len = 0
        for doc in self._docs:
            total_len += len(doc)
            for term in set(doc):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        n = len(self._docs)
        self._avgdl = total_len / n if n else 0.0
        # Standard BM25 IDF with +1 smoothing; terms in every doc get ~0.
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    def add(self, doc: Iterable[str]) -> None:
        """Append a single already-tokenized document and recompute stats."""
        self._docs.append(tuple(doc))
        self._recompute_stats()

    def rebuild(self, corpus: Iterable[Iterable[str]]) -> None:
        self._docs = [tuple(c) for c in corpus]
        self._recompute_stats()

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def corpus(self) -> list[tuple[str, ...]]:
        return list(self._docs)

    # ------------------------------------------------------------------ #
    def _score_doc(self, terms: tuple[str, ...], freq: dict[str, int], dl: int) -> float:
        """BM25 score for a single doc given query terms and term frequencies."""
        score = 0.0
        for term in terms:
            if term not in freq:
                continue
            idf = self._idf.get(term, 0.0)
            tf = freq[term]
            # BM25 with delta smoothing (Han et al.) to avoid zeroing empty docs
            tf_delta = tf * (1 + self.delta)
            denom = self.k1 * (
                (1 - self.b) + self.b * (dl / self._avgdl if self._avgdl else 1.0)
            ) + tf_delta
            score += idf * (tf_delta / denom) if denom else 0.0
        return score

    def score(self, query: str, doc: Iterable[str]) -> float:
        """Score a single external (already-tokenized) doc against a query string."""
        terms = tokenize(query)
        tokens = tuple(doc)
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        return self._score_doc(tuple(terms), freq, len(tokens))

    def search(self, query: str, top_k: int | None = None) -> list[BM25Result]:
        """Return docs ranked by BM25 score, best first.

        Results carry their ``index`` in the corpus so callers can map back to
        source records.
        """
        terms = tuple(tokenize(query))
        if not terms or not self._docs:
            return []
        scored = []
        for i, doc in enumerate(self._docs):
            freq: dict[str, int] = {}
            for t in doc:
                freq[t] = freq.get(t, 0) + 1
            dl = len(doc)
            s = self._score_doc(terms, freq, dl)
            if s > 0.0 or len(terms) == 0:  # keep zero-score only when no query terms
                scored.append(BM25Result(index=i, score=s, tokens=doc))
        scored.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored
