"""Text retrieval over ingested documents: BM25 + dense (9002) + fusion.

This is Issue #2 scope only. It is intentionally independent of visual
retrieval (#3) and reranking (#4) -- those will build on top later.

The index is built from the pages stored by the Milestone 1 ingestion
pipeline (``DocumentStore`` + ``DocumentPage``). Every returned candidate
carries the citation metadata needed for the QA feature: ``document_id``,
``filename``, ``page_no``, and the page ``text``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .bm25 import BM25, tokenize
from .config import Settings, get_settings
from .embedding import EmbeddingError, get_text_embedding
from .store import DocumentStore


class EmptyIndexError(RuntimeError):
    """Raised when retrieval is attempted on an index with no documents."""


class RetrievalError(RuntimeError):
    """Raised for retrieval failures (e.g. embedding service down)."""


@dataclass
class TextCandidate:
    """A ranked retrieval result, fully wired to its source for citations."""

    document_id: str
    filename: str
    page_no: int
    text: str
    score: float
    source: str  # 'bm25' | 'dense' | 'fusion'

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_no": self.page_no,
            "text": self.text,
            "score": self.score,
            "source": self.source,
        }

    @property
    def citation(self) -> str:
        return f"{self.filename} · page {self.page_no}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class TextIndex:
    """Keyed BM25 + in-memory dense vectors over the document store."""

    def __init__(self, store: DocumentStore, settings: Settings | None = None):
        self.store = store
        self.settings = settings or get_settings()
        self._bm25 = BM25()
        # Parallel arrays mapping corpus index -> chunk record
        self._chunks: list[dict] = []
        self._texts: list[str] = []
        self._vectors: list[list[float]] = []
        self._built_for: set[str] = set()

    # ------------------------------------------------------------------ #
    # Index construction / refresh
    # ------------------------------------------------------------------ #
    def rebuild(self) -> None:
        """Rebuild the BM25 corpus from every document in the store."""
        docs = self.store.list_all()
        self._chunks = []
        self._texts = []
        self._vectors = []
        for doc in docs:
            for page in doc.pages:
                text = _clean(page.extracted_text)
                # Skip pages with no extracted text: blank slides are not
                # useful keyword/dense candidates, and the 9002 endpoint
                # rejects empty prompts.
                if not text:
                    continue
                self._chunks.append({
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "page_no": page.page_no,
                    "text": text,
                })
                self._texts.append(text)
        self._bm25.rebuild([tokenize(text) for text in self._texts])
        self._built_for = {d.document_id for d in docs}

    def is_built(self) -> bool:
        return len(self._chunks) > 0

    def doc_count(self) -> int:
        return len(self._built_for)

    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def search_bm25(self, query: str, top_k: int = 8) -> list[TextCandidate]:
        """Keyword retrieval using pure BM25 over page text."""
        if not self.is_built():
            raise EmptyIndexError("Index is empty - ingest documents first.")
        results = self._bm25.search(_clean(query), top_k=top_k)
        return [
            TextCandidate(
                document_id=self._chunks[r.index]["document_id"],
                filename=self._chunks[r.index]["filename"],
                page_no=self._chunks[r.index]["page_no"],
                text=self._chunks[r.index]["text"],
                score=r.score,
                source="bm25",
            )
            for r in results
        ]

    def search_dense(self, query: str, top_k: int = 8) -> list[TextCandidate]:
        """Dense retrieval using 9002 embeddings + cosine similarity.

        Lazy-builds the dense vectors on first call (fetched from 9002 and
        cached in memory). Raises :class:`RetrievalError` if 9002 is down.
        """
        if not self.is_built():
            raise EmptyIndexError("Index is empty - ingest documents first.")
        if not self._vectors:
            self._build_vectors()
        qv = self._embed([_clean(query)])[0]
        scored = _cosine_scores(qv.vector, self._vectors)
        top = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:top_k]
        return [
            TextCandidate(
                document_id=self._chunks[i]["document_id"],
                filename=self._chunks[i]["filename"],
                page_no=self._chunks[i]["page_no"],
                text=self._chunks[i]["text"],
                score=scored[i],
                source="dense",
            )
            for i in top
        ]

    def search(self, query: str, top_k: int = 8,
               bm25_weight: float = 0.5) -> list[TextCandidate]:
        """Combine BM25 + dense scores (score-sum fusion) into one ranking.

        Replaces the per-source scores with a zero-to-one normalized fusion
        score and tags results as ``source='fusion'``.
        """
        if not self.is_built():
            raise EmptyIndexError("Index is empty - ingest documents first.")

        bm25 = self.search_bm25(query, top_k=max(top_k * 3, top_k))
        dense = self.search_dense(query, top_k=max(top_k * 3, top_k))

        merged = _fuse(bm25, dense, bm25_weight=bm25_weight)
        return merged[:top_k]

    # ------------------------------------------------------------------ #
    def _build_vectors(self) -> None:
        self._vectors = self._embed_all(self._texts)

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        # Batch in chunks of 32 to stay well under the endpoint's limits.
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            try:
                res = get_text_embedding(batch, settings=self.settings)
            except EmbeddingError as exc:
                raise RetrievalError(f"Dense retrieval failed: {exc}") from exc
            vecs.extend(r.vector for r in res)
        return vecs

    def _embed(self, texts: list[str]) -> list:
        try:
            return get_text_embedding(texts, settings=self.settings)
        except EmbeddingError as exc:
            raise RetrievalError(f"Dense retrieval failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cosine_scores(query_vec: list[float], vectors: list[list[float]]) -> list[float]:
    """Cosine similarity between a query vector and each stored vector (0..1)."""
    qn = _norm(query_vec)
    return [
        _dot(query_vec, v) / (qn * _norm(v)) if qn and _norm(v) else 0.0
        for v in vectors
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _fuse(bm25: list[TextCandidate], dense: list[TextCandidate],
          bm25_weight: float) -> list[TextCandidate]:
    """Score-sum fusion over two ranked lists keyed by (doc_id, page_no).

    Each list's scores are min-max normalized to [0,1], then combined as
    ``weight*bm25 + (1-weight)*dense``. Candidates present in only one list get
    the other list's score = 0. Results are deduplicated by (doc_id, page_no).
    """
    if bm25_weight < 0 or bm25_weight > 1:
        raise ValueError("bm25_weight must be in [0, 1]")

    def norm(items: list[TextCandidate]) -> dict[tuple, float]:
        if not items:
            return {}
        scores = {_key(c): c.score for c in items}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            # Degenerate range: either a single candidate or every candidate
            # tied for the same score. Min-max would push them all to 0.0,
            # which is misleading (a lone result is the top result). Map every
            # tied/only candidate to 1.0 deterministically instead.
            return {k: 1.0 for k in scores}
        span = hi - lo
        return {k: (v - lo) / span for k, v in scores.items()}

    b = norm(bm25)
    d = norm(dense)

    combined: dict[tuple, float] = {}
    for key in set(b) | set(d):
        combined[key] = bm25_weight * b.get(key, 0.0) + (1 - bm25_weight) * d.get(key, 0.0)

    # Build lookup of the fuller record for each key
    record = {_key(c): c for c in bm25 + dense}
    ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for key, score in ordered:
        cand = record[key]
        out.append(TextCandidate(
            document_id=cand.document_id,
            filename=cand.filename,
            page_no=cand.page_no,
            text=cand.text,
            score=score,
            source="fusion",
        ))
    return out


def _key(c: TextCandidate) -> tuple:
    return (c.document_id, c.page_no)
