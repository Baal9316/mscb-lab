"""Text retrieval over ingested documents: BM25 + dense (9002) + fusion.

This is Issue #2 scope only. It is intentionally independent of visual
retrieval (#3) and reranking (#4) -- those will build on top later.

The index is built from the pages stored by the Milestone 1 ingestion
pipeline (``DocumentStore`` + ``DocumentPage``). Every returned candidate
carries the citation metadata needed for the QA feature: ``document_id``,
``filename``, ``page_no``, and the page ``text``. Long page text is split
into smaller retrieval chunks (with a per-page ``chunk_index``) so BM25 and
dense retrieval operate over focused sub-passages rather than one oversized
vector per page, while preserving page-level provenance.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .bm25 import BM25, tokenize
from .config import Settings, get_settings
from .embedding import EmbeddingError, get_text_embedding
from .store import DocumentStore

# Default chunking parameters (configurable at build time).
DEFAULT_CHUNK_SIZE = 800   # target characters per chunk
DEFAULT_CHUNK_OVERLAP = 100  # characters of overlap between adjacent chunks


class EmptyIndexError(RuntimeError):
    """Raised when retrieval is attempted on an index with no documents."""


class RetrievalError(RuntimeError):
    """Raised for retrieval failures (e.g. embedding service down)."""


@dataclass
class TextCandidate:
    """A ranked retrieval result, fully wired to its source for citations.

    ``chunk_index`` is the chunk's position within its page (0 when the page
    is small enough to be a single chunk). ``citation`` shows page-level
    provenance; the chunk index stays available internally for pinpointing.
    """

    document_id: str
    filename: str
    page_no: int
    text: str
    score: float
    source: str  # 'bm25' | 'dense' | 'fusion'
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_no": self.page_no,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }

    @property
    def citation(self) -> str:
        return f"{self.filename} · page {self.page_no}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _splitpoints(text: str) -> list[int]:
    """Offsets that mark good cutting points (paragraph then sentence ends)."""
    positions = []
    # Paragraph/sentence boundaries, preferring longer ones for cleaner cuts.
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"\u201c])|(?<=\n)\s*", text):
        positions.append(m.end())
    # Sort by preference: prefer paragraph breaks over sentence breaks.
    # We don't distinguish here; dedupe and keep ascending.
    positions = sorted(set(positions))
    return positions


def chunk_text(
    text: str,
    *,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into retrieval chunks at paragraph/sentence boundaries.

    - Text at or below ``size`` is returned as a single chunk.
    - Longer text is split at sentence/paragraph boundaries, accumulating
      segments until the target ``size`` is reached, then emitting a chunk.
      ``overlap`` characters from the previous chunk are carried forward so
      no context is lost across a cut.

    Args:
        text: cleaned page text.
        size: target characters per chunk.
        overlap: characters of overlap between consecutive chunks.

    Returns:
        list of chunk strings (non-empty).
    """
    text = _clean(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    overlap = max(0, min(overlap, max(0, size)))  # clamp overlap to size
    # Segments split at sentence/paragraph boundaries (preserving separators).
    boundaries = _splitpoints(text)
    seg_starts = [0] + boundaries
    seg_tails = boundaries + [len(text)]
    segments = [text[a:b].strip() for a, b in zip(seg_starts, seg_tails)]
    segments = [s for s in segments if s]

    chunks: list[str] = []
    cur = ""
    carry = ""  # overlap carried into the next chunk
    for seg in segments:
        candidate = cur + ("" if not cur else " ") + seg
        if len(candidate) <= size:
            cur = candidate
            continue
        # Current accumulator is full: flush it, then start the next chunk with
        # the trailing `overlap` characters of the chunk just emitted.
        if cur:
            chunks.append(cur.strip())
        # Carry the tail (word-safe) of this segment-forward as overlap.
        carry = cur.strip()[-overlap:] if overlap and cur else ""
        # Try to fit the segment into the new chunk (with carry).
        new_base = (carry + " " + seg) if carry else seg
        if len(new_base) <= size:
            cur = new_base
        else:
            # Segment alone exceeds size: hard-split it word-safely.
            if carry:
                chunks.append(carry.strip())
                carry = ""
                new_base = seg
            while len(new_base) > size:
                window = new_base[:size]
                m = re.search(r"(?<=\S)\s+(?=\S)", window)
                cut = (m.start() + 1) if m else size
                chunks.append(window[:cut].strip())
                new_base = new_base[cut:].strip()
            cur = new_base
    if cur and cur.strip():
        chunks.append(cur.strip())
    return chunks


def _chunk_page(page_index: int, text: str,
                chunk_size: int, chunk_overlap: int) -> list[tuple[int, str]]:
    """Return [(chunk_index, chunk_text), ...] for one page's cleaned text."""
    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
    return [(ci, c) for ci, c in enumerate(chunks)]


class TextIndex:
    """Keyed BM25 + in-memory dense vectors over the document store."""

    def __init__(self, store: DocumentStore, settings: Settings | None = None,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.store = store
        self.settings = settings or get_settings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
        """Rebuild the BM25 corpus over page text chunks.

        Each page becomes one or more chunks: short pages stay a single chunk,
        long pages are split via :func:`chunk_text` (preserving page provenance).
        """
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
                for ci, chunk in _chunk_page(page.page_no, text,
                                             self.chunk_size, self.chunk_overlap):
                    self._chunks.append({
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "page_no": page.page_no,
                        "chunk_index": ci,
                        "text": chunk,
                    })
                    self._texts.append(chunk)
        self._bm25.rebuild([tokenize(text) for text in self._texts])
        self._built_for = {d.document_id for d in docs}

    def is_built(self) -> bool:
        return len(self._chunks) > 0

    def doc_count(self) -> int:
        return len(self._built_for)

    def chunk_count(self) -> int:
        return len(self._chunks)

    def sample_texts(self, limit: int = 10) -> list[str]:
        """Return a diverse spread of chunk texts (one per page when possible)
        for downstream topic generation. Keeps only non-empty chunks and
        spreads across pages."""
        out: list[str] = []
        seen_pages: set[tuple] = set()
        for chunk in self._chunks:
            key = (chunk.get("document_id"), chunk.get("page_no"))
            if key in seen_pages:
                continue
            txt = str(chunk.get("text", "")).strip()
            if not txt:
                continue
            seen_pages.add(key)
            out.append(txt)
            if len(out) >= limit:
                break
        # If a single page has few chunks, top up from remaining chunks.
        if len(out) < limit and len(self._chunks):
            added = set()
            for chunk in self._chunks:
                if len(out) >= limit:
                    break
                txt = str(chunk.get("text", "")).strip()
                if not txt or any(txt == c for c in out):
                    continue
                out.append(txt)
        return out[:limit]

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def search_bm25(self, query: str, top_k: int = 8) -> list[TextCandidate]:
        """Keyword retrieval using pure BM25 over page-text chunks."""
        if not self.is_built():
            raise EmptyIndexError("Index is empty - ingest documents first.")
        results = self._bm25.search(_clean(query), top_k=top_k)
        return [
            TextCandidate(
                document_id=self._chunks[r.index]["document_id"],
                filename=self._chunks[r.index]["filename"],
                page_no=self._chunks[r.index]["page_no"],
                chunk_index=self._chunks[r.index]["chunk_index"],
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
                chunk_index=self._chunks[i]["chunk_index"],
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
    """Score-sum fusion over two ranked lists keyed by (doc_id, page_no, chunk).

    Each list's scores are min-max normalized to [0,1], then combined as
    ``weight*bm25 + (1-weight)*dense``. Candidates present in only one list get
    the other list's score = 0. Results are deduplicated by (doc_id, page,
    chunk_index) so multiple sub-chunks of the same page are kept distinct.
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
            chunk_index=cand.chunk_index,
            text=cand.text,
            score=score,
            source="fusion",
        ))
    return out


def _key(c: TextCandidate) -> tuple:
    return (c.document_id, c.page_no, c.chunk_index)
