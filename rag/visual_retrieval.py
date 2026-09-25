"""Visual retrieval over slide/page images using 9003 Qwen3-VL embeddings.

Issue #3 scope. Independent of reranking (#4) and QA (#5), which build on top.

Design (approved):
- Visual embeddings are generated LAZILY (not at upload): when the index is
  first built/queried, each rendered page image produced by Milestone 1 is read,
  and we check for a cached per-page embedding first.
- A persistent cache lives under each document's indexed-content directory so
  that ``DocumentStore.delete()`` removes the cache with the document:
      <data_dir>/documents/<doc_id>/index/visual/<page_no>.json
- Loaded vectors are kept in memory for the life of the process, so a restart
  re-loads from disk instead of re-hitting 9003.

Prioritized: text-to-image retrieval (text query -> 9003 -> cosine-sort against
slide-image embeddings). Image-to-image is supported via the same client but kept
minimal.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .store import DocumentStore
from .visual import VisualEmbeddingError, get_visual_embedding


class EmptyVisualIndexError(RuntimeError):
    """Raised when visual retrieval is attempted on an index with no images."""


class VisualRetrievalError(RuntimeError):
    """Raised for visual retrieval failures (e.g. the 9003 service is down)."""


@dataclass
class VisualCandidate:
    """A ranked visual result, fully wired to its source for citations."""

    document_id: str
    filename: str
    page_no: int
    image_path: str
    image_url: str
    score: float
    source: str  # 'visual'
    text: str = ""  # optional page text, when available, for context

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_no": self.page_no,
            "image_path": self.image_path,
            "image_url": self.image_url,
            "score": self.score,
            "source": self.source,
            "text": self.text,
        }

    @property
    def citation(self) -> str:
        return f"{self.filename} · page {self.page_no}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class VisualIndex:
    """Lazy, disk-cached visual index over the document store."""

    def __init__(self, store: DocumentStore, settings: Settings | None = None):
        self.store = store
        self.settings = settings or get_settings()
        # Parallel arrays: one entry per indexed page image
        self._records: list[dict] = []
        self._vectors: list[list[float]] = []
        self._built = False
        self._cached_paths: set[Path] = set()

    # ------------------------------------------------------------------ #
    # Index construction (lazy)
    # ------------------------------------------------------------------ #
    def build(self, force_reload: bool = False) -> None:
        """Load or generate cached embeddings for every stored page image.

        For each page with a rendered image: if a cached vector exists on disk,
        load it; otherwise call 9003 once, save it under the document's
        ``index/visual/`` dir, and keep it in memory. ``force_reload`` rebuilds
        from scratch even if already loaded (used after deletions).
        """
        if self._built and not force_reload:
            return
        self._records = []
        self._vectors = []
        self._cached_paths = set()

        for doc in self.store.list_all():
            for page in doc.pages:
                if not page.image_path or not Path(page.image_path).exists():
                    continue
                vec = self._load_cached(doc.document_id, page.page_no)
                if vec is None:
                    vec = self._generate_and_cache(doc, page)
                self._records.append({
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "page_no": page.page_no,
                    "image_path": page.image_path,
                    "image_url": page.image_url,
                    "text": page.extracted_text,
                })
                self._vectors.append(vec)
        self._built = True

    def is_built(self) -> bool:
        return self._built and len(self._records) > 0

    def image_count(self) -> int:
        return len(self._records)

    def doc_count(self) -> int:
        return len({r["document_id"] for r in self._records})

    def rebuild(self) -> None:
        """Reconcile the in-memory index with the store (e.g. after a delete).

        When a document is deleted, ``DocumentStore.delete()`` removes its whole
        directory (including the ``index/visual`` cache). This drops it from the
        in-memory index and reloads any remaining cached vectors (no re-hitting
        9003 for pages whose cache survives).
        """
        # Drop cache files that no longer exist, rebuild fresh from disk/store.
        self._built = False
        self.build(force_reload=True)

    def cached_vector_count(self) -> int:
        """Number of page vectors currently materialized (memory + disk)."""
        return len(self._cached_paths)

    # ------------------------------------------------------------------ #
    # Disk cache helpers
    # ------------------------------------------------------------------ #
    def _cache_dir(self, document_id: str) -> Path:
        return self.store.document_dir(document_id) / "index" / "visual"

    def _cache_file(self, document_id: str, page_no: int) -> Path:
        return self._cache_dir(document_id) / f"page_{page_no:03d}.json"

    def _load_cached(self, document_id: str, page_no: int):
        """Return the cached vector for a page, or None if not cached."""
        path = self._cache_file(document_id, page_no)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            vec = data["vector"]
            if isinstance(vec, list) and vec and all(
                    isinstance(x, (int, float)) for x in vec):
                self._cached_paths.add(path)
                return [float(x) for x in vec]
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            # Corrupt/partial cache => treat as missing and regenerate.
            return None
        return None

    def _generate_and_cache(self, doc, page) -> list[float]:
        """Embed a page image via 9003 and persist the vector to its doc cache."""
        try:
            emb = get_visual_embedding(image_url=page.image_url, settings=self.settings)
        except VisualEmbeddingError as exc:
            raise VisualRetrievalError(f"Visual embedding failed: {exc}") from exc
        path = self._cache_file(doc.document_id, page.page_no)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vector": emb.vector, "document_id": doc.document_id,
                                    "page_no": page.page_no}), encoding="utf-8")
        self._cached_paths.add(path)
        return emb.vector

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def search_text_to_image(self, query: str, top_k: int = 8) -> list[VisualCandidate]:
        """Text query -> 9003 -> cosine-sort against slide-image embeddings."""
        self.build()
        if not self._records:
            raise EmptyVisualIndexError("Visual index is empty - ingest documents first.")
        qv = self._embed_query_text(_clean(query))
        return self._rank(qv, top_k=top_k)

    def search_image_to_image(self, image_url: str, top_k: int = 8) -> list[VisualCandidate]:
        """Image query -> 9003 -> cosine-sort against slide-image embeddings."""
        self.build()
        if not self._records:
            raise EmptyVisualIndexError("Visual index is empty - ingest documents first.")
        try:
            emb = get_visual_embedding(image_url=image_url, settings=self.settings)
        except VisualEmbeddingError as exc:
            raise VisualRetrievalError(f"Visual query embedding failed: {exc}") from exc
        return self._rank(emb.vector, top_k=top_k)

    # ------------------------------------------------------------------ #
    def _embed_query_text(self, text: str) -> list[float]:
        try:
            return get_visual_embedding(text=text, settings=self.settings).vector
        except VisualEmbeddingError as exc:
            raise VisualRetrievalError(f"Visual query embedding failed: {exc}") from exc

    def _rank(self, query_vec: list[float], top_k: int) -> list[VisualCandidate]:
        scored = _cosine_scores(query_vec, self._vectors)
        top = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:top_k]
        out = []
        for i in top:
            r = self._records[i]
            out.append(VisualCandidate(
                document_id=r["document_id"],
                filename=r["filename"],
                page_no=r["page_no"],
                image_path=r["image_path"],
                image_url=r["image_url"],
                score=scored[i],
                source="visual",
                text=r.get("text", ""),
            ))
        return out


def _cosine_scores(query_vec: list[float], vectors: list[list[float]]) -> list[float]:
    qn = _norm(query_vec)
    return [_dot(query_vec, v) / (qn * _norm(v)) if qn and _norm(v) else 0.0 for v in vectors]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))
