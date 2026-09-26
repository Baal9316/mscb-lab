"""Multimodal reranking: combine text + visual candidates, rerank via 9004.

Issue #4 scope. This module ties together the two prior retrieval routes:
  - :class:`rag.retrieval.TextIndex`   (Issue #2, text/BM25/dense)
  - :class:`rag.visual_retrieval.VisualIndex` (Issue #3, slide images)
and reranks a merged pool with the class 9004 multimodal reranker.

Flow for ``rerank_evidence``:
  1. Retrieve a WIDER pool from each index (default top 15 each) so the reranker
     sees enough raw candidates.
  2. Merge + dedupe by ``(document_id, page_no)``; a page present in both text
     and visual retrieval becomes ONE multimodal candidate carrying both text and
     image (and its image_path).
  3. Cap the combined pool (default 25).
  4. Convert candidates to the 9004 ``documents`` payload and rerank with
     ``top_n`` (default 5, configurable).
  5. Map ``results[i].index`` back to original candidates, re-attach citation
     metadata, tag ``source='rerank'``.
  On 9004 failure, fall back to Reciprocal Rank Fusion (RRF) over the text and
  visual pools (still multimodal) and tag results ``source='rerank-fallback'``.
  Raw text and visual scores are NEVER compared directly -- RRF works on ranks,
  so differing score scales are fine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Settings, get_settings
from .retrieval import EmptyIndexError, TextIndex
from .reranker import RerankError, rerank_candidates
from .visual_retrieval import EmptyVisualIndexError, VisualIndex


@dataclass
class EvidenceCandidate:
    """A single multimodal evidence item ready to rerank and to cite."""

    document_id: str
    filename: str
    page_no: int
    text: str = ""
    image_path: str = ""
    image_url: str = ""

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_no": self.page_no,
            "text": self.text,
            "image_path": self.image_path,
            "image_url": self.image_url,
        }

    @property
    def citation(self) -> str:
        return f"{self.filename} · page {self.page_no}"

    @property
    def is_multimodal(self) -> bool:
        return bool(self.text) and bool(self.image_url)

    def to_rerank_document(self) -> str | dict:
        """Convert this candidate into the 9004 ``documents`` payload form.

        Text-only -> plain string; image-only -> {content:[image_url]};
        text+image -> {content:[text, image_url]}.
        """
        content: list[dict] = []
        if self.text:
            content.append({"type": "text", "text": self.text})
        if self.image_url:
            content.append({"type": "image_url", "image_url": {"url": self.image_url}})
        if not content:
            content.append({"type": "text", "text": ""})
        if len(content) == 1 and content[0]["type"] == "text":
            return content[0]["text"]
        return {"content": content}


@dataclass
class RerankedEvidence:
    candidate: EvidenceCandidate
    score: float
    source: str  # 'rerank' | 'rerank-fallback'

    def to_dict(self) -> dict:
        d = self.candidate.to_dict()
        d["score"] = self.score
        d["source"] = self.source
        return d

    @property
    def citation(self) -> str:
        return self.candidate.citation


def _collect_candidates(text_idx: TextIndex, visual_idx: VisualIndex,
                        query: str, text_pool: int, visual_pool: int) -> list[EvidenceCandidate]:
    """Gather a wider pool from both indexes and dedupe by (doc_id, page_no).

    Empty indexes (either route) are handled gracefully: if an index raises
    EmptyIndexError/EmptyVisualIndexError (no docs), it simply contributes no
    candidates rather than failing the whole rerank.
    """
    merged: dict[tuple, EvidenceCandidate] = {}

    # Text candidates (BM25 + dense fused already inside TextIndex.search)
    try:
        text_hits = text_idx.search(query, top_k=text_pool)
    except EmptyIndexError:
        text_hits = []
    for tc in text_hits:
        key = (tc.document_id, tc.page_no)
        if key not in merged:
            merged[key] = EvidenceCandidate(
                document_id=tc.document_id,
                filename=tc.filename,
                page_no=tc.page_no,
                text=tc.text,
            )
        else:
            if tc.text and not merged[key].text:
                merged[key].text = tc.text

    # Visual candidates
    try:
        visual_hits = visual_idx.search_text_to_image(query, top_k=visual_pool)
    except EmptyVisualIndexError:
        visual_hits = []
    for vc in visual_hits:
        key = (vc.document_id, vc.page_no)
        if key not in merged:
            merged[key] = EvidenceCandidate(
                document_id=vc.document_id,
                filename=vc.filename,
                page_no=vc.page_no,
                text=vc.text,
                image_path=vc.image_path,
                image_url=vc.image_url,
            )
        else:
            # enrich an existing text-only candidate with the image
            merged[key].image_path = vc.image_path
            merged[key].image_url = vc.image_url
            if vc.text and not merged[key].text:
                merged[key].text = vc.text

    if not merged:
        return []
    # Deterministic ordering of the pool (by doc id, page) before capping.
    ordered = sorted(merged.values(), key=lambda c: (c.document_id, c.page_no))
    return ordered


def _rrf_score(lists: list[list[EvidenceCandidate]], k: float = 60.0) -> dict[tuple, float]:
    """Reciprocal Rank Fusion over rank-ordered candidate lists.

    Only ranks matter, so text and visual score scales are never compared.
    ``k=60`` is the standard RRF constant.
    """
    scores: dict[tuple, float] = {}
    for ranked in lists:
        for rank, cand in enumerate(ranked, start=1):
            key = (cand.document_id, cand.page_no)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def _summarize(query: str, text_idx: TextIndex, visual_idx: VisualIndex, pool: int):
    """Return named, rank-ordered candidate lists for RRF fallback."""
    try:
        t = text_idx.search(query, top_k=pool)
    except EmptyIndexError:
        t = []
    try:
        v = visual_idx.search_text_to_image(query, top_k=pool)
    except EmptyVisualIndexError:
        v = []
    t_cands = [
        EvidenceCandidate(document_id=c.document_id, filename=c.filename,
                          page_no=c.page_no, text=c.text) for c in t
    ]
    v_cands = [
        EvidenceCandidate(document_id=c.document_id, filename=c.filename,
                          page_no=c.page_no, text=c.text,
                          image_path=c.image_path, image_url=c.image_url) for c in v
    ]
    return t_cands, v_cands


def rerank_evidence(
    query: str,
    text_idx: TextIndex,
    visual_idx: VisualIndex,
    *,
    top_n: int = 5,
    text_pool: int = 15,
    visual_pool: int = 15,
    max_pool: int = 25,
    reranker=rerank_candidates,
    settings: Settings | None = None,
) -> list[RerankedEvidence]:
    """Combine + rerank text and visual candidates via 9004, with RRF fallback.

    Args:
        query: user question.
        text_idx: built Issue #2 TextIndex.
        visual_idx: built Issue #3 VisualIndex.
        top_n: final evidence count returned (default 5, configurable).
        text_pool / visual_pool: pre-rerank pool size per index (default 15).
        max_pool: cap on combined candidates sent to the reranker (default 25).
        reranker: injectable reranker function (defaults to 9004 client) for tests.
        settings: injectable settings.

    Returns:
        list[RerankedEvidence] best-first, with citation metadata intact.
    """
    pool_candidates = _collect_candidates(text_idx, visual_idx, query,
                                          text_pool, visual_pool)
    if not pool_candidates:
        return []

    pool_candidates = pool_candidates[:max_pool]

    # Build the 9004 documents payload preserving input order.
    documents = [c.to_rerank_document() for c in pool_candidates]

    try:
        outcomes = reranker(query, documents, top_n=top_n, settings=settings)
    except RerankError:
        # Graceful multimodal fallback: RRF over text + visual rank lists.
        t_cands, v_cands = _summarize(query, text_idx, visual_idx, max_pool)
        rrf = _rrf_score([t_cands, v_cands])
        # Build candidate lookup by key from the merged pool
        by_key = {(c.document_id, c.page_no): c for c in pool_candidates}
        # Also include candidates from the RRF lists if not in the capped pool
        for c in t_cands + v_cands:
            by_key.setdefault((c.document_id, c.page_no), c)
        ranked_keys = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [
            RerankedEvidence(candidate=by_key[key], score=score, source="rerank-fallback")
            for key, score in ranked_keys if key in by_key
        ]

    # Map results[i].index back to pool position, keep citation metadata.
    result = []
    for o in outcomes:
        if 0 <= o.index < len(pool_candidates):
            cand = pool_candidates[o.index]
            result.append(RerankedEvidence(candidate=cand, score=o.score, source="rerank"))
    # If 9004 returned nothing usable, fall back too.
    if not result:
        t_cands, v_cands = _summarize(query, text_idx, visual_idx, max_pool)
        rrf = _rrf_score([t_cands, v_cands])
        by_key = {(c.document_id, c.page_no): c for c in pool_candidates + t_cands + v_cands}
        ranked_keys = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        result = [RerankedEvidence(candidate=by_key[key], score=s, source="rerank-fallback")
                  for key, s in ranked_keys if key in by_key]
    return result
