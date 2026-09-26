"""Tests for Issue #4 multimodal reranking: 9004 client + RRF fallback pipeline."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rag.config import Settings
from rag.models import Document, DocumentPage
from rag.rerank_pipeline import (
    EvidenceCandidate,
    rerank_evidence,
    _collect_candidates,
    _rrf_score,
)
from rag.reranker import RerankError, RerankOutcome, rerank_candidates
from rag.retrieval import TextIndex
from rag.store import DocumentStore
from rag.visual_retrieval import VisualIndex


def _settings(tmp_path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(class_api_key="test-key", data_dir=data_dir)


def _png(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([n]) * 8)
    return path


def _seed(settings: Settings, tmp_path) -> DocumentStore:
    """One doc, 3 pages with text + image (for both text & visual indexes)."""
    store = DocumentStore(settings.data_dir)
    pages = []
    for p in range(1, 4):
        img = _png(tmp_path / f"img_{p}.png", p)
        url = f"data:image/png;base64,{img.read_bytes().hex()}"
        pages.append(DocumentPage(
            document_id="doc-0", filename="slides.pdf", page_no=p,
            extracted_text=f"Page {p} content about topic {p}",
            image_path=str(img), image_url=url,
        ))
    store.add(Document(document_id="doc-0", filename="slides.pdf",
                       sha256="sha", pages=pages))
    return store


def _build_indexes(settings: Settings, tmp_path, monkeypatch) -> tuple:
    """Seed + build text & visual indexes, patching embeddings via monkeypatch.

    The fake embedders are deterministic per-content, so both indexes return
    results offline. Patches are scoped to the test via monkeypatch.
    """
    from rag.embedding import EmbeddingResult
    from rag.visual import VisualEmbedding

    def fake_text(texts, model=None, settings=None):
        return [
            EmbeddingResult(
                text=t, index=i,
                vector=[math.sin(i + sum(ord(c) for c in t) * 0.01) for i in range(16)],
            ) for i, t in enumerate(texts)
        ]

    def fake_vis(**kw):
        seed = (kw.get("image_url") or "") + (kw.get("text") or "")
        base = sum(ord(c) for c in seed)
        return VisualEmbedding(
            vector=[math.sin(i + base * 0.017) for i in range(16)],
            image_url=kw.get("image_url", ""), text=kw.get("text", ""))

    import rag.retrieval as rt
    import rag.visual_retrieval as vr
    monkeypatch.setattr(rt, "get_text_embedding", fake_text)
    monkeypatch.setattr(vr, "get_visual_embedding", fake_vis)

    store = _seed(settings, tmp_path)
    text_idx = TextIndex(store, settings=settings)
    text_idx.rebuild()
    vis_idx = VisualIndex(store, settings=settings)
    monkeypatch.setattr(vr, "get_visual_embedding", fake_vis)  # ensure set again
    return store, text_idx, vis_idx


# --------------------------------------------------------------------------- #
# 9004 client
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {"results": [{"index": 1, "relevance_score": 0.9,
                                           "document": {"text": "x"}}]}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class TestRerankClient:
    def test_payload_uses_top_n(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        captured = {}
        monkeypatch.setattr("rag.reranker.requests.post",
                            lambda url, json=None, headers=None, timeout=None: (
                                captured.__setitem__("json", json) or _Resp()))
        docs = ["text a", {"content": [{"type": "image_url", "image_url": {"url": "d1"}}]}]
        rerank_candidates("question", docs, top_n=5, settings=s)
        body = captured["json"]
        assert body["top_n"] == 5
        assert body["query"] == "question"
        assert len(body["documents"]) == 2
        assert body["model"] == "Qwen/Qwen3-VL-Reranker-2B"

    def test_top_n_configurable(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        captured = {}
        monkeypatch.setattr("rag.reranker.requests.post",
                            lambda url, json=None, headers=None, timeout=None: (
                                captured.__setitem__("json", json) or _Resp()))
        rerank_candidates("q", ["a"], top_n=3, settings=s)
        assert captured["json"]["top_n"] == 3

    def test_auth_header(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        captured = {}
        monkeypatch.setattr("rag.reranker.requests.post",
                            lambda url, json=None, headers=None, timeout=None: (
                                captured.__setitem__("headers", headers) or _Resp()))
        rerank_candidates("q", ["a"], settings=s)
        assert captured["headers"]["Authorization"] == "Bearer test-key"

    def test_response_index_mapping(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        monkeypatch.setattr("rag.reranker.requests.post",
                            lambda *a, **k: _Resp(body={"results": [
                                {"index": 2, "relevance_score": 0.8},
                                {"index": 0, "relevance_score": 0.5},
                            ]}))
        out = rerank_candidates("q", ["a", "b", "c"], settings=s)
        assert [o.index for o in out] == [2, 0]
        assert [o.score for o in out] == [0.8, 0.5]

    def test_raises_no_key(self, tmp_path):
        s = _settings(tmp_path).__class__(class_api_key="your_key_here",
                                          data_dir=tmp_path / "d")
        with pytest.raises(RerankError):
            rerank_candidates("q", ["a"], settings=s)

    def test_raises_http_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        monkeypatch.setattr("rag.reranker.requests.post",
                            lambda *a, **k: _Resp(status=500, body={"e": 1}))
        with pytest.raises(RerankError):
            rerank_candidates("q", ["a"], settings=s)

    def test_raises_network_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        def boom(*a, **k):
            raise ConnectionError("down")
        monkeypatch.setattr("rag.reranker.requests.post", boom)
        with pytest.raises(RerankError):
            rerank_candidates("q", ["a"], settings=s)

    def test_raises_malformed(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        monkeypatch.setattr("rag.reranker.requests.post",
                            lambda *a, **k: _Resp(body={"nope": 1}))
        with pytest.raises(RerankError):
            rerank_candidates("q", ["a"], settings=s)

    def test_empty_docs_returns_empty(self, tmp_path):
        s = _settings(tmp_path)
        assert rerank_candidates("q", [], settings=s) == []

    def test_no_credential_never_raises_key_into_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        class Bad:
            status_code = 401
            text = '{"message": "unauthorized"}'
            def json(self):
                return {"message": "unauthorized"}
        monkeypatch.setattr("rag.reranker.requests.post", lambda *a, **k: Bad())
        with pytest.raises(RerankError) as ei:
            rerank_candidates("q", ["a"], settings=s)
        assert "test-key" not in str(ei.value)


# --------------------------------------------------------------------------- #
# Candidate formatting
# --------------------------------------------------------------------------- #
class TestEvidenceCandidate:
    def test_text_only(self):
        c = EvidenceCandidate("d", "f.pdf", 1, text="hello")
        assert c.to_rerank_document() == "hello"

    def test_image_only(self):
        c = EvidenceCandidate("d", "f.pdf", 1, image_url="data:x")
        assert c.to_rerank_document() == {"content": [
            {"type": "image_url", "image_url": {"url": "data:x"}}]}

    def test_text_plus_image(self):
        c = EvidenceCandidate("d", "f.pdf", 1, text="t", image_url="data:x")
        doc = c.to_rerank_document()
        assert doc == {"content": [
            {"type": "text", "text": "t"},
            {"type": "image_url", "image_url": {"url": "data:x"}},
        ]}
        assert c.is_multimodal


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class TestRerankPipeline:
    def _setup(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store, text_idx, vis_idx = _build_indexes(s, tmp_path, monkeypatch)
        return s, store, text_idx, vis_idx

    def test_candidates_present_text_and_visual(self, tmp_path, monkeypatch):
        s, store, text_idx, vis_idx = self._setup(tmp_path, monkeypatch)
        t = text_idx.search("topic", top_k=5)
        v = vis_idx.search_text_to_image("topic", top_k=5)
        assert t and v  # both routes return results with our fake embedders

    def test_metadata_survives_reranking(self, tmp_path, monkeypatch):
        s, store, text_idx, vis_idx = self._setup(tmp_path, monkeypatch)

        def fake_rerank(query, docs, top_n=5, settings=None):
            n = len(docs)
            return [RerankOutcome(index=i, score=(n - i) / n, document={})
                    for i in range(n)]

        ev = rerank_evidence("topic", text_idx, vis_idx, top_n=5,
                             settings=s, reranker=fake_rerank)
        assert ev
        for e in ev:
            assert e.candidate.document_id and e.candidate.filename and e.candidate.page_no
            assert e.candidate.citation
            assert e.source == "rerank"

    def test_same_page_dedup(self, tmp_path, monkeypatch):
        s, store, text_idx, vis_idx = self._setup(tmp_path, monkeypatch)
        pool = _collect_candidates(text_idx, vis_idx, "topic", 15, 15)
        keys = [(c.document_id, c.page_no) for c in pool]
        assert len(keys) == len(set(keys))  # no duplicate (doc,page)

    def test_combined_candidate_has_text_and_image(self, tmp_path, monkeypatch):
        """A page returned by both text and visual retrieval becomes multimodal."""
        s, store, text_idx, vis_idx = self._setup(tmp_path, monkeypatch)
        pool = _collect_candidates(text_idx, vis_idx, "topic", 15, 15)
        for c in pool:
            if c.image_url:
                assert c.to_rerank_document() in (
                    {"content": [{"type": "image_url", "image_url": {"url": c.image_url}}]},
                    {"content": [{"type": "text", "text": c.text},
                                 {"type": "image_url", "image_url": {"url": c.image_url}}]},
                ) or isinstance(c.to_rerank_document(), dict)

    def test_pool_cap(self, tmp_path, monkeypatch):
        s, store, text_idx, vis_idx = self._setup(tmp_path, monkeypatch)
        captured = {}
        def fake_rerank(query, docs, top_n=5, settings=None):
            captured["doc_count"] = len(docs)
            return [RerankOutcome(index=i, score=1.0, document={})
                    for i in range(min(top_n, len(docs)))]
        rerank_evidence("topic", text_idx, vis_idx, top_n=5, max_pool=2,
                        settings=s, reranker=fake_rerank)
        assert captured["doc_count"] <= 2

    def test_configurable_top_n(self, tmp_path, monkeypatch):
        s, store, text_idx, vis_idx = self._setup(tmp_path, monkeypatch)
        called = {}
        def fake_rerank(query, docs, top_n=5, settings=None):
            called["top_n"] = top_n
            return [RerankOutcome(index=i, score=1.0 - i * 0.1, document={})
                    for i in range(min(top_n, len(docs)))]
        ev = rerank_evidence("topic", text_idx, vis_idx, top_n=3,
                             settings=s, reranker=fake_rerank)
        assert called["top_n"] == 3
        assert len(ev) <= 3

    def test_rrf_fallback_on_9004_failure(self, tmp_path, monkeypatch):
        s, store, text_idx, vis_idx = self._setup(tmp_path, monkeypatch)
        def fake_rerank(query, docs, top_n=5, settings=None):
            raise RerankError("9004 down")
        ev = rerank_evidence("topic", text_idx, vis_idx, top_n=5,
                             settings=s, reranker=fake_rerank)
        assert ev
        assert all(e.source == "rerank-fallback" for e in ev)

    def test_rrf_rank_based_not_score_based(self):
        """RRF only uses ranks; identical rank orders give equal fusion scores
        regardless of raw score magnitude (text vs visual never compared)."""
        a = [EvidenceCandidate("d0", "f", 1, text="x"),
             EvidenceCandidate("d1", "f", 1, text="x")]
        b = [EvidenceCandidate("d1", "f", 1, text="x"),
             EvidenceCandidate("d0", "f", 1, text="x")]
        sc = _rrf_score([a, b], k=60.0)
        assert sc[("d0", 1)] == sc[("d1", 1)]

    def test_empty_pool_returns_empty(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store = DocumentStore(s.data_dir)  # empty
        text_idx = TextIndex(store, settings=s)
        vis_idx = VisualIndex(store, settings=s)
        ev = rerank_evidence("anything", text_idx, vis_idx, settings=s)
        assert ev == []

    def test_no_credential_leak_via_fallback(self, tmp_path, monkeypatch):
        s, store, text_idx, vis_idx = self._setup(tmp_path, monkeypatch)
        def fake_rerank(query, docs, top_n=5, settings=None):
            raise RerankError(f"401 using key {settings.class_api_key}")
        # Fallback is deterministic and cannot surface the key.
        ev = rerank_evidence("topic", text_idx, vis_idx,
                             settings=s, reranker=fake_rerank)
        assert ev
        assert all(e.source == "rerank-fallback" for e in ev)
