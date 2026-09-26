"""Tests for Issue #3 visual retrieval: 9003 client + lazy disk-cached VisualIndex.

The 9003 HTTP layer is mocked (offline, deterministic). Cache on-disk behavior
is exercised with real temp files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rag.config import Settings
from rag.models import Document, DocumentPage
from rag.store import DocumentStore
from rag.visual import VisualEmbeddingError, get_visual_embedding
from rag.visual_retrieval import (
    EmptyVisualIndexError,
    VisualIndex,
    VisualRetrievalError,
)


def _settings(tmp_path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(class_api_key="test-key", data_dir=data_dir)


def _png(path: Path, n: int = 1):
    """Write a tiny placeholder PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([n]) * 8)
    return path


def _dummy_url(path: Path) -> str:
    return f"data:image/png;base64,{path.read_bytes().hex()}"


def _seed_images(tmp_path, settings: Settings, pages_per_doc=2):
    """Seed N documents, each with images + data URLs."""
    store = DocumentStore(settings.data_dir)
    docs = []
    for di, fname in enumerate(["slidesA.pdf", "slidesB.pdf"]):
        doc_id = f"doc-{di}"
        doc_pages = []
        for p in range(1, pages_per_doc + 1):
            png = _png(tmp_path / f"img_{di}_{p}.png", p)
            doc_pages.append(DocumentPage(
                document_id=doc_id, filename=fname, page_no=p,
                extracted_text=f"Page {p} of {fname}",
                image_path=str(png),
                image_url=_dummy_url(png),
            ))
        doc = Document(document_id=doc_id, filename=fname, sha256=f"sha{di}", pages=doc_pages)
        store.add(doc)
        docs.append(doc)
    return store, docs


def _fake_embeddor(settings=None):
    """Deterministic fake: vector derived from the image/text content."""
    from rag.visual import VisualEmbedding

    def embed(*, image_url=None, text=None, model=None, settings=None):
        seed = (image_url or "") + (text or "")
        base = sum(ord(c) for c in seed)
        vec = [math.sin(i + base * 0.013) for i in range(16)]
        return VisualEmbedding(vector=vec, image_url=image_url or "", text=text or "")
    return embed


# --------------------------------------------------------------------------- #
# 9003 client
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body or {"data": [{"embedding": [0.5] * 2048}], "usage": {}}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class TestVisualClient:
    def test_image_only_payload(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        captured = {}
        def fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            captured["url"] = url
            captured["headers"] = headers
            return _Resp()
        monkeypatch.setattr("rag.visual.requests.post", fake_post)

        emb = get_visual_embedding(image_url="data:image/png;base64,AAAA", settings=s)
        body = captured["json"]
        assert body["model"] == "Qwen/Qwen3-VL-Embedding-2B"
        content = body["messages"][0]["content"]
        assert content == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert emb.dimension == 2048

    def test_text_only_payload(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        captured = {}
        monkeypatch.setattr("rag.visual.requests.post",
                            lambda url, json=None, headers=None, timeout=None: (
                                captured.__setitem__("json", json) or _Resp()))
        emb = get_visual_embedding(text="a slide about computers", settings=s)
        content = captured["json"]["messages"][0]["content"]
        assert content == [{"type": "text", "text": "a slide about computers"}]
        assert emb.dimension == 2048

    def test_mixed_payload(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        captured = {}
        monkeypatch.setattr("rag.visual.requests.post",
                            lambda url, json=None, headers=None, timeout=None: (
                                captured.__setitem__("json", json) or _Resp()))
        get_visual_embedding(image_url="data:image/png;base64,BB", text="cap", settings=s)
        content = captured["json"]["messages"][0]["content"]
        assert content == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB"}},
            {"type": "text", "text": "cap"},
        ]

    def test_requires_input(self, tmp_path):
        s = _settings(tmp_path)
        with pytest.raises(VisualEmbeddingError):
            get_visual_embedding(settings=s)

    def test_raises_without_key(self, tmp_path):
        s = _settings(tmp_path).__class__(class_api_key="your_key_here",
                                          data_dir=(tmp_path / "d"))
        with pytest.raises(VisualEmbeddingError):
            get_visual_embedding(text="x", settings=s)

    def test_raises_on_http_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        monkeypatch.setattr("rag.visual.requests.post",
                            lambda *a, **k: _Resp(status=500, body={"error": "boom"}))
        with pytest.raises(VisualEmbeddingError):
            get_visual_embedding(text="x", settings=s)

    def test_raises_on_network_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        def boom(*a, **k):
            raise ConnectionError("down")
        monkeypatch.setattr("rag.visual.requests.post", boom)
        with pytest.raises(VisualEmbeddingError):
            get_visual_embedding(text="x", settings=s)

    def test_raises_on_malformed(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        monkeypatch.setattr("rag.visual.requests.post",
                            lambda *a, **k: _Resp(body={"data": [{"embedding": "nope"}]}))
        with pytest.raises(VisualEmbeddingError):
            get_visual_embedding(text="x", settings=s)

        monkeypatch.setattr("rag.visual.requests.post",
                            lambda *a, **k: _Resp(body={"unexpected": 1}))
        with pytest.raises(VisualEmbeddingError):
            get_visual_embedding(text="x", settings=s)


# --------------------------------------------------------------------------- #
# VisualIndex: cache + generation
# --------------------------------------------------------------------------- #
class TestIndexCache:
    def test_generates_and_persists_cache_when_missing(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store, _ = _seed_images(tmp_path, s)
        calls = {"n": 0}
        def fake_embed(**kw):
            calls["n"] += 1
            return _fake_embeddor()(**kw)
        monkeypatch.setattr("rag.visual_retrieval.get_visual_embedding", fake_embed)

        idx = VisualIndex(store, settings=s)
        idx.build()
        assert idx.image_count() == 4
        assert calls["n"] == 4  # generated for each page
        # cache files persisted
        cache_files = list((store.document_dir("doc-0") / "index" / "visual").glob("*.json"))
        assert len(cache_files) == 2

    def test_loads_cache_no_repeat_call(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store, _ = _seed_images(tmp_path, s)
        calls = {"n": 0}
        def fake_embed(**kw):
            calls["n"] += 1
            return _fake_embeddor()(**kw)
        monkeypatch.setattr("rag.visual_retrieval.get_visual_embedding", fake_embed)

        idx = VisualIndex(store, settings=s)
        idx.build()
        assert calls["n"] == 4

        # Fresh index (simulating restart) -> caches exist, no re-calls
        idx2 = VisualIndex(store, settings=s)
        idx2.build()
        assert calls["n"] == 4  # unchanged: no repeated 9003 call
        assert idx2.image_count() == 4

    def test_cache_missing_one_page_regenerates_only_it(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store, _ = _seed_images(tmp_path, s)
        calls = {"n": 0}
        def fake_embed(**kw):
            calls["n"] += 1
            return _fake_embeddor()(**kw)
        monkeypatch.setattr("rag.visual_retrieval.get_visual_embedding", fake_embed)

        idx = VisualIndex(store, settings=s); idx.build()
        assert calls["n"] == 4
        # delete one cache file -> only that page regenerates
        f = (store.document_dir("doc-0") / "index" / "visual" / "page_001.json")
        f.unlink()
        idx2 = VisualIndex(store, settings=s); idx2.build()
        assert calls["n"] == 5  # only the missing one regenerated

    def test_delete_document_cleans_cache_and_index(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store, _ = _seed_images(tmp_path, s)
        # generate caches first
        idx = VisualIndex(store, settings=s)
        monkeypatch.setattr("rag.visual_retrieval.get_visual_embedding",
                            lambda **kw: _fake_embeddor()(**kw))
        idx.build()
        ddir = store.document_dir("doc-1")
        assert (ddir / "index" / "visual").exists()

        store.delete("doc-1")  # DocumentStore removes the whole doc dir
        assert not ddir.exists()  # cache gone
        idx.rebuild()  # reconcile
        assert idx.image_count() == 2  # only doc-0 remains
        assert all(r["document_id"] == "doc-0" for r in idx._records)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class TestRetrieval:
    def _index(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store, _ = _seed_images(tmp_path, s)
        idx = VisualIndex(store, settings=s)
        monkeypatch.setattr("rag.visual_retrieval.get_visual_embedding",
                            lambda **kw: _fake_embeddor()(**kw))
        return store, idx

    def test_text_to_image_ranking_and_metadata(self, tmp_path, monkeypatch):
        _, idx = self._index(tmp_path, monkeypatch)
        cands = idx.search_text_to_image("slide about files", top_k=3)
        assert cands
        c = cands[0]
        # every candidate carries full citation metadata
        assert c.document_id and c.filename and c.page_no
        assert c.image_path and c.image_url
        assert c.source == "visual"
        assert c.citation == f"{c.filename} · page {c.page_no}"
        assert -1.0 <= c.score <= 1.0  # cosine similarity

    def test_image_to_image_retrieval(self, tmp_path, monkeypatch):
        _, idx = self._index(tmp_path, monkeypatch)
        cands = idx.search_image_to_image("data:image/png;base64,aa", top_k=2)
        assert cands and len(cands) <= 2

    def test_empty_index_raises(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store = DocumentStore(s.data_dir)  # no docs
        idx = VisualIndex(store, settings=s)
        monkeypatch.setattr("rag.visual_retrieval.get_visual_embedding",
                            lambda **kw: _fake_embeddor()(**kw))
        with pytest.raises(EmptyVisualIndexError):
            idx.search_text_to_image("anything")

    def test_service_failure_is_raised(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store, _ = _seed_images(tmp_path, s)
        idx = VisualIndex(store, settings=s)

        def boom(**kw):
            raise VisualEmbeddingError("9003 down")
        monkeypatch.setattr("rag.visual_retrieval.get_visual_embedding", boom)
        with pytest.raises(VisualRetrievalError):
            idx.search_text_to_image("question")

    def test_deleted_doc_removed_from_results(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store, _ = _seed_images(tmp_path, s)
        idx = VisualIndex(store, settings=s)
        monkeypatch.setattr("rag.visual_retrieval.get_visual_embedding",
                            lambda **kw: _fake_embeddor()(**kw))
        store.delete("doc-0")
        idx.rebuild()
        cands = idx.search_text_to_image("query", top_k=5)
        assert all(c.document_id != "doc-0" for c in cands)
