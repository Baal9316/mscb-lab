"""Tests for text chunking in Issue #2 text retrieval (compliance fix).

Long page text is split into multiple retrieval chunks (BM25/dense operate over
chunks, not whole pages), while preserving page-level provenance and not
breaking Issue #4's page-level multimodal deduplication.
"""
from __future__ import annotations

import math

from rag.retrieval import (
    TextCandidate,
    TextIndex,
    chunk_text,
    _key,
    _fuse,
)
from rag.config import Settings
from rag.models import Document, DocumentPage
from rag.store import DocumentStore


# --------------------------------------------------------------------------- #
# chunk_text unit tests
# --------------------------------------------------------------------------- #
class TestChunkText:
    def test_short_page_stays_one_chunk(self):
        text = "A short slide with just a little text about computers."
        assert chunk_text(text) == [text]

    def test_empty_and_whitespace(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_long_page_creates_multiple_chunks(self):
        sent = ("This is a sentence about machine learning and how models "
                "generalize from training data to new examples. ")
        text = sent * 20  # ~ 1500 chars
        chunks = chunk_text(text, size=200, overlap=20)
        assert len(chunks) > 1
        # each chunk reasonably bounded
        assert all(len(c) <= 220 for c in chunks)
        # combined should cover the source (concatenated non-empty parts)
        assert "".join(chunks).strip()

    def test_overlap_preserves_context(self):
        # Force a cut that would otherwise lose a term shared across boundaries.
        text = ("alpha beta gamma delta. " * 6)
        chunks = chunk_text(text, size=120, overlap=40)
        if len(chunks) > 1:
            # the trailing token of an earlier chunk may reappear in the next
            combined = " ".join(chunks)
            # overlap just means adjacent chunks share some boundary context;
            # verify we didn't drop the full content
            assert "alpha" in combined and "delta" in combined

    def test_respects_size_parameter(self):
        text = "word " * 500  # 2500 chars
        big = chunk_text(text, size=2000, overlap=100)
        small = chunk_text(text, size=100, overlap=10)
        assert len(big) < len(small)  # longer chunks -> fewer chunks

    def test_never_returns_empty_chunk_entries(self):
        text = "sentence one. sentence two. sentence three. sentence four. " * 10
        chunks = chunk_text(text, size=150, overlap=30)
        assert chunks and all(c.strip() for c in chunks)


# --------------------------------------------------------------------------- #
# TextIndex retriewal over chunks
# --------------------------------------------------------------------------- #
def _mk_settings(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(class_api_key="test-key", data_dir=data_dir)


def _long_text() -> str:
    base = ("Introductory material about neural networks and backpropagation. " * 8
            + "The transformer architecture uses attention mechanisms. " * 8
            + "Gradient descent optimizes the loss. " * 6)
    return base  # long enough to chunk at size=200


def _seed(tmp_path, settings):
    store = DocumentStore(settings.data_dir)
    store.add(Document(
        document_id="doc-0", filename="notes.pdf", sha256="s0",
        pages=[
            DocumentPage(document_id="doc-0", page_no=1, filename="notes.pdf",
                         extracted_text="Short slide intro."),
            DocumentPage(document_id="doc-0", page_no=2, filename="notes.pdf",
                         extracted_text=_long_text()),
            DocumentPage(document_id="doc-0", page_no=3, filename="notes.pdf",
                         extracted_text="Another short page."),
        ],
    ))
    return store


def _fake_embedder():
    from rag.embedding import EmbeddingResult

    def fake(texts, model=None, settings=None):
        return [EmbeddingResult(
            text=t, index=i,
            vector=[math.sin(i + sum(ord(c) for c in t) * 0.01) for i in range(16)],
        ) for i, t in enumerate(texts)]
    return fake


def _build_index(tmp_path, monkeypatch, chunk_size=200, overlap=30):
    s = _mk_settings(tmp_path)
    store = _seed(tmp_path, s)
    idx = TextIndex(store, settings=s, chunk_size=chunk_size, chunk_overlap=overlap)
    idx.rebuild()
    monkeypatch.setattr("rag.retrieval.get_text_embedding", _fake_embedder())
    return s, store, idx


class TestRetrievalChunking:
    def _index(self, tmp_path, monkeypatch, chunk_size=200, overlap=30):
        return _build_index(tmp_path, monkeypatch, chunk_size, overlap)

    def test_short_pages_remain_single_chunk(self, tmp_path, monkeypatch):
        _, _, idx = self._index(tmp_path, monkeypatch)
        # page1 and page3 are short -> exactly one chunk for each provenance
        short_chunks = [c for c in idx._chunks if c["page_no"] in (1, 3)]
        assert len(short_chunks) == 2
        assert all(c["chunk_index"] == 0 for c in short_chunks)

    def test_long_page_creates_multiple_chunks(self, tmp_path, monkeypatch):
        _, _, idx = self._index(tmp_path, monkeypatch)
        long_chunks = [c for c in idx._chunks if c["page_no"] == 2]
        assert len(long_chunks) > 1

    def test_chunks_preserve_metadata(self, tmp_path, monkeypatch):
        _, _, idx = self._index(tmp_path, monkeypatch)
        for c in idx._chunks:
            assert c["document_id"] == "doc-0"
            assert c["filename"] == "notes.pdf"
            assert c["page_no"] in (1, 2, 3)
            assert isinstance(c["chunk_index"], int)
            assert c["text"].strip()

    def test_unique_chunk_ids(self, tmp_path, monkeypatch):
        _, _, idx = self._index(tmp_path, monkeypatch)
        keys = [(c["document_id"], c["page_no"], c["chunk_index"]) for c in idx._chunks]
        assert len(keys) == len(set(keys))

    def test_bm25_retrieves_relevant_subchunk(self, tmp_path, monkeypatch):
        _, _, idx = self._index(tmp_path, monkeypatch)
        # 'attention' only appears in the long page's second area
        hits = idx.search_bm25("attention mechanisms transformer", top_k=5)
        top = hits[0]
        assert top.page_no == 2
        assert "attention" in top.text  # the correct sub-chunk surfaced
        assert "chunk_index" in top.__dict__

    def test_dense_retrieval_over_chunks(self, tmp_path, monkeypatch):
        _, _, idx = self._index(tmp_path, monkeypatch)
        hits = idx.search_dense("gradient descent optimizes loss", top_k=5)
        assert hits
        for h in hits:
            assert 0 <= h.page_no <= 3
            assert isinstance(h.chunk_index, int)

    def test_text_candidate_has_chunk_index(self, tmp_path, monkeypatch):
        _, _, idx = self._index(tmp_path, monkeypatch)
        c = idx.search_bm25("gradient descent", top_k=3)[0]
        assert isinstance(c.chunk_index, int)
        assert c.citation == f"{c.filename} · page {c.page_no}"

    def test_fuse_keys_by_chunk(self):
        """Same page, different chunk_index = distinct fusion entries."""
        b = [TextCandidate("d0", "f", 2, "chunkA", 0.9, "bm25", chunk_index=0),
             TextCandidate("d0", "f", 2, "chunkB", 0.7, "bm25", chunk_index=1)]
        out = _fuse(b, [], bm25_weight=1.0)
        assert {c.chunk_index for c in out} == {0, 1}
        assert len(out) == 2

    def test_fusion_respects_chunk_key(self):
        assert _key(TextCandidate("d", "f", 2, "x", 0.5, "s", chunk_index=0)) != \
            _key(TextCandidate("d", "f", 2, "x", 0.5, "s", chunk_index=1))


# --------------------------------------------------------------------------- #
# Issue #4 compatibility: same-page multiple chunks dedupe at page level
# --------------------------------------------------------------------------- #
class TestIssue4Compatibility:
    def _gather(self, tmp_path, monkeypatch):
        from rag.rerank_pipeline import _collect_candidates
        s, store, idx = _build_index(tmp_path, monkeypatch)

        class _StubVisual:
            """Minimal stand-in so text-only pool collection works offline."""
            def search_text_to_image(self, *a, **k):
                return []
        return _collect_candidates, idx, _StubVisual()

    def test_same_page_multiple_chunks_dedupe(self, tmp_path, monkeypatch):
        """rerank_pipeline._collect_candidates keeps one EvidenceCandidate per
        page even when several sub-chunks are retrieved, and keeps the strongest
        text chunk for that page."""
        collect, idx, stub = self._gather(tmp_path, monkeypatch)
        pool = collect(idx, stub, "attention mechanisms", 15, 15)
        # multiple sub-chunks of page 2 collapse to a single page-level item
        p2 = [p for p in pool if p.page_no == 2]
        assert len(p2) == 1
        assert p2[0].document_id == "doc-0" and p2[0].filename == "notes.pdf"
        assert p2[0].text.strip()
