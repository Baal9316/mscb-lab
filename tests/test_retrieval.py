"""Tests for Issue #2 text retrieval: BM25 + 9002 embeddings + fusion."""
from __future__ import annotations

import math

import pytest

from rag.bm25 import BM25, tokenize
from rag.config import Settings
from rag.embedding import EmbeddingError, get_text_embedding
from rag.models import Document, DocumentPage
from rag.retrieval import EmptyIndexError, RetrievalError, TextIndex, _fuse
from rag.store import DocumentStore


def _settings(tmp_path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(class_api_key="test-key", data_dir=data_dir)


SEED_PAGES = [
    ("lecture1.pdf", [
        "Support vector machines find the optimal separating hyperplane in high dimensional feature space.",
        "The kernel trick lets SVMs handle non linear decision boundaries.",
    ]),
    ("lecture2.pdf", [
        "Gradient boosting builds an ensemble of weak learners sequentially.",
        "Regularization reduces overfitting in gradient boosting models.",
    ]),
    ("notes.pdf", [
        "Neural networks learn hierarchical representations of data.",
        "Dropout is a regularization technique for neural networks.",
    ]),
]


def _seed_store(store: DocumentStore, pages=SEED_PAGES, doc_prefix="doc-"):
    added = []
    for i, (fname, texts) in enumerate(pages):
        doc_id = f"{doc_prefix}{i}"
        doc = Document(
            document_id=doc_id,
            filename=fname,
            sha256=f"sha{i}",
            pages=[
                DocumentPage(
                    document_id=doc_id, page_no=j + 1, filename=fname,
                    extracted_text=t,
                )
                for j, t in enumerate(texts)
            ],
        )
        store.add(doc)
        added.append(doc)
    return added


def _fake_embedder(settings=None):
    """Deterministic fake for get_text_embedding based on character codes."""
    def embed(texts, model=None, settings=None):
        vecs = []
        for t in texts:
            # stable pseudo-vector: value depends on chars
            base = sum(ord(c) for c in t)
            vec = [math.sin(i + base * 0.01) for i in range(16)]
            vecs.append(vec)
        from rag.embedding import EmbeddingResult
        return [EmbeddingResult(text=t, vector=v, index=i)
                for i, (t, v) in enumerate(zip(texts, vecs))]
    return embed


class TestBM25:
    def test_ranks_relevant_docs_first(self):
        bm = BM25([["vector", "database", "index"],
                   ["machine", "learning", "model"],
                   ["vector", "search", "similarity"]])
        res = bm.search("vector index", top_k=3)
        # doc1 ("machine learning model") shares no query terms -> zero score, dropped
        assert len(res) == 2
        by_doc = {r.index: r.score for r in res}
        assert by_doc[0] > by_doc[2]
        assert res[0].index == 0  # 'vector' + 'index' beats 'vector' alone

    def test_empty_query_returns_empty(self):
        bm = BM25([["a", "b"], ["c", "d"]])
        assert bm.search("") == []
        assert bm.search("   ", top_k=5) == []

    def test_tokenize_lowercases(self):
        assert tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]

    def test_rebuild_replaces_corpus(self):
        bm = BM25([["apple"], ["banana"]])
        bm.rebuild([["cherry"], ["apple"]])
        res = bm.search("apple")
        assert res and res[0].index == 1  # after rebuild, apple is idx 1


class TestEmbedClient:
    def test_payload_and_auth_header(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        captured = {}

        class _Resp:
            status_code = 200
            text = ""
            def json(self):
                return {"embeddings": {"float": [[0.1]*2048, [0.2]*2048]},
                        "texts": ["a", "b"]}
        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()
        monkeypatch.setattr("rag.embedding.requests.post", fake_post)

        res = get_text_embedding(["a", "b"], settings=s)
        assert captured["url"] == s.text_embedding_url
        assert captured["headers"]["Authorization"] == f"Bearer test-key"
        assert captured["json"]["texts"] == ["a", "b"]
        assert captured["json"]["model"] == "nvidia/Nemotron-3-Embed-1B-BF16"
        assert len(res) == 2 and res[0].dimension == 2048

    def test_raises_without_key(self, tmp_path):
        s = _settings(tmp_path)
        s = s.__class__(class_api_key="your_key_here", data_dir=s.data_dir)
        with pytest.raises(EmbeddingError):
            get_text_embedding(["x"], settings=s)

    def test_raises_on_http_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        class _Err:
            status_code = 500
            text = '{"error":"boom"}'
            def json(self):
                return {"error": "boom"}
        monkeypatch.setattr("rag.embedding.requests.post",
                            lambda *a, **k: _Err())
        with pytest.raises(EmbeddingError):
            get_text_embedding(["x"], settings=s)

    def test_raises_on_network_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        def boom(*a, **k):
            raise ConnectionError("refused")
        monkeypatch.setattr("rag.embedding.requests.post", boom)
        with pytest.raises(EmbeddingError):
            get_text_embedding(["x"], settings=s)

    def test_raises_on_count_mismatch(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        class _Resp:
            status_code = 200
            text = ""
            def json(self):
                return {"embeddings": {"float": [[0.1]*8]}, "texts": ["a"]}
        monkeypatch.setattr("rag.embedding.requests.post",
                            lambda *a, **k: _Resp())
        with pytest.raises(EmbeddingError):
            get_text_embedding(["a", "b"], settings=s)

    def test_empty_input_returns_empty(self, tmp_path):
        s = _settings(tmp_path)
        assert get_text_embedding([], settings=s) == []


class TestRetrievalMetadata:
    def _index(self, tmp_path):
        s = _settings(tmp_path)
        store = DocumentStore(s.data_dir)
        _seed_store(store)
        idx = TextIndex(store, settings=s)
        idx.rebuild()
        return idx

    def test_bm25_returns_metadata(self, tmp_path):
        idx = self._index(tmp_path)
        cands = idx.search_bm25("support vector machines hyperplane")
        assert cands
        c = cands[0]
        assert c.document_id == "doc-0"
        assert c.filename == "lecture1.pdf"
        assert c.page_no == 1
        assert c.source == "bm25"
        assert c.citation == "lecture1.pdf · page 1"
        assert "hyperplane" in c.text

    def test_dense_returns_metadata_and_scores(self, tmp_path, monkeypatch):
        idx = self._index(tmp_path)
        monkeypatch.setattr("rag.retrieval.get_text_embedding", _fake_embedder())
        cands = idx.search_dense("linear", top_k=3)
        assert cands
        for c in cands:
            assert c.source == "dense"
            assert c.document_id and c.filename and c.page_no
            assert 0.0 <= c.score <= 1.0

    def test_fusion_dedupes_and_keeps_metadata(self, tmp_path, monkeypatch):
        idx = self._index(tmp_path)
        monkeypatch.setattr("rag.retrieval.get_text_embedding", _fake_embedder())
        cands = idx.search("machine learning", top_k=4)
        assert cands
        assert all(c.source == "fusion" for c in cands)
        # no duplicate (doc_id, page)
        keys = [(c.document_id, c.page_no) for c in cands]
        assert len(keys) == len(set(keys))


class TestEdgeCases:
    def test_empty_index_raises_on_bm25(self, tmp_path):
        s = _settings(tmp_path)
        store = DocumentStore(s.data_dir)  # no docs
        idx = TextIndex(store, settings=s)
        idx.rebuild()
        assert not idx.is_built()
        with pytest.raises(EmptyIndexError):
            idx.search_bm25("anything")
        with pytest.raises(EmptyIndexError):
            idx.search("anything")

    def test_missing_deleted_document_rebuilds(self, tmp_path):
        s = _settings(tmp_path)
        store = DocumentStore(s.data_dir)
        _seed_store(store)
        idx = TextIndex(store, settings=s)
        idx.rebuild()
        assert idx.doc_count() == 3
        # delete a doc, rebuild, verify it disappears
        store.delete("doc-1")
        idx.rebuild()
        assert idx.doc_count() == 2
        cands = idx.search_bm25("gradient boosting")
        for c in cands:
            assert c.document_id != "doc-1"
            assert c.filename != "lecture2.pdf"

    def test_dense_service_failure_is_raised(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store = DocumentStore(s.data_dir)
        _seed_store(store)
        idx = TextIndex(store, settings=s)
        idx.rebuild()
        assert idx.is_built()

        def boom(*a, **k):
            raise EmbeddingError("9002 down")
        monkeypatch.setattr("rag.retrieval.get_text_embedding", boom)
        with pytest.raises(RetrievalError):
            idx.search_dense("machine")

    def test_empty_text_pages_skipped_from_index(self, tmp_path):
        """Blank slides (no extracted text) must not enter the index — otherwise
        9002 rejects empty prompts ('decoder prompt cannot be empty')."""
        s = _settings(tmp_path)
        store = DocumentStore(s.data_dir)
        doc = Document(
            document_id="d-blank", filename="blank.pdf", sha256="s1",
            pages=[
                # The 2nd and 3rd pages have empty text (blank slides)
                DocumentPage(document_id="d-blank", page_no=1, filename="blank.pdf",
                             extracted_text="This is a real slide with content."),
                DocumentPage(document_id="d-blank", page_no=2, filename="blank.pdf",
                             extracted_text="   \n  "),
                DocumentPage(document_id="d-blank", page_no=3, filename="blank.pdf",
                             extracted_text=""),
            ],
        )
        store.add(doc)
        idx = TextIndex(store, settings=s)
        idx.rebuild()
        assert idx.chunk_count() == 1  # only the non-empty page
        cands = idx.search_bm25("slide content")
        assert len(cands) == 1 and cands[0].page_no == 1


class TestFusionUnit:
    def test_fuse_weights(self):
        from rag.retrieval import TextCandidate
        def mk(doc, page, s):
            return TextCandidate(doc, f"{doc}.pdf", page, "t", s, "src")
        a = [mk("d0", 1, 1.0), mk("d1", 1, 0.8)]
        b = [mk("d1", 1, 1.0), mk("d0", 1, 0.6)]
        fused = _fuse(a, b, bm25_weight=0.5)
        scores = {c.document_id: c.score for c in fused}
        # d1: 0.5*0.0(bm25 mock already normalized)+0.5*1.0 ; d0 0.5*1+0.5*0
        # both present in both lists via different relative order
        assert set(scores) == {"d0", "d1"}
        assert all(c.source == "fusion" for c in fused)

    def test_fusion_single_candidate_normalizes_to_one(self):
        """A lone (single) candidate must not fuse to 0.0 — it's the top result."""
        from rag.retrieval import TextCandidate
        def mk(doc, page, s):
            return TextCandidate(doc, f"{doc}.pdf", page, "t", s, "src")
        one = [mk("d0", 1, 0.9)]
        fused = _fuse(one, one, bm25_weight=0.5)
        assert len(fused) == 1
        assert fused[0].document_id == "d0"
        assert fused[0].score == 1.0  # hi == lo -> mapped to 1.0, not 0.0

    def test_fusion_equal_scores_do_not_collapse_to_zero(self):
        """Multiple candidates all tied at the same score must stay 1.0 (tied
        top), deterministically, rather than being min-maxed to 0.0."""
        from rag.retrieval import TextCandidate
        def mk(doc, page, s):
            return TextCandidate(doc, f"{doc}.pdf", page, "t", s, "src")
        tied = [mk("d0", 1, 0.7), mk("d1", 1, 0.7), mk("d2", 1, 0.7)]
        fused = _fuse(tied, [], bm25_weight=1.0)  # pure BM25 path, equal scores
        assert len(fused) == 3
        # every tied candidate normalized to 1.0 (deterministic tie), never 0.0
        assert all(c.score == 1.0 for c in fused)
        assert {c.document_id for c in fused} == {"d0", "d1", "d2"}
