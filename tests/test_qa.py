"""Tests for Issue #5 question answering + citations.

Covers the 9001 client, strict source-id validation, grounding behavior, and
the end-to-end answer_question pipeline (with 9001 + retrieval mocked).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rag.config import Settings
from rag.models import Document, DocumentPage
from rag.qa import (
    QAGenerationError,
    Source,
    StructuredAnswer,
    answer_question,
    build_sources,
    generate_llm_response,
    parse_structured_answer,
    prompt_from_evidence,
)
from rag.rerank_pipeline import EvidenceCandidate, RerankedEvidence
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


def _seed(tmp_path, settings: Settings) -> DocumentStore:
    store = DocumentStore(settings.data_dir)
    pages = []
    for p in range(1, 4):
        img = _png(tmp_path / f"img_{p}.png", p)
        url = f"data:image/png;base64,{img.read_bytes().hex()}"
        pages.append(DocumentPage(
            document_id="doc-0", filename="slides.pdf", page_no=p,
            extracted_text=f"Page {p} content with some course material about topic {p}",
            image_path=str(img), image_url=url))
    store.add(Document(document_id="doc-0", filename="slides.pdf",
                       sha256="sha", pages=pages))
    return store


def _build_evidence(n=3):
    """Synthetic reranked evidence, independent of retrieval, for unit tests."""
    ev = []
    for i in range(n):
        ev.append(RerankedEvidence(
            candidate=EvidenceCandidate(
                document_id=f"doc-{i}", filename=f"deck{i}.pdf", page_no=i + 1,
                text=f"Evidence text number {i + 1} about course material",
                image_path=f"/tmp/page_{i+1}.png",
                image_url=f"data:image/png;base64,img{i + 1}"),
            score=round(1.0 - i * 0.1, 3), source="rerank"))
    return ev


# --------------------------------------------------------------------------- #
# 9001 client
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status=200, content="ok"):
        self.status_code = status
        self._content = content
        self.text = json.dumps({"choices": [{"message": {"content": content}}]})

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class TestLLMClient:
    def test_payload_shape_and_auth(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        captured = {}
        def fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            captured["headers"] = headers
            captured["url"] = url
            return _Resp(content="hello")
        monkeypatch.setattr("rag.qa.requests.post", fake_post)
        out = generate_llm_response([{"role": "user", "content": "hi"}], settings=s)
        assert out == "hello"
        assert captured["url"] == s.llm_url
        assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert captured["json"]["max_tokens"] == 500
        assert captured["json"]["model"] == "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
        assert captured["headers"]["Authorization"] == "Bearer test-key"

    def test_raises_no_key(self, tmp_path):
        s = _settings(tmp_path).__class__(class_api_key="your_key_here",
                                          data_dir=tmp_path / "d")
        with pytest.raises(QAGenerationError):
            generate_llm_response([{"role": "user", "content": "x"}], settings=s)

    def test_raises_http_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        monkeypatch.setattr("rag.qa.requests.post",
                            lambda *a, **k: _Resp(status=500, content="boom"))
        with pytest.raises(QAGenerationError):
            generate_llm_response([{"role": "user", "content": "x"}], settings=s)

    def test_raises_network_error(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        def boom(*a, **k):
            raise ConnectionError("down")
        monkeypatch.setattr("rag.qa.requests.post", boom)
        with pytest.raises(QAGenerationError):
            generate_llm_response([{"role": "user", "content": "x"}], settings=s)

    def test_raises_empty_content(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        monkeypatch.setattr("rag.qa.requests.post",
                            lambda *a, **k: _Resp(content="   "))
        with pytest.raises(QAGenerationError):
            generate_llm_response([{"role": "user", "content": "x"}], settings=s)

    def test_raises_malformed(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        class _Bad:
            status_code = 200
            text = "not json"
            def json(self):
                return {"no": "choices"}
        monkeypatch.setattr("rag.qa.requests.post", lambda *a, **k: _Bad())
        with pytest.raises(QAGenerationError):
            generate_llm_response([{"role": "user", "content": "x"}], settings=s)


# --------------------------------------------------------------------------- #
# parse_structured_answer + source validation
# --------------------------------------------------------------------------- #
class TestParsing:
    def test_parse_grounded(self):
        r = parse_structured_answer('{"answer": "Temp controls sampling.", "source_ids": [1,3], "insufficient": false}')
        assert r["answer"] == "Temp controls sampling."
        assert r["source_ids"] == [1, 3]
        assert r["insufficient"] is False

    def test_parse_code_fence(self):
        r = parse_structured_answer('```json\n{"answer": "x", "source_ids": [2], "insufficient": false}\n```')
        assert r["source_ids"] == [2]

    def test_parse_insufficient(self):
        r = parse_structured_answer('{"answer": "not enough info", "source_ids": [], "insufficient": true}')
        assert r["insufficient"] is True
        assert r["source_ids"] == []

    def test_parse_insufficient_forces_empty_ids(self):
        r = parse_structured_answer('{"answer": "a", "source_ids": [1], "insufficient": true}')
        assert r["source_ids"] == []  # insufficient clears ids

    def test_parse_no_json_raises(self):
        with pytest.raises(QAGenerationError):
            parse_structured_answer("the model just rambled with no JSON")

    def test_parse_unbalanced_raises(self):
        with pytest.raises(QAGenerationError):
            parse_structured_answer('{"answer": "unbalanced')

    def test_parse_invalid_json_raises(self):
        with pytest.raises(QAGenerationError):
            parse_structured_answer('{"answer": oops}')


class TestSourceValidation:
    def test_valid_ids_build_sources(self):
        ev = _build_evidence(3)
        src = build_sources([1, 3], ev)
        assert len(src) == 2
        assert src[0].document_id == "doc-0" and src[0].page_no == 1
        assert src[0].filename == "deck0.pdf"
        assert src[1].document_id == "doc-2"
        assert src[0].citation == "deck0.pdf · page 1"

    def test_out_of_range_and_negative_rejected(self):
        ev = _build_evidence(2)
        src = build_sources([0, 99, -3, 1], ev)  # 0 invalid, 99 invalid, -3 invalid
        assert len(src) == 1 and src[0].page_no == 1

    def test_duplicate_ids_deduped_preserving_order(self):
        """[1,1,2] must dedupe to [1,2], order preserved, so the UI never
        shows the same source twice."""
        ev = _build_evidence(3)
        src = build_sources([1, 1, 2], ev)
        assert len(src) == 2
        assert [s.page_no for s in src] == [1, 2]  # order preserved, deduped
        # also [1,2,2,1] 
        src2 = build_sources([1, 2, 2, 1], ev)
        assert [s.page_no for s in src2] == [1, 2]

    def test_source_fields(self):
        ev = _build_evidence(1)
        src = build_sources([1], ev)[0]
        assert src.document_id and src.filename and src.page_no
        assert src.excerpt
        assert src.image_path and src.image_url
        assert src.citation


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
class TestPrompt:
    def test_includes_question_and_images(self):
        ev = _build_evidence(3)
        msgs = prompt_from_evidence("what is temperature?", ev)
        # system + user
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        content = msgs[1]["content"]
        text_parts = [c for c in content if c["type"] == "text"]
        img_parts = [c for c in content if c["type"] == "image_url"]
        assert any("what is temperature?" in c["text"] for c in text_parts)
        assert len(img_parts) == 3  # 3 evidence items carry images

    def test_uses_indices_not_filenames_in_prompt_model_contract(self):
        ev = _build_evidence(2)
        msgs = prompt_from_evidence("q", ev)
        user_text = next(c["text"] for c in msgs[1]["content"] if c["type"] == "text")
        assert "[1]" in user_text and "[2]" in user_text

    def test_system_prompt_forbids_citation_metadata_in_answer(self):
        """The model is told never to author filenames/pages/citation text."""
        from rag.qa import _SYSTEM_PROMPT
        p = _SYSTEM_PROMPT.lower()
        # Explicit prohibitions present
        assert "do not mention" in p
        assert "page numbers" in p or "slide numbers" in p
        assert "citation" in p
        assert "source_ids" in p
        assert "never invent" in p

    def test_images_labeled_with_evidence_index(self):
        """Each slide image must be tied to its numbered evidence id so the model
        cannot attribute an image to the wrong source."""
        ev = _build_evidence(3)
        msgs = prompt_from_evidence("q", ev)
        content = msgs[1]["content"]
        text_items = [c for c in content if c["type"] == "text"]
        assert any("Slide image for evidence [1]:" in t["text"] for t in text_items)
        assert any("Slide image for evidence [2]:" in t["text"] for t in text_items)
        assert any("Slide image for evidence [3]:" in t["text"] for t in text_items)
        imgs = [c for c in content if c["type"] == "image_url"]
        assert len(imgs) == 3

    def test_prompt_contains_no_citation_metadata(self):
        """The 9001 model must never see filenames, page numbers, paths, citation
        strings, or document ids — only evidence index + text + image.

        Checks the USER evidence content specifically (the system prompt
        necessarily mentions 'page numbers' only to forbid them, so it is not
        the subject of this test)."""
        ev = _build_evidence(3)  # filenames like deck0.pdf, pages 1..3
        msgs = prompt_from_evidence("q", ev)
        user_content = next(c["content"] for c in msgs if c["role"] == "user")
        dumped = json.dumps(user_content)
        # The evidence filenames must not appear anywhere the model sees.
        for candidate in ev:
            fname = candidate.candidate.filename  # e.g. "deck0.pdf"
            assert fname not in dumped, f"filename leaked to model: {fname}"
        # Explicitly forbid citation/page/path patterns in the user (evidence) block.
        assert "· page" not in dumped
        assert "page " not in dumped
        assert ".pdf" not in dumped
        assert "document_id" not in dumped


# --------------------------------------------------------------------------- #
# End-to-end answer_question
# --------------------------------------------------------------------------- #
class TestAnswerQuestion:
    def _pipeline(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store = _seed(tmp_path, s)
        import rag.retrieval as rt
        import rag.visual_retrieval as vr
        from rag.embedding import EmbeddingResult
        from rag.visual import VisualEmbedding

        def fake_text(texts, model=None, settings=None):
            return [EmbeddingResult(
                text=t, index=i,
                vector=[math.sin(i + sum(ord(c) for c in t) * 0.01) for i in range(16)],
            ) for i, t in enumerate(texts)]
        def fake_vis(**kw):
            seed = (kw.get("image_url") or "") + (kw.get("text") or "")
            base = sum(ord(c) for c in seed)
            return VisualEmbedding(
                vector=[math.sin(i + base * 0.017) for i in range(16)],
                image_url=kw.get("image_url", ""), text=kw.get("text", ""))
        monkeypatch.setattr(rt, "get_text_embedding", fake_text)
        monkeypatch.setattr(vr, "get_visual_embedding", fake_vis)

        text_idx = TextIndex(store, settings=s); text_idx.rebuild()
        vis_idx = VisualIndex(store, settings=s)
        return s, text_idx, vis_idx

    def test_grounded_answer(self, tmp_path, monkeypatch):
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)

        def fake_llm(messages, max_tokens=500, settings=None):
            return '{"answer": "The model is grounded.", "source_ids": [1], "insufficient": false}'
        out = answer_question("q", ti, vi, llm=fake_llm, settings=s)
        assert out.answer == "The model is grounded."
        assert len(out.sources) >= 1
        assert out.insufficient is False

    def test_visual_answer_source_has_image(self, tmp_path, monkeypatch):
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)
        def fake_llm(messages, max_tokens=500, settings=None):
            return '{"answer": "based on the slide image", "source_ids": [1], "insufficient": false}'
        out = answer_question("show me the meme slide", ti, vi, llm=fake_llm, settings=s)
        assert out.sources  # at least one source with image
        assert out.sources[0].image_url.startswith("data:")

    def test_fake_out_of_range_source_rejected(self, tmp_path, monkeypatch):
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)
        def fake_llm(messages, max_tokens=500, settings=None):
            return '{"answer": "x", "source_ids": [999], "insufficient": false}'
        out = answer_question("q", ti, vi, llm=fake_llm, settings=s)
        assert out.sources == []  # fabricated id dropped, no crash

    def test_unanswerable(self, tmp_path, monkeypatch):
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)
        def fake_llm(messages, max_tokens=500, settings=None):
            return '{"answer": "not enough info", "source_ids": [], "insufficient": true}'
        out = answer_question("unsupported", ti, vi, llm=fake_llm, settings=s)
        assert out.insufficient is True
        assert out.sources == []

    def test_insufficient_with_source_ids_forces_empty_sources(self, tmp_path, monkeypatch):
        """If the model returns insufficient=true but wrongly includes source_ids,
        Python must return sources=[] — an insufficient answer never cites."""
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)
        def fake_llm(messages, max_tokens=500, settings=None):
            return ('{"answer": "The materials do not contain enough information.", '
                    '"source_ids": [1], "insufficient": true}')
        out = answer_question("q", ti, vi, llm=fake_llm, settings=s)
        assert out.insufficient is True
        assert out.sources == []  # source_ids ignored when insufficient

    def test_no_documents(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        store = DocumentStore(s.data_dir)  # empty
        ti = TextIndex(store, settings=s); vi = VisualIndex(store, settings=s)
        out = answer_question("q", ti, vi, settings=s)
        assert out.insufficient is True
        assert out.sources == []
        assert "enough information" in out.answer.lower()

    def test_9001_failure_propagates(self, tmp_path, monkeypatch):
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)
        def fake_llm(messages, max_tokens=500, settings=None):
            raise QAGenerationError("9001 down")
        with pytest.raises(QAGenerationError):
            answer_question("q", ti, vi, llm=fake_llm, settings=s)

    def test_malformed_9001_output_raises(self, tmp_path, monkeypatch):
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)
        def fake_llm(messages, max_tokens=500, settings=None):
            return "the model produced no usable json at all"
        with pytest.raises(QAGenerationError):
            answer_question("q", ti, vi, llm=fake_llm, settings=s)

    def test_9004_fallback_feeds_qa(self, tmp_path, monkeypatch):
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)
        from rag.rerank_pipeline import rerank_evidence
        from rag.reranker import RerankError

        def fake_rerank(query, text_idx, visual_idx, **kw):
            # force a fallback-style result via the real RRF path
            return rerank_evidence(query, text_idx, visual_idx, settings=s,
                                   reranker=lambda *a, **k: (_ for _ in ()).throw(
                                       RerankError("9004 simulated down")))
        def fake_llm(messages, max_tokens=500, settings=None):
            return '{"answer": "fallback answer", "source_ids": [1], "insufficient": false}'
        out = answer_question("q", ti, vi, llm=fake_llm, rerank=fake_rerank, settings=s)
        assert out.used_fallback is True
        assert out.sources  # still produced valid trusted sources

    def test_no_credential_leak_in_prompt_or_error(self, tmp_path, monkeypatch):
        s, ti, vi = self._pipeline(tmp_path, monkeypatch)
        captured = {}
        def fake_llm(messages, max_tokens=500, settings=None):
            joined = json.dumps(messages)
            assert "test-key" not in joined  # key must never enter the prompt
            return '{"answer": "safe", "source_ids": [1], "insufficient": false}'
        out = answer_question("q", ti, vi, llm=fake_llm, settings=s)
        assert out.answer == "safe"
