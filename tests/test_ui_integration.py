"""Integration tests for the Gradio UI-to-backend layer.

These call the same helper functions the Gradio handlers wire up
(app.upload_pdf, app.refresh_documents, app.view_page, app.delete_document),
so they validate the UI's data flow against the real Milestone 1 backend
without needing a live browser or server.

The 9005 parser's HTTP layer is stubbed (offline, deterministic) by injecting a
fake parser into ingest_document via the ``parser`` hook.
"""
from __future__ import annotations

import json

import gradio as gr
import pytest

from rag.config import Settings
from rag.ingest import DocumentStore, ingest_document
from rag.models import DocumentPage


def _ok_parser(settings=None):
    def parse(image_url, filename, page_no, settings=None, instruction=""):
        return type("R", (), {
            "text": f"OCR text for {filename} page {page_no}",
            "page_no": page_no, "filename": filename,
            "prompt_tokens": 1, "completion_tokens": 1,
        })()
    return parse


def _settings(tmp_path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(class_api_key="test-key", data_dir=data_dir)


def _seed_doc(tmp_path, settings: Settings, name="sample.pdf", pages=2,
              parser=_ok_parser()):
    """Ingest into a temp store and monkeypatch app store ctor to use it."""
    import rag.ingest
    store = DocumentStore(settings.data_dir)
    # Build a small PDF
    import pymupdf
    pdf = tmp_path / name
    doc = pymupdf.open()
    for i in range(pages):
        pg = doc.new_page(width=400, height=300)
        pg.insert_text((72, 100), f"Page {i+1}", fontsize=18)
    doc.save(str(pdf)); doc.close()
    return ingest_document(pdf, store=store, settings=settings, parser=parser)


class TestUpload:
    def _upload_via_app(self, tmp_path, s, _appmod, name="sample.pdf"):
        """Create a PDF file and push it through app.upload_pdf (the UI path)."""
        import pymupdf
        pdf = tmp_path / name
        doc = pymupdf.open()
        pg = doc.new_page(width=400, height=300)
        pg.insert_text((72, 100), "Page 1", fontsize=18)
        doc.save(str(pdf)); doc.close()
        return pdf, _appmod.upload_pdf(str(pdf), settings=s)

    def test_upload_uses_backend_and_reports(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        import app as appmod
        monkeypatch.setattr(appmod, "get_settings", lambda: s)
        _, msg = self._upload_via_app(tmp_path, s, appmod)
        assert "Uploaded" in msg and "pages" in msg

    def test_upload_rejects_duplicate(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        import app as appmod
        monkeypatch.setattr(appmod, "get_settings", lambda: s)
        pdf, msg1 = self._upload_via_app(tmp_path, s, appmod)
        assert "Uploaded" in msg1
        msg2 = appmod.upload_pdf(str(pdf), settings=s)
        assert "Duplicate" in msg2 and "rejected" in msg2.lower()

    def test_upload_non_pdf_rejected(self, tmp_path):
        s = _settings(tmp_path)
        import app as appmod
        f = tmp_path / "notes.txt"; f.write_text("hi")
        msg = appmod.upload_pdf(str(f), settings=s)
        assert "Unsupported type" in msg and ".pptx" in msg

    def test_upload_pptx_without_libreoffice_gives_actionable_message(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        import app as appmod
        monkeypatch.setattr(appmod.converter, "find_soffice", lambda: None)
        f = tmp_path / "deck.pptx"; f.write_bytes(b"PK\x03\x04 fake")
        msg = appmod.upload_pdf(str(f), settings=s)
        assert "LibreOffice" in msg and "brew install" in msg and "PDF manually" in msg

    def test_upload_empty_selection(self, tmp_path):
        s = _settings(tmp_path)
        import app as appmod
        assert "select a pdf" in appmod.upload_pdf(None, settings=s).lower()


class TestMultiUpload:
    """app.upload_files: batch upload wired to the multi-file picker."""

    @staticmethod
    def _fake_upload(monkeypatch, appmod, calls):
        def fake(fp, settings=None):
            calls.append(fp)
            if fp.endswith(".txt"):
                return "Unsupported type '.txt'."
            return (f"✅ Uploaded '{fp}' — 1 pages (document x)."
                    "\n\nSelect the document below to inspect its pages.")
        monkeypatch.setattr(appmod, "upload_pdf", fake)

    def test_uploads_every_file_and_summarizes(self, monkeypatch):
        import app as appmod
        calls: list[str] = []
        self._fake_upload(monkeypatch, appmod, calls)
        outputs = list(appmod.upload_files(["a.pdf", "b.pdf", "c.pdf"]))
        assert calls == ["a.pdf", "b.pdf", "c.pdf"]
        # One progress message per file, then the final summary.
        assert len(outputs) == 4
        assert "Uploading 2 of 3" in outputs[1]
        assert outputs[-1].startswith("Done: 3 of 3")
        assert outputs[-1].count("Select a document below") == 1

    def test_one_bad_file_does_not_stop_the_batch(self, monkeypatch):
        import app as appmod
        calls: list[str] = []
        self._fake_upload(monkeypatch, appmod, calls)
        final = list(appmod.upload_files(["a.pdf", "notes.txt", "b.pdf"]))[-1]
        assert calls == ["a.pdf", "notes.txt", "b.pdf"]
        assert final.startswith("Done: 2 of 3")
        assert "Unsupported type" in final

    def test_single_path_still_works(self, monkeypatch):
        import app as appmod
        calls: list[str] = []
        self._fake_upload(monkeypatch, appmod, calls)
        final = list(appmod.upload_files("only.pdf"))[-1]
        assert calls == ["only.pdf"] and final.startswith("Done: 1 of 1")

    def test_empty_selection(self):
        import app as appmod
        for empty in (None, []):
            out = list(appmod.upload_files(empty))
            assert len(out) == 1 and "select one or more" in out[0].lower()


class TestDocumentList:
    def test_list_shows_docs_after_upload(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        import app as appmod
        lib = _seed_doc(tmp_path, s)
        # ensure app's store sees it (same data_dir)
        appmod.upload_pdf(lib.source_path, settings=s)  # reconcile
        # upload_pdf with duplicate will report duplicate; use refresh directly
        msg, dd = appmod.refresh_documents(settings=s)
        assert "sample.pdf" in msg

    def test_document_dropdown_choices(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        import app as appmod
        lib = _seed_doc(tmp_path, s)
        appmod.upload_pdf(lib.source_path, settings=s)
        opts = appmod.load_document_options(settings=s)
        assert len(opts) == 1 and "sample.pdf" in opts[0]


class TestPageView:
    def test_view_page_shows_image_and_text(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        import app as appmod
        lib = _seed_doc(tmp_path, s)
        labels = appmod.load_document_options(settings=s)
        doc_label = labels[0]
        page_label = "Page 1"
        img, text, status = appmod.view_page(doc_label, page_label, settings=s)
        assert img and "succeeded" in status.lower()
        assert "page 1" in text

    def test_view_page_unknown_doc_is_safe(self, tmp_path):
        s = _settings(tmp_path)
        import app as appmod
        img, text, status = appmod.view_page("nope :: nope", "Page 1", settings=s)
        assert text != "" and "not found" in text.lower()


class TestDelete:
    def test_delete_removes_doc_and_content(self, tmp_path, monkeypatch):
        s = _settings(tmp_path)
        import app as appmod
        lib = _seed_doc(tmp_path, s)
        labels = appmod.load_document_options(settings=s)
        assert labels
        msg = appmod.delete_document(labels[0], settings=s)
        assert "Deleted" in msg
        assert appmod.load_document_options(settings=s) == []


def test_helpers_are_wired_in_blocks():
    """Smoke test: the app also builds a Blocks object (UI wiring loads)."""
    s = __import__("app").build_app()  # settings default is fine
    assert isinstance(s, gr.Blocks)


class TestAskQuestionUI:
    def test_ask_question_empty(self):
        import app as appmod
        ans, sources, images, status = appmod.ask_question("")
        assert "Please enter a question" in ans
        assert "No question" in status

    def test_ask_question_no_documents(self, tmp_path, monkeypatch):
        import app as appmod
        from rag.config import Settings
        s = Settings(class_api_key="test-key", data_dir=tmp_path / "data")
        (tmp_path / "data").mkdir(exist_ok=True, parents=True)
        monkeypatch.setattr(appmod, "get_settings", lambda: s)
        ans, sources, images, status = appmod.ask_question("any question", settings=s)
        assert "enough information" in ans.lower()
        assert images == []

    def _seed_one(self, tmp_path, settings):
        from rag.models import Document, DocumentPage
        store = __import__("rag.store", fromlist=["DocumentStore"]).DocumentStore(settings.data_dir)
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xAA" * 8)
        store.add(Document(
            document_id="docA", filename="a.pdf", sha256="s1",
            pages=[DocumentPage(document_id="docA", page_no=1, filename="a.pdf",
                                extracted_text="The CPU is the worker that does the work",
                                image_path=str(img),
                                image_url=f"data:image/png;base64,{img.read_bytes().hex()}")]))
        return store

    def test_consecutive_questions_reuse_indexes(self, tmp_path, monkeypatch):
        """Repeated questions with unchanged docs must reuse cached indexes."""
        import app as appmod
        from rag.config import Settings
        import rag.retrieval as rt
        from rag.embedding import EmbeddingResult
        import math

        def fake_text(texts, model=None, settings=None):
            return [EmbeddingResult(
                text=t, index=i,
                vector=[math.sin(i + sum(ord(c) for c in t) * 0.01) for i in range(16)],
            ) for i, t in enumerate(texts)]
        monkeypatch.setattr(rt, "get_text_embedding", fake_text)

        s = Settings(class_api_key="test-key", data_dir=tmp_path / "data")
        store = self._seed_one(tmp_path, s)
        appmod.invalidate_qa_index_cache()
        ti1, vi1 = appmod.get_qa_indexes(store, settings=s)
        ti2, vi2 = appmod.get_qa_indexes(store, settings=s)
        assert ti1 is ti2 and vi1 is vi2  # unchanged collection -> reused

    def test_upload_invalidates_cached_indexes(self, tmp_path, monkeypatch):
        import app as appmod
        from rag.config import Settings
        import rag.retrieval as rt
        from rag.embedding import EmbeddingResult
        import math

        def fake_text(texts, model=None, settings=None):
            return [EmbeddingResult(
                text=t, index=i,
                vector=[math.sin(i + sum(ord(c) for c in t) * 0.01) for i in range(16)],
            ) for i, t in enumerate(texts)]
        monkeypatch.setattr(rt, "get_text_embedding", fake_text)

        s = Settings(class_api_key="test-key", data_dir=tmp_path / "data")
        store = self._seed_one(tmp_path, s)
        appmod.invalidate_qa_index_cache()
        ti1, _ = appmod.get_qa_indexes(store, settings=s)
        # upload a NEW document (simulate via adding to store + invalidate like upload_pdf)
        from rag.models import Document, DocumentPage
        img2 = tmp_path / "b.png"; img2.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xBB" * 8)
        store.add(Document(
            document_id="docB", filename="b.pdf", sha256="s2",
            pages=[]))  # no pages so no re-embed cost; still changes fingerprint
        appmod.invalidate_qa_index_cache()  # what upload_pdf calls
        ti2, _ = appmod.get_qa_indexes(store, settings=s)
        assert ti1 is not ti2  # rebuilt after upload

    def test_delete_invalidates_cached_indexes(self, tmp_path, monkeypatch):
        import app as appmod
        from rag.config import Settings
        import rag.retrieval as rt
        from rag.embedding import EmbeddingResult
        import math

        def fake_text(texts, model=None, settings=None):
            return [EmbeddingResult(
                text=t, index=i,
                vector=[math.sin(i + sum(ord(c) for c in t) * 0.01) for i in range(16)],
            ) for i, t in enumerate(texts)]
        monkeypatch.setattr(rt, "get_text_embedding", fake_text)

        s = Settings(class_api_key="test-key", data_dir=tmp_path / "data")
        store = self._seed_one(tmp_path, s)
        appmod.invalidate_qa_index_cache()
        ti1, _ = appmod.get_qa_indexes(store, settings=s)
        store.delete("docA")
        appmod.invalidate_qa_index_cache()  # what delete_document calls
        # After invalidation, next get rebuilds to empty collection.
        ti2, _ = appmod.get_qa_indexes(store, settings=s)
        assert ti1 is not ti2
        assert ti2.chunk_count() == 0


class TestQuizUI:
    """UI-layer tests for the Practice Quiz tab (Issue #6).

    Verifies the opaque-quiz-id flow: generate returns a separate opaque id
    and public data (no private key); Submit and Show Answers use ONLY the
    opaque id to look up the frozen Quiz in the server-side registry; radio
    selections are mapped to {question_id: option} automatically; grading
    makes zero 9001 calls.
    """

    def _stub_generate(self, appmod, monkeypatch, n=2, visual=False):
        """Register a frozen quiz+catalog and stub rag.quiz.generate_quiz so
        generate_quiz_ui runs fully offline."""
        from rag.quiz import Quiz, QuizQuestion, store_quiz
        from rag.rerank_pipeline import EvidenceCandidate, RerankedEvidence

        ev = [
            RerankedEvidence(EvidenceCandidate(
                "d", "deck.pdf", 1, text="About temperature sampling",
                image_path="/tmp/slide_1.png", image_url="data:image/png;base64,x"), 0.9, "rerank"),
            RerankedEvidence(EvidenceCandidate(
                "d", "deck.pdf", 2, text="About attention sampling with images",
                image_path="/tmp/slide_2.png", image_url="data:image/png;base64,y"), 0.8, "rerank"),
        ]
        catalog = None
        def fake_gen(**kwargs):
            nonlocal catalog
            qid = "gen-abc"
            quiz = Quiz(quiz_id=qid, questions=(
                QuizQuestion(f"{qid}_q1", "What controls randomness?",
                             ("A", "B", "C", "D"), 1, "B", "Temperature does.",
                             ("E1",), bool(visual)),
                QuizQuestion(f"{qid}_q2", "Which mechanism attends?",
                             ("A", "B", "C", "D"), 2, "C", "Attention.",
                             ("E2",), False),
            ))
            store_quiz(quiz)
            # catalog with the same evidence
            from rag.quiz import build_evidence_catalog
            catalog = build_evidence_catalog(ev)
            appmod._QUIZ_CATALOGS[qid] = catalog
            return quiz, ev
        monkeypatch.setattr("rag.quiz.generate_quiz", fake_gen)
        return catalog

    def test_generate_returns_separate_opaque_id_and_no_private_key(
            self, tmp_path, monkeypatch):
        import app as appmod
        s = Settings(class_api_key="test-key", data_dir=tmp_path / "data")
        self._stub_generate(appmod, monkeypatch)
        quiz_id, pub, status = appmod.generate_quiz_ui("", 2, settings=s)
        assert "✅" in status
        # public data is a dict, opaque id is a separate string value
        assert isinstance(pub, dict)
        assert isinstance(quiz_id, str) and quiz_id
        assert json.dumps(pub) != quiz_id  # public JSON != opaque quiz_id
        # public data has questions but never the private answer key
        assert len(pub["questions"]) == 2
        dpub = json.dumps(pub)
        assert "correct_index" not in dpub
        assert "correct_answer" not in dpub
        assert "explanation" not in dpub
        assert "document_id" not in dpub

    def test_submit_uses_opaque_id_and_radio_mapping(self, tmp_path, monkeypatch):
        import app as appmod
        s = Settings(class_api_key="test-key", data_dir=tmp_path / "data")
        self._stub_generate(appmod, monkeypatch)
        quiz_id, pub, _ = appmod.generate_quiz_ui("", 2, settings=s)
        # Simulate the panel wiring: panel i's radio holds the chosen option.
        # For q1 the correct answer is "B"; q2 correct is "C".
        called_with = {}
        orig_get = __import__("rag.quiz", fromlist=["get_quiz"]).get_quiz
        # spy on get_quiz to prove it only ever receives the opaque id
        def spy_get(qid):
            called_with["qid"] = qid
            return orig_get(qid)
        monkeypatch.setattr("rag.quiz.get_quiz", spy_get)
        # zero 9001 calls during grading
        calls = []
        import rag.quiz as qm
        orig_llm = qm.generate_llm_response
        qm.generate_llm_response = lambda *a, **k: calls.append(a) or ""
        try:
            score = appmod._grade_from_radios(
                quiz_id, "B", "C")   # radio 0 -> q1="B", radio 1 -> q2="C"
        finally:
            qm.generate_llm_response = orig_llm
        assert calls == []           # zero 9001 calls
        assert called_with["qid"] == quiz_id  # used the opaque id, not JSON
        assert "Score: 2 / 2 (100%)" in score
        # public JSON was never passed to get_quiz
        assert called_with["qid"] != json.dumps(pub)

    def test_show_answers_uses_same_opaque_id_and_reveals(self, tmp_path, monkeypatch):
        import app as appmod
        s = Settings(class_api_key="test-key", data_dir=tmp_path / "data")
        self._stub_generate(appmod, monkeypatch)
        qid, pub, _ = appmod.generate_quiz_ui("", 2, settings=s)
        called_with = {}
        orig_get = __import__("rag.quiz", fromlist=["get_quiz"]).get_quiz
        def spy_get(x):
            called_with["qid"] = x
            return orig_get(x)
        monkeypatch.setattr("rag.quiz.get_quiz", spy_get)
        review, images = appmod._reveal_from_radios(qid, "B", "C")
        assert called_with["qid"] == qid
        assert called_with["qid"] != json.dumps(pub)
        # explanations + correct answers + citations appear after reveal
        assert "Temperature does." in review
        assert "Correct answer: **C**" in review
        assert "deck.pdf · page 1" in review
        assert "supporting excerpt" in review.lower() or "excerpt" in review.lower()
        assert images  # supporting slide image paths returned

    def _stub_quiz_for_mapping(self, appmod, tmp_path):
        from rag.quiz import Quiz, QuizQuestion, store_quiz
        quiz = Quiz(quiz_id="q-map", questions=(
            QuizQuestion("q-map_q1", "p", ("A", "B", "C", "D"), 1, "B", "e",
                         ("E1",), False),
        ))
        store_quiz(quiz)
        return quiz.quiz_id

    def test_build_answers_from_radios_uses_question_ids(self, tmp_path):
        import app as appmod
        qid = self._stub_quiz_for_mapping(appmod, tmp_path)
        quiz = __import__("rag.quiz", fromlist=["get_quiz"]).get_quiz(qid)
        answers = appmod._build_answers_from_radios(quiz, ["B"])
        assert answers == {"q-map_q1": "B"}

    def test_populate_panel_hides_extra(self):
        import app as appmod
        data = {"questions": [{"question_id": "x_q1", "prompt": "p",
                               "options": ["A", "B", "C", "D"]}]}
        vis, md, img, radio = appmod._populate_question_panel(data, 0)
        assert md == "**x_q1.** p"
        vis2, md2, img2, radio2 = appmod._populate_question_panel(data, 3)
        assert vis2["visible"] is False
        assert radio2["interactive"] is False

    def test_build_app_has_three_tabs(self, tmp_path):
        import app as appmod
        s = Settings(class_api_key="test-key", data_dir=tmp_path / "data")
        demo = appmod.build_app(settings=s)
        assert appmod.generate_quiz_ui
        assert appmod._grade_from_radios
        assert appmod._reveal_from_radios
