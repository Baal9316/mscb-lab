"""Integration tests for the Gradio UI-to-backend layer.

These call the same helper functions the Gradio handlers wire up
(app.upload_pdf, app.refresh_documents, app.view_page, app.delete_document),
so they validate the UI's data flow against the real Milestone 1 backend
without needing a live browser or server.

The 9005 parser's HTTP layer is stubbed (offline, deterministic) by injecting a
fake parser into ingest_document via the ``parser`` hook.
"""
from __future__ import annotations

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
        assert "Only PDF" in appmod.upload_pdf(str(f), settings=s)

    def test_upload_empty_selection(self, tmp_path):
        s = _settings(tmp_path)
        import app as appmod
        assert "select a pdf" in appmod.upload_pdf(None, settings=s).lower()


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
