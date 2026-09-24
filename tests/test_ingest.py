"""Integration tests for the ingestion pipeline (parser + renderer are stubbed
so tests run offline and deterministically)."""
from __future__ import annotations

from rag.ingest import DocumentStore, ingest_document


def _fake_parser(settings=None):
    """Return a parser stub that records calls and returns deterministic text."""

    def parse(image_url, filename, page_no, settings=None, instruction=""):
        return type("R", (), {
            "text": f"OCR text for {filename} page {page_no}",
            "page_no": page_no,
            "filename": filename,
            "prompt_tokens": 1,
            "completion_tokens": 1,
        })()

    return parse


def test_ingest_creates_document_with_metadata(sample_pdf, settings):
    store = DocumentStore(settings.data_dir)
    doc = ingest_document(sample_pdf, store=store, settings=settings,
                          parser=_fake_parser())

    assert doc.filename == "sample.pdf"
    assert doc.page_count == 2
    assert len(doc.pages) == 2
    for i, page in enumerate(doc.pages, start=1):
        assert page.page_no == i
        assert page.extracted_text == f"OCR text for sample.pdf page {i}"
        # image path exists on disk
        import os
        assert os.path.exists(page.image_path)
        # metadata preserved
        assert page.filename == "sample.pdf"

    # persisted in store
    assert store.get(doc.document_id) is not None
    # original PDF copied into store
    assert (store.document_dir(doc.document_id) / "sample.pdf").exists()


def test_ingest_duplicate_returns_existing(sample_pdf, settings):
    store = DocumentStore(settings.data_dir)
    first = ingest_document(sample_pdf, store=store, settings=settings,
                            parser=_fake_parser())
    second = ingest_document(sample_pdf, store=store, settings=settings,
                             parser=_fake_parser())
    assert second.document_id == first.document_id
    assert len(store.list_all()) == 1  # not double-ingested


def test_ingest_then_delete_removes_everything(sample_pdf, settings):
    store = DocumentStore(settings.data_dir)
    doc = ingest_document(sample_pdf, store=store, settings=settings,
                          parser=_fake_parser())
    ddir = store.document_dir(doc.document_id)
    assert ddir.exists()

    assert store.delete(doc.document_id) is True
    assert store.get(doc.document_id) is None
    assert not ddir.exists()  # page images + original all gone


def test_page_text_saved_to_disk(sample_pdf, settings):
    store = DocumentStore(settings.data_dir)
    doc = ingest_document(sample_pdf, store=store, settings=settings,
                          parser=_fake_parser())
    txt_paths = sorted(p for p in
                       (store.document_dir(doc.document_id) / "pages").glob("*.txt"))
    assert len(txt_paths) == 2
    assert b"OCR text for sample.pdf page 1" in txt_paths[0].read_bytes()


def test_every_page_document_id_matches_parent(sample_pdf, settings):
    store = DocumentStore(settings.data_dir)
    doc = ingest_document(sample_pdf, store=store, settings=settings,
                          parser=_fake_parser())
    assert doc.pages
    for page in doc.pages:
        assert page.document_id == doc.document_id


def test_serialization_roundtrip_preserves_document_id_and_status(sample_pdf, settings):
    store = DocumentStore(settings.data_dir)
    doc = ingest_document(sample_pdf, store=store, settings=settings,
                          parser=_fake_parser())
    # Round-trip through the store (to_dict -> from_dict)
    reloaded = store.get(doc.document_id)
    assert reloaded is not None
    for page in reloaded.pages:
        assert page.document_id == reloaded.document_id
        assert page.parse_status == "success"
        assert page.parse_error is None


def test_parse_failure_is_recorded_not_hidden(sample_pdf, settings):
    """A failed 9005 parse must keep the page but record the failure."""
    from rag.parser import DocumentParserError

    def failing_parser(image_url, filename, page_no, settings=None, instruction=""):
        raise DocumentParserError("parser endpoint returned HTTP 500: server error")

    store = DocumentStore(settings.data_dir)
    doc = ingest_document(sample_pdf, store=store, settings=settings,
                          parser=failing_parser)
    assert doc.pages
    for page in doc.pages:
        assert page.parse_status == "failed"
        assert page.parse_error is not None
        assert page.extracted_text == ""  # no text, but page retained
        # image still usable for visual retrieval
        assert page.image_url.startswith("data:")


def test_parse_error_does_not_leak_credentials(sample_pdf, settings):
    """The recorded parse_error must never contain the API key."""
    from rag.parser import DocumentParserError

    def leaking_parser(image_url, filename, page_no, settings=None, instruction=""):
        raise DocumentParserError(f"401 unauthorized, used key {settings.class_api_key}")

    store = DocumentStore(settings.data_dir)
    doc = ingest_document(sample_pdf, store=store, settings=settings,
                          parser=leaking_parser)
    for page in doc.pages:
        assert settings.class_api_key not in (page.parse_error or "")
