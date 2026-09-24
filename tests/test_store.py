"""Tests for the DocumentStore persistence layer."""
from __future__ import annotations

import json
from pathlib import Path

from rag.models import Document
from rag.store import DocumentStore


def _make_doc(doc_id="doc1", filename="a.pdf", sha="abc123", pages=2):
    from rag.models import DocumentPage
    return Document(
        document_id=doc_id,
        filename=filename,
        sha256=sha,
        source_path="/src/a.pdf",
        pages=[
            DocumentPage(document_id=doc_id, page_no=i, filename=filename,
                         extracted_text=f"page {i}")
            for i in range(1, pages + 1)
        ],
    )


def test_metadata_file_created_on_init(tmp_path):
    store = DocumentStore(tmp_path / "data")
    assert (tmp_path / "data" / "metadata.json").exists()


def test_add_and_get_roundtrip(tmp_path):
    store = DocumentStore(tmp_path / "data")
    doc = _make_doc()
    store.add(doc)
    got = store.get("doc1")
    assert got is not None
    assert got.filename == "a.pdf"
    assert got.sha256 == "abc123"
    assert len(got.pages) == 2
    assert got.pages[0].page_no == 1


def test_add_and_list(tmp_path):
    store = DocumentStore(tmp_path / "data")
    store.add(_make_doc("d1", "a.pdf", "aaa", pages=1))
    store.add(_make_doc("d2", "b.pdf", "bbb", pages=3))
    listing = store.list_all()
    assert len(listing) == 2
    assert {d.document_id for d in listing} == {"d1", "d2"}


def test_duplicate_detection_by_sha256(tmp_path):
    store = DocumentStore(tmp_path / "data")
    store.add(_make_doc("d1", "a.pdf", "abc123"))
    found = store.find_by_sha256("abc123")
    assert found is not None and found.document_id == "d1"
    assert store.find_by_sha256("nope") is None


def test_delete_removes_metadata_and_content(tmp_path):
    store = DocumentStore(tmp_path / "data")
    doc = _make_doc("d1", "a.pdf", "abc123")
    store.add(doc)

    # Simulate indexed content that must also be removed.
    ddir = store.document_dir("d1")
    idx = ddir / "index"
    idx.mkdir(parents=True, exist_ok=True)
    (idx / "chunks.json").write_text("[]", encoding="utf-8")
    (ddir / "original.pdf").write_text("pdf", encoding="utf-8")

    assert store.get("d1") is not None
    assert store.delete("d1") is True
    assert store.get("d1") is None
    assert not ddir.exists()


def test_delete_missing_returns_false(tmp_path):
    store = DocumentStore(tmp_path / "data")
    assert store.delete("missing") is False


def test_list_indexed_content(tmp_path):
    store = DocumentStore(tmp_path / "data")
    store.add(_make_doc("d1"))
    ddir = store.document_dir("d1")
    idx = ddir / "index"
    idx.mkdir(parents=True, exist_ok=True)
    (idx / "a.json").write_text("1", encoding="utf-8")
    (idx / "b.json").write_text("2", encoding="utf-8")
    paths = store.list_indexed_content("d1")
    assert len(paths) == 2


def test_metadata_is_valid_json(tmp_path):
    store = DocumentStore(tmp_path / "data")
    store.add(_make_doc())
    raw = json.loads((tmp_path / "data" / "metadata.json").read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) == 1
