"""DocumentStore: JSON-backed persistence for uploaded documents.

Responsibilities:
- persist document metadata to ``metadata.json`` under ``data/``
- detect duplicate uploads by SHA-256
- list all documents
- delete a document and cleanup its files, rendered pages, and any content
  that downstream indexing staged next to it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .models import Document


class DocumentStore:
    """Store document metadata in a JSON registry plus per-document folders.

    Layout::

        <data_dir>/
            metadata.json
            documents/<document_id>/
                original.pdf            (the uploaded file)
                source/
                pages/
                    page_001.png
                    page_001.txt
                index/                   (searchable / indexed content)
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.documents_dir = self.data_dir / "documents"
        self.metadata_path = self.data_dir / "metadata.json"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        if not self.metadata_path.exists():
            self._write_metadata([])

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _write_metadata(self, docs: list[Document]) -> None:
        self.metadata_path.write_text(
            json.dumps([d.to_dict() for d in docs], indent=2), encoding="utf-8")

    def _load_metadata(self) -> list[Document]:
        if not self.metadata_path.exists():
            return []
        try:
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [Document.from_dict(d) for d in raw]

    def _save(self, docs: list[Document]) -> None:
        self._write_metadata(docs)

    def document_dir(self, document_id: str) -> Path:
        return self.documents_dir / document_id

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def sha256_of(path: str | Path) -> str:
        """Return the SHA-256 hex digest of a file (streamed, memory-safe)."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def find_by_sha256(self, sha256: str) -> Document | None:
        for doc in self._load_metadata():
            if doc.sha256 == sha256:
                return doc
        return None

    def add(self, doc: Document) -> None:
        """Persist a document. Overwrites any existing record with the same id."""
        docs = self._load_metadata()
        docs = [d for d in docs if d.document_id != doc.document_id]
        docs.append(doc)
        self._save(docs)

    def get(self, document_id: str) -> Document | None:
        for doc in self._load_metadata():
            if doc.document_id == document_id:
                return doc
        return None

    def list_all(self) -> list[Document]:
        docs = self._load_metadata()
        docs.sort(key=lambda d: d.uploaded_at, reverse=True)
        return docs

    def delete(self, document_id: str) -> bool:
        """Delete a document's metadata and all on-disk content.

        Removes the registry entry and the document folder (original file,
        rendered page images, extracted text, and any indexed content). Returns
        True if the document existed and was removed.
        """
        docs = self._load_metadata()
        existing = [d for d in docs if d.document_id == document_id]
        if not existing:
            return False
        docs = [d for d in docs if d.document_id != document_id]
        self._save(docs)
        ddir = self.document_dir(document_id)
        if ddir.exists():
            shutil.rmtree(ddir, ignore_errors=True)
        return True

    def list_indexed_content(self, document_id: str) -> list[Path]:
        """Return artifact paths belonging to a document's index/ directory."""
        ddir = self.document_dir(document_id)
        idx = ddir / "index"
        if not idx.exists():
            return []
        return sorted(p for p in idx.rglob("*") if p.is_file())
