"""Core data models for documents and pages.

These dataclasses define the shape that every part of the pipeline relies on:
each stored page carries enough metadata to trace any retrieved chunk or image
back to its source document and page/slide number.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DocumentPage:
    """A single page/slide of an uploaded document.

    ``image_url`` is a ``data:`` URL of the rendered page, used to send the page
    to the visual embedding and document-parser endpoints. ``document_id`` links
    the page back to its parent :class:`Document`.

    ``parse_status`` records whether OCR (9005) succeeded for this page so we can
    distinguish an empty slide from a failed parse request:
      - ``"success"``  -> text was extracted normally (may legitimately be empty)
      - ``"failed"``   -> the parser endpoint failed; ``parse_error`` explains why
      - ``"skipped"``  -> parsing was not attempted for this page
    """

    document_id: str
    filename: str
    page_no: int
    extracted_text: str = ""
    image_path: str = ""
    image_url: str = ""
    parse_status: str = "success"
    parse_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    """A document as stored in the knowledge base."""

    document_id: str
    filename: str
    sha256: str
    source_path: str = ""
    uploaded_at: str = field(default_factory=_utcnow)
    page_count: int = 0
    pages: list[DocumentPage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "source_path": self.source_path,
            "uploaded_at": self.uploaded_at,
            "page_count": self.page_count,
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        pages = [DocumentPage(**pg) for pg in data.get("pages", [])]
        return cls(
            document_id=data["document_id"],
            filename=data["filename"],
            sha256=data["sha256"],
            source_path=data.get("source_path", ""),
            uploaded_at=data.get("uploaded_at", _utcnow()),
            page_count=data.get("page_count", len(pages)),
            pages=pages,
        )
