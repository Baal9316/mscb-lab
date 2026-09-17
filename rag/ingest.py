"""Ingestion pipeline: upload a PDF, render pages, parse text, store metadata.

High-level flow for ``ingest_document``:

1. Compute SHA-256 of the source PDF.
2. If a document with the same hash already exists, skip (duplicate detection).
3. Copy the PDF into the store under a fresh document id.
4. Render each page to PNG.
5. Parse each page via the 9005 document-parser client.
6. Build a :class:`Document` with full page metadata and persist it.

Any searchable/indexed content that later components (embeddings, BM25) create
is placed under ``<doc dir>/index/`` so ``delete_document`` removes it too.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import Settings, get_settings
from .models import Document, DocumentPage
from .parser import DocumentParserError, ParserResult, parse_document_page
from .renderer import render_pdf_pages
from .store import DocumentStore


class IngestionError(RuntimeError):
    """Raised when a document cannot be ingested."""


def _new_id(filename: str, sha256: str) -> str:
    return f"{Path(filename).stem}-{sha256[:10]}"


def ingest_document(
    pdf_path: str | Path,
    *,
    store: DocumentStore | None = None,
    settings: Settings | None = None,
    parser=parse_document_page,
    renderer=render_pdf_pages,
) -> Document:
    """Ingest a single PDF.

    Returns the stored :class:`Document`. If a duplicate (same SHA-256) already
    exists, the existing document is returned and nothing new is created.
    """
    s = settings or get_settings()
    st = store or DocumentStore(s.data_dir)
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise IngestionError(f"File not found: {pdf_path}")

    sha256 = DocumentStore.sha256_of(pdf_path)

    # Duplicate detection
    existing = st.find_by_sha256(sha256)
    if existing is not None:
        return existing

    document_id = _new_id(pdf_path.name, sha256)
    doc_dir = st.document_dir(document_id)
    pages_dir = doc_dir / "pages"
    index_dir = doc_dir / "index"
    pages_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy the original file into the store
    stored_pdf = doc_dir / pdf_path.name
    shutil.copy2(pdf_path, stored_pdf)

    try:
        # 2. Render pages to PNG
        rendered = renderer(pdf_path, pages_dir)

        # 3. Parse each page via 9005
        pages: list[DocumentPage] = []
        for rp in rendered:
            parse_status = "success"
            parse_error: str | None = None
            try:
                result: ParserResult = parser(
                    image_url=rp.image_url,
                    filename=pdf_path.name,
                    page_no=rp.page_no,
                    settings=s,
                )
                text = result.text
            except DocumentParserError as exc:
                # Keep the page (graceful degradation: the image/URL remain
                # usable for visual retrieval), but surface the failure in
                # metadata rather than silently hiding it.
                text = ""
                parse_status = "failed"
                # Defensively redact the API key from any error text recorded
                # to disk, in case an upstream message ever embeds it.
                message = str(exc)
                if s.class_api_key:
                    message = message.replace(s.class_api_key, "[REDACTED]")
                parse_error = message
            # Save extracted text next to the page image
            (pages_dir / f"page_{rp.page_no:03d}.txt").write_text(text, encoding="utf-8")
            pages.append(DocumentPage(
                document_id=document_id,
                page_no=rp.page_no,
                filename=pdf_path.name,
                extracted_text=text,
                image_path=rp.image_path,
                image_url=rp.image_url,
                parse_status=parse_status,
                parse_error=parse_error,
            ))

    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(doc_dir, ignore_errors=True)
        raise IngestionError(f"Ingestion failed for {pdf_path.name}: {exc}") from exc

    doc = Document(
        document_id=document_id,
        filename=pdf_path.name,
        sha256=sha256,
        source_path=str(pdf_path),
        page_count=len(pages),
        pages=pages,
    )
    st.add(doc)
    return doc
