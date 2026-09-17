"""Rendering PDF pages to PNG images.

Uses PyMuPDF to render each page of a PDF into a PNG. This image is the input
to the visual-embedding (9003), reranker (9004) and document-parser (9005)
endpoints, so clients build a ``data:`` URL from it.
"""
from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF


@dataclass
class RenderedPage:
    page_no: int
    image_path: str
    width: int
    height: int

    @property
    def image_url(self) -> str:
        """A ``data:image/png;base64,...`` URL for this page image."""
        return data_url_of(self.image_path)


DEFAULT_ZOOM = 2.0


def render_pdf_pages(pdf_path: str | Path, out_dir: str | Path,
                     zoom: float = DEFAULT_ZOOM) -> list[RenderedPage]:
    """Render every page of ``pdf_path`` to PNG and write into ``out_dir``.

    Returns one :class:`RenderedPage` per page, ordered by page number (1-based).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages: list[RenderedPage] = []
    with pymupdf.open(str(pdf_path)) as pdf_doc:
        for i in range(pdf_doc.page_count):
            page = pdf_doc.load_page(i)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            name = f"page_{i + 1:03d}.png"
            img_path = out / name
            pix.save(str(img_path))
            pages.append(RenderedPage(
                page_no=i + 1,
                image_path=str(img_path),
                width=pix.width,
                height=pix.height,
            ))
    return pages


def data_url_of(image_path: str | Path) -> str:
    """Return a ``data:image/png;base64,...`` URL from a PNG file."""
    path = Path(image_path)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"
