"""Tests for the PDF page renderer."""
from __future__ import annotations

import inspect
from pathlib import Path

from rag import renderer


def test_render_pdf_pages_creates_one_png_per_page(sample_pdf, tmp_path):
    pages = renderer.render_pdf_pages(sample_pdf, tmp_path / "out")
    assert len(pages) == 2
    for p in pages:
        assert p.page_no in (1, 2)
        assert Path(p.image_path).is_file()
        # 1-based, sequential
    assert [p.page_no for p in pages] == [1, 2]
    assert Path(pages[0].image_path).suffix == ".png"


def test_rendered_page_has_dimensions(sample_pdf, tmp_path):
    pages = renderer.render_pdf_pages(sample_pdf, tmp_path / "out")
    for p in pages:
        assert p.width > 0 and p.height > 0


def test_image_url_is_data_url(sample_pdf, tmp_path):
    pages = renderer.render_pdf_pages(sample_pdf, tmp_path / "out")
    url = pages[0].image_url
    assert url.startswith("data:image/png;base64,")
    assert len(url) > 40


def test_data_url_of_returns_png_mimetype(sample_pdf, tmp_path):
    pages = renderer.render_pdf_pages(sample_pdf, tmp_path / "out")
    assert renderer.data_url_of(pages[0].image_path).startswith("data:image/png;base64,")


def test_renderer_is_module_level_function():
    # The class-endpoints instruction asks for separate functions such as
    # parse_document_page / get_text_embedding; render is a standalone function.
    assert inspect.isfunction(renderer.render_pdf_pages)
