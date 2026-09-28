"""Tests for rag/converter.py — pptx -> pdf conversion boundary.

These stay offline (LibreOffice is not part of the test run):
- .pdf files pass through unchanged (no conversion).
- missing/unsupported formats raise the right errors.
- a .pptx with no LibreOffice available raises a clear LibreOfficeNotFoundError
  (so the graceful-degradation path is covered without installing anything).
A real soffice hand-in-hand conversion is exercised live in convert_demo.py.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from rag import converter


def test_pdf_passes_through(tmp_path):
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    dst = converter.to_pdf(src, tmp_path / "out")
    assert dst.suffix == ".pdf"
    assert dst.read_bytes() == b"%PDF-1.4 fake"


def test_unsupported_format_raises(tmp_path):
    src = tmp_path / "a.doc"
    src.write_bytes(b"x")
    with pytest.raises(converter.UnsupportedFormatError):
        converter.to_pdf(src, tmp_path / "out")


def test_missing_file_raises(tmp_path):
    with pytest.raises(converter.ConversionError):
        converter.to_pdf(tmp_path / "nope.pdf", tmp_path / "out")


def test_pptx_without_libreoffice_raises_clear_error(tmp_path, monkeypatch):
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"PK fakepptx")
    monkeypatch.setattr(converter, "find_soffice", lambda: None)
    with pytest.raises(converter.LibreOfficeNotFoundError) as einfo:
        converter.to_pdf(src, tmp_path / "out")
    msg = str(einfo.value)
    assert "LibreOffice" in msg and "brew install" in msg


def test_requires_conversion(tmp_path):
    assert converter.requires_conversion(tmp_path / "d.pptx") is True
    assert converter.requires_conversion(tmp_path / "d.pdf") is False
    assert converter.requires_conversion(tmp_path / "d") is False
