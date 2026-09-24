"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a small 2-page PDF and return its path."""
    import pymupdf
    output = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page(width=400, height=300)
        page.insert_text((72, 100), f"Page {i + 1} heading", fontsize=20)
        page.insert_text((72, 140), "Some body text for the course material.")
    doc.save(str(output))
    doc.close()
    return output


@pytest.fixture
def settings(tmp_path):
    """Settings pointing at a temp data dir and a fake (non-empty) key."""
    from rag.config import Settings
    data_dir = tmp_path / "data"
    s = Settings(class_api_key="test-key", data_dir=data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return s
