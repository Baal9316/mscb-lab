"""Tests for the 9005 document-parser client (mock the HTTP layer)."""
from __future__ import annotations

import json

import pytest

from rag.parser import DocumentParserError, parse_document_page


class _FakeResponse:
    def __init__(self, status_code, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def test_parse_document_page_builds_correct_payload(settings, monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse(200, {
            "choices": [{"message": {"content": "  extracted text here  "}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    monkeypatch.setattr("rag.parser.requests.post", fake_post)

    result = parse_document_page(
        "data:image/png;base64,AAAA",
        filename="slides.pdf",
        page_no=3,
        settings=settings,
    )

    # URL comes from settings
    assert captured["url"] == settings.document_parser_url
    # Auth header carries the key
    assert captured["headers"]["Authorization"] == f"Bearer {settings.class_api_key}"
    # Model + chat messages with image content
    body = captured["json"]
    assert body["model"] == settings.document_parser_model
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AAAA"
    # Content is stripped
    assert result.text == "extracted text here"
    assert result.page_no == 3
    assert result.filename == "slides.pdf"
    assert result.prompt_tokens == 10 and result.completion_tokens == 5


def test_parse_document_page_raises_without_key(settings):
    settings = settings.__class__(class_api_key="your_key_here", data_dir=settings.data_dir)
    with pytest.raises(DocumentParserError):
        parse_document_page("data:image/png;base64,AAAA",
                            filename="x.pdf", page_no=1, settings=settings)


def test_parse_document_page_raises_on_http_error(settings, monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResponse(500, {"error": "boom"})

    monkeypatch.setattr("rag.parser.requests.post", fake_post)
    with pytest.raises(DocumentParserError):
        parse_document_page("data:image/png;base64,AAAA",
                            filename="x.pdf", page_no=1, settings=settings)


def test_parse_document_page_raises_on_network_error(settings, monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        raise ConnectionError("refused")

    monkeypatch.setattr("rag.parser.requests.post", fake_post)
    with pytest.raises(DocumentParserError):
        parse_document_page("data:image/png;base64,AAAA",
                            filename="x.pdf", page_no=1, settings=settings)


def test_parse_document_page_raises_on_bad_shape(settings, monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResponse(200, {"unexpected": True})

    monkeypatch.setattr("rag.parser.requests.post", fake_post)
    with pytest.raises(DocumentParserError):
        parse_document_page("data:image/png;base64,AAAA",
                            filename="x.pdf", page_no=1, settings=settings)
