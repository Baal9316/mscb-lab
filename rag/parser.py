"""Document-parsing client for the class OCR endpoint (port 9005).

The endpoint is OpenAI-chat-completions style:
POST {DOCUMENT_PARSER_URL}  (default http://dobolyi.com:9005/v1/chat/completions)
Authorization: Bearer <key>
body: {"model": "dots.mocr", "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "<instruction>"},
            {"type": "image_url", "image_url": {"url": "<data url>"}}] }],
       "max_tokens": N}

Response: {"choices": [{"message": {"content": "<extracted text>"}}], "usage": ...}
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import Settings, get_settings


@dataclass
class ParserResult:
    text: str
    page_no: int
    filename: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "page_no": self.page_no,
            "filename": self.filename,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class DocumentParserError(RuntimeError):
    """Raised when the document-parser endpoint cannot be reached or errors."""


def parse_document_page(
    image_url: str,
    filename: str,
    page_no: int,
    instruction: str = "Extract all text on this page verbatim, preserving the layout as much as possible.",
    *,
    settings: Settings | None = None,
) -> ParserResult:
    """Extract text from a page image via the 9005 parser endpoint.

    ``image_url`` must be a ``data:image/...;base64,...`` URL (see renderer).
    """
    s = settings or get_settings()
    if not s.is_api_key_set:
        raise DocumentParserError(
            "CLASS_API_KEY is not configured. Set it in a local .env file.")

    payload = {
        "model": s.document_parser_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {s.class_api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(s.document_parser_url, json=payload, headers=headers, timeout=120)
    except (requests.RequestException, OSError, ConnectionError) as exc:  # network / connection errors
        raise DocumentParserError(f"Request to parser endpoint failed: {exc}") from exc

    if resp.status_code != 200:
        raise DocumentParserError(
            f"Parser endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise DocumentParserError(f"Unexpected parser response shape: {resp.text[:300]}") from exc

    return ParserResult(
        text=text.strip(),
        page_no=page_no,
        filename=filename,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )
