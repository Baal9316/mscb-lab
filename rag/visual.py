"""Visual-embedding client for the class 9003 endpoint (Qwen3-VL).

Endpoint contract (reconfirmed live):
    POST {VISUAL_EMBEDDING_URL}   (default http://dobolyi.com:9003/v1/embeddings)
    Authorization: Bearer <key>
    body: {"model": "Qwen/Qwen3-VL-Embedding-2B",
           "messages": [{"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": "<data url>"}},   # optional
               {"type": "text", "text": "<text>"}                           # optional
           ]}]}

Both text-only (content with only a text item) and image-only (content with only
an image_url item) — and image+text — work with this same ``messages`` shape
(verified live; dim 2048). If neither image nor text is supplied the content is
a single empty text item.

Response (OpenAI-style):
    {"id": "...", "object": "list", "data": [{"index": 0, "object": "embedding",
                                               "embedding": [...2048 floats...]}],
     "usage": {"prompt_tokens": N, "total_tokens": N}}
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import Settings, get_settings


class VisualEmbeddingError(RuntimeError):
    """Raised when the visual-embedding endpoint cannot be reached or errors."""


@dataclass
class VisualEmbedding:
    vector: list[float]
    image_url: str = ""
    text: str = ""
    dimension: int = 2048

    def __post_init__(self) -> None:
        self.dimension = len(self.vector)

    def to_dict(self) -> dict:
        return {"vector": self.vector, "image_url": self.image_url, "text": self.text}


def get_visual_embedding(
    *,
    image_url: str | None = None,
    text: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> VisualEmbedding:
    """Embed an image and/or a text query via the 9003 Qwen3-VL endpoint.

    Args:
        image_url: optional ``data:image/...;base64,...`` URL of an image.
        text: optional text (used alone for a text-only query, or alongside an
            image for a mixed query).
        model: optional model id override.
        settings: injectable settings.

    Returns:
        A :class:`VisualEmbedding` with the 2048-dim vector.

    Raises:
        VisualEmbeddingError: if the key is missing, the request fails, or the
            response is malformed.
    """
    s = settings or get_settings()
    if not s.is_api_key_set:
        raise VisualEmbeddingError(
            "CLASS_API_KEY is not configured. Set it in a local .env file.")

    if not image_url and not text:
        raise VisualEmbeddingError("Need an image_url and/or text to embed.")

    content: list[dict] = []
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": ""})

    payload = {
        "model": model or "Qwen/Qwen3-VL-Embedding-2B",
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "Authorization": f"Bearer {s.class_api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(s.visual_embedding_url, json=payload, headers=headers, timeout=120)
    except (requests.RequestException, OSError, ConnectionError) as exc:
        raise VisualEmbeddingError(
            f"Request to visual-embedding endpoint failed: {exc}") from exc

    if resp.status_code != 200:
        raise VisualEmbeddingError(
            f"Visual-embedding endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        vec = data["data"][0]["embedding"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise VisualEmbeddingError(
            f"Unexpected visual-embedding response shape: {resp.text[:300]}") from exc

    if not isinstance(vec, list) or not vec or not all(
            isinstance(x, (int, float)) for x in vec):
        raise VisualEmbeddingError("Malformed embedding response: vector is not a float list.")

    return VisualEmbedding(vector=[float(x) for x in vec], image_url=image_url or "",
                           text=text or "")
