"""Text-embedding client for the class 9002 endpoint (Nemotron).

Endpoint contract (verified live):
    POST {TEXT_EMBEDDING_URL}   (default http://dobolyi.com:9002/v2/embed)
    Authorization: Bearer <key>
    body: {"model": "nvidia/Nemotron-3-Embed-1B-BF16", "texts": ["...", ...]}

Response:
    {"id": "...",
     "embeddings": {"float": [[...], ...]},          # one 2048-dim vector per text
     "texts": [...],
     "meta": {"billed_units": {...}},
     "response_type": "embeddings_by_type"}

This client returns the dense vectors; callers pair them with source metadata.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import Settings, get_settings


class EmbeddingError(RuntimeError):
    """Raised when the text-embedding endpoint cannot be reached or errors."""


@dataclass
class EmbeddingResult:
    """One text and its embedding, plus the source text echoed back."""

    text: str
    vector: list[float]
    index: int = 0

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def to_dict(self) -> dict:
        return {"text": self.text, "vector": self.vector, "index": self.index}


def get_text_embedding(
    texts: list[str],
    *,
    model: str | None = None,
    settings: Settings | None = None,
) -> list[EmbeddingResult]:
    """Embed one or more texts via the 9002 Nemotron endpoint.

    Args:
        texts: list of text strings to embed (order preserved in the result).
        model: optional model id override (defaults to settings value).
        settings: injectable settings (defaults to cached get_settings()).

    Returns:
        list[EmbeddingResult] with the same length as ``texts``.

    Raises:
        EmbeddingError: if the key is missing, the request fails, or the
            response is malformed.
    """
    s = settings or get_settings()
    if not s.is_api_key_set:
        raise EmbeddingError(
            "CLASS_API_KEY is not configured. Set it in a local .env file.")

    if not texts:
        return []

    payload = {
        "model": model or "nvidia/Nemotron-3-Embed-1B-BF16",
        "texts": list(texts),
    }
    headers = {
        "Authorization": f"Bearer {s.class_api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(s.text_embedding_url, json=payload,
                             headers=headers, timeout=120)
    except (requests.RequestException, OSError, ConnectionError) as exc:
        raise EmbeddingError(f"Request to text-embedding endpoint failed: {exc}") from exc

    if resp.status_code != 200:
        raise EmbeddingError(
            f"Text-embedding endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        vecs = data["embeddings"]["float"]
    except (ValueError, KeyError, TypeError) as exc:
        raise EmbeddingError(f"Unexpected embedding response shape: {resp.text[:300]}") from exc

    if len(vecs) != len(texts):
        raise EmbeddingError(
            f"Embedding count mismatch: sent {len(texts)} text(s), "
            f"received {len(vecs)} vector(s).")

    return [
        EmbeddingResult(text=t, vector=v, index=i)
        for i, (t, v) in enumerate(zip(texts, vecs))
    ]


def embed_query(query: str, **kwargs) -> EmbeddingResult:
    """Convenience: embed a single query string."""
    return get_text_embedding([query], **kwargs)[0]
