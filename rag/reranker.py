"""Reranker client for the class 9004 endpoint (Jina-style rerank).

Endpoint contract (reconfirmed live):
    POST {RERANKER_URL}   (default http://dobolyi.com:9004/rerank)
    Authorization: Bearer <key>
    body: {"model": "Qwen/Qwen3-VL-Reranker-2B",
           "query": "<text>",
           "documents": [ <string> | {"content": [{"type": "text", ...}|{"type": "image_url", ...}]} ],
           "top_n": 5}            # NOTE: top_n, NOT top_k

Response:
    {"id": "...", "model": "...", "usage": {...},
     "results": [{"index": <orig position>, "document": {...}, "relevance_score": <0..1>}]}

Each result's ``index`` maps back to the caller's ``documents`` list position.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import Settings, get_settings


class RerankError(RuntimeError):
    """Raised when the reranker endpoint cannot be reached or errors."""


@dataclass
class RerankOutcome:
    index: int
    score: float
    document: dict


def rerank_candidates(
    query: str,
    documents: list,
    *,
    top_n: int = 5,
    model: str | None = None,
    settings: Settings | None = None,
) -> list[RerankOutcome]:
    """Rerank a list of candidates (strings or {content:[...]} multimodal objs).

    Args:
        query: the user question (text).
        documents: candidate list; each is either a plain string (text-only) or a
            dict ``{"content": [{"type":"text"...},{type:"image_url"...}]}``.
        top_n: how many top results to return (configurable; default 5).
        model: optional model id override.
        settings: injectable settings.

    Returns:
        list[RerankOutcome] ordered best-first, each with the original index and
        relevance score.

    Raises:
        RerankError: on missing key, network failure, HTTP error, or malformed
            response.
    """
    s = settings or get_settings()
    if not s.is_api_key_set:
        raise RerankError("CLASS_API_KEY is not configured. Set it in a local .env file.")

    if not documents:
        return []

    payload = {
        "model": model or "Qwen/Qwen3-VL-Reranker-2B",
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    headers = {
        "Authorization": f"Bearer {s.class_api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(s.reranker_url, json=payload, headers=headers, timeout=120)
    except (requests.RequestException, OSError, ConnectionError) as exc:
        raise RerankError(f"Request to reranker endpoint failed: {exc}") from exc

    if resp.status_code != 200:
        raise RerankError(f"Reranker endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        results = data["results"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RerankError(f"Unexpected reranker response shape: {resp.text[:300]}") from exc

    out: list[RerankOutcome] = []
    for r in results:
        try:
            idx = int(r["index"])
            score = float(r["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError(f"Malformed rerank result: {r}") from exc
        doc = r.get("document", {})
        out.append(RerankOutcome(index=idx, score=score, document=doc))
    return out
