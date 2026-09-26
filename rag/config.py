"""Application configuration.

Reads all secrets and endpoint URLs from environment variables (loaded from a
local ``.env`` file when present). The real API key is NEVER hard-coded here
and must not be committed; see ``.env.example`` for the shape.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load optional local .env (does nothing if absent).
load_dotenv()

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _get(name: str, default: str) -> str:
    """Return env var ``name`` or ``default`` (never None)."""
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    """Typed view over environment configuration."""

    #: Shared class API key. Must only come from the environment / .env.
    class_api_key: str = field(default_factory=lambda: _get("CLASS_API_KEY", ""))
    text_embedding_url: str = field(default_factory=lambda: _get(
        "TEXT_EMBEDDING_URL", "http://dobolyi.com:9002/v2/embed"))
    visual_embedding_url: str = field(default_factory=lambda: _get(
        "VISUAL_EMBEDDING_URL", "http://dobolyi.com:9003/v1/embeddings"))
    reranker_url: str = field(default_factory=lambda: _get(
        "RERANKER_URL", "http://dobolyi.com:9004/rerank"))
    document_parser_url: str = field(default_factory=lambda: _get(
        "DOCUMENT_PARSER_URL", "http://dobolyi.com:9005/v1/chat/completions"))
    #: Parser model id.
    document_parser_model: str = field(default_factory=lambda: _get(
        "DOCUMENT_PARSER_MODEL", "dots.mocr"))

    #: Answer-generation (vision) LLM endpoint + model (9001).
    llm_url: str = field(default_factory=lambda: _get(
        "LLM_URL", "http://dobolyi.com:9001/v1/chat/completions"))
    llm_model: str = field(default_factory=lambda: _get(
        "LLM_MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"))

    #: max_tokens for the single 9001 quiz-generation call (whole quiz).
    quiz_max_tokens: int = field(default_factory=lambda: int(
        _get("QUIZ_MAX_TOKENS", "2500")))

    #: Directory where uploaded documents, rendered pages and indexes live.
    data_dir: Path = field(default_factory=lambda: Path(
        _get("DATA_DIR", str(_DEFAULT_DATA_DIR))))

    @property
    def is_api_key_set(self) -> bool:
        return bool(self.class_api_key) and self.class_api_key != "your_key_here"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) application settings."""
    return Settings()


def default_data_dir() -> Path:
    return _DEFAULT_DATA_DIR
