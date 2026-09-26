"""Question answering + citations for the multimodal RAG.

Issue #5 scope. Connects the verified retrieval/reranking pipeline to a
vision-capable answer-generation endpoint (9001) and produces a structured,
grounded answer with trusted sources.

Design (approved):
- The LLM receives the user question plus the top reranked evidence (text +
  slide images when available). Thinking is disabled and ``max_tokens`` ~500.
- The model is instructed to reply with STRICT JSON:
      {"answer": "...", "source_ids": [1, 3], "insufficient": false}
- The model NEVER emits filenames/pages/excerpts/paths. Python validates every
  ``source_id`` against the actual reranked evidence and builds the final
  trusted :class:`Source` list itself.
- Grounding: the model answers only from supplied evidence and must report
  ``insufficient`` rather than inventing content or citations.

9001 endpoint contract (verified live):
    POST {LLM_URL}   (default http://dobolyi.com:9001/v1/chat/completions)
    Authorization: Bearer <key>
    body: {"model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
           "messages": [...],
           "max_tokens": 500, "temperature": 0.0,
           "chat_template_kwargs": {"enable_thinking": False}}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import requests

from .config import Settings, get_settings
from .rerank_pipeline import RerankedEvidence, rerank_evidence
from .retrieval import TextIndex
from .visual_retrieval import VisualIndex

# 9001 defaults (verified).
DEFAULT_LLM_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
DEFAULT_LLM_URL = "http://dobolyi.com:9001/v1/chat/completions"


class QAGenerationError(RuntimeError):
    """Raised when the answer-generation endpoint fails or returns unusable output."""


@dataclass
class Source:
    """A validated citation, built only from trusted reranked evidence."""

    document_id: str
    filename: str
    page_no: int
    excerpt: str
    image_path: str
    image_url: str
    chunk_index: int = 0
    citation: str = ""

    def __post_init__(self) -> None:
        if not self.citation:
            self.citation = f"{self.filename} · page {self.page_no}"

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_no": self.page_no,
            "chunk_index": self.chunk_index,
            "excerpt": self.excerpt,
            "image_path": self.image_path,
            "image_url": self.image_url,
            "citation": self.citation,
        }


@dataclass
class StructuredAnswer:
    """Final validated result from the QA pipeline."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    insufficient: bool = False
    used_fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "insufficient": self.insufficient,
            "used_fallback": self.used_fallback,
        }


# --------------------------------------------------------------------------- #
# 9001 client
# --------------------------------------------------------------------------- #
def generate_llm_response(
    messages: list[dict],
    *,
    max_tokens: int = 500,
    settings: Settings | None = None,
) -> str:
    """Call the 9001 chat-completions endpoint and return message content.

    Raises:
        QAGenerationError: on missing key, request failure, HTTP error, or empty
            content.
    """
    s = settings or get_settings()
    if not s.is_api_key_set:
        raise QAGenerationError(
            "CLASS_API_KEY is not configured. Set it in a local .env file.")

    llm_url = getattr(s, "llm_url", None) or DEFAULT_LLM_URL
    llm_model = getattr(s, "llm_model", None) or DEFAULT_LLM_MODEL

    payload = {
        "model": llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {s.class_api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(llm_url, json=payload, headers=headers, timeout=180)
    except (requests.RequestException, OSError, ConnectionError) as exc:
        raise QAGenerationError(f"Request to answer-generation endpoint failed: {exc}") from exc

    if resp.status_code != 200:
        raise QAGenerationError(
            f"Answer-generation endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise QAGenerationError(
            f"Malformed answer-generation response: {resp.text[:300]}") from exc

    if not content or not content.strip():
        raise QAGenerationError("Answer-generation endpoint returned empty content.")
    return content.strip()


# --------------------------------------------------------------------------- #
# Strict JSON parse of the model's answer block
# --------------------------------------------------------------------------- #
def parse_structured_answer(text: str) -> dict:
    """Parse the model's JSON answer into {answer, source_ids, insufficient}.

    Tolerates the model wrapping JSON in backticks or extra prose: extracts the
    first ``{...}`` block and parses it. Missing fields default reasonably.
    Raises :class:`QAGenerationError` if no usable JSON is found.
    """
    if not text or not text.strip():
        raise QAGenerationError("Empty model output.")
    # Strip code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    # Find the first balanced JSON object
    start = cleaned.find("{")
    if start == -1:
        raise QAGenerationError("Model output contains no JSON object.")
    depth, end = 0, -1
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise QAGenerationError("Unbalanced JSON object in model output.")
    block = cleaned[start:end + 1]

    def _coerce_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"):
            return True
        return False

    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        raise QAGenerationError(f"Model returned invalid JSON: {exc}") from exc

    answer = str(obj.get("answer", "")).strip()
    if not answer:
        raise QAGenerationError("Model JSON has empty 'answer'.")

    raw_ids = obj.get("source_ids", [])
    if isinstance(raw_ids, int):
        raw_ids = [raw_ids]
    source_ids = []
    for rid in raw_ids:
        try:
            source_ids.append(int(rid))
        except (TypeError, ValueError):
            continue

    insufficient = _coerce_bool(obj.get("insufficient", False))
    # If insufficient, ignore source ids (no valid sources allowed).
    return {
        "answer": answer,
        "source_ids": [] if insufficient else source_ids,
        "insufficient": insufficient,
    }


# --------------------------------------------------------------------------- #
# Source validation
# --------------------------------------------------------------------------- #
def build_sources(source_ids: list[int],
                  evidence: list[RerankedEvidence]) -> list[Source]:
    """Build trusted Sources from reranked evidence, validating every source_id.

    ``source_id`` is 1-based into the ``evidence`` list (matching how the model
    is instructed to index evidence). Out-of-range or negative ids are skipped —
    the model cannot inject a source that was not actually supplied.
    """
    sources: list[Source] = []
    seen = set()
    for sid in source_ids:
        idx = sid - 1
        if idx < 0 or idx >= len(evidence):
            continue  # reject out-of-range / fabricated id
        if idx in seen:
            continue
        seen.add(idx)
        e = evidence[idx].candidate
        sources.append(Source(
            document_id=e.document_id,
            filename=e.filename,
            page_no=e.page_no,
            chunk_index=e.chunk_index,
            excerpt=e.text or "",
            image_path=e.image_path,
            image_url=e.image_url,
        ))
    return sources


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are a course-material assistant. Answer the user's question using "
    "ONLY the supplied course evidence, which is provided as numbered items "
    "[1], [2], ... (each may include text and/or a slide image). "
    "Answer the substantive question in a concise, grounded way. "
    "When the evidence is insufficient to answer, set insufficient=true and "
    "give a clear answer saying the course materials do not contain enough "
    "information. "
    "Never invent facts, filenames, page numbers, slide numbers, file paths, "
    "or source labels. "
    "Do NOT mention filenames, slide numbers, page numbers, paths, or citation "
    "labels inside your answer text at all - leave all citation and location "
    "display to the caller. Refer to specific evidence only through its "
    "numbered index in source_ids, never by name or page. "
    "Reply with ONLY a JSON object of the form: "
    '{"answer": "...", "source_ids": [1, 3], "insufficient": false}. '
    "source_ids are the 1-based indices of the evidence items you actually "
    "used. If insufficient, source_ids must be []."
)


def prompt_from_evidence(question: str, evidence: list[RerankedEvidence]) -> list[dict]:
    """Build the 9001 messages: system prompt + user (question + evidence).

    The model-facing prompt exposes ONLY the numbered evidence index, the text,
    and any associated slide image. Filenames, page numbers, file paths,
    citation strings, and document ids are NEVER sent to 9001 — Python is the
    only component that maps a source_id back to trusted citation metadata.

    Each evidence item's label, text, and slide image are bundled together so
    the model can associate an image with its numbered index.
    """
    # Question + evidence instructions read as one text block (no citation info).
    user_text_parts = ["Answer this question:", question, "", "Evidence (numbered):"]
    for i, e in enumerate(evidence, start=1):
        user_text_parts.append(f"\nEvidence [{i}]:")
        if e.candidate.text:
            user_text_parts.append(f"Text: {e.candidate.text}")
    user_text = "\n".join(user_text_parts)

    # Content array: numbered label, then its slide image, per evidence item.
    content: list[dict] = [{"type": "text", "text": user_text}]
    for i, e in enumerate(evidence[:5], start=1):
        if e.candidate.image_url:
            content.append({
                "type": "text",
                "text": f"Slide image for evidence [{i}]:",
            })
            content.append({"type": "image_url", "image_url": {"url": e.candidate.image_url}})

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def answer_question(
    question: str,
    text_idx: TextIndex,
    visual_idx: VisualIndex,
    *,
    top_n: int = 5,
    llm=generate_llm_response,
    rerank=rerank_evidence,
    settings: Settings | None = None,
) -> StructuredAnswer:
    """End-to-end: retrieve, rerank, generate, validate, and structure.

    Returns:
        StructuredAnswer — answer + trusted sources + flags.

    Raises:
        QAGenerationError: if the generation endpoint fails/unusable output.
    """
    evidence = rerank(question, text_idx, visual_idx, top_n=top_n, settings=settings)
    used_fallback = bool(evidence) and all(e.source == "rerank-fallback" for e in evidence)

    if not evidence:
        # No candidates at all -> nothing to ground on.
        return StructuredAnswer(
            answer="The course materials do not contain enough information to answer this question.",
            sources=[],
            insufficient=True,
            used_fallback=False,
        )

    messages = prompt_from_evidence(question, evidence)
    content = llm(messages, max_tokens=500, settings=settings)
    parsed = parse_structured_answer(content)

    if parsed["insufficient"]:
        return StructuredAnswer(
            answer=parsed["answer"],
            sources=[],
            insufficient=True,
            used_fallback=used_fallback,
        )

    sources = build_sources(parsed["source_ids"], evidence)
    return StructuredAnswer(
        answer=parsed["answer"],
        sources=sources,
        insufficient=False,
        used_fallback=used_fallback,
    )
