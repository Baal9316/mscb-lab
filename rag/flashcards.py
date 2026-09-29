"""Flashcard generation (study cards) grounded in course evidence.

Mirrors the Issue #6 quiz conventions with no duplicated retrieval or LLM
logic: it reuses the verified Issue #2/#3/#4/#5 pipeline exactly like
``rag.quiz`` does:

- :func:`rag.qa.generate_llm_response` — the verified 9001 client.
- :func:`rag.rerank_pipeline.rerank_evidence` — text + visual -> reranked
  evidence (with multimodal RRF fallback when 9004 is down).
- :func:`rag.quiz.build_evidence_catalog` / ``suggest_topics`` — deduplicated,
  globally-unique evidence ids (``E1, E2, ...``) and the Auto-mode topic
  suggester.

Design decisions (same spirit as Issue #6):
- ONE 9001 call generates the whole deck; cards are frozen at generation time.
- The model sees only ``Evidence E<N>:`` + text + slide image — NEVER
  filenames, page numbers, paths, or citation strings. ``source_ids`` are
  private catalog ids resolved server-side when the user reveals a card.
- ``visual=true`` only when answering requires the slide image.
- Malformed cards are rejected, never silently repaired.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import Settings, get_settings
from .qa import QAGenerationError, generate_llm_response
from .quiz import (
    EvidenceCatalog, QuizError, _extract_json_block, build_evidence_catalog,
    suggest_topics, _sample_diverse_text,
)
from .rerank_pipeline import RerankedEvidence, rerank_evidence
from .retrieval import TextIndex
from .visual_retrieval import VisualIndex

# Defaults.
DEFAULT_CARD_COUNT = 8
DEFAULT_MAX_TOKENS = 2000


class FlashcardError(RuntimeError):
    """Raised when flashcard generation or validation fails."""


# --------------------------------------------------------------------------- #
# Data model (immutable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Flashcard:
    """A single validated study card.

    ``front`` is the prompt/concept; ``back`` is the answer. ``source_ids`` are
    PRIVATE evidence-catalog ids (e.g. ``("E1", "E3")``) resolved server-side —
    they never appear in what the model sees or what the browser receives
    (the UI shows resolved citations instead).
    """

    card_id: str
    front: str
    back: str
    source_ids: tuple[str, ...]
    visual: bool


@dataclass(frozen=True)
class FlashcardDeck:
    deck_id: str
    cards: tuple[Flashcard, ...]


# --------------------------------------------------------------------------- #
# Prompt construction — model sees ONLY evidence entries, no filenames/pages
# --------------------------------------------------------------------------- #
_FLASHCARD_SYSTEM_PROMPT = (
    "You are a course-material assistant that writes study flashcards from "
    "the supplied evidence. Write flashcards that help a student memorize and "
    "understand the key concepts.\n"
    "For each card:\n"
    "- FRONT: a short prompt, question, or concept name.\n"
    "- BACK: the correct, concise answer (at most two sentences), faithful ONLY "
    "to the supplied evidence. Never invent facts that are not in the evidence.\n"
    "- source_ids: the E-ids of the evidence entries the answer is based on "
    "(1-2 entries).\n"
    "- visual: true ONLY if the answer genuinely requires looking at a slide "
    "image (e.g. a diagram), otherwise false.\n"
    "Reply with ONLY a single JSON object (no text before or after):\n"
    '{"cards": [{"front": "...", "back": "...", '
    '"source_ids": ["E1"], "visual": false}]}'
)


def _catalog_block(catalog: EvidenceCatalog) -> str:
    """Render the catalog the model may cite from (evidence ids only)."""
    lines = []
    for eid, entry in catalog.entries():
        lines.append(f"Evidence {eid}: {entry.text}")
        if entry.has_image:
            lines.append(f"(slide image available for Evidence {eid})")
    return "\n".join(lines)


def flashcard_prompt_from_catalog(
    topic: str, catalog: EvidenceCatalog, n_cards: int,
    extra_topics: Sequence[str] = (),
) -> list[dict]:
    user = [
        "Create study flashcards for this topic: " + topic
        if topic else "Create study flashcards from the course evidence below.",
    ]
    topic_list = [t for t in (extra_topics or ()) if t and t != topic]
    if topic and topic_list:
        user.append("Also cover these related topics: " + "; ".join(topic_list))
    user.append(f"Number of cards: {n_cards}")
    user.append("Evidence (only sources you may use):\n" + _catalog_block(catalog))
    return [
        {"role": "system", "content": _FLASHCARD_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user)},
    ]


# --------------------------------------------------------------------------- #
# Strict parsing + validation (no silent repairs)
# --------------------------------------------------------------------------- #
def parse_flashcards_response(content: str) -> list[dict]:
    """Parse the model's JSON reply into a list of raw card dicts."""
    if not content or not content.strip():
        raise FlashcardError("Model returned an empty response.")
    try:
        block = _extract_json_block(content)
    except QuizError as exc:
        raise FlashcardError(str(exc)) from exc
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        raise FlashcardError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("cards"), list):
        raise FlashcardError("Expected a JSON object with a 'cards' array.")
    return obj["cards"]


def validate_card(
    raw: dict, *, deck_id: str, index: int, catalog: EvidenceCatalog,
) -> Flashcard:
    """Validate one raw card against the evidence catalog (strict)."""
    if not isinstance(raw, dict):
        raise FlashcardError(f"Card {index + 1}: not an object.")
    front = raw.get("front")
    back = raw.get("back")
    if not isinstance(front, str) or not front.strip():
        raise FlashcardError(f"Card {index + 1}: missing/invalid 'front'.")
    if not isinstance(back, str) or not back.strip():
        raise FlashcardError(f"Card {index + 1}: missing/invalid 'back'.")
    source_ids = raw.get("source_ids", [])
    if not isinstance(source_ids, list) or not source_ids:
        raise FlashcardError(
            f"Card {index + 1}: 'source_ids' must be a non-empty list.")
    ids = []
    for sid in source_ids:
        if not isinstance(sid, str) or not catalog.has(sid):
            raise FlashcardError(
                f"Card {index + 1}: unknown evidence id {sid!r}.")
        ids.append(sid)
    visual = raw.get("visual", False)
    if not isinstance(visual, bool):
        visual = bool(visual)
    return Flashcard(
        card_id=f"{deck_id}:c{index + 1}",
        front=front.strip(), back=back.strip(),
        source_ids=tuple(ids), visual=visual,
    )


# --------------------------------------------------------------------------- #
# Generation orchestrator — ONE 9001 call for the whole deck
# --------------------------------------------------------------------------- #
def generate_flashcards(
    *,
    topic: str | None,
    text_idx: TextIndex,
    visual_idx: VisualIndex,
    n_cards: int = DEFAULT_CARD_COUNT,
    top_k: int = 5,
    rerank=rerank_evidence,
    llm=generate_llm_response,
    settings: Settings | None = None,
    extra_topics: Sequence[str] = (),
) -> tuple[FlashcardDeck, list[RerankedEvidence]]:
    """Generate a grounded flashcard deck using a single 9001 call.

    Returns ``(FlashcardDeck, evidence_used)``. Raises :class:`FlashcardError`
    on generation/validation failure or insufficient evidence.
    """
    s = settings or get_settings()
    if n_cards < 1:
        raise FlashcardError("n_cards must be at least 1.")

    # 1) Retrieve + rerank evidence (same path as the quiz pipeline).
    queries = [topic] if topic else None
    evidence: list[RerankedEvidence] = []
    if queries:
        for q in queries:
            evidence.extend(rerank(q, text_idx, visual_idx, top_n=top_k, settings=s))
    else:
        sample = _sample_diverse_text(text_idx)
        topics = suggest_topics(sample, llm=llm, settings=s)
        for t in topics + list(extra_topics):
            evidence.extend(rerank(t, text_idx, visual_idx, top_n=top_k, settings=s))

    if not evidence:
        raise FlashcardError(
            "Not enough course material to generate flashcards. "
            "Upload a document first.")

    # 2) One global, deduplicated evidence catalog.
    catalog = build_evidence_catalog(evidence)

    # 3) Single 9001 call (one safe regenerate attempt on validation failure).
    prompt_topic = topic or "the supplied course evidence"
    messages = flashcard_prompt_from_catalog(
        prompt_topic, catalog, n_cards, extra_topics=extra_topics)
    max_tokens = getattr(s, "flashcard_max_tokens", None) or DEFAULT_MAX_TOKENS

    deck: FlashcardDeck | None = None
    last_error: FlashcardError | None = None
    for _attempt in range(2):
        deck_id = str(uuid.uuid4())
        try:
            content = llm(messages, max_tokens=max_tokens, settings=s)
        except QAGenerationError as exc:  # noqa: BLE001
            raise FlashcardError(f"LLM call failed: {exc}") from exc
        raw_cards = parse_flashcards_response(content)
        cards: list[Flashcard] = []
        skipped = 0
        for i, raw in enumerate(raw_cards):
            try:
                cards.append(validate_card(raw, deck_id=deck_id, index=i,
                                           catalog=catalog))
            except FlashcardError:
                skipped += 1
        if len(cards) >= n_cards:
            deck = FlashcardDeck(deck_id=deck_id, cards=tuple(cards[:n_cards]))
            break
        last_error = FlashcardError(
            f"Only {len(cards)} of {n_cards} cards were valid "
            f"({skipped} rejected). Not enough grounded material to generate "
            "the requested deck.")

    if deck is None:
        raise last_error or FlashcardError("Flashcard generation failed.")
    return deck, evidence


# --------------------------------------------------------------------------- #
# Starred-card registry (persisted, survives sessions)
# --------------------------------------------------------------------------- #
# Entries are keyed by a content hash (front+back), so re-generated decks
# don't duplicate stars. Origin tracks how the card got in:
#   "manual"    -> student starred it in the Flashcards tab
#   "quiz-miss" -> auto-created from a missed practice-quiz question
STARRED_CARDS_FILE = "starred_cards.json"


def starred_cards_path(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return Path(s.data_dir) / STARRED_CARDS_FILE


def load_starred_cards(settings: Settings | None = None) -> list[dict]:
    """Load persisted starred cards (newest first)."""
    path = starred_cards_path(settings)
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return items if isinstance(items, list) else []


def _save_starred_cards(settings, items: list[dict]) -> None:
    path = starred_cards_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def card_key(front: str, back: str) -> str:
    """Stable identity for a card (star dedupe across regenerated decks)."""
    import hashlib
    return hashlib.sha256(f"{front.strip()}\n{back.strip()}".encode()).hexdigest()[:16]


def star_card(
    settings: Settings | None,
    *,
    front: str,
    back: str,
    sources: Sequence[str] = (),
    origin: str = "manual",
) -> dict:
    """Star a card. Returns the (possibly pre-existing) stored entry."""
    s = settings or get_settings()
    key = card_key(front, back)
    items = [it for it in load_starred_cards(s) if it.get("key") != key]
    entry = {
        "key": key, "front": front.strip(), "back": back.strip(),
        "sources": list(sources), "origin": origin,
    }
    items.insert(0, entry)
    _save_starred_cards(s, items)
    return entry


def unstar_card(settings: Settings | None, key: str) -> bool:
    """Remove a starred card by key. Returns True if it existed."""
    s = settings or get_settings()
    items = load_starred_cards(s)
    kept = [it for it in items if it.get("key") != key]
    if len(kept) == len(items):
        return False
    _save_starred_cards(s, kept)
    return True


def add_missed_quiz_cards(
    settings: Settings | None,
    missed: Sequence[dict],
) -> tuple[int, list[str]]:
    """Auto-star cards for missed quiz questions.

    ``missed`` is a list of ``{"front", "back", "sources"}``. Already-starred
    cards are not duplicated. Returns (newly_added, keys).
    """
    s = settings or get_settings()
    existing = {it.get("key") for it in load_starred_cards(s)}
    added = 0
    keys: list[str] = []
    for m in missed:
        entry = star_card(s, front=m["front"], back=m["back"],
                          sources=m.get("sources", ()), origin="quiz-miss")
        keys.append(entry["key"])
        if entry["key"] not in existing:
            added += 1
    return added, keys
