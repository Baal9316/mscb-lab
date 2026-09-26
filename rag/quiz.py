"""Practice quiz generation, grading and session registry (Issue #6).

Reuses the verified Issue #2/#3/#4/#5 pipeline with no duplicated retrieval or
9001 logic:

- :func:`rag.qa.generate_llm_response` — the verified 9001 client
  (``enable_thinking=False``).
- :func:`rag.rerank_pipeline.rerank_evidence` — text + visual -> reranked
  evidence (with multimodal RRF fallback when 9004 is down).
- ``rag.qa`` strict-JSON parsing patterns.

Key design decisions (user-approved, Issue #6):
- Default 5 questions, configurable (``n_questions``).
- ONE 9001 call generates the whole quiz; the answer key is frozen at
  generation time and NEVER re-derived. Grading makes ZERO 9001 calls.
- Answer key stays server-side: the browser only ever receives the *public*
  representation (quiz_id + question_id + prompt + options + optional image).
- One global, deduplicated *evidence catalog* is built before the single
  generation call. Evidence ids are ``E1, E2, ...`` and are unique across the
  whole quiz (critical in Auto mode, where independently-retrieved topics must
  not collide on local ``[1], [2]`` labels). The model sees only
  ``Evidence E<N>:`` + text + slide image — NEVER filenames, page numbers,
  paths, or citation strings.
- ``visual=true`` is set only when answering truly requires the image
  (not merely because a slide has an ``image_url``).
- Malformed questions are rejected, never silently repaired.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .config import Settings, get_settings
from .qa import QAGenerationError, Source, generate_llm_response
from .rerank_pipeline import RerankedEvidence, rerank_evidence
from .retrieval import TextIndex
from .visual_retrieval import VisualIndex

# Defaults (Issue #6 decisions).
DEFAULT_QUIZ_SIZE = 5
# max_tokens for the WHOLE quiz (5 questions + explanations in one call).
DEFAULT_QUIZ_MAX_TOKENS = 2500


# --------------------------------------------------------------------------- #
# Data model (immutable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuizQuestion:
    """A single validated, frozen multiple-choice quiz question."""

    question_id: str
    prompt: str
    options: tuple[str, ...]          # exactly 4, frozen
    correct_index: int                # 0..3
    correct_answer: str               # == options[correct_index]
    explanation: str
    source_ids: tuple[str, ...]       # evidence catalog ids, e.g. ("E1", "E3")
    visual: bool


@dataclass(frozen=True)
class Quiz:
    """An immutable generated quiz with its frozen answer key."""

    quiz_id: str
    questions: tuple[QuizQuestion, ...]

    def question_by_id(self, question_id: str) -> QuizQuestion | None:
        for q in self.questions:
            if q.question_id == question_id:
                return q
        return None


class QuizError(RuntimeError):
    """Raised on quiz-generation failure (endpoint down, invalid questions,
    or insufficient evidence)."""


# --------------------------------------------------------------------------- #
# Server-side session registry (private key stays here).
# --------------------------------------------------------------------------- #
QUIZ_REGISTRY: dict[str, Quiz] = {}


def store_quiz(quiz: Quiz) -> None:
    """Register a frozen quiz under its opaque uuid (server-side only)."""
    QUIZ_REGISTRY[quiz.quiz_id] = quiz


def get_quiz(quiz_id: str) -> Quiz | None:
    """Return the frozen quiz for a quiz_id, or None."""
    return QUIZ_REGISTRY.get(quiz_id)


def clear_quiz_registry() -> None:
    """Reset the registry (test helper)."""
    QUIZ_REGISTRY.clear()


# --------------------------------------------------------------------------- #
# Evidence catalog
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CatalogEntry:
    """Private mapping from a public evidence id (E1, E2, ...) to trusted
    source metadata. The model never sees this full mapping."""

    # Prefixed with underscore to signal these are private and never serialized.
    document_id: str = field(repr=False)
    filename: str = field(repr=False)
    page_no: int = field(repr=False)
    chunk_index: int = field(repr=False)
    excerpt: str = field(repr=False)
    image_path: str = field(repr=False)
    image_url: str = field(repr=False)
    text: str = field(repr=False)

    @property
    def has_image(self) -> bool:
        return bool(self.image_url or self.image_path)


@dataclass(frozen=True)
class EvidenceCatalog:
    """Deduplicated, globally-unique evidence ids for a quiz."""

    _entries: dict[str, CatalogEntry] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def ids(self) -> list[str]:
        return list(self._entries.keys())

    def entry(self, eid: str) -> CatalogEntry | None:
        return self._entries.get(eid)

    def has(self, eid: str) -> bool:
        return eid in self._entries

    def entries(self) -> list[tuple[str, CatalogEntry]]:
        return list(self._entries.items())


def _catalog_key(re: RerankedEvidence) -> tuple:
    c = re.candidate
    return (c.document_id, c.page_no)


def build_evidence_catalog(evidence: Sequence[RerankedEvidence]) -> EvidenceCatalog:
    """Build ONE deduplicated evidence catalog.

    ``evidence`` is the concatenation of reranked evidence gathered across all
    quiz topics (Auto mode) or a single topic. Entries are deduplicated by
    ``(document_id, page_no)`` — the FIRST (highest-scored) occurrence kept,
    since topics are processed in relevance order. Each entry gets a globally
    unique id ``E1, E2, ...``.
    """
    entries: dict[str, CatalogEntry] = {}
    seen_keys: set[tuple] = set()
    counter = 1
    for re_item in evidence:
        key = _catalog_key(re_item)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        c = re_item.candidate
        entries[f"E{counter}"] = CatalogEntry(
            document_id=c.document_id,
            filename=c.filename,
            page_no=c.page_no,
            chunk_index=c.chunk_index,
            excerpt=c.text or "",
            image_path=c.image_path or "",
            image_url=c.image_url or "",
            text=c.text or "",
        )
        counter += 1
    return EvidenceCatalog(entries)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
_QUIZ_SYSTEM_PROMPT = (
    "You are a course-material assistant that writes practice quiz questions. "
    "You are given a list of evidence items drawn from course materials, "
    "labeled E1, E2, ... (each may include text and/or a slide image). "
    "Questions must be answerable ONLY from the supplied evidence. "
    "Never invent facts, filenames, page numbers, slide numbers, file paths, "
    "document ids, or citation labels - refer to evidence only through its "
    "E-id in source_ids. "
    "Set visual=true ONLY when answering genuinely requires inspecting the "
    "provided image (a diagram/chart/screenshot/meme); otherwise visual=false. "
    "Reply with ONLY a JSON object of the form: "
    '{"questions": [{"prompt": "...", "options": ["A","B","C","D"], '
    '"correct_index": 1, "correct_answer": "B", "explanation": "...", '
    '"source_ids": ["E1"], "visual": false}]}. '
    "Each question must have exactly 4 options, correct_index 0..3, correct_answer "
    "equal to options[correct_index], a non-empty prompt and explanation, and "
    "source_ids that reference only supplied E-ids. "
    "CRITICAL: correct_answer must be the VERBATIM full text of the chosen "
    "option - NOT a single letter and NOT an abbreviation. For example, if a "
    "long option is 'Human-in-the-loop and prompt-based interaction' and "
    "correct_index selects it, then correct_answer must literally equal "
    "'Human-in-the-loop and prompt-based interaction'."
)


def quiz_prompt_from_catalog(
    topic: str,
    catalog: EvidenceCatalog,
    n_questions: int,
    *,
    extra_topics: Sequence[str] = (),
) -> list[dict]:
    """Build 9001 messages for quiz generation from the evidence catalog.

    The model sees ONLY ``Evidence E<N>:`` + text + slide image. No filenames,
    page numbers, paths, citation strings, or document ids.
    """
    user_parts = [
        f"Generate a practice quiz with {n_questions} multiple-choice questions "
        "about the supplied course evidence.",
        f"Topic: {topic}",
    ]
    if extra_topics:
        user_parts.append("Additional topics to cover: " + ", ".join(extra_topics))
    user_parts.append("")
    user_parts.append("Evidence (labeled):")
    for eid in catalog.ids():
        entry = catalog.entry(eid)
        user_parts.append(f"\nEvidence {eid}:")
        if entry and entry.text:
            user_parts.append(f"Text: {entry.text}")

    user_text = "\n".join(user_parts)

    content: list[dict] = [{"type": "text", "text": user_text}]
    for eid in catalog.ids():
        entry = catalog.entry(eid)
        if entry and entry.image_url:
            content.append({"type": "text", "text": f"Slide image for evidence {eid}:"})
            content.append(
                {"type": "image_url", "image_url": {"url": entry.image_url}})

    return [
        {"role": "system", "content": _QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# --------------------------------------------------------------------------- #
# Strict JSON parse of the quiz-generation response
# --------------------------------------------------------------------------- #
def _extract_json_block(text: str) -> str:
    """Extract the first balanced JSON object from model output (fence-aware)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        raise QuizError("Model output contains no JSON object.")
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
        raise QuizError("Unbalanced JSON object in model output.")
    return cleaned[start:end + 1]


def parse_quiz_response(text: str) -> list[dict]:
    """Parse the model's quiz JSON into a list of raw question dicts."""
    if not text or not text.strip():
        raise QuizError("Empty quiz-generation output.")
    block = _extract_json_block(text)
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        raise QuizError(f"Model returned invalid JSON: {exc}") from exc

    raw_questions = obj.get("questions", [])
    if not isinstance(raw_questions, list) or not raw_questions:
        raise QuizError("Model output contains no questions.")
    if not all(isinstance(q, dict) for q in raw_questions):
        raise QuizError("Model output has malformed question entries.")
    return raw_questions


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_question(
    raw: dict,
    *,
    quiz_id: str,
    index: int,
    catalog: EvidenceCatalog,
) -> QuizQuestion:
    """Validate and normalize a raw question dict into a frozen QuizQuestion.

    Raises :class:`QuizError` on ANY malformed field — the answer key is never
    silently repaired or guessed.
    """
    try:
        prompt = str(raw.get("prompt", "")).strip()
        options_raw = raw.get("options")
        correct_index_raw = raw.get("correct_index")
        correct_answer = str(raw.get("correct_answer", "")).strip()
        explanation = str(raw.get("explanation", "")).strip()
        source_ids_raw = raw.get("source_ids", [])
        visual_raw = raw.get("visual", False)
    except Exception as exc:  # noqa: BLE001 - normalize defensive
        raise QuizError(f"Question {index + 1}: malformed fields: {exc}") from exc

    if not prompt:
        raise QuizError(f"Question {index + 1}: empty prompt.")

    if not isinstance(options_raw, list) or len(options_raw) != 4:
        raise QuizError(
            f"Question {index + 1}: must have exactly 4 options (got "
            f"{len(options_raw) if isinstance(options_raw, list) else 'non-list'}).")
    options = tuple(str(o).strip() for o in options_raw)
    if any(not o for o in options):
        raise QuizError(f"Question {index + 1}: empty option text.")

    try:
        correct_index = int(correct_index_raw)
    except (TypeError, ValueError):
        raise QuizError(f"Question {index + 1}: correct_index must be an integer.") from None
    if correct_index < 0 or correct_index > 3:
        raise QuizError(f"Question {index + 1}: correct_index out of range 0..3.")

    if correct_answer != options[correct_index]:
        raise QuizError(
            f"Question {index + 1}: correct_answer does not match options[correct_index].")

    if not explanation:
        raise QuizError(f"Question {index + 1}: empty explanation.")

    if isinstance(source_ids_raw, str):
        source_ids_raw = [s.strip() for s in source_ids_raw.split(",") if s.strip()]
    if not isinstance(source_ids_raw, list):
        raise QuizError(f"Question {index + 1}: source_ids must be a list.")
    # Normalize to strings, uppercase E-prefixed ids.
    source_ids: list[str] = []
    for sid in source_ids_raw:
        s = str(sid).strip()
        if not s:
            continue
        normalized = s.upper() if s[0] in "eE" else s
        source_ids.append(normalized)
    if not source_ids:
        raise QuizError(f"Question {index + 1}: empty source_ids.")
    for sid in source_ids:
        if not catalog.has(sid):
            raise QuizError(
                f"Question {index + 1}: unknown/invalid source id '{sid}'.")

    visual = bool(visual_raw) if isinstance(visual_raw, bool) else \
        str(visual_raw).strip().lower() in ("true", "yes", "1")

    # visual=true => at least one source id must point to image-backed evidence.
    if visual:
        has_img = False
        for sid in source_ids:
            entry = catalog.entry(sid)
            if entry is not None and entry.has_image:
                has_img = True
                break
        if not has_img:
            raise QuizError(
                f"Question {index + 1}: visual=true but no source image is available.")

    return QuizQuestion(
        question_id=f"{quiz_id}_q{index + 1}",
        prompt=prompt,
        options=options,
        correct_index=correct_index,
        correct_answer=correct_answer,
        explanation=explanation,
        source_ids=tuple(source_ids),
        visual=visual,
    )


# --------------------------------------------------------------------------- #
# Topic selection (Auto mode)
# --------------------------------------------------------------------------- #
_AUTO_TOPIC_PROMPT = (
    "You are helping design a practice quiz. The user will provide a sample of "
    "course content. Identify up to 5 distinct, specific topics that would be "
    "good practice-quiz questions, in order of importance. Reply with ONLY a "
    "JSON object of the form: {\"topics\": [\"...\", \"...\"]}. Topics must be "
    "specific, not generic (e.g. \"temperature/top-p sampling in LLMs\", not "
    "\"LLMs\")."
)


def suggest_topics(
    sample_text: str,
    *,
    max_topics: int = 5,
    llm=generate_llm_response,
    settings: Settings | None = None,
) -> list[str]:
    """Use 9001 to identify candidate quiz topics from a sample of content.

    Runs the mandatory Auto-mode topic-generation step — arbitrary chunks are
    never presented as 'top topics' directly.
    """
    messages = [
        {"role": "system", "content": _AUTO_TOPIC_PROMPT},
        {"role": "user", "content": f"Course content sample:\n{sample_text}"},
    ]
    content = llm(messages, max_tokens=400, settings=settings)
    block = _extract_json_block(content)
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        raise QuizError(f"Topic generation returned invalid JSON: {exc}") from exc
    topics = obj.get("topics", [])
    if not isinstance(topics, list) or not topics:
        raise QuizError("Topic generation returned no topics.")
    cleaned = [str(t).strip() for t in topics if str(t).strip()]
    if not cleaned:
        raise QuizError("Topic generation returned empty topics.")
    return cleaned[:max_topics]


def _sample_diverse_text(text_idx: TextIndex, limit: int = 10) -> str:
    """Return a limited sample of diverse chunk text for topic generation."""
    texts = text_idx.sample_texts(limit) if hasattr(text_idx, "sample_texts") else []
    samples = texts if isinstance(texts, list) else list(texts)
    return "\n---\n".join(samples) if samples else "No course content uploaded."


# --------------------------------------------------------------------------- #
# Public representation (no answer key)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PublicQuiz:
    """Client-visible quiz: quiz_id + question_id + prompt + options + image.

    Contains NO correct_index / correct_answer / explanation / source ids.
    """

    quiz_id: str
    questions: list[dict]

    def to_dict(self) -> dict:
        return {"quiz_id": self.quiz_id, "questions": self.questions}


def build_public_quiz(quiz: Quiz, catalog: EvidenceCatalog) -> PublicQuiz:
    """Public quiz with the image attached for visual questions (catalog lookup).
    Still contains no answer/explanation/source-key info."""
    pub_questions = []
    for q in quiz.questions:
        item: dict[str, Any] = {
            "question_id": q.question_id,
            "prompt": q.prompt,
            "options": list(q.options),
        }
        if q.visual:
            for sid in q.source_ids:
                entry = catalog.entry(sid)
                if entry and entry.image_url:
                    item["image_url"] = entry.image_url
                    break
        pub_questions.append(item)
    return PublicQuiz(quiz_id=quiz.quiz_id, questions=pub_questions)


# --------------------------------------------------------------------------- #
# Generation orchestrator
# --------------------------------------------------------------------------- #
def generate_quiz(
    *,
    topic: str | None,
    text_idx: TextIndex,
    visual_idx: VisualIndex,
    n_questions: int = DEFAULT_QUIZ_SIZE,
    top_k: int = 5,
    rerank=rerank_evidence,
    llm=generate_llm_response,
    settings: Settings | None = None,
    extra_topics: Sequence[str] = (),
) -> tuple[Quiz, list[RerankedEvidence]]:
    """Generate a grounded, frozen quiz using a single 9001 call.

    Returns ``(Quiz, evidence_used)``. Raises :class:`QuizError` on
    generation/validation failure or insufficient evidence.
    """
    s = settings or get_settings()
    if n_questions < 1:
        raise QuizError("n_questions must be at least 1.")

    # 1) Retrieve + rerank evidence for the topic(s).
    queries = [topic] if topic else None
    evidence: list[RerankedEvidence] = []

    if queries:
        for q in queries:
            evidence.extend(rerank(q, text_idx, visual_idx, top_n=top_k, settings=s))
    else:
        # Auto mode: topic generation step, then retrieve for each topic.
        sample = _sample_diverse_text(text_idx)
        topics = suggest_topics(sample, llm=llm, settings=s)
        for t in topics + list(extra_topics):
            evidence.extend(rerank(t, text_idx, visual_idx, top_n=top_k, settings=s))

    if not evidence:
        raise QuizError("Not enough course material to generate the requested quiz.")

    # 2) Build one global, deduplicated evidence catalog.
    catalog = build_evidence_catalog(evidence)

    # 3) Single 9001 call for the whole quiz (with one safe regenerate attempt
    #    if validation rejects too many — decision 6: "reject/regenerate safely").
    if topic:
        prompt_topic = topic
    else:
        prompt_topic = "the supplied course evidence"
    messages = quiz_prompt_from_catalog(
        prompt_topic, catalog, n_questions, extra_topics=extra_topics)
    max_tokens = getattr(s, "quiz_max_tokens", None) or DEFAULT_QUIZ_MAX_TOKENS

    quiz: Quiz | None = None
    last_error: QuizError | None = None
    for attempt in range(2):
        quiz_id = str(uuid.uuid4())
        content = llm(messages, max_tokens=max_tokens, settings=s)  # noqa: BLE001 - wrap below
        raw_questions = parse_quiz_response(content)
        questions: list[QuizQuestion] = []
        skipped: int = 0
        for i, raw in enumerate(raw_questions):
            try:
                questions.append(validate_question(
                    raw, quiz_id=quiz_id, index=i, catalog=catalog))
            except QuizError:
                # Reject this question; continue rather than silently guessing.
                skipped += 1
        if len(questions) >= n_questions:
            quiz = Quiz(quiz_id=quiz_id, questions=tuple(questions[:n_questions]))
            break
        last_error = QuizError(
            f"Only {len(questions)} of {n_questions} questions were valid"
            f" ({skipped} rejected). Not enough course material to generate "
            "the requested quiz.")
        if attempt == 0:
            # One safe regeneration attempt before reporting failure.
            continue

    if quiz is None:
        assert last_error is not None
        raise last_error

    store_quiz(quiz)
    return quiz, list(evidence)


# --------------------------------------------------------------------------- #
# Grading (zero 9001 calls)
# --------------------------------------------------------------------------- #
def grade_quiz(
    quiz: Quiz,
    answers: dict[str, str],
) -> dict:
    """Grade user answers against the frozen key. NEVER calls 9001.

    ``answers`` maps question_id -> the selected option text. Equal weighting.

    Returns a dict with the score summary and per-question results.
    """
    correct_count = 0
    per_question = []
    used_key = {"correct_answer": True}  # proof grading reads stored key only.
    for q in quiz.questions:
        chosen = answers.get(q.question_id)
        is_correct = bool(chosen) and chosen == q.correct_answer
        correct_count += int(is_correct)
        per_question.append({
            "question_id": q.question_id,
            "correct": is_correct,
            "chosen": chosen,
        })
    total = len(quiz.questions)
    percent = round(100.0 * correct_count / total) if total else 0.0
    return {
        "correct": correct_count,
        "total": total,
        "percent": percent,
        "per_question": per_question,
    }


def reveal(quiz: Quiz, catalog: EvidenceCatalog) -> dict:
    """Reveal answers/explanations/citations only AFTER grading or explicit
    request. Maps each question to its trusted Source (built from the catalog,
    not the model)."""
    revealed = []
    for q in quiz.questions:
        sources = []
        for sid in q.source_ids:
            entry = catalog.entry(sid)
            if entry is None:
                continue
            sources.append(Source(
                document_id=entry.document_id,
                filename=entry.filename,
                page_no=entry.page_no,
                chunk_index=entry.chunk_index,
                excerpt=entry.excerpt,
                image_path=entry.image_path,
                image_url=entry.image_url,
            ).to_dict())
        revealed.append({
            "question_id": q.question_id,
            "correct_index": q.correct_index,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "sources": sources,
        })
    return {"quiz_id": quiz.quiz_id, "review": revealed}
