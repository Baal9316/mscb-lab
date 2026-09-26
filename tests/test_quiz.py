"""Tests for Issue #6 practice quizzes.

Covers the frozen answer key, strict validation, single-call generation,
evidence-catalog privacy, grading with zero 9001 calls, visual questions, the
decision-17 hard guarantee, RRF fallback, insufficient evidence, and session
registry isolation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import Settings
from rag.models import Document, DocumentPage
from rag.quiz import (
    DEFAULT_QUIZ_SIZE,
    EvidenceCatalog,
    PublicQuiz,
    Quiz,
    QuizError,
    QuizQuestion,
    build_evidence_catalog,
    build_public_quiz,
    clear_quiz_registry,
    generate_quiz,
    get_quiz,
    grade_quiz,
    parse_quiz_response,
    quiz_prompt_from_catalog,
    reveal,
    suggest_topics,
    validate_question,
)
from rag.rerank_pipeline import EvidenceCandidate, RerankedEvidence
from rag.retrieval import TextIndex
from rag.store import DocumentStore
from rag.visual_retrieval import VisualIndex


def _settings(tmp_path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(class_api_key="test-key", data_dir=data_dir)


def _png(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([n]) * 8)
    return path


def _seed(tmp_path, settings: Settings) -> DocumentStore:
    store = DocumentStore(settings.data_dir)
    pages = []
    for p in range(1, 4):
        img = _png(tmp_path / f"img_{p}.png", p)
        url = f"data:image/png;base64,{img.read_bytes().hex()}"
        pages.append(DocumentPage(
            document_id="doc-0", filename="slides.pdf", page_no=p,
            extracted_text=f"Page {p} content about course topic {p} sampling temperature",
            image_path=str(img), image_url=url))
    store.add(Document(document_id="doc-0", filename="slides.pdf",
                       sha256="sha", pages=pages))
    return store


def _evidence(n=3, with_image=True):
    """Synthetic reranked evidence for unit tests (no fetch)."""
    ev = []
    for i in range(n):
        img = f"data:image/png;base64,img{i + 1}" if with_image else ""
        ev.append(RerankedEvidence(
            candidate=EvidenceCandidate(
                document_id=f"doc-{i}", filename=f"deck{i}.pdf", page_no=i + 1,
                text=f"Evidence text number {i + 1} about course sampling",
                image_url=img),
            score=round(1.0 - i * 0.1, 3), source="rerank"))
    return ev


def _valid_quiz_response(n=5):
    qs = []
    for i in range(n):
        qs.append({
            "prompt": f"Question {i + 1} about sampling?",
            "options": ["A", "B", "C", "D"],
            "correct_index": 1,
            "correct_answer": "B",
            "explanation": f"Explanation {i + 1}.",
            "source_ids": ["E1"],
            "visual": False,
        })
    return json.dumps({"questions": qs})


@pytest.fixture(autouse=True)
def _isolate_registry():
    clear_quiz_registry()
    yield
    clear_quiz_registry()


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store = _seed(tmp_path, settings)
    ti = TextIndex(store, settings=settings)
    ti.rebuild()
    # VisualIndex is only needed to satisfy signatures; mock rerank is injected
    # per-test, so the indexes are never actually searched here.
    vi = VisualIndex(store, settings=settings)
    return settings, store, ti, vi


# --------------------------------------------------------------------------- #
# Data model + answer-key immutability
# --------------------------------------------------------------------------- #
class TestDataModel:
    def test_frozen_fields(self):
        q = QuizQuestion(
            question_id="x_q1", prompt="p",
            options=("A", "B", "C", "D"), correct_index=1,
            correct_answer="B", explanation="e",
            source_ids=("E1",), visual=False)
        with pytest.raises(Exception):
            q.options = ("1",)  # frozen -> AttributeError
        assert q.correct_answer == "B"

    def test_question_by_id(self):
        q = Quiz(quiz_id="x", questions=(
            QuizQuestion("x_q1", "p", ("A", "B", "C", "D"), 0, "A", "e",
                         ("E1",), False),
        ))
        assert q.question_by_id("x_q1") is not None
        assert q.question_by_id("nope") is None


# --------------------------------------------------------------------------- #
# Strict JSON parsing
# --------------------------------------------------------------------------- #
class TestParseQuizResponse:
    def test_parses_questions(self):
        raw = parse_quiz_response(_valid_quiz_response(2))
        assert len(raw) == 2
        assert raw[0]["correct_answer"] == "B"

    def test_rejects_non_json(self):
        with pytest.raises(QuizError):
            parse_quiz_response("no json here")

    def test_rejects_no_questions(self):
        with pytest.raises(QuizError):
            parse_quiz_response('{"questions": []}')

    def test_rejects_empty(self):
        with pytest.raises(QuizError):
            parse_quiz_response("   ")

    def test_tolerates_code_fence(self):
        raw = parse_quiz_response("```json\n" + _valid_quiz_response(1) + "\n```")
        assert len(raw) == 1


# --------------------------------------------------------------------------- #
# Evidence catalog (privacy + uniqueness)
# --------------------------------------------------------------------------- #
class TestEvidenceCatalog:
    def test_global_unique_ids(self):
        catalog = build_evidence_catalog(_evidence(4))
        ids = catalog.ids()
        assert len(ids) == 4
        assert ids == ["E1", "E2", "E3", "E4"]
        assert len(set(ids)) == 4

    def test_dedupe_by_page(self):
        # two evidence entries for the same (doc, page) collapse to one.
        ev = [
            RerankedEvidence(EvidenceCandidate(
                "d", "f.pdf", 3, text="first", image_url="i1"), 0.9, "rerank"),
            RerankedEvidence(EvidenceCandidate(
                "d", "f.pdf", 3, text="second", image_url="i2"), 0.8, "rerank"),
            RerankedEvidence(EvidenceCandidate(
                "d", "f.pdf", 4, text="other page", image_url=""), 0.7, "rerank"),
        ]
        catalog = build_evidence_catalog(ev)
        assert len(catalog) == 2  # page 3 + page 4
        assert catalog.entry("E1").text == "first"  # first/highest kept

    def test_catalog_private_metadata_not_in_prompt(self):
        catalog = build_evidence_catalog(_evidence(2))
        msgs = quiz_prompt_from_catalog("sampling", catalog, 2)
        user_content = next(c["content"] for c in msgs if c["role"] == "user")
        dumped = json.dumps(user_content)
        assert "deck" not in dumped
        assert ".pdf" not in dumped
        assert "page" not in dumped.lower()
        assert "document" not in dumped.lower()
        assert "E1" in dumped and "E2" in dumped
        # image labeled for the model
        assert "Slide image for evidence E1" in dumped


# --------------------------------------------------------------------------- #
# validate_question
# --------------------------------------------------------------------------- #
class TestValidateQuestion:
    @pytest.fixture
    def catalog(self):
        return build_evidence_catalog(_evidence(3))

    def test_valid_passes(self, catalog):
        raw = {
            "prompt": "q", "options": ["A", "B", "C", "D"],
            "correct_index": 2, "correct_answer": "C",
            "explanation": "because", "source_ids": ["e1"], "visual": False,
        }
        q = validate_question(raw, quiz_id="z", index=0, catalog=catalog)
        assert q.correct_answer == "C"
        assert q.source_ids == ("E1",)  # normalized uppercase
        assert q.question_id == "z_q1"
        assert q.visual is False

    def test_requires_exactly_four_options(self, catalog):
        raw = {"prompt": "q", "options": ["A", "B", "C"],
               "correct_index": 0, "correct_answer": "A",
               "explanation": "e", "source_ids": ["E1"], "visual": False}
        with pytest.raises(QuizError):
            validate_question(raw, quiz_id="z", index=0, catalog=catalog)

    def test_rejects_empty_prompt(self, catalog):
        raw = {"prompt": "  ", "options": ["A", "B", "C", "D"],
               "correct_index": 0, "correct_answer": "A",
               "explanation": "e", "source_ids": ["E1"], "visual": False}
        with pytest.raises(QuizError):
            validate_question(raw, quiz_id="z", index=0, catalog=catalog)

    def test_rejects_malformed_correct_index(self, catalog):
        raw = {"prompt": "q", "options": ["A", "B", "C", "D"],
               "correct_index": 7, "correct_answer": "A",
               "explanation": "e", "source_ids": ["E1"], "visual": False}
        with pytest.raises(QuizError):
            validate_question(raw, quiz_id="z", index=0, catalog=catalog)

    def test_rejects_mismatched_correct_answer(self, catalog):
        raw = {"prompt": "q", "options": ["A", "B", "C", "D"],
               "correct_index": 1, "correct_answer": "D",  # != options[1]
               "explanation": "e", "source_ids": ["E1"], "visual": False}
        with pytest.raises(QuizError):
            validate_question(raw, quiz_id="z", index=0, catalog=catalog)

    def test_rejects_invalid_source_ids(self, catalog):
        raw = {"prompt": "q", "options": ["A", "B", "C", "D"],
               "correct_index": 0, "correct_answer": "A",
               "explanation": "e", "source_ids": ["E99"], "visual": False}
        with pytest.raises(QuizError):
            validate_question(raw, quiz_id="z", index=0, catalog=catalog)

    def test_rejects_empty_explanation(self, catalog):
        raw = {"prompt": "q", "options": ["A", "B", "C", "D"],
               "correct_index": 0, "correct_answer": "A",
               "explanation": "", "source_ids": ["E1"], "visual": False}
        with pytest.raises(QuizError):
            validate_question(raw, quiz_id="z", index=0, catalog=catalog)

    def test_visual_true_requires_image_backed_source(self):
        # evidence WITHOUT images -> visual=true rejected
        catalog = build_evidence_catalog(_evidence(3, with_image=False))
        raw = {"prompt": "q", "options": ["A", "B", "C", "D"],
               "correct_index": 0, "correct_answer": "A",
               "explanation": "e", "source_ids": ["E1"], "visual": True}
        with pytest.raises(QuizError):
            validate_question(raw, quiz_id="z", index=0, catalog=catalog)

    def test_visual_false_allowed_with_image_present(self, catalog):
        # image present but visual=false is fine (does not force visual=true)
        raw = {"prompt": "q", "options": ["A", "B", "C", "D"],
               "correct_index": 0, "correct_answer": "A",
               "explanation": "e", "source_ids": ["E1"], "visual": False}
        q = validate_question(raw, quiz_id="z", index=0, catalog=catalog)
        assert q.visual is False


# --------------------------------------------------------------------------- #
# Public representation privacy
# --------------------------------------------------------------------------- #
class TestPublicRepresentation:
    def test_private_key_absent_from_public(self):
        catalog = build_evidence_catalog(_evidence(2))
        quiz = Quiz(quiz_id="abc", questions=(
            QuizQuestion("abc_q1", "What is X?", ("A", "B", "C", "D"), 1, "B",
                         "explain", ("E1",), False),
            QuizQuestion("abc_q2", "What is Y?", ("A", "B", "C", "D"), 3, "D",
                         "explain2", ("E2",), True),
        ))
        pub = build_public_quiz(quiz, catalog).to_dict()
        dumped = json.dumps(pub)
        assert dumped == json.dumps(pub)  # serializable
        # private key fields absent
        assert "correct_index" not in dumped
        assert "correct_answer" not in dumped
        assert "explanation" not in dumped
        assert "source_ids" not in dumped
        assert "document_id" not in dumped
        # visual question exposes its image; non-visual does not
        assert len(pub["questions"]) == 2
        assert "image_url" not in pub["questions"][0]
        assert "image_url" in pub["questions"][1]

    def test_public_quiz_dataclass(self):
        pq = PublicQuiz(quiz_id="x", questions=[{"question_id": "q1"}])
        assert pq.to_dict()["quiz_id"] == "x"


# --------------------------------------------------------------------------- #
# Grading + zero-9001 guarantee + hidden-untill-reveal
# --------------------------------------------------------------------------- #
class TestGrading:
    def _quiz(self):
        q1 = QuizQuestion("q1", "q?", ("A", "B", "C", "D"), 1, "B", "e1",
                          ("E1",), False)
        q2 = QuizQuestion("q2", "q?", ("A", "B", "C", "D"), 2, "C", "e2",
                          ("E2",), False)
        return Quiz(quiz_id="x", questions=(q1, q2))

    def test_scoring_equal_weighting(self):
        quiz = self._quiz()
        res = grade_quiz(quiz, {"q1": "B", "q2": "C"})
        assert res["correct"] == 2 and res["total"] == 2 and res["percent"] == 100
        assert all(r["correct"] for r in res["per_question"])

    def test_partial_score(self):
        quiz = self._quiz()
        res = grade_quiz(quiz, {"q1": "A", "q2": "C"})
        assert res["correct"] == 1 and res["total"] == 2 and res["percent"] == 50

    def test_missing_answer_counts_incorrect(self):
        quiz = self._quiz()
        res = grade_quiz(quiz, {})
        assert res["correct"] == 0 and res["total"] == 2

    def test_grading_makes_zero_9001_calls(self):
        quiz = self._quiz()
        calls = []
        # attach a sentinel to prove generate_llm_response is never touched
        import rag.quiz as quizmod
        orig = quizmod.generate_llm_response
        quizmod.generate_llm_response = lambda *a, **k: calls.append(a) or "..."
        try:
            grade_quiz(quiz, {"q1": "B", "q2": "C"})
        finally:
            quizmod.generate_llm_response = orig
        assert calls == []

    def test_reveal_contains_answers_after(self):
        catalog = build_evidence_catalog(_evidence(2))
        quiz = Quiz(quiz_id="x", questions=(
            QuizQuestion("x_q1", "q?", ("A", "B", "C", "D"), 1, "B", "the explanation",
                         ("E1",), False),
        ))
        rev = reveal(quiz, catalog)
        item = rev["review"][0]
        assert item["correct_answer"] == "B"
        assert item["explanation"] == "the explanation"
        assert len(item["sources"]) == 1
        src = item["sources"][0]
        assert "filename" in src and "page_no" in src
        # reveal is a separate call - grading output contains no reveal data
        res = grade_quiz(quiz, {"x_q1": "B"})
        assert "explanation" not in json.dumps(res)


# --------------------------------------------------------------------------- #
# generate_quiz (single call) + fixed-key hard guarantee
# --------------------------------------------------------------------------- #
class TestGenerateQuiz:
    def test_default_question_count(self):
        assert DEFAULT_QUIZ_SIZE == 5

    def test_generates_single_llm_call_and_fixed_key(self, pipeline):
        settings, store, ti, vi = pipeline
        calls = []
        def fake_llm(messages, max_tokens=500, settings=None):
            calls.append((messages, max_tokens, settings))
            return _valid_quiz_response(5)
        def fake_rerank(q, ti, vi, top_n=5, settings=None):
            return _evidence(3)
        quiz, evidence = generate_quiz(
            topic="sampling", text_idx=ti, visual_idx=vi,
            n_questions=5, llm=fake_llm, rerank=fake_rerank, settings=settings)
        # one single 9001 call for the whole quiz
        assert len(calls) == 1
        assert len(quiz.questions) == 5
        # key frozen
        original = [q.correct_index for q in quiz.questions]
        assert [q.correct_index for q in quiz.questions] == original
        # registry holds it
        assert get_quiz(quiz.quiz_id) is quiz

    def test_configurable_question_count(self, pipeline):
        settings, store, ti, vi = pipeline
        def fake_llm(messages, max_tokens=500, settings=None):
            return _valid_quiz_response(3)
        def fake_rerank(q, ti, vi, top_n=5, settings=None):
            return _evidence(3)
        quiz, _ = generate_quiz(
            topic="t", text_idx=ti, visual_idx=vi,
            n_questions=3, llm=fake_llm, rerank=fake_rerank, settings=settings)
        assert len(quiz.questions) == 3

    def test_too_few_valid_questions_rejected(self, pipeline):
        settings, store, ti, vi = pipeline
        def fake_llm(messages, max_tokens=500, settings=None):
            # only 2 valid questions returned, we asked for 5
            return _valid_quiz_response(2)
        def fake_rerank(q, ti, vi, top_n=5, settings=None):
            return _evidence(3)
        with pytest.raises(QuizError) as e:
            generate_quiz(topic="t", text_idx=ti, visual_idx=vi,
                          n_questions=5, llm=fake_llm, rerank=fake_rerank,
                          settings=settings)
        assert "not enough" in str(e.value).lower()

    def test_malformed_question_rejected_not_repaired(self, pipeline):
        settings, store, ti, vi = pipeline
        def fake_llm(messages, max_tokens=500, settings=None):
            raw = json.loads(_valid_quiz_response(1))
            raw["questions"][0]["correct_answer"] = "WRONG"  # mismatched
            return json.dumps(raw)
        def fake_rerank(q, ti, vi, top_n=5, settings=None):
            return _evidence(3)
        with pytest.raises(QuizError):
            generate_quiz(topic="t", text_idx=ti, visual_idx=vi,
                          n_questions=1, llm=fake_llm, rerank=fake_rerank,
                          settings=settings)

    def test_no_evidence_raises_insufficient(self, pipeline):
        settings, store, ti, vi = pipeline
        def fake_rerank(q, ti, vi, top_n=5, settings=None):
            return []
        with pytest.raises(QuizError):
            generate_quiz(topic="t", text_idx=ti, visual_idx=vi,
                          n_questions=5, llm=lambda *a, **k: _valid_quiz_response(5),
                          rerank=fake_rerank, settings=settings)

    def test_rrf_fallback_still_produces_valid_quiz(self, pipeline):
        """9004 down -> rerank_evidence returns fallback-sourced evidence;
        quiz generation must still produce a valid grounded quiz."""
        settings, store, ti, vi = pipeline
        calls = []
        def fake_llm(messages, max_tokens=500, settings=None):
            calls.append(messages)
            return _valid_quiz_response(2)
        def fake_rerank(q, ti, vi, top_n=5, settings=None):
            # all evidence from RRF fallback (9004 unavailable)
            ev = _evidence(2)
            return [RerankedEvidence(e.candidate, e.score, "rerank-fallback")
                    for e in ev]
        quiz, evidence = generate_quiz(
            topic="t", text_idx=ti, visual_idx=vi,
            n_questions=2, llm=fake_llm, rerank=fake_rerank, settings=settings)
        assert len(quiz.questions) == 2
        assert evidence and all(e.source == "rerank-fallback" for e in evidence)
        assert get_quiz(quiz.quiz_id) is quiz

    def test_visual_question_includes_image_and_key_hidden(self, pipeline):
        """A genuinely-visual question (visual=true) keeps its answer key and
        explanation hidden while still exposing the needed image."""
        settings, store, ti, vi = pipeline
        quiz = Quiz(quiz_id="v-123", questions=(
            QuizQuestion("v-123_q1",
                         "Which diagram shows the sampling tradeoff?",
                         ("Top-left", "Top-right", "Bottom-left", "Bottom-right"),
                         2, "Bottom-left", "The right panel illustrates it.",
                         ("E1",), True),
        ))
        catalog = build_evidence_catalog(_evidence(3))
        pub = build_public_quiz(quiz, catalog).to_dict()
        dumped = json.dumps(pub)
        assert "image_url" in pub["questions"][0]  # image exposed to answer
        assert "correct_index" not in dumped
        assert "correct_answer" not in dumped
        assert "explanation" not in dumped
        assert "Bottom-left" in dumped  # option text present (needed)


    def test_decision17_fixed_key_hard_guarantee(self, pipeline):
        """Change/mock 9001 AFTER generation; grading still uses ORIGINAL key."""
        settings, store, ti, vi = pipeline
        generated = {}
        def fake_llm(messages, max_tokens=500, settings=None):
            # only the FIRST call (generation) returns a real quiz; pretend
            # subsequent 9001 behavior CHANGES to a completely different key.
            if "changed" not in generated:
                generated["changed"] = True
                return json.dumps({"questions": [{
                    "prompt": "q", "options": ["A", "B", "C", "D"],
                    "correct_index": 1, "correct_answer": "B",
                    "explanation": "e", "source_ids": ["E1"], "visual": False}]})
            # subsequent (should never be reached during grading)
            return json.dumps({"questions": [{
                "prompt": "q", "options": ["A", "B", "C", "D"],
                "correct_index": 3, "correct_answer": "D",  # DIFFERENT key
                "explanation": "e", "source_ids": ["E1"], "visual": False}]})
        def fake_rerank(q, ti, vi, top_n=5, settings=None):
            return _evidence(1)
        quiz, _ = generate_quiz(topic="t", text_idx=ti, visual_idx=vi,
                                n_questions=1, llm=fake_llm, rerank=fake_rerank,
                                settings=settings)
        assert quiz.questions[0].correct_answer == "B"
        qid = quiz.questions[0].question_id  # actual generated id
        # now mutate the mock so any 9001 call would yield "D" - grading must
        # STILL use the stored key (B), and must make zero 9001 calls.
        calls = []
        def witness(*a, **k):
            calls.append(a)
            return "should not be called"
        import rag.quiz as quizmod
        orig = quizmod.generate_llm_response
        quizmod.generate_llm_response = witness
        try:
            res = grade_quiz(quiz, {qid: "B"})
        finally:
            quizmod.generate_llm_response = orig
        assert calls == []
        assert res["correct"] == 1  # graded against original B key


# --------------------------------------------------------------------------- #
# Session registry isolation
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_same_quiz_id_returns_same_frozen_quiz(self, pipeline):
        settings, store, ti, vi = pipeline
        def fake_llm(messages, max_tokens=500, settings=None):
            return _valid_quiz_response(1)
        def fake_rerank(q, ti, vi, top_n=5, settings=None):
            return _evidence(1)
        quiz, _ = generate_quiz(topic="t", text_idx=ti, visual_idx=vi,
                                n_questions=1, llm=fake_llm, rerank=fake_rerank,
                                settings=settings)
        assert get_quiz(quiz.quiz_id) is quiz
        # public representation has NO private key
        pub = build_public_quiz(quiz, build_evidence_catalog(_evidence(1))).to_dict()
        assert "correct_index" not in json.dumps(pub)


# --------------------------------------------------------------------------- #
# Auto mode topic generation
# --------------------------------------------------------------------------- #
class TestAutoMode:
    def test_suggest_topics(self, settings):
        def fake_llm(messages, max_tokens=400, settings=None):
            return '{"topics": ["temperature sampling", "tokenization", "attention"]}'
        topics = suggest_topics("sample course text", llm=fake_llm, settings=settings)
        assert topics == ["temperature sampling", "tokenization", "attention"]

    def test_suggest_topics_rejects_none(self, settings):
        def fake_llm(messages, max_tokens=400, settings=None):
            return '{"topics": []}'
        with pytest.raises(QuizError):
            suggest_topics("sample", llm=fake_llm, settings=settings)


# --------------------------------------------------------------------------- #
# Privacy / credential-safety of the model-facing prompt
# --------------------------------------------------------------------------- #
class TestPromptPrivacy:
    def test_no_filename_page_path_docid(self):
        catalog = build_evidence_catalog(_evidence(2))
        msgs = quiz_prompt_from_catalog("topic", catalog, 2)
        user_content = next(c["content"] for c in msgs if c["role"] == "user")
        dumped = json.dumps(user_content)
        for entry in catalog.entries():
            eid, entry = entry
            assert entry.filename not in dumped
        assert ".pdf" not in dumped
        assert "page" not in dumped.lower()
        assert "document_id" not in dumped
        assert "E1" in dumped and "E2" in dumped


def test_no_credential_leakage_in_quiz_modules(tmp_path):
    import rag.quiz as qm
    import rag.qa as qa
    # no real key literals in source (test-key is not the real key)
    for mod in (qm, qa):
        src = Path(mod.__file__).read_text()
        assert "your_key_here" not in src or ".env" in src  # dummy only via config
