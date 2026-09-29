"""Tests for rag.flashcards — grounded flashcard generation.

Covers strict JSON parsing, validation, single-call generation, evidence-catalog
privacy (no filenames/pages in what the model sees), malformed-card rejection,
and insufficient-evidence handling. Fully offline: llm/rerank are fakes.
"""
from __future__ import annotations

import json

import pytest

from rag.config import Settings
from rag.flashcards import (
    DEFAULT_CARD_COUNT,
    FlashcardError,
    flashcard_prompt_from_catalog,
    generate_flashcards,
    parse_flashcards_response,
    validate_card,
)
from rag.quiz import build_evidence_catalog, QuizError
from rag.rerank_pipeline import EvidenceCandidate, RerankedEvidence


def _settings(tmp_path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(class_api_key="test-key", data_dir=data_dir)


def _evidence(n=3, with_image=True):
    ev = []
    for i in range(1, n + 1):
        ev.append(RerankedEvidence(
            EvidenceCandidate(
                document_id="doc-0", filename="slides.pdf", page_no=i,
                text=f"Evidence text {i} about CAPM and beta.",
                image_url=f"data:image/png;base64,{i}" if with_image else ""),
            score=0.9 - i * 0.1, source="rerank"))
    return ev


def _valid_response(n=2):
    cards = []
    for i in range(1, n + 1):
        cards.append({"front": f"Concept {i}?",
                      "back": f"Answer {i}.",
                      "source_ids": [f"E{i}"],
                      "visual": False})
    return json.dumps({"cards": cards})


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
class TestParse:
    def test_parses_cards(self):
        raw = parse_flashcards_response(_valid_response(2))
        assert len(raw) == 2
        assert raw[0]["front"] == "Concept 1?"

    def test_rejects_non_json(self):
        with pytest.raises(FlashcardError):
            parse_flashcards_response("not json")

    def test_rejects_missing_cards_key(self):
        with pytest.raises(FlashcardError):
            parse_flashcards_response('{"cardsx": []}')

    def test_rejects_empty(self):
        with pytest.raises(FlashcardError):
            parse_flashcards_response("   ")

    def test_tolerates_code_fence(self):
        raw = parse_flashcards_response("```json\n" + _valid_response(1) + "\n```")
        assert len(raw) == 1


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class TestValidate:
    def _catalog(self):
        return build_evidence_catalog(_evidence(3))

    def test_valid_card(self):
        catalog = self._catalog()
        card = validate_card(
            {"front": "Q", "back": "A", "source_ids": ["E1", "E2"], "visual": False},
            deck_id="d", index=0, catalog=catalog)
        assert card.front == "Q" and card.back == "A"
        assert card.source_ids == ("E1", "E2")

    def test_missing_front_rejected(self):
        with pytest.raises(FlashcardError):
            validate_card({"back": "A", "source_ids": ["E1"]},
                          deck_id="d", index=0, catalog=self._catalog())

    def test_unknown_source_rejected(self):
        with pytest.raises(FlashcardError):
            validate_card({"front": "Q", "back": "A", "source_ids": ["E99"]},
                          deck_id="d", index=0, catalog=self._catalog())

    def test_empty_sources_rejected(self):
        with pytest.raises(FlashcardError):
            validate_card({"front": "Q", "back": "A", "source_ids": []},
                          deck_id="d", index=0, catalog=self._catalog())


# --------------------------------------------------------------------------- #
# Generation: single call, grounding, privacy, errors
# --------------------------------------------------------------------------- #
class TestGenerate:
    def test_single_call_generates_deck(self, tmp_path):
        s = _settings(tmp_path)
        calls = []

        def fake_llm(messages, max_tokens=500, settings=None):
            calls.append(messages)
            return _valid_response(3)

        def fake_rerank(query, text_idx, visual_idx, top_n=5, settings=None):
            return _evidence(3)

        deck, _ = generate_flashcards(
            topic="CAPM", text_idx=object(), visual_idx=object(),
            n_cards=3, llm=fake_llm, rerank=fake_rerank, settings=s)
        assert len(calls) == 1  # ONE 9001 call for the whole deck
        assert len(deck.cards) == 3

    def test_cards_grounded_in_catalog(self, tmp_path):
        s = _settings(tmp_path)

        def fake_llm(messages, max_tokens=500, settings=None):
            return _valid_response(2)

        def fake_rerank(query, text_idx, visual_idx, top_n=5, settings=None):
            return _evidence(2)

        deck, _ = generate_flashcards(
            topic="CAPM", text_idx=object(), visual_idx=object(),
            n_cards=2, llm=fake_llm, rerank=fake_rerank, settings=s)
        assert deck.cards[0].source_ids == ("E1",)

    def test_prompt_never_leaks_filenames_or_pages(self, tmp_path):
        s = _settings(tmp_path)
        seen = {}

        def fake_llm(messages, max_tokens=500, settings=None):
            seen["msgs"] = messages
            return _valid_response(1)

        def fake_rerank(query, text_idx, visual_idx, top_n=5, settings=None):
            return _evidence(2)

        generate_flashcards(topic="CAPM", text_idx=object(), visual_idx=object(),
                            n_cards=1, llm=fake_llm, rerank=fake_rerank, settings=s)
        dumped = json.dumps(seen["msgs"])
        assert ".pdf" not in dumped
        assert "slides" not in dumped
        assert "page" not in dumped.lower()

    def test_all_invalid_cards_raises(self, tmp_path):
        s = _settings(tmp_path)

        def fake_llm(messages, max_tokens=500, settings=None):
            return '{"cards": [{"front": "", "back": "A", "source_ids": ["E1"]}]}'

        def fake_rerank(query, text_idx, visual_idx, top_n=5, settings=None):
            return _evidence(1)

        with pytest.raises(FlashcardError):
            generate_flashcards(topic="CAPM", text_idx=object(),
                                visual_idx=object(), n_cards=1,
                                llm=fake_llm, rerank=fake_rerank, settings=s)

    def test_insufficient_evidence_raises(self, tmp_path):
        s = _settings(tmp_path)

        def fake_rerank(query, text_idx, visual_idx, top_n=5, settings=None):
            return []

        with pytest.raises(FlashcardError) as einfo:
            generate_flashcards(topic="CAPM", text_idx=object(),
                                visual_idx=object(), n_cards=1,
                                rerank=fake_rerank, settings=s)
        assert "material" in str(einfo.value)

    def test_defaults(self):
        assert DEFAULT_CARD_COUNT == 8


# --------------------------------------------------------------------------- #
# Starred-card registry
# --------------------------------------------------------------------------- #
class TestStarredRegistry:
    def test_star_and_load_persist(self, tmp_path):
        from rag.flashcards import card_key, load_starred_cards, star_card
        s = _settings(tmp_path)
        star_card(s, front="What is beta?", back="Sensitivity to market risk.",
                  sources=["slides.pdf · page 2"], origin="manual")
        items = load_starred_cards(s)
        assert len(items) == 1
        assert items[0]["front"] == "What is beta?"
        assert items[0]["sources"] == ["slides.pdf · page 2"]
        assert items[0]["key"] == card_key("What is beta?", "Sensitivity to market risk.")

    def test_star_same_card_dedupes(self, tmp_path):
        from rag.flashcards import load_starred_cards, star_card
        s = _settings(tmp_path)
        star_card(s, front="Q", back="A")
        star_card(s, front="Q", back="A")   # same content -> same key
        assert len(load_starred_cards(s)) == 1

    def test_unstar_removes(self, tmp_path):
        from rag.flashcards import load_starred_cards, star_card, unstar_card
        s = _settings(tmp_path)
        e = star_card(s, front="Q", back="A")
        assert unstar_card(s, e["key"]) is True
        assert load_starred_cards(s) == []
        assert unstar_card(s, e["key"]) is False  # gone

    def test_quiz_miss_autostars_with_origin(self, tmp_path):
        from rag.flashcards import load_starred_cards, add_missed_quiz_cards
        s = _settings(tmp_path)
        missed = [{"front": "What is CAPM?", "back": "E(r) = Rf + β(Rm − Rf)",
                   "sources": ["slides.pdf · page 5"]}]
        added, _ = add_missed_quiz_cards(s, missed)
        assert added == 1
        items = load_starred_cards(s)
        assert items[0]["origin"] == "quiz-miss"
        assert items[0]["sources"] == ["slides.pdf · page 5"]

    def test_quiz_miss_resubmit_does_not_duplicate(self, tmp_path):
        from rag.flashcards import add_missed_quiz_cards, load_starred_cards, star_card
        s = _settings(tmp_path)
        missed = [{"front": "Q?", "back": "A."}]
        star_card(s, front="Q?", back="A.")      # already starred manually
        added, _ = add_missed_quiz_cards(s, missed)
        assert added == 0
        assert len(load_starred_cards(s)) == 1
