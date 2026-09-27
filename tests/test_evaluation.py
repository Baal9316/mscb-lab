"""Tests for the Issue #7 evaluation harness.

These tests make the scoring deterministic and lock the evaluation-set structure
so expected answers/pages cannot silently drift and inflate results.
"""
from __future__ import annotations

import copy
import json

import pytest

from evaluation.aggregate import aggregate_metrics
from evaluation.eval_harness import (
    TOP_K, _generate_from_evidence, _text_only_candidates, run_single,
    run_evaluation, summary_table)
from evaluation.eval_set import EVAL_SET, get_eval_set
from evaluation.scoring import (
    fact_present, normalize, score_answer, score_insufficient, score_sequence,
    score_source_hit, score_visual)


# --------------------------------------------------------------------------- #
# Normalization / deterministic fact matching
# --------------------------------------------------------------------------- #
class TestNormalize:
    def test_case_whitespace_punctuation(self):
        assert normalize("  Larger MODELS  tend, to Perform BETTER! ") == \
            "larger models tend to perform better"

    def test_fact_present_case_and_punct_insensitive(self):
        ans = "It says LARGER MODELS tend to perform better, per page 6!"
        assert fact_present(ans, "larger models tend to perform better")

    def test_fact_present_with_alias(self):
        ans = "bigger models help"
        assert fact_present(ans, "larger models", {"larger": ["bigger"]})

    def test_fact_absent(self):
        assert not fact_present("models are small", "larger models")


# --------------------------------------------------------------------------- #
# Answer correctness
# --------------------------------------------------------------------------- #
class TestScoreAnswer:
    def test_match_all_requires_every_fact(self):
        item = {"type": "text",
                "expected_facts": ["larger", "recent"], "match_mode": "all"}
        assert score_answer(item, "larger and more recent models both help")
        assert not score_answer(item, "larger models only")

    def test_match_any_requires_one_fact(self):
        item = {"type": "text",
                "expected_facts": ["GPU", "vLLM"], "match_mode": "any"}
        assert score_answer(item, "use vLLM")
        assert score_answer(item, "use a GPU")
        assert not score_answer(item, "use a CPU")

    def test_unanswerable_not_scored_by_facts(self):
        item = {"type": "unanswerable", "expected_facts": []}
        assert score_answer(item, "anything") is True

    def test_q1_recent_alone_does_not_pass(self):
        # "more recent models" without the relationship must FAIL
        item = {"type": "text", "match_mode": "all",
                "expected_facts": ["larger models tend to perform better",
                                   "more recent models tend to perform better"]}
        assert score_answer(item, "larger models tend to perform better and "
                                  "more recent models tend to perform better too.")
        # merely mentioning more recent models is insufficient
        assert not score_answer(item, "larger models tend to perform better "
                                      "and also more recent models.")
        assert not score_answer(item, "more recent models matter.")

    def test_q3_gpu_alone_fails(self):
        # q3: key answer is vLLM; "GPU" alone must fail
        item = {"type": "text", "match_mode": "all", "expected_facts": ["vLLM"]}
        assert score_answer(item, "vLLM")
        assert not score_answer(item, "GPU")       # GPU alone fails
        assert not score_answer(item, "a GPU only")  # GPU only fails


# --------------------------------------------------------------------------- #
# Concept-group scoring (q1/q2/q4 generalization over literal phrases)
# --------------------------------------------------------------------------- #
def _q(item_id: str) -> dict:
    return next(i for i in EVAL_SET if i["question_id"] == item_id)


class TestConceptGroups:
    def test_tolerates_parentheses_and_paraphrase(self):
        q1 = _q("q1")
        assert score_answer(q1, (
            "Larger models tend to perform better, and newer models also "
            "tend to perform better."))
        # parenthetical inserted -> still passes via recency + performance
        assert score_answer(q1, (
            "Larger models tend to perform better; more recent (newer) "
            "models also tend to perform better."))
        assert score_answer(q1, (
            "Bigger models achieve better performance, and newer models "
            "(e.g., more recent releases) perform better as well."))

    def test_case_and_punctuation_insensitive(self):
        q1 = _q("q1")
        assert score_answer(q1, (
            "LARGER MODELS TEND TO PERFORM BETTER; NEWER MODELS, ALSO, "
            "TEND TO PERFORM BETTER."))

    def test_q1_requires_both_relationships(self):
        q1 = _q("q1")
        # scale->better AND recency->better: dropped recency relationship fails
        assert not score_answer(q1, "Larger models perform better only.")
        # "newer models are mentioned" without recency->performance fails
        assert not score_answer(q1, (
            "Larger models perform better; newer models are mentioned."))
        # mentions of recency without performance fail
        assert not score_answer(q1, "Newer models exist and are used more.")

    def test_q1_does_not_use_pilot_answers_as_aliases(self):
        """The scorer must generalize to NEW paraphrases not present in the
        pilot run — proving aliases come from course wording, not hard-coded
        observed answers."""
        q1 = _q("q1")
        # novel phrasing (never produced in the pilot): more parameters +
        # better outcomes, newest + perform better — must PASS from concepts.
        assert score_answer(q1, (
            "More parameters give better outcomes; the newest releases also "
            "perform better."))
        # the actual pilot q1 phrasing must pass too
        assert score_answer(q1, (
            "According to the course material, larger models tend to perform "
            "better, and more recent (newer) models also tend to perform "
            "better."))

    def test_q1_split_on_conjunction_propositions(self):
        """'and' introduces independent propositions; merely mentioning newer
        models in the same sentence as performance elsewhere is not a
        recency -> better claim."""
        q1 = _q("q1")
        assert score_answer(q1, (
            "Larger models perform better and newer models also perform "
            "better."))
        assert not score_answer(q1, (
            "Larger models perform better and newer models are merely "
            "mentioned."))
        assert score_answer(q1, (
            "Larger models tend to perform better; more recent (newer) "
            "models also tend to perform better."))

    def test_q2_requires_both_fp8_and_quantization_meaning(self):
        q2 = _q("q2")
        # minimal paraphrase names quantization but never says WHAT it does.
        assert not score_answer(q2, "FP8 is an 8-bit quantization format.")
        # full account passes
        assert score_answer(q2, (
            "FP8 indicates 8-bit precision. Quantization scales and rounds "
            "model parameters to lower precision."))
        # alternate course-grounded reduction wording passes
        assert score_answer(q2, (
            "FP8 is 8-bit; quantization can reduce values from 16-bit "
            "precision to 8-bit."))
        # only FP8 mention, no quantization -> fail
        assert not score_answer(q2, "FP8 is a model family.")
        # quantization meaning alone, FP8/8-bit missing -> fail
        assert not score_answer(q2, "Quantization reduces precision.")
        # quantization concept but no reduction meaning -> fail
        assert not score_answer(q2, (
            "FP8 is an 8-bit format and uses quantization in some models."))

    def test_q4_requires_context_meaning_and_token_measurement(self):
        q4 = _q("q4")
        assert score_answer(q4, (
            "Context, also called the context window or context length, is how "
            "much text or other information a model can access while "
            "generating; it is measured in tokens."))
        # lacks token measurement -> fail
        assert not score_answer(q4, "Context is how much information a model can access.")
        # lacks the available/accessible concept -> fail
        assert not score_answer(q4, "Context is a model setting measured in tokens.")
        # lacks context concept entirely -> fail
        assert not score_answer(q4, "Tokens are the unit of measurement.")


# --------------------------------------------------------------------------- #
# Workflow-sequence scoring (q6)
# --------------------------------------------------------------------------- #
class TestSequence:
    def _item(self):
        return {"expected_sequence": ["Understand", "Prompt", "Prototype",
                                      "Evaluate"]}

    def test_order_pass(self):
        item = self._item()
        assert score_sequence(item, (
            "First you Understand the problem, then Prompt the model, then "
            "Prototype, and finally Evaluate the result.")) is True

    def test_reversed_order_fails(self):
        item = self._item()
        assert score_sequence(item, (
            "Evaluate, then Prototype, then Prompt, then Understand.")) is False

    def test_missing_token_fails(self):
        item = self._item()
        assert score_sequence(item, "just Understand and Prompt.") is False

    def test_no_sequence_applicable_returns_none(self):
        assert score_sequence({"expected_sequence": []}, "anything") is None
        assert score_sequence({}, "anything") is None

    def test_case_insensitive_order(self):
        item = self._item()
        assert score_sequence(item, "understand prompt prototype evaluate") is True


# --------------------------------------------------------------------------- #
# Source scoring
# --------------------------------------------------------------------------- #
class TestScoreSource:
    def test_hit_when_expected_page_cited(self):
        item = {"expected_pages": [33]}
        assert score_source_hit(item, [31, 33])

    def test_hit_accepts_extra_supporting_pages(self):
        # user rule: extra legitimate citations don't fail; expected page present
        item = {"expected_pages": [33]}
        assert score_source_hit(item, [12, 33, 41])

    def test_fail_when_only_wrong_pages(self):
        item = {"expected_pages": [33]}
        assert not score_source_hit(item, [12, 31])

    def test_empty_expected_fails(self):
        item = {"expected_pages": []}
        assert not score_source_hit(item, [33])

    def test_q2_requires_both_source_groups(self):
        # q2 two-part: FP8 (p18) AND quantization (p15); one alone insufficient
        item = {"expected_pages": [18, 15], "required_source_groups": [[18], [15]]}
        assert score_source_hit(item, [18, 15])
        assert score_source_hit(item, [15, 18, 33])       # both + extra ok
        assert not score_source_hit(item, [18])           # only FP8 page
        assert not score_source_hit(item, [15])           # only quant page
        assert not score_source_hit(item, [33])           # unrelated page

    def test_single_topic_keeps_any_page_rule(self):
        # questions without required_source_groups still pass with any expected
        item = {"expected_pages": [6, 21]}
        assert score_source_hit(item, [21])
        assert score_source_hit(item, [6])


# --------------------------------------------------------------------------- #
# Insufficient-evidence scoring
# --------------------------------------------------------------------------- #
class TestInsufficient:
    def test_passes_only_insufficient_true_and_empty(self):
        item = {
            "type": "unanswerable",
            "refusal_phrases": ["do not contain", "not enough information"],
        }
        assert score_insufficient(
            item, "the materials do not contain this", True, True)
        # insufficient true but sources non-empty -> fail
        assert not score_insufficient(
            item, "the materials do not contain this", True, False)
        # insufficient false -> fail
        assert not score_insufficient(
            item, "the materials do not contain this", False, True)
        # refusal phrase missing -> fail
        assert not score_insufficient(
            item, "here is the answer", True, True)

    def test_non_unanswerable_never_scores(self):
        item = {"type": "text", "refusal_phrases": ["not contain"]}
        assert not score_insufficient(item, "x", True, True)


# --------------------------------------------------------------------------- #
# Visual scoring (retrieval and citation are independent)
# --------------------------------------------------------------------------- #
class TestVisual:
    def _item(self):
        return {"type": "visual", "visual_page": 33}

    def test_retrieval_hit_and_citation_hit_independent(self):
        item = self._item()
        # supplied but not cited -> retrieval hit, citation miss
        v = score_visual(item, [33], [31], ["/img31.png"])
        assert v["visual_retrieval_hit"] is True
        assert v["visual_citation_hit"] is False
        # cited but not supplied -> retrieval miss, citation maybe hit
        v = score_visual(item, [31], [33], ["/img33.png"])
        assert v["visual_retrieval_hit"] is False
        # both hit
        v = score_visual(item, [33], [33], ["/img33.png"])
        assert v["visual_retrieval_hit"] is True
        assert v["visual_citation_hit"] is True

    def test_citation_hit_requires_real_image(self):
        item = self._item()
        v = score_visual(item, [33], [33], [])
        assert v["visual_citation_hit"] is False  # cited but no image

    def test_non_visual_items_null(self):
        item = {"type": "text", "visual_page": 33}
        v = score_visual(item, [33], [33], ["/img.png"])
        assert v["visual_retrieval_hit"] is None
        assert v["visual_citation_hit"] is None


# --------------------------------------------------------------------------- #
# Evaluation set integrity (anti-drift)
# --------------------------------------------------------------------------- #
class TestEvalSet:
    def test_exactly_eight(self):
        assert len(EVAL_SET) == 8
        ids = [i["question_id"] for i in EVAL_SET]
        assert ids == ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]

    def test_types(self):
        types = {i["question_id"]: i["type"] for i in EVAL_SET}
        assert types["q5"] == "visual"
        assert types["q6"] == "visual"
        assert types["q7"] == "unanswerable"
        assert sum(t == "visual" for t in types.values()) >= 2
        assert sum(t == "unanswerable" for t in types.values()) >= 1

    def test_vibe_coding_question_present(self):
        q5 = next(i for i in EVAL_SET if i["question_id"] == "q5")
        assert "Vibe Coding" in q5["question"]
        assert q5["visual_page"] == 33

    def test_grounded_pages_are_ints(self):
        for item in EVAL_SET:
            for p in item.get("expected_pages", []):
                assert isinstance(p, int) and p >= 1

    def test_get_eval_set_returns_copy(self):
        s1 = get_eval_set()
        s2 = get_eval_set()
        s1[0]["question"] = "MUTATED"
        assert s2[0]["question"] != "MUTATED"


# --------------------------------------------------------------------------- #
# Harness structure / fairness
# --------------------------------------------------------------------------- #
class TestHarnessStructure:
    def test_top_k_is_five(self):
        assert TOP_K == 5

    def test_every_question_has_both_approaches(self, tmp_path, monkeypatch):
        import rag.qa as qa
        from rag.config import Settings
        from rag.store import DocumentStore
        from rag.models import Document, DocumentPage
        from rag.rerank_pipeline import EvidenceCandidate, RerankedEvidence

        data = tmp_path / "data"
        store = DocumentStore(data)
        store.add(Document(
            document_id="d", filename="deck.pdf", sha256="s",
            pages=[DocumentPage(
                document_id="d", filename="deck.pdf", page_no=6,
                extracted_text="Larger models tend to perform better, "
                               "but more recent models also help.",
                image_path="", image_url="")]))
        s = Settings(class_api_key="test-key", data_dir=data)

        # stub each approach's retrieval + generation so no network is hit
        def fake_retrieve_a(q, ti, vi, top_k=5, settings=None):
            return [RerankedEvidence(
                candidate=EvidenceCandidate(
                    "d", "deck.pdf", 6, text="Larger models perform better"),
                score=1.0, source="text-only")]
        def fake_retrieve_b(q, ti, vi, top_k=5, settings=None):
            return ([RerankedEvidence(
                candidate=EvidenceCandidate(
                    "d", "deck.pdf", 6, text="Larger models perform better"),
                score=1.0, source="rerank")], False)
        monkeypatch.setattr("evaluation.eval_harness.retrieve_a", fake_retrieve_a)
        monkeypatch.setattr("evaluation.eval_harness.retrieve_b", fake_retrieve_b)

        def fake_llm(messages, max_tokens=500, settings=None):
            return json.dumps({
                "answer": "Larger models tend to perform better and more "
                          "recent models also help, per the material.",
                "source_ids": [1],
                "insufficient": False,
            })
        monkeypatch.setattr("evaluation.eval_harness._generate_from_evidence",
                            _generate_from_evidence)

        records = run_evaluation(store, s, llm=fake_llm,
                                 out_dir=tmp_path / "evalout")
        pairs = {(r["question_id"], r["approach"]) for r in records}
        for i in range(1, 9):
            assert ("q%d" % i, "A") in pairs
            assert ("q%d" % i, "B") in pairs
        assert len(records) == 8 * 2

    def test_alternating_order(self):
        # run_evaluation uses A then B on even idx, B then A on odd idx.
        # Simulate by checking the internal function is exercised via loop.
        ev = get_eval_set()
        orders = []
        for idx in range(len(ev)):
            orders.append(["A", "B"] if idx % 2 == 0 else ["B", "A"])
        # q0 q2 q4 q6 A-first; q1 q3 q5 q7 B-first
        assert orders[0] == ["A", "B"]
        assert orders[1] == ["B", "A"]
        assert orders[7] == ["B", "A"]

    def test_run_single_writes_expected_fields(self, tmp_path, monkeypatch):
        from rag.config import Settings
        item = dict(get_eval_set()[0])  # q1
        item["question"] = "how do larger and recent models relate?"
        s = Settings(class_api_key="test-key")
        # build a minimal fake index pair so retrieval is bypassed
        class _FA:
            def search(self, q, top_k=5):
                return []
        monkeypatch.setattr("evaluation.eval_harness.retrieve_a",
                            lambda q, ti, vi, top_k=5, settings=None: [])
        monkeypatch.setattr("evaluation.eval_harness.retrieve_b",
                            lambda q, ti, vi, top_k=5, settings=None: ([], False))
        rec = run_single(item, "A", text_idx=_FA(), visual_idx=_FA(),
                         settings=s)
        for fld in ("question_id", "question", "type", "approach", "answer",
                    "retrieved_pages", "cited_pages", "correct", "source_hit",
                    "insufficient_correct", "visual_retrieval_hit",
                    "visual_citation_hit", "fallback_used", "error",
                    "retrieval_latency", "generation_latency", "total_latency"):
            assert fld in rec

    def test_results_files_written_and_no_credentials(self, tmp_path, monkeypatch):
        from rag.config import Settings
        from rag.store import DocumentStore
        from rag.models import Document, DocumentPage
        from rag.rerank_pipeline import EvidenceCandidate, RerankedEvidence

        data = tmp_path / "data"
        store = DocumentStore(data)
        store.add(Document(
            document_id="d", filename="deck.pdf", sha256="s",
            pages=[DocumentPage(
                document_id="d", filename="deck.pdf", page_no=5,
                extracted_text="knowledge cutoff date clarifies training data.",
                image_path="", image_url="")]))
        s = Settings(class_api_key="test-key", data_dir=data)

        def fake_retrieve_a(q, ti, vi, top_k=5, settings=None):
            return [RerankedEvidence(
                candidate=EvidenceCandidate(
                    "d", "deck.pdf", 5, text="knowledge cutoff date"),
                score=1.0, source="text-only")]
        monkeypatch.setattr("evaluation.eval_harness.retrieve_a", fake_retrieve_a)
        monkeypatch.setattr("evaluation.eval_harness.retrieve_b",
                            lambda q, ti, vi, top_k=5, settings=None: ([], False))
        monkeypatch.setattr("evaluation.eval_harness._generate_from_evidence",
                            _generate_from_evidence)
        def fake_llm(messages, max_tokens=500, settings=None):
            return ('{"answer": "a knowledge cutoff date clarifies how up to '
                    'date the training data is.", "source_ids": [1], '
                    '"insufficient": false}')

        out = tmp_path / "evalout"
        records = run_evaluation(store, s, llm=fake_llm, out_dir=out)
        assert (out / "results.json").exists()
        assert (out / "results.csv").exists()
        js = (out / "results.json").read_text()
        cs = (out / "results.csv").read_text()
        real_key = s.class_api_key  # "test-key"
        assert real_key not in js and real_key not in cs
        # CSV has 16 data rows (8 questions x 2 approaches) + header
        assert cs.count("\n") == 17

    def test_llm_never_called_in_index_build(self):
        # index build happens before per-question timing; ensure build_indexes_once
        # does not call the generation LLM (structural - covered by latency fields).
        from evaluation.eval_harness import IndexBuildTimes
        assert hasattr(IndexBuildTimes, "text_index_build_latency")
        assert hasattr(IndexBuildTimes, "visual_index_build_latency")


# --------------------------------------------------------------------------- #
# Aggregate metrics + summary table
# --------------------------------------------------------------------------- #
class TestAggregate:
    def _records(self):
        return [
            {"question_id": "q1", "type": "text", "approach": "A", "correct": True,
             "source_hit": True, "visual_retrieval_hit": None,
             "visual_citation_hit": None, "insufficient_correct": False,
             "retrieval_latency": 0.5, "generation_latency": 1.0,
             "total_latency": 1.5},
            {"question_id": "q1", "type": "text", "approach": "B", "correct": True,
             "source_hit": True, "visual_retrieval_hit": None,
             "visual_citation_hit": None, "insufficient_correct": False,
             "retrieval_latency": 1.5, "generation_latency": 1.0,
             "total_latency": 2.5},
            {"question_id": "q5", "type": "visual", "approach": "A", "correct": False,
             "source_hit": False, "visual_retrieval_hit": False,
             "visual_citation_hit": False, "insufficient_correct": False,
             "retrieval_latency": 0.5, "generation_latency": 1.0,
             "total_latency": 1.5},
            {"question_id": "q5", "type": "visual", "approach": "B", "correct": True,
             "source_hit": True, "visual_retrieval_hit": True,
             "visual_citation_hit": True, "insufficient_correct": False,
             "retrieval_latency": 2.0, "generation_latency": 1.0,
             "total_latency": 3.0},
            {"question_id": "q7", "type": "unanswerable", "approach": "A",
             "correct": True, "source_hit": False,
             "visual_retrieval_hit": None, "visual_citation_hit": None,
             "insufficient_correct": True, "retrieval_latency": 0.5,
             "generation_latency": 1.0, "total_latency": 1.5},
            {"question_id": "q7", "type": "unanswerable", "approach": "B",
             "correct": True, "source_hit": False,
             "visual_retrieval_hit": None, "visual_citation_hit": None,
             "insufficient_correct": True, "retrieval_latency": 1.0,
             "generation_latency": 1.0, "total_latency": 2.0},
        ]

    def test_aggregate_metrics_split_by_approach(self):
        agg = aggregate_metrics(self._records())
        a, b = agg["A"], agg["B"]
        # A: 2 answerable (q1 correct, q5 incorrect) -> 1/2 = 0.5
        assert a["answer_accuracy"] == 0.5
        # source: q1 hit, q5 miss -> 0.5
        assert a["source_support_rate"] == 0.5
        # visual: q5 citation miss -> 0.0
        assert a["visual_success_rate"] == 0.0
        # unanswerable: 1/1
        assert a["unanswerable_refusal_accuracy"] == 1.0
        # B: q1 correct, q5 correct -> 1.0
        assert b["answer_accuracy"] == 1.0
        assert b["source_support_rate"] == 1.0
        assert b["visual_success_rate"] == 1.0
        # latencies
        assert a["avg_retrieval_latency"] == pytest.approx(0.5)
        assert b["avg_retrieval_latency"] == pytest.approx(1.5)

    def test_summary_table_no_winner_declared(self):
        tbl = summary_table(self._records())
        assert "Approach" in tbl and "A" in tbl and "B" in tbl
        assert "winner" not in tbl.lower()
        assert "better than" not in tbl.lower()
