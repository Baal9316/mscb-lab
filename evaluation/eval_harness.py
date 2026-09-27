"""A/B evaluation harness for Issue #7.

Approach A — Text-only baseline: BM25 + text embeddings/fusion, NO visual
retrieval, NO multimodal 9004 reranking.
Approach B — Full multimodal RAG: BM25 + text embeddings + visual embeddings +
multimodal 9004 reranking.

This is a baseline-vs-full-system comparison. We do NOT assume B wins; we
MEASURE. Run-order alternates (q1 A→B, q2 B→A, ...) so warm-cache/order effects
do not consistently favor one approach.

Fairness: same question set, same documents, same 9001 model + generation
settings, same top_k=5, latency recorded separately for retrieval and
generation. Index construction is done ONCE up front and reported as one-time
setup cost — never included in per-question retrieval latency.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .eval_set import get_eval_set
from .scoring import (score_answer, score_insufficient, score_sequence,
                      score_source_hit, score_visual)

TOP_K = 5


@dataclass
class IndexBuildTimes:
    text_index_build_latency: float = 0.0
    visual_index_build_latency: float = 0.0


# --------------------------------------------------------------------------- #
# Retrieval paths
# --------------------------------------------------------------------------- #
def _text_only_candidates(text_candidates) -> list:
    """Collapse TextIndex hits to one EvidenceCandidate per page (strongest
    chunk) and wrap as reranked-evidence shaped objects."""
    from rag.rerank_pipeline import EvidenceCandidate, RerankedEvidence
    groups: dict = {}
    for tc in text_candidates:
        key = (tc.document_id, tc.page_no)
        cur = groups.get(key)
        if cur is None or tc.score > getattr(cur, "_text_score", -1):
            groups[key] = EvidenceCandidate(
                document_id=tc.document_id, filename=tc.filename,
                page_no=tc.page_no, text=tc.text,
                chunk_index=tc.chunk_index, _text_score=tc.score)
    out = [RerankedEvidence(candidate=c, score=float(getattr(c, "_text_score", 0.0)),
                            source="text-only") for c in groups.values()]
    out.sort(key=lambda e: e.score, reverse=True)
    return out


def retrieve_a(question: str, text_idx, visual_idx, top_k: int = TOP_K,
               settings=None) -> list:
    """Approach A: BM25 + dense text fusion, no visual, no rerank."""
    try:
        hits = text_idx.search(question, top_k=max(top_k * 3, top_k))
    except Exception:  # noqa: BLE001 - empty index -> no candidates
        hits = []
    return _text_only_candidates(hits)[:top_k]


def retrieve_b(question: str, text_idx, visual_idx, top_k: int = TOP_K,
               settings=None) -> tuple:
    """Approach B: full multimodal rerank (9004) with RRF fallback."""
    from rag.rerank_pipeline import rerank_evidence
    evidence = rerank_evidence(question, text_idx, visual_idx,
                               top_n=top_k, settings=settings)
    used_fallback = bool(evidence) and all(
        e.source == "rerank-fallback" for e in evidence)
    return evidence, used_fallback


def _generate_from_evidence(question: str, evidence, *, settings, llm=None) -> tuple:
    """Reuse Issue #5 generation/validation/citation on given evidence.

    Returns (StructuredAnswer, generation_latency).
    """
    from rag.qa import (StructuredAnswer, build_sources, generate_llm_response,
                        parse_structured_answer, prompt_from_evidence)
    if not evidence:
        return StructuredAnswer(
            answer="The course materials do not contain enough information to "
                   "answer this question.",
            sources=[], insufficient=True), 0.0
    messages = prompt_from_evidence(question, evidence)
    t0 = time.perf_counter()
    content = (llm or generate_llm_response)(messages, max_tokens=500,
                                             settings=settings)
    gen_latency = time.perf_counter() - t0
    parsed = parse_structured_answer(content)
    if parsed["insufficient"]:
        return (StructuredAnswer(answer=parsed["answer"], sources=[],
                                 insufficient=True), gen_latency)
    sources = build_sources(parsed["source_ids"], evidence)
    return (StructuredAnswer(answer=parsed["answer"], sources=sources,
                             insufficient=False), gen_latency)


def _evidence_pages(evidence) -> list[int]:
    pages = []
    for e in evidence:
        try:
            pages.append(e.candidate.page_no)
        except AttributeError:
            try:
                pages.append(e.page_no)
            except AttributeError:
                continue
    return sorted(set(pages))


def _evidence_images(evidence) -> list[str]:
    imgs = []
    for e in evidence:
        try:
            if e.candidate.image_path:
                imgs.append(e.candidate.image_path)
        except AttributeError:
            try:
                if e.image_path:
                    imgs.append(e.image_path)
            except AttributeError:
                continue
    return imgs


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def build_indexes_once(store, settings) -> tuple:
    """Build text + visual indexes once; return (ti, vi, IndexBuildTimes)."""
    from rag.retrieval import TextIndex
    from rag.visual_retrieval import VisualIndex

    times = IndexBuildTimes()
    t0 = time.perf_counter()
    ti = TextIndex(store, settings=settings)
    ti.rebuild()
    times.text_index_build_latency = time.perf_counter() - t0

    t0 = time.perf_counter()
    vi = VisualIndex(store, settings=settings)
    vi.build()
    times.visual_index_build_latency = time.perf_counter() - t0

    return ti, vi, times


# --------------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------------- #
def run_single(item: dict, approach: str, *, text_idx, visual_idx,
               settings, llm=None, top_k: int = TOP_K) -> dict:
    """Run one question through one approach; produce the result record."""
    from rag.qa import QAGenerationError, StructuredAnswer

    error = None
    used_fallback = False
    evidence = []
    t_total0 = time.perf_counter()

    try:
        if approach == "A":
            t0 = time.perf_counter()
            evidence = retrieve_a(item["question"], text_idx, visual_idx,
                                  top_k=top_k, settings=settings)
            retrieval_latency = time.perf_counter() - t0
            sa, gen_latency = _generate_from_evidence(
                item["question"], evidence, settings=settings, llm=llm)
        else:
            t0 = time.perf_counter()
            evidence, used_fallback = retrieve_b(
                item["question"], text_idx, visual_idx, top_k=top_k,
                settings=settings)
            retrieval_latency = time.perf_counter() - t0
            sa, gen_latency = _generate_from_evidence(
                item["question"], evidence, settings=settings, llm=llm)
            sa.used_fallback = used_fallback
    except QAGenerationError as exc:
        error = str(exc)
        sa = StructuredAnswer(answer="", sources=[], insufficient=False)
        retrieval_latency = 0.0
        gen_latency = 0.0
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        sa = StructuredAnswer(answer="", sources=[], insufficient=False)
        retrieval_latency = 0.0
        gen_latency = 0.0
    except (KeyboardInterrupt, SystemExit):
        raise

    total = time.perf_counter() - t_total0

    # Pages/images SUPPLIED to generation (from evidence), vs CITED (trusted
    # sources final). visual_retrieval_hit uses supplied pages.
    supplied_pages = _evidence_pages(evidence)
    supplied_images = _evidence_images(evidence)
    cited_pages = [s.page_no for s in sa.sources]
    cited_images = [s.image_path for s in sa.sources if s.image_path]
    vis = score_visual(item, supplied_pages, cited_pages, cited_images)

    correct = score_answer(item, sa.answer) if not error else False
    # workflow-order questions (e.g. q6) are scored on sequence, not just
    # token presence.
    if item.get("expected_sequence") and not error:
        correct = bool(score_sequence(item, sa.answer))
    source_hit = score_source_hit(item, cited_pages) if not error else False
    insufficient_correct = score_insufficient(
        item, sa.answer, sa.insufficient, len(sa.sources) == 0) if not error else False

    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "type": item["type"],
        "expected_facts": item.get("expected_facts", []),
        "expected_pages": item.get("expected_pages", []),
        "approach": approach,
        "top_k": top_k,
        "model": getattr(settings, "llm_model", ""),
        "answer": sa.answer,
        "retrieved_pages": supplied_pages,
        "cited_pages": cited_pages,
        "correct": correct,
        "source_hit": source_hit,
        "insufficient_correct": insufficient_correct,
        "insufficient": sa.insufficient,
        "visual_retrieval_hit": vis["visual_retrieval_hit"],
        "visual_citation_hit": vis["visual_citation_hit"],
        "fallback_used": used_fallback,
        "error": error,
        "retrieval_latency": round(retrieval_latency, 4),
        "generation_latency": round(gen_latency, 4),
        "total_latency": round(total, 4),
    }


# --------------------------------------------------------------------------- #
# Orchestration + persistence
# --------------------------------------------------------------------------- #
def run_evaluation(store, settings, *, llm=None,
                   out_dir: Path = Path("evaluation"),
                   eval_set=None) -> list[dict]:
    """Run all 8 questions x both approaches with alternating order.

    Returns a flat list of result records (one per (question_id, approach)).
    """
    items = eval_set if eval_set is not None else get_eval_set()
    ti, vi, times = build_indexes_once(store, settings)

    records: list[dict] = []
    for idx, item in enumerate(items):
        order = ["A", "B"] if idx % 2 == 0 else ["B", "A"]
        for approach in order:
            rec = run_single(item, approach, text_idx=ti, visual_idx=vi,
                             settings=settings, llm=llm)
            rec["_text_index_build_latency"] = round(times.text_index_build_latency, 4)
            rec["_visual_index_build_latency"] = round(times.visual_index_build_latency, 4)
            records.append(rec)

    _write_results(records, times, out_dir)
    return records


def _write_results(records: list[dict], times: IndexBuildTimes,
                   out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    public = [{k: v for k, v in r.items() if not k.startswith("_")}
              for r in records]

    doc = {
        "meta": {
            "text_index_build_latency": round(times.text_index_build_latency, 4),
            "visual_index_build_latency": round(times.visual_index_build_latency, 4),
            "top_k": TOP_K,
        },
        "results": public,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(doc, f, indent=2, default=str)

    fields = [
        "question_id", "type", "approach", "correct", "source_hit",
        "insufficient_correct", "visual_retrieval_hit", "visual_citation_hit",
        "fallback_used", "retrieved_pages", "cited_pages",
        "retrieval_latency", "generation_latency", "total_latency", "error",
    ]
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in public:
            w.writerow({k: r.get(k, "") for k in fields})


def summary_table(records: list[dict]) -> str:
    """README-ready markdown summary table, one row per approach."""
    from .aggregate import aggregate_metrics
    agg = aggregate_metrics(records)
    header = ("| Approach | Answer accuracy | Source-support | Visual | "
              "Unanswerable | Avg retrieval | Avg generation | Avg total |")
    sep = "|:---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"
    rows = []
    for approach in ("A", "B"):
        a = agg.get(approach, {})
        rows.append(
            f"| {approach} | {a.get('answer_accuracy', 0):.2f} | "
            f"{a.get('source_support_rate', 0):.2f} | "
            f"{a.get('visual_success_rate', 0):.2f} | "
            f"{a.get('unanswerable_refusal_accuracy', 0):.2f} | "
            f"{a.get('avg_retrieval_latency', 0):.2f}s | "
            f"{a.get('avg_generation_latency', 0):.2f}s | "
            f"{a.get('avg_total_latency', 0):.2f}s |")
    return "\n".join([header, sep] + rows)
