"""Aggregate evaluation metrics (Issue #7)."""
from __future__ import annotations


def _num(records, key):
    return [r[key] for r in records if isinstance(r.get(key), (int, float))]


def aggregate_metrics(records: list[dict]) -> dict:
    """Per-approach aggregate metrics over a flat record list."""
    out = {}
    for approach in ("A", "B"):
        recs = [r for r in records if r.get("approach") == approach]
        if not recs:
            out[approach] = {"n": 0}
            continue
        answerable = [r for r in recs if r.get("type") != "unanswerable"]
        visual = [r for r in recs if r.get("type") == "visual"]
        unans = [r for r in recs if r.get("type") == "unanswerable"]

        def rate(items, key):
            if not items:
                return 0.0
            vals = [i.get(key) for i in items]
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        out[approach] = {
            "n": len(recs),
            "answer_accuracy": rate(answerable, "correct"),
            "source_support_rate": rate(answerable, "source_hit"),
            "visual_success_rate": rate(visual, "visual_citation_hit"),
            "visual_retrieval_rate": rate(visual, "visual_retrieval_hit"),
            "unanswerable_refusal_accuracy": rate(unans, "insufficient_correct"),
            "avg_retrieval_latency": round(
                (sum(_num(recs, "retrieval_latency")) /
                 len(_num(recs, "retrieval_latency"))), 4)
                if _num(recs, "retrieval_latency") else 0.0,
            "avg_generation_latency": round(
                (sum(_num(recs, "generation_latency")) /
                 len(_num(recs, "generation_latency"))), 4)
                if _num(recs, "generation_latency") else 0.0,
            "avg_total_latency": round(
                (sum(_num(recs, "total_latency")) /
                 len(_num(recs, "total_latency"))), 4)
                if _num(recs, "total_latency") else 0.0,
        }
    return out
