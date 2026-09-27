"""Evaluation package for Issue #7 (testing & evaluation).

Provides a fixed, grounded evaluation set (``eval_set.py``), a deterministic
scoring harness (``scoring.py``), an A/B pipeline runner (``eval_harness.py``)
and aggregate metrics (``aggregate.py``).

Grounding rule: every expected fact/source page is verified against the actual
Week 2 deck; no guessed pages. The A-vs-B comparison is a baseline-vs-full-system
measurement, not an isolated causal test of visual embeddings, and no winning
approach is assumed ahead of measurement.
"""
