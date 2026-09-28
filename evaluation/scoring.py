"""Deterministic scoring for Issue #7 evaluation.

No LLM judge, no fuzzy/semantic similarity. Correctness uses normalized fact
sets (exact substring match after normalizing case/whitespace/punctuation),
with an optional alias map. Source scoring requires at least one cited page in
expected_pages; extra legitimate supporting citations do not cause failure.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[,!?.;:'\"()\[\]{}*/\\|#@^&~`+-]")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation (deterministic)."""
    t = _PUNCT.sub(" ", (text or ""))
    t = _WS.sub(" ", t)
    return t.strip().lower()


def _fact_variants(fact: str, aliases: dict) -> list[str]:
    """Return normalized variants of ``fact`` after substituting any alias
    phrase that occurs inside it with each of its aliased values.

    Example: fact "larger models are best" with alias {"larger": ["bigger"]}
    yields normalized "larger models are best" and "bigger models are best".
    """
    base = normalize(fact)
    variants = [base]
    ran_subst = False
    for key, vals in aliases.items():
        nk = normalize(key)
        # build a variant for each alias value only when the key appears in fact
        if nk and nk in " " + base + " ":  # whole-token-ish occurrence
            for v in vals:
                nv = normalize(v)
                if nv:
                    variants.append(base.replace(nk, nv))
                    ran_subst = True
    # if no alias key matched, keep only the base
    return variants if ran_subst else [base]


def fact_present(answer: str, fact: str, aliases: dict | None = None) -> bool:
    """True if the (possibly-aliased) fact substring appears in answer."""
    n_answer = normalize(answer or "")
    if not n_answer or not fact:
        return False
    return any(v and v in n_answer for v in _fact_variants(fact, aliases or {}))


# --------------------------------------------------------------------------- #
# Deterministic concept-group scoring (replaces brittle literal-phrase matching)
#
# A concept group is a list of alternative wordings for ONE semantic concept,
# defined from the course material/question terminology. A question passes only
# when at least one member from EVERY required group appears (after the existing
# case/whitespace/punctuation normalization). No LLM judge, no embedding
# similarity, no fuzzy scoring. Aliases come from course wording/question
# terminology — NOT from observed model answers.
# --------------------------------------------------------------------------- #
def _group_present(answer: str, group: list[str]) -> bool:
    """True if at least one member of a concept group appears in the answer."""
    if not group:
        return False
    n_answer = normalize(answer or "")
    if not n_answer:
        return False
    return any(normalize(m) and normalize(m) in n_answer for m in group)


def score_concept_groups(item: dict, answer: str) -> bool:
    """Concept-group correctness: every required group must have >=1 member
    present in the answer (deterministic, after normalization).

    Example: required_concept_groups = [
        ["term A", "alias A1"],
        ["term B", "alias B1", "alias B2"],
    ]
    passes only when at least one member of group 0 AND at least one member of
    group 1 appear.
    """
    groups = item.get("required_concept_groups")
    if not groups:
        return True  # not applicable -> falls through to fact matching
    for group in groups:
        if not _group_present(answer, group):
            return False
    return True


# --------------------------------------------------------------------------- #
# Relationship scoring (q1: two performance relationships)
#
# Some questions require a *relationship* between concepts, not just the
# presence of terms. ``required_relationships`` is a list of group-pairs:
#     [ [scale_terms], [better_performance_terms] ]
# A relationship is satisfied when one clause of the answer contains a member
# of BOTH groups (clause = punctuation- or "and"-delimited fragment, normalized
# as usual). All listed relationships must be satisfied. This is fully
# deterministic and defined only from course/question terminology.
# --------------------------------------------------------------------------- #
_CLAUSE_SPLIT = re.compile(
    r"[.;!?]+|\s+(?:and|but|whereas|while|however|although)\s+")


def _clauses(answer: str) -> list[str]:
    """Split into proposition fragments for relationship scoring.

    Splits on strong sentence boundaries (\".;!?\") AND on conjunctions that
    introduce independent propositions (\"and / but / whereas / while /
    however / although\"), so \"Larger models perform better and newer models
    are merely mentioned.\" is separated into two propositions — the second of
    which does NOT establish recency -> better performance.

    Commas and parentheticals are intentionally left intact, so a phrase like
    \"newer models, also, tend to perform better\" stays one proposition.
    \"and\" inside a word (e.g. \"concatenate\") is not affected because the
    pattern requires surrounding whitespace.
    """
    parts = _CLAUSE_SPLIT.split(answer or "")
    out: list[str] = []
    for part in parts:
        n = normalize(part)
        if n:
            out.append(n)
    return out


def score_relationships(item: dict, answer: str) -> bool:
    """Relationship-group correctness: every required relationship (a pair of
    concept groups) must be satisfied within a single clause of the answer."""
    rels = item.get("required_relationships")
    if not rels:
        return True  # not applicable
    clauses = _clauses(answer)
    if not clauses:
        return False
    for group_a, group_b in rels:
        satisfied = False
        for clause in clauses:
            if _group_present(clause, group_a) and _group_present(clause, group_b):
                satisfied = True
                break
        if not satisfied:
            return False
    return True


def score_answer(item: dict, answer: str) -> bool:
    """Answer correctness per score mode.

    Priority:
    1. ``required_relationships`` — each (concept-pair) relationship must hold
       within a clause (q1).
    2. ``required_concept_groups`` — every group must contribute (q2/q4).
    3. ``expected_facts`` + ``match_mode`` (all/any) — other questions.
    Unanswerable items score via insufficiency instead.
    """
    if item.get("type") == "unanswerable":
        return True  # correctness measured by insufficient_correct instead
    if item.get("required_relationships"):
        return score_relationships(item, answer)
    if item.get("required_concept_groups"):
        return score_concept_groups(item, answer)
    facts = item.get("expected_facts") or []
    if not facts:
        return False
    aliases = item.get("aliases") or {}
    mode = item.get("match_mode", "all")
    if mode == "any":
        return any(fact_present(answer or "", f, aliases) for f in facts)
    return all(fact_present(answer or "", f, aliases) for f in facts)


def score_source_hit(item: dict, cited_pages: list[int]) -> bool:
    """Source-support: at least one cited page in expected_pages.

    Extra legitimate supporting citations do NOT cause failure (user rule).
    Text-only unanswerable returns False (no expected pages).

    For multi-part questions with ``required_source_groups`` (e.g. q2: FP8 on
    p18 AND quantization on p15), the final citations must satisfy EVERY group
    — a single unrelated expected page cannot establish support for the whole
    multi-part answer. Single-topic questions keep the ordinary
    any-expected-page rule.
    """
    groups = item.get("required_source_groups")
    if groups:
        cited = set(cited_pages or [])
        return all(bool(cited & set(g)) for g in groups)
    expected = set(item.get("expected_pages") or [])
    if not expected:
        return False
    return bool(expected & set(cited_pages or []))


def score_sequence(item: dict, answer: str) -> bool | None:
    """Deterministic sequence check for workflow-order questions.

    Verifies the tokens in ``expected_sequence`` occur in that relative order
    within the normalized answer. Only used when the item defines
    ``expected_sequence`` (e.g. q6 Iterating the Vibes); otherwise not
    applicable (returns None).
    """
    seq = item.get("expected_sequence")
    if not seq:
        return None
    n_answer = normalize(answer or "")
    if not n_answer:
        return False
    pos = -1
    for token in seq:
        ntok = normalize(token)
        idx = n_answer.find(ntok)
        if idx == -1:
            return False
        if idx < pos:  # out of order
            return False
        pos = idx
    return True


def score_insufficient(item: dict, answer: str, insufficient: bool,
                       source_ids_empty: bool) -> bool:
    """Unanswerable passes only when insufficient==True AND sources==[] AND the
    answer communicates the course material lacks the information."""
    if item.get("type") != "unanswerable":
        return False
    refusal = any(fr in (answer or "").lower() for fr in
                  (item.get("refusal_phrases") or ["not contain"]))
    return bool(insufficient) and bool(source_ids_empty) and refusal


def score_visual(item: dict, supplied_pages: list[int],
                 cited_pages: list[int], cited_images: list[str]) -> dict:
    """Independent visual metrics:

    ``visual_retrieval_hit`` — expected visual page was supplied to generation.
    ``visual_citation_hit`` — expected visual page in trusted sources AND its
    real slide image is present.
    """
    visual_page = item.get("visual_page")
    if not visual_page or item.get("type") != "visual":
        return {"visual_retrieval_hit": None, "visual_citation_hit": None}
    retrieval_hit = visual_page in (supplied_pages or [])
    citation_hit = (visual_page in (cited_pages or [])
                    and any(p for p in (cited_images or []) if p))
    return {
        "visual_retrieval_hit": retrieval_hit,
        "visual_citation_hit": citation_hit,
    }
