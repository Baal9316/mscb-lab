"""Fixed, grounded 8-question evaluation set for Issue #7.

Every expected page / expected fact below was verified against the ACTUAL
Week 2 deck (the course deck used for this assignment's evaluations).
DO NOT guess or edit pages without re-inspecting the deck — the harness tests
assert this file changes only with explicit review.

q4 was finalized as "context / context window" (page 8) after the originally
approved "temperature / top-p" topic was confirmed absent from the deck.
"""
from __future__ import annotations

# Neutral label for the course deck used to ground pages (for human
# readability only; results never embed credentials or private filenames).
WEEK2_FILENAME = "Week 2 deck"

# ---------------------------------------------------------------------------- #
# Evaluation set (order fixed). Each item:
#   question_id, question, type, expected_facts[], match_mode, aliases{},
#   expected_pages[], and (for visual) the required retrieval/citation page.
# ---------------------------------------------------------------------------- #
EVAL_SET = [
    {
        "question_id": "q1",
        "question": (
            "According to the course material, how do larger models and newer "
            "(more recent) models each relate to performance?"),
        "type": "text",
        "expected_facts": [
            "larger models tend to perform better",
            "more recent models tend to perform better",
        ],
        "match_mode": "all",
        "aliases": {
            "larger": ["bigger", "more parameters"],
            "recent": ["newer", "more recent"],
        },
        #: two performance relationships, each must hold within one clause.
        #: scale->better AND recency->better (both course-derived terms).
        "required_relationships": [
            [["larger", "bigger", "more parameters"],
             ["better", "perform better", "performs better", "better performance",
              "tend to perform better", "achieve better performance"]],
            [["more recent", "newer", "recent", "newest"],
             ["better", "perform better", "performs better", "better performance",
              "tend to perform better", "achieve better performance"]],
        ],
        "expected_pages": [6],
        "expected_source": "Week 2 · page 6",
    },
    {
        "question_id": "q2",
        "question": (
            "In a model name like Qwen3-Coder-30B-A3B-Instruct-FP8, what does "
            "the FP8 component indicate, and what is quantization?"),
        "type": "text",
        "expected_facts": [
            "FP8 is 8-bit",
            "quantization scales and rounds model parameters",
            "models trained at 16-bit can be reduced to 8-bit",
        ],
        "match_mode": "all",
        "aliases": {
            "8-bit": ["8 bits", "eight-bit"],
            "scales": ["rounding"],
        },
        #: semantic requirement 1: FP8 indicates 8-bit precision/format (p18).
        #: semantic requirement 2: quantization reduces numeric precision by
        #: scaling/rounding parameters incl. 16-bit -> 8-bit (p15).
        #: Both parts of the two-part question must actually be answered:
        #: FP8 meaning AND what quantization DOES. Quantization-meaning aliases
        #: are course-grounded (p15 wording), not taken from model answers.
        "required_concept_groups": [
            ["fp8", "fp 8"],                             # FP8 mentioned
            ["8-bit", "8 bit", "eight-bit", "8 bits"],   # 8-bit precision/format
            ["quantization", "quantized",                # quantization concept
             "quantize", "quantises", "quantizes"],
            ["scales and rounds", "scaling and rounding",  # what quantization does
             "scales and round", "scale and rounds",
             "reduces precision", "reduced precision",
             "lower precision", "reduce precision",
             "16-bit to 8-bit", "16 bit to 8 bit",
             "reduce values", "reduces values"],
        ],
        "expected_pages": [18, 15],
        "expected_source": "Week 2 · page 15 / 18",
        #: two-part question: support requires both groups of source pages.
        "required_source_groups": [[18], [15]],
    },
    {
        "question_id": "q3",
        "question": (
            "According to the course material, which exact tools are "
            "recommended for technical users running local models on a GPU?"),
        "type": "text",
        "expected_facts": [
            "vLLM",
        ],
        "match_mode": "all",
        "aliases": {
            "vllm": ["VLLM", "v-llm"],
        },
        "expected_pages": [26],
        "expected_source": "Week 2 · page 26",
    },
    {
        "question_id": "q4",
        "question": (
            "What is 'context' in the course material, and how is it measured?"),
        "type": "text",
        "expected_facts": [
            "how much text or other information a model can access",
            "measured in tokens",
            "referred to as context length or context window",
        ],
        "match_mode": "all",
        "aliases": {
            "tokens": ["text chunks"],
            "context window": ["context length"],
        },
        #: concept groups (course p8 wording):
        #: context = amount of text/info available/accessible to the model;
        #: measured in tokens; also called context length/context window.
        "required_concept_groups": [
            ["context", "context window", "context length", "contextual"],
            ["text", "information"],                 # text/information available
            ["available", "access", "accessible",    # ...to the model
             "available to the model", "can access"],
            ["tokens", "token", "text chunks"],      # measured in tokens
        ],
        "expected_pages": [8],
        "expected_source": "Week 2 · page 8",
    },
    {
        "question_id": "q5",
        "question": (
            "Find the meme about Vibe Coding on 'Prod' in the course slides. "
            "What does it show?"),
        "type": "visual",
        "expected_facts": [
            "Vibe Coding on Prod",
            "production-grade enterprise app",
        ],
        "match_mode": "any",
        "aliases": {
            "prod": ["production"],
            "one does not simply": ["vibe coding on prod"],
        },
        "expected_pages": [33],
        "expected_source": "Week 2 · page 33",
        "visual_page": 33,
    },
    {
        "question_id": "q6",
        "question": (
            "According to the 'Iterating the Vibes' workflow diagram, what is "
            "the sequence of steps in the vibe coding process?"),
        "type": "visual",
        "expected_facts": [
            "Understand",
            "Prompt",
            "Prototype",
            "Evaluate",
        ],
        "match_mode": "any",
        "aliases": {},
        "expected_pages": [31],
        "expected_source": "Week 2 · page 31",
        "visual_page": 31,
        #: workflow-order: the four steps must appear in this sequence.
        "expected_sequence": ["Understand", "Prompt", "Prototype", "Evaluate"],
    },
    {
        "question_id": "q7",
        "question": (
            "How does photosynthesis convert sunlight into chemical energy in "
            "chloroplasts?"),
        "type": "unanswerable",
        "expected_facts": [],  # not used; refusal expected
        "match_mode": "all",
        "aliases": {},
        "expected_pages": [],
        "expected_source": None,
        #: refusal must communicate insufficient course evidence
        "refusal_phrases": [
            "do not contain",
            "do not have",
            "not contain enough information",
            "not enough information",
            "course materials do not contain",
        ],
    },
    {
        "question_id": "q8",
        "question": (
            "How do model providers clarify how up-to-date (or not) a model's "
            "training data is, per the course material?"),
        "type": "text",
        "expected_facts": [
            "knowledge cutoff date",
            "training data",
        ],
        "match_mode": "all",
        "aliases": {
            "knowledge cutoff date": ["knowledge cutoff", "cutoff date"],
        },
        "expected_pages": [5],
        "expected_source": "Week 2 · page 5",
    },
]


def get_eval_set() -> list[dict]:
    """Return a copy of the fixed evaluation set (never mutate in place)."""
    return [dict(item) for item in EVAL_SET]
