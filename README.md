# Hybrid Multimodal RAG — Course Material Assistant (MBAX 6418, Assignment 2)

A retrieval-augmented generation (RAG) system that answers questions and
generates practice quizzes/flashcards over uploaded course material (slide
decks), with every answer grounded in citable source documents, pages, and
images.

The pipeline is **hybrid multimodal**:

- **Keyword + dense text retrieval** — BM25 and text embeddings over
  source-preserving text chunks.
- **Visual retrieval** — embeddings of the actual slide/page images, so
  diagrams, charts, and memes are retrievable by description.
- **Multimodal reranking** — a reranker combines text + visual evidence and
  picks the strongest candidates (with a deterministic RRF fallback if the
  reranker is unavailable).
- **Grounded generation** — a vision-capable LLM answers from the supplied
  evidence; Python validates every source id and builds trusted citations.

## Architecture

```
Upload course material (PDF / PPTX)
        ↓
Parse / OCR / render pages
        ↓
┌─────────────────────────────┬────────────────────────────┐
Text chunks                  Page images                    │
  → BM25                      → Visual embeddings           │
  → Text embeddings                                        │
        └──────────────────────┬───────────────────────────┘
                         Candidate fusion
                               ↓
                     Multimodal reranker
                               ↓
                       Grounded evidence
                     ↙                    ↘
                QA (answers)          Quiz / Flashcards
                + trusted sources     (frozen server-side keys)
```

See [`docs/architecture.svg`](docs/architecture.svg) for the full diagram.

## Features

### 📁 Document Manager
- Upload **PDF** (native) or **PowerPoint `.pptx`** (auto-converted to PDF via
  LibreOffice), then per-page rendering + OCR.
- SHA-256 **duplicate detection** — re-uploading the same deck is rejected.
- Per-page preview: original rendered slide image + extracted text.
- Delete a document — removes it *and* its indexed content (embeddings/caches)
  so it can no longer be retrieved.

### ❓ Ask a Question
- Answers grounded ONLY in the retrieved course evidence.
- Sources are shown as trusted citations: `document · page N`, the supporting
  excerpt, and the actual slide/page image.
- **Insufficient evidence**: if the materials don't contain the answer, the
  system says so (`insufficient = true`) and returns no sources — it never
  fabricates an answer or citation.

### 📝 Practice Quiz
- Generates a multiple-choice quiz (default 5 questions, configurable) grounded
  in the course material, in a single generation call.
- **Fixed, server-side answer key**: the questions/answers are frozen when the
  quiz is generated and are never regenerated; grading makes **zero** LLM
  calls.
- Answer using plain radio buttons (no JSON entry); score plus per-question
  feedback appears on submit; explanation, citation, excerpt, and slide image
  are revealed only after you submit or click **Show Answers**.

### 🃏 Flashcards
- Flashcard decks generated from the same evidence pipeline (question/answer
  cards), with **star/unstar** so you can study cards you missed; star state is
  tracked per session.

## Evaluation

A fixed 8-question evaluation (5 text / 2 visual / 1 unanswerable) compares a
**text-only baseline** (BM25 + text embeddings) against the **full multimodal
pipeline** (text + visual embeddings + multimodal reranking) on the Week 2
course deck — same questions, documents, model, and `top_k=5`, with alternating
execution order and index construction excluded from per-question latency.

| Metric | A: Text-only | B: Full multimodal |
|---|---:|---:|
| Answer accuracy | 0.71 | 1.00 |
| Source support | 0.86 | 1.00 |
| Visual citation success | 0.00 | 1.00 |
| Unanswerable accuracy | 1.00 | 1.00 |
| Avg retrieval latency | 0.69s | 3.56s |
| Avg generation latency | 0.58s | 1.35s |
| Avg total latency | 1.27s | 4.92s |

One-time setup (reported separately, outside per-question latency):
- Text index build: **0.015s**
- Visual index build: **25.34s**

**Interpretation:** in this 8-question evaluation the full multimodal pipeline
achieved higher answer accuracy and visual-source success, while the text-only
baseline had lower retrieval and end-to-end latency. These results are specific
to this evaluation set and the Week 2 course material — they are not a claim
that one approach is universally better.

Full per-question results and the harness live in [`evaluation/`](evaluation/):
- `evaluation/results.json` (16 records), `evaluation/results.csv`
- `evaluation/eval_set.py` — the frozen, grounded question set
- `evaluation/eval_harness.py` — deterministic A/B runner
- `evaluation/scoring.py`, `evaluation/aggregate.py` — deterministic scoring
  (no LLM judge; normalized concept/relationship/sequence matching)

## Installation

Requires **Python ≥ 3.10**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # use .venv/Scripts/activate on Windows
pip install -r requirements.txt
cp .env.example .env               # then put the real class API key in .env
```

**Never commit the real API key.** `.env` is git-ignored; only the placeholder
`.env.example` is tracked. Keys must stay in `.env` (or your environment) —
never in code, logs, tests, screenshots, or GitHub.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `CLASS_API_KEY` | Shared class API key (from `.env` only) | — |
| `TEXT_EMBEDDING_URL` | Text embeddings (9002, Nemotron) | `http://dobolyi.com:9002/v2/embed` |
| `VISUAL_EMBEDDING_URL` | Visual embeddings (9003, Qwen3-VL) | `http://dobolyi.com:9003/v1/embeddings` |
| `RERANKER_URL` | Multimodal reranker (9004, Qwen3-VL) | `http://dobolyi.com:9004/rerank` |
| `DOCUMENT_PARSER_URL` | Doc parser / OCR (9005, dots.mocr) | `http://dobolyi.com:9005/v1/chat/completions` |
| `LLM_URL` | Answer-generation LLM (9001, Qwen3.6) | `http://dobolyi.com:9001/v1/chat/completions` |
| `LLM_MODEL` | Model id for 9001 | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` |
| `DATA_DIR` | Storage root (documents, pages, indexes) | `<repo>/data` |

These are the **class-provided services** (styles: 9001/9002/9005 serve
vLLM/DGI-compatible APIs; 9004 serves a Jina-style rerank API). They must be
reachable from your machine.

## Supported formats & conversion (PowerPoint)

| Upload format | How it's handled | Extra software needed |
|---|---|---|
| `.pdf` | Used directly (native). | — |
| `.pptx` | Auto-converted to PDF via headless LibreOffice, then rendered/OCR'd. | **LibreOffice** |
| anything else (`.doc`, `.docx`, …) | Rejected; export to PDF first. | — |

**Required software:** LibreOffice is needed **only for `.pptx` uploads** —
not for PDFs and not to run the test suite.

```bash
brew install --cask libreoffice     # macOS
```

or download from <https://www.libreoffice.org/download/>. `rag/converter.py`
locates `soffice` on `PATH` (with fallbacks for standard macOS install
locations) and raises a clear error if it's missing.

**Workaround (no install):** export the deck to PDF in PowerPoint
(`File > Export > PDF`) and upload the PDF.

## Running the app

```bash
python -m app
```

Opens the Gradio UI at <http://127.0.0.1:7860> with four tabs:

1. **📁 Document Manager** — upload (PDF/PPTX), browse documents/pages, preview
   slide images + extracted text, delete documents.
2. **❓ Ask a Question** — type a question about the uploaded material; see the
   answer, trusted citations (`document · page N`), supporting excerpt, and
   slide images.
3. **📝 Practice Quiz** — pick a topic (blank = auto) and question count →
   *Generate Quiz* → answer with radio buttons → *Submit Quiz* → score and
   per-question feedback → *Show Answers* → explanations, citations, excerpts,
   and images.
4. **🃏 Flashcards** — generate a study deck, flip cards, star/unstar the ones
   you missed.

### Workflow tips
- **Upload material first** (Document Manager), then ask questions / generate
  quizzes — retrieval and generation run over whatever is in the collection.
- Re-uploads of an identical file are rejected as duplicates (SHA-256).
- Deleting a document removes it from retrieval: ask again afterward and it
  will no longer be cited.
- If the reranker is unavailable, the system automatically falls back to a
  deterministic reciprocal-rank fusion over text + visual evidence (still
  multimodal) and marks the answer accordingly.

## Tests

```bash
python -m pytest tests/ -q
```

**269 tests** (offline, deterministic): ingest/render/OCR, store + dedupe,
BM25/dense/visual retrieval, rerank + RRF fallback, QA grounding + source
validation, quiz fixed-key behavior, flashcards, converter passthrough/error
paths, the evaluation harness/scoring, and UI-layer integration.

A live converter smoke test is available as `convert_demo.py` (requires
LibreOffice if you want to run it; it writes only git-ignored demo artifacts).

## Repository layout

```
app.py                      Gradio UI (4 tabs)
convert_demo.py             PPTX→PDF demo (git-ignored outputs)
rag/                        core package (config, ingest, retrieval, QA,
                            quiz, flashcards, converter, reranker, visual…)
evaluation/                 frozen eval set, harness, scoring, results
tests/                      pytest suite (269 tests)
docs/architecture.svg       pipeline diagram
docs/screenshots/           UI screenshots
.env.example                placeholder config (never contains real keys)
```

## Limitations

- Evaluation results are specific to the fixed 8-question set over the Week 2
  deck; absolute numbers will differ on other material.
- Visual retrieval quality depends on the deck's image quality and how
  describable the diagrams are.
- `.pptx` uploads require LibreOffice installed locally (see above).
- The class endpoints must be reachable from your machine; when 9001/9004 are
  down the app degrades gracefully (offline-friendly error messages, RRF
  fallback), but real answers need the live services.
- Quiz/flashcard answer keys and star state are **per-session** (in-memory);
  they are not persisted across restarts.

## Security

- API keys are read **only** from `.env` / environment variables
  (`rag/config.py`, `get_settings()`); they are never committed.
- `.gitignore` excludes `.env`, `data/`, `__pycache__/`, probe/dev artifacts,
  and generated demo files.
- Final-tree gitleaks scan: **no leaks found**.
- The repository stays **private**; do not make it public without an audit.
