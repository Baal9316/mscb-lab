# Hybrid Multimodal RAG — Course Material Q&A (MBAX 6418, Assignment 2)

A retrieval-augmented generation system that answers questions over course
PDFs using **text** retrieval (BM25 + Nemotron embeddings), **visual** retrieval
(Qwen3-VL embeddings of slide images), and a **multimodal reranker**, with every
answer grounded in citable source documents and pages.

## Status — Milestone 1: Document Ingestion

Implemented on branch `feature/document-ingestion` (not merged):

- Python project structure (`rag/` package + `tests/`)
- Secure configuration via environment variables / `.env`
- PDF upload with SHA-256 duplicate detection
- PDF pages rendered to PNG images (PyMuPDF)
- Document parsing via the class OCR endpoint (port 9005, `dots.mocr`)
- Filename + page-number metadata preserved for citations
- Document listing and deletion (removes rendered pages + indexed content)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # use .venv/Scripts/activate on Windows
pip install -r requirements.txt
cp .env.example .env               # then put the real key in .env
```

**Never commit the real API key.** `.env` is git-ignored; only the placeholder
`.env.example` is tracked.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `CLASS_API_KEY` | Shared class API key (from `.env` only) | — |
| `TEXT_EMBEDDING_URL` | Text embeddings (9002, Nemotron) | `http://dobolyi.com:9002/v2/embed` |
| `VISUAL_EMBEDDING_URL` | Visual embeddings (9003, Qwen3-VL) | `http://dobolyi.com:9003/v1/embeddings` |
| `RERANKER_URL` | Multimodal reranker (9004, Qwen3-VL) | `http://dobolyi.com:9004/rerank` |
| `DOCUMENT_PARSER_URL` | Doc parser / OCR (9005, dots.mocr) | `http://dobolyi.com:9005/v1/chat/completions` |
| `DATA_DIR` | Storage root | `<repo>/data` |

## Supported formats & conversion (PowerPoint)

The ingestion pipeline operates on **PDFs** (pages are rendered to images and
OCR'd). To let users upload slides directly, the app also accepts **PowerPoint
(`.pptx`)** files and converts them to PDF before rendering/OCR.

| Upload format | How it's handled | Extra software needed |
|---|---|---|
| `.pdf` | Used directly (native). | — |
| `.pptx` | Auto-converted to PDF via headless LibreOffice, then rendered/OCR'd (wired into `app.py` upload). | **LibreOffice** |
| anything else (`.doc`, `.docx`, etc.) | Rejected; export to PDF first. | — |

**Conversion step** (`rag/converter.py` → `to_pdf`):

```bash
soffice --headless --convert-to pdf --outdir <out_dir> <deck.pptx>
```

The resulting PDF follows the normal path: `render_pdf_pages()` → page PNGs →
9005 OCR → stored per-page metadata. Native `.pdf` uploads skip conversion.
Uploading the **same deck twice is not duplicated** — SHA-256 dedupe applies to
the converted PDF, and the UI reports duplicates instead of re-ingesting.

**Required software:** LibreOffice is the *only* extra dependency, and it's
needed **only for `.pptx` uploads** — not for PDFs and not to run the test
suite. Install it with Homebrew:

```bash
brew install --cask libreoffice
```

or download from <https://www.libreoffice.org/download/>. `rag/converter.py`
locates `soffice` on `PATH` (with fallbacks for the standard macOS install
location) and both the app and the converter raise a clear, actionable error
if it's missing.

**Workaround (no install):** manually export the deck to PDF in PowerPoint
(`File > Export > PDF`) and upload the PDF — fully supported.

**Verification:** after conversion, rendered slide images and their extracted
text should be visually checked against the original deck (fonts/layout
fidelity). `convert_demo.py` converts a sample deck and emits a slide-viewer
HTML for inspection.

## Tests

```bash
python -m pytest tests/ -q
```

The suite runs fully offline — the 9005 parser client's HTTP layer is mocked,
and converter tests cover the passthrough/error paths without requiring
LibreOffice. A live conversion is exercised by `convert_demo.py` (requires
LibreOffice installed).

## Roadmap

See GitHub Issues #1–#8 for the planned work areas (ingestion, text/visual
retrieval, reranking, QA + citations, quizzes, evaluation, docs). Not yet built.
