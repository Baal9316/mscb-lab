"""Scientific-cleaner Gradio UI for Milestone 1 document ingestion.

This is a UI ONLY around the already-merged backend. It delegates all document
logic to :func:`rag.ingest.ingest_document` and :class:`rag.store.DocumentStore`
-- it does NOT re-implement ingestion, rendering, or parsing.

Launch:
    source .venv/bin/activate
    python app.py
Then open the printed local URL (default http://127.0.0.1:7860).
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from rag.config import Settings, get_settings
from rag.ingest import DocumentStore, DocumentParserError, ingest_document


# --------------------------------------------------------------------------- #
# Backend-facing helpers (kept separate so they are unit-testable without the
# Gradio server). All write to the same store the UI shows.
# --------------------------------------------------------------------------- #
class UIError(RuntimeError):
    pass


def _new_store(settings: Settings | None = None) -> DocumentStore:
    s = settings or get_settings()
    return DocumentStore(s.data_dir)


# --------------------------------------------------------------------------- #
# Runtime QA index cache (Issue #5 performance).
#
# Building TextIndex/VisualIndex re-embeds every document through 9002/9003 on
# each call, so we cache the built indexes at the app level and reuse them for
# repeated questions with unchanged documents. The cache is invalidated by a
# change in the collection fingerprint (doc id + uploaded_at), which changes
# on upload or delete.
# --------------------------------------------------------------------------- #
_QA_CACHE: dict = {}


def _collection_fingerprint(store: DocumentStore) -> tuple:
    """Stable signature of the current document collection."""
    docs = store.list_all()
    sig = tuple(sorted((d.document_id, d.uploaded_at) for d in docs))
    return sig


def get_qa_indexes(store: DocumentStore,
                   settings: Settings | None = None):
    """Return cached or freshly-built (TextIndex, VisualIndex).

    Reuses the runtime indexes when the collection is unchanged; rebuilds only
    when the fingerprint changes (i.e. after an upload or delete).
    """
    s = settings or get_settings()
    fingerprint = _collection_fingerprint(store)

    cached = _QA_CACHE.get("indexes")
    if cached and cached.get("fingerprint") == fingerprint:
        return cached["text_idx"], cached["visual_idx"]

    from rag.retrieval import TextIndex
    from rag.visual_retrieval import VisualIndex

    text_idx = TextIndex(store, settings=s)
    text_idx.rebuild()
    visual_idx = VisualIndex(store, settings=s)
    _QA_CACHE["indexes"] = {
        "fingerprint": fingerprint,
        "text_idx": text_idx,
        "visual_idx": visual_idx,
    }
    return text_idx, visual_idx


def invalidate_qa_index_cache() -> None:
    """Force a rebuild on the next question (used after upload/delete in tests)."""
    _QA_CACHE.pop("indexes", None)


def upload_pdf(file_path, settings: Settings | None = None) -> str:
    """Upload a PDF via the Milestone 1 ingest_document backend.

    Returns a human-readable message. Duplicates are detected up front (by
    SHA-256) and reported clearly without re-ingesting.
    """
    if not file_path:
        return "Please select a PDF file first."

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return f"Only PDF files are supported (got '{path.suffix}')."

    if not path.exists():
        return "The selected file could not be read."

    s = settings or get_settings()
    store = _new_store(s)

    try:
        sha256 = DocumentStore.sha256_of(path)
    except OSError as exc:
        return f"Could not read file: {exc}"

    # Clear duplicate report (no re-ingest).
    existing = store.find_by_sha256(sha256)
    if existing is not None:
        return (f"⚠️ Duplicate upload rejected: '{existing.filename}' "
                f"(document {existing.document_id}) is already in the collection "
                "with the exact same content.")

    if not s.is_api_key_set:
        return ("Configuration error: CLASS_API_KEY is not set. Add it to a "
                "local .env file (see .env.example) and restart.")

    try:
        doc = ingest_document(path, store=store, settings=s)
    except Exception as exc:  # noqa: BLE001 - surface safely to the user
        return f"Upload failed: {exc}"

    # New document added -> cached QA indexes are stale; force a rebuild.
    invalidate_qa_index_cache()

    failed = sum(1 for p in doc.pages if p.parse_status == "failed")
    msg = (f"✅ Uploaded '{doc.filename}' — {doc.page_count} pages "
           f"(document {doc.document_id}).")
    if failed:
        msg += f"\n⚠️ {failed} page(s) could not be OCR-parsed (see page view)."
    return msg + "\n\nSelect the document below to inspect its pages."


def refresh_documents(settings: Settings | None = None) -> tuple[str, gr.Dropdown]:
    """Reload the document list. Returns (message, dropdown choices)."""
    store = _new_store(settings)
    docs = store.list_all()
    if not docs:
        msg = "No documents yet — upload a PDF above."
        return msg, gr.Dropdown(choices=[], value=None)
    lines = []
    for d in docs:
        lines.append(f"• {d.filename}  —  {d.page_count} pages  ({d.document_id})")
    choices = [f"{d.filename} ({d.page_count}p) :: {d.document_id}" for d in docs]
    return "\n".join(lines), gr.Dropdown(choices=choices, value=None)


def load_document_options(settings: Settings | None = None) -> list[str]:
    store = _new_store(settings)
    return [f"{d.filename} ({d.page_count}p) :: {d.document_id}" for d in store.list_all()]


def get_pages_for_document(label: str | None,
                           settings: Settings | None = None) -> gr.Dropdown:
    """Given a document dropdown label, return the page-dropdown options."""
    if not label:
        return gr.Dropdown(choices=[], value=None)
    document_id = _id_from_label(label)
    store = _new_store(settings)
    doc = store.get(document_id)
    if not doc:
        return gr.Dropdown(choices=[], value=None)
    choices = [_page_label(p.page_no) for p in doc.pages]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def _page_label(page_no: int) -> str:
    return f"Page {page_no}"


def _id_from_label(label: str) -> str:
    return label.split("::")[-1].strip()


def _page_from_label(label: str) -> int:
    return int(label.replace("Page ", "").strip())


def view_page(doc_label: str | None, page_label: str | None,
              settings: Settings | None = None) -> tuple:
    """Return (image_path_or_None, text, status_html) for a selected page.

    Never returns the API key or raw error headers; parse errors are shown as a
    short, safe description already recorded on the page.
    """
    if not doc_label or not page_label:
        return (None, "Select a document and a page.", "Status: —")
    store = _new_store(settings)
    doc = store.get(_id_from_label(doc_label))
    if not doc:
        return (None, "Document not found (it may have been deleted).", "Status: —")
    page_no = _page_from_label(page_label)
    page = next((p for p in doc.pages if p.page_no == page_no), None)
    if page is None:
        return (None, f"Page {page_no} not found.", "Status: —")

    if page.parse_status == "failed":
        status = (f"⚠️ Parsing failed: {page.parse_error or 'unknown error'}\n"
                  "Showing the original slide image; text unavailable.")
        text = page.extracted_text or "(no text extracted)"
    else:
        status = "✅ Parsing succeeded."
        text = page.extracted_text or "(empty page — no text extracted)"

    image = page.image_path if Path(page.image_path).exists() else None
    return image, text, status


def delete_document(doc_label: str | None,
                    settings: Settings | None = None) -> str:
    """Delete a document (removes files, pages, and indexed content)."""
    if not doc_label:
        return "Select a document to delete."
    store = _new_store(settings)
    document_id = _id_from_label(doc_label)
    if store.delete(document_id):
        invalidate_qa_index_cache()  # deleted doc -> QA indexes stale
        return f"🗑️ Deleted document {document_id}."
    return "Document not found (it may already be deleted)."


def ask_question(question: str, settings: Settings | None = None) -> tuple:
    """Ask a question through the full retrieval+rerank+QA pipeline.

    Returns (answer_text, sources_block, images, status). ``images`` is a list
    of source image file paths (one per cited source) for the UI to render.
    """
    from rag.qa import answer_question, QAGenerationError

    if not question or not question.strip():
        return ("Please enter a question.", "", [], "⚠️ No question")
    s = settings or get_settings()
    store = _new_store(s)

    try:
        ti, vi = get_qa_indexes(store, settings=s)
    except Exception as exc:  # noqa: BLE001
        return (f"Could not build retrieval indexes: {exc}", "", [], "⚠️ Error")

    try:
        result = answer_question(question.strip(), ti, vi, settings=s)
    except QAGenerationError as exc:
        return (f"⚠️ Answer generation failed: {exc}", "", [],
                "⚠️ Generation unavailable")
    except Exception as exc:  # noqa: BLE001
        return (f"⚠️ QA pipeline error: {exc}", "", [], "⚠️ Error")

    if result.insufficient:
        return (result.answer, "_(no sources — answer marked insufficient)_",
                [], "ℹ️ Insufficient evidence")

    lines = []
    images = []
    for src in result.sources:
        lines.append(f"**{src.citation}**  — chunk {src.chunk_index}")
        excerpt = (src.excerpt[:300] + "…") if len(src.excerpt) > 300 else src.excerpt
        lines.append(f"> {excerpt}")
        if src.image_path and Path(src.image_path).exists():
            images.append(src.image_path)

    sources_block = "\n\n".join(lines) if lines else "_(no sources cited)_"
    status = "✅ Fallback used (reranker unavailable)" if result.used_fallback else "✅ OK"
    return (result.answer, sources_block, images, status)


# --------------------------------------------------------------------------- #
# Gradio app
# --------------------------------------------------------------------------- #
def build_app(settings: Settings | None = None) -> gr.Blocks:
    """Construct the Gradio UI. ``settings`` is injectable for tests.

    Two tabs: (1) Document manager (Milestone 1), (2) Ask a Question (Issue #5).
    """
    s = settings or get_settings()

    with gr.Blocks(title="Course Material Manager – MBAX 6418") as demo:
        gr.Markdown("# 📚 Course Material Manager")
        gr.Markdown("Hybrid multimodal RAG over course materials — manage course "
                    "PDFs and ask grounded questions with citations.")

        if not s.is_api_key_set:
            gr.Markdown("**⚠️ CLASS_API_KEY is not set.** Set it in a local `.env` "
                        "(see `.env.example`) and restart.")

        with gr.Tabs():
            # ---------------- TAB 1: Document manager (unchanged) ---------------
            with gr.Tab("📁 Document Manager"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1. Upload a PDF")
                        file_input = gr.File(label="PDF file", file_types=[".pdf"],
                                             type="filepath")
                        upload_btn = gr.Button("Upload", variant="primary")
                        upload_result = gr.Textbox(label="Upload result", lines=3,
                                                   interactive=False)

                        gr.Markdown("### 2. Manage documents")
                        doc_dropdown = gr.Dropdown(label="Document", choices=[],
                                                   interactive=True)
                        refresh_btn = gr.Button("↻ Refresh list")
                        doc_list = gr.Textbox(label="In collection", lines=5,
                                              interactive=False)
                        delete_btn = gr.Button("🗑️ Delete selected document",
                                               variant="stop")
                        delete_result = gr.Textbox(label="Delete result", lines=1,
                                                   interactive=False)

                    with gr.Column(scale=3):
                        gr.Markdown("### 3. Page preview")
                        page_dropdown = gr.Dropdown(label="Page", choices=[],
                                                    interactive=True)
                        page_status = gr.Markdown("Status: —")
                        page_image = gr.Image(label="Original rendered slide/page",
                                              type="filepath", height=480)
                        page_text = gr.Textbox(label="Extracted text", lines=10,
                                               interactive=False)

                # Document-manager wiring
                upload_btn.click(upload_pdf, inputs=[file_input], outputs=[upload_result])
                upload_btn.click(refresh_documents, outputs=[doc_list, doc_dropdown])
                refresh_btn.click(refresh_documents, outputs=[doc_list, doc_dropdown])
                doc_dropdown.change(get_pages_for_document, inputs=[doc_dropdown],
                                    outputs=[page_dropdown])
                doc_dropdown.change(view_page, inputs=[doc_dropdown, page_dropdown],
                                    outputs=[page_image, page_text, page_status])
                page_dropdown.change(view_page, inputs=[doc_dropdown, page_dropdown],
                                    outputs=[page_image, page_text, page_status])
                delete_btn.click(delete_document, inputs=[doc_dropdown],
                                 outputs=[delete_result])
                delete_btn.click(refresh_documents, outputs=[doc_list, doc_dropdown])

            # ---------------- TAB 2: Ask a Question (Issue #5) -----------------
            with gr.Tab("❓ Ask a Question"):
                gr.Markdown("Ask about the ingested course materials. The answer is "
                            "grounded in retrieved evidence with trusted citations.")
                with gr.Column():
                    question_input = gr.Textbox(label="Ask a Question", lines=3,
                                                placeholder="e.g. What is the CPU in a computer?")

                    gr.Markdown(
                        "Optional: attach a slide image to ask a visual question."
                    )
                    ask_btn = gr.Button("Ask", variant="primary")
                    qa_status = gr.Markdown("Status: —")
                    answer_out = gr.Textbox(label="Answer", lines=6, interactive=False)
                    sources_out = gr.Markdown(label="Sources")
                    source_images = gr.Gallery(label="Supporting slide/page images",
                                               columns=1, height="auto")

                ask_btn.click(
                    ask_question,
                    inputs=[question_input],
                    outputs=[answer_out, sources_out, source_images, qa_status],
                )

    return demo


def main() -> None:
    demo = build_app()
    # Launch and force Gradio to print the local URL explicitly.
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inbrowser=True)


if __name__ == "__main__":
    main()
