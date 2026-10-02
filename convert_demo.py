"""Live PPTX -> PDF -> images demo, with a slide viewer for visual checks.

Requires LibreOffice installed on the machine (see README "Supported formats &
conversion"). Builds a small 2-slide deck, converts it to PDF via
rag.converter, runs it through the normal render + OCR pipeline, and writes a
self-contained slide-viewer HTML so you can visually confirm the converted
slides still look correct (fonts/layout fidelity).

Run:   .venv/bin/python convert_demo.py   (Installs: brew install --cask libreoffice)
"""
from __future__ import annotations

import base64
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

from rag import converter
from rag.renderer import render_pdf_pages

HERE = Path(__file__).resolve().parent
DECK = HERE / "_demo_deck.pptx"
WORK = HERE / "_convert_work"
VIEWER = HERE / "converted_slides.html"


def make_deck() -> Path:
    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[5])  # blank
    tb = s1.shapes.add_textbox(Inches(0.6), Inches(0.5), Inches(8), Inches(1))
    tb.text_frame.text = "MBAX 6418 — Portfolio Theory"
    p = s1.shapes.add_textbox(Inches(0.6), Inches(2.2), Inches(8), Inches(2))
    p.text_frame.text = ("Efficient portfolios maximize expected return for a "
                         "given level of risk.\nSlides convert & render images.")
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    b = s2.shapes.add_textbox(Inches(0.6), Inches(1.0), Inches(8), Inches(1))
    b.text_frame.text = "Slide 2 — Payoff diagram"
    tb2 = s2.shapes.add_textbox(Inches(0.6), Inches(2.5), Inches(3), Inches(1.4))
    tb2.text_frame.text = "Risk-free  vs.  Risky asset"
    prs.save(str(DECK))
    return DECK


def data_url(path: str | Path) -> str:
    """Base64 data URL from a PNG path (accepts str or Path — the renderer
    returns image_path as str)."""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def viewer(pages) -> str:
    imgs = "".join(
        f'<div class="slide"><h3>Rendered page {p.page_no}</h3>'
        f'<img src="{data_url(p.image_path)}" alt="page {p.page_no}"></div>'
        for p in pages)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>PPTX->PDF conversion check</title><style>
body{{font-family:-apple-system,sans-serif;background:#f6f7f9;color:#1f2937;margin:0}}
.wrap{{max-width:1000px;margin:0 auto;padding:22px}}
h1{{font-size:19px}}.note{{color:#6b7280;font-size:13px;margin-bottom:18px}}
.slide{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin-bottom:18px}}
.slide img{{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}}
</style></head><body><div class="wrap">
<h1>PowerPoint → PDF → rendered slide images</h1>
<div class="note">Source: <code>{DECK.name}</code> (2 slides) · converted by {
converter.find_soffice() or "LibreOffice"} · rendered with PyMuPDF. Visually confirm text/layout fidelity.</div>
{imgs}
</div></body></html>"""


def main() -> None:
    make_deck()
    WORK.mkdir(exist_ok=True)
    print("Converting deck to PDF via LibreOffice ...")
    pdf = converter.to_pdf(DECK, WORK)          # pptx -> pdf
    print(f"  PDF produced: {pdf} ({pdf.stat().st_size} bytes)")
    pages = render_pdf_pages(pdf, WORK / "pages")   # pdf -> PNGs
    print(f"  Rendered {len(pages)} page image(s)")
    for p in pages:
        print(f"    page {p.page_no}: {p.width}x{p.height}px at {p.image_path}")
    VIEWER.write_text(viewer(pages), encoding="utf-8")
    print(f"  Slide viewer -> {VIEWER}")


if __name__ == "__main__":
    main()
