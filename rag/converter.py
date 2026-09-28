"""Boundary-ish note: not in the notebook.

PowerPoint (`.pptx`) → PDF conversion so slides can be rendered as images.

The pipeline's core (render → OCR → store) works on PDFs. To also accept
PowerPoint decks directly, we convert `.pptx` to PDF first using a headless
LibreOffice install, then feed the resulting PDF into the normal pipeline.

Design
------
- `.pdf`  -> passthrough (no conversion needed).
- `.pptx` -> converted to PDF via ``soffice --headless --convert-to pdf``.
- Anything else -> unsupported (raises UnsupportedFormatError).

LibreOffice is an *extra* install required only for `.pptx` uploads (see
README "Supported formats & conversion"). If it isn't present we raise a clear
error telling the user to install it or re-export the deck to PDF manually
(`File > Export > PDF`) — the documented workaround.

Run:            .venv/bin/python -m pytest tests/test_converter.py -q
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".pptx"}
PPTX_EXT = ".pptx"


class ConversionError(RuntimeError):
    """Raised when a deck (pptx) cannot be converted and rendering cannot proceed."""


class UnsupportedFormatError(ConversionError):
    """Raised for files that are neither PDF nor PPTX."""


class LibreOfficeNotFoundError(ConversionError):
    """Raised when soffice is required but not installed."""


def find_soffice() -> str | None:
    """Locate the LibreOffice `soffice` binary.

    Checks PATH first, then common macOS locations. Returns binary path or None.
    """
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if exe:
        return exe
    candidates = (
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice.bin",
        "/opt/homebrew/bin/soffice",
        "/usr/local/bin/soffice",
        "/usr/bin/soffice",
    )
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def requires_conversion(path: str | Path) -> bool:
    return Path(path).suffix.lower() == PPTX_EXT


def to_pdf(src: str | Path, out_dir: str | Path,
           soffice: str | None = None) -> Path:
    """Convert ``src`` to a PDF in ``out_dir`` and return its path.

    - ``.pdf``   -> copied verbatim into ``out_dir`` (no conversion).
    - ``.pptx``  -> converted via headless LibreOffice.
    - otherwise  -> raises :class:`UnsupportedFormatError`.
    """
    src = Path(src)
    ext = src.suffix.lower()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not src.is_file():
        raise ConversionError(f"File not found: {src}")

    if ext == ".pdf":
        dest = out / src.name
        shutil.copy2(src, dest)
        return dest

    if ext != PPTX_EXT:
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Supported upload types: "
            f".pdf, .pptx. (For other document types, export to PDF first.)")

    binary = soffice or find_soffice()
    if binary is None:
        raise LibreOfficeNotFoundError(
            "PowerPoint conversion requires LibreOffice, which was not found. "
            "Install it (e.g. `brew install --cask libreoffice`) OR export the "
            "deck to PDF manually (File > Export > PDF) and upload the PDF.")

    cmd = [binary, "--headless", "--convert-to", "pdf",
           "--outdir", str(out), str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ConversionError(
            f"LibreOffice conversion failed for {src.name}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}")

    pdf_name = src.stem + ".pdf"
    result = out / pdf_name
    if not result.exists():
        raise ConversionError(
            f"LibreOffice reported success but no PDF was produced: {result}")
    return result
