"""Script untuk menghasilkan file PDF dummy dari file TXT.

Membaca file .txt di data/dummy/ dan mengonversinya menjadi
file .pdf yang rapi.

Jalankan:
    python -m scripts.generate_pdfs
"""

from __future__ import annotations

import logging
from pathlib import Path

from fpdf import FPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DUMMY_DIR = Path("data/dummy")


def _sanitize(text: str) -> str:
    """Replace non-latin1 characters."""
    replacements = {
        "\u2013": "-", "\u2014": "--", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00d7": "x",
        "\u2022": "-", "\u2192": "->", "\u00a0": " ",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def txt_to_pdf(txt_path: Path, pdf_path: Path) -> None:
    """Konversi file TXT menjadi PDF."""
    content = txt_path.read_text(encoding="utf-8")
    title = txt_path.stem.replace("_", " ").title()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(w=0, h=12, text=_sanitize(title), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    for line in content.split("\n"):
        stripped = line.strip()
        safe = _sanitize(stripped)

        # Skip separator lines
        if safe and len(safe) > 3 and all(c in ("=", "-") for c in safe):
            continue

        # ALL CAPS section headers
        if (safe and safe.isupper() and len(safe) > 3
                and not safe.startswith("-") and not safe.startswith("|")):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(0, 51, 102)
            pdf.ln(3)
            pdf.multi_cell(w=usable_w, h=7, text=safe, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            continue

        # Sub-headers ending with colon
        if (safe.endswith(":") and len(safe) < 80
                and not safe.startswith("-") and not safe.startswith("*")):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(w=usable_w, h=6, text=safe, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            continue

        # Table rows
        if safe.startswith("|"):
            pdf.set_font("Courier", "", 7)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(w=usable_w, h=4, text=safe, new_x="LMARGIN", new_y="NEXT")
            continue

        # Bullet points & list items
        if safe.startswith("- ") or safe.startswith("* "):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(30, 30, 30)
            indent = 4
            pdf.set_x(pdf.l_margin + indent)
            pdf.multi_cell(w=usable_w - indent, h=5, text=safe, new_x="LMARGIN", new_y="NEXT")
            continue

        # Numbered items
        if safe and safe[0].isdigit() and ". " in safe[:5]:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(w=usable_w, h=5, text=safe, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            continue

        # Q&A
        if safe.startswith("T: ") or safe.startswith("J: "):
            pdf.set_font("Helvetica", "B" if safe.startswith("T: ") else "", 9)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(w=usable_w, h=5, text=safe, new_x="LMARGIN", new_y="NEXT")
            if safe.startswith("J: "):
                pdf.ln(2)
            continue

        # Empty lines
        if not safe:
            pdf.ln(3)
            continue

        # Normal text
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(w=usable_w, h=5, text=safe, new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(pdf_path))
    logger.info(f"  PDF dibuat: {pdf_path}")


def main() -> None:
    """Generate PDF dari semua file TXT di data/dummy/."""
    logger.info("=" * 50)
    logger.info("Generating PDF files dari dummy data...")
    logger.info("=" * 50)

    txt_files = sorted(DUMMY_DIR.glob("*.txt"))
    if not txt_files:
        logger.warning("Tidak ada file .txt di data/dummy/")
        return

    for txt_file in txt_files:
        pdf_path = txt_file.with_suffix(".pdf")
        logger.info(f"Mengonversi: {txt_file.name} -> {pdf_path.name}")
        try:
            txt_to_pdf(txt_file, pdf_path)
        except Exception as exc:
            logger.error(f"Gagal: {txt_file.name}: {exc}")
            import traceback
            traceback.print_exc()

    logger.info("Selesai!")


if __name__ == "__main__":
    main()
