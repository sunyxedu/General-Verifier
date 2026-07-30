#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the report PDF: headless Chrome print + page-number overlay.

Usage:  python3 report/build.py
Output: "Test-time Verification 技术报告.pdf" at the repo root.
Deps:   Google Chrome, pypdf, reportlab (system python3 is fine).
"""
import io
import os
import subprocess
import sys

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "report.html")
RAW = os.path.join(HERE, ".raw.pdf")
OUT = os.path.join(ROOT, "Test-time Verification 技术报告.pdf")


def print_to_pdf() -> None:
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu",
         f"--print-to-pdf={RAW}", "--no-pdf-header-footer",
         "file://" + SRC],
        check=True, capture_output=True, timeout=120,
    )


def overlay_page_numbers() -> int:
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    reader = PdfReader(RAW)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages, start=1):
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.45, 0.44, 0.42)
        c.drawCentredString(A4[0] / 2, 24, str(i))
        c.save()
        buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)
    with open(OUT, "wb") as f:
        writer.write(f)
    return len(reader.pages)


def main() -> None:
    print_to_pdf()
    pages = overlay_page_numbers()
    os.remove(RAW)
    print(f"built: {OUT} ({pages} pages)")


if __name__ == "__main__":
    sys.exit(main())
