"""Render a document (a list of blocks) as PDF, DOCX or plain text.

Blocks are tuples: ("h1", text), ("h2", text), ("p", text), ("table", rows).
Keeping the content format-neutral means one template can be saved in any
file type, which is useful for testing the parsers in phase 3.
"""
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FIXED_TIME = datetime(2026, 1, 1)  # stored in file metadata so reruns give the same bytes


def write_pdf(path: Path, blocks: list, title: str) -> None:
    styles = getSampleStyleSheet()
    story = []
    for kind, value in blocks:
        if kind == "h1":
            story.append(Paragraph(escape(value), styles["Title"]))
        elif kind == "h2":
            story.append(Paragraph(escape(value), styles["Heading2"]))
        elif kind == "p":
            story.append(Paragraph(escape(value), styles["BodyText"]))
            story.append(Spacer(1, 4))
        elif kind == "table":
            table = Table(value, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]))
            story.append(table)
            story.append(Spacer(1, 8))
    # invariant=1 removes the creation date and random id, so output is repeatable
    SimpleDocTemplate(str(path), pagesize=A4, title=title, author="OpsCopilot demo data",
                      invariant=1).build(story)


def write_docx(path: Path, blocks: list, title: str) -> None:
    doc = Document()
    doc.core_properties.title = title
    doc.core_properties.author = "OpsCopilot demo data"
    doc.core_properties.created = FIXED_TIME
    doc.core_properties.modified = FIXED_TIME
    for kind, value in blocks:
        if kind == "h1":
            doc.add_heading(value, level=0)
        elif kind == "h2":
            doc.add_heading(value, level=2)
        elif kind == "p":
            doc.add_paragraph(value)
        elif kind == "table":
            table = doc.add_table(rows=len(value), cols=len(value[0]))
            table.style = "Table Grid"
            for r, row in enumerate(value):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
    doc.save(str(path))


def write_txt(path: Path, blocks: list, title: str) -> None:
    lines = []
    for kind, value in blocks:
        if kind == "table":
            lines.extend(" | ".join(row) for row in value)
        elif kind == "h2":
            lines.append(f"\n{value}")
        else:
            lines.append(value)
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


WRITERS = {".pdf": write_pdf, ".docx": write_docx, ".txt": write_txt}


def write_doc(doc, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / doc.filename
    WRITERS[path.suffix](path, doc.blocks, doc.filename)
    return path
