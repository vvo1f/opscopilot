import pytest

from opscopilot.datagen.documents import make_contract, make_invoice, make_minutes
from opscopilot.datagen.writers import write_doc
from opscopilot.ingestion.parsers import EmptyDocument, UnsupportedFileType, parse_file


def text_of(path):
    return "\n".join(p.text for p in parse_file(path))


def test_pdf_text_and_accents(tmp_path):
    doc = make_invoice(0)  # Italian invoice: has accented letters and the euro sign
    text = text_of(write_doc(doc, tmp_path))
    assert doc.truth["invoice_number"] in text
    assert "€" in text and "Qtà" in text


def test_docx_keeps_the_table_content(tmp_path):
    doc = make_contract(0)  # contract_01 is a docx
    assert doc.filename.endswith(".docx")
    text = text_of(write_doc(doc, tmp_path))
    assert doc.truth["parties"][0] in text


def test_txt(tmp_path):
    doc = make_minutes(0)
    assert doc.filename.endswith(".txt")
    assert doc.truth["attendees"][0] in text_of(write_doc(doc, tmp_path))


def test_unsupported_extension(tmp_path):
    path = tmp_path / "notes.exe"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedFileType):
        parse_file(path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n  ")
    with pytest.raises(EmptyDocument):
        parse_file(path)
