import io

import pytest
from docx import Document

from rag.loaders import UnsupportedFileError, load_bytes


def test_txt():
    pages = load_bytes("note.txt", "مرحبا بالعالم".encode())
    assert pages[0].text == "مرحبا بالعالم"


def test_docx_paragraphs_and_tables():
    doc = Document()
    doc.add_paragraph("سياسة الإجازات")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "سنوية"
    table.rows[0].cells[1].text = "21 يوم"
    buf = io.BytesIO()
    doc.save(buf)
    text = load_bytes("p.docx", buf.getvalue())[0].text
    assert "سياسة الإجازات" in text and "سنوية | 21 يوم" in text


def test_unsupported():
    with pytest.raises(UnsupportedFileError):
        load_bytes("image.png", b"...")
