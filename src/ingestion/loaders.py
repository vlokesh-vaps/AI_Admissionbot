"""Load supported knowledge-base documents into normalized text records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re




SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


@dataclass(frozen=True)
class LoadedDocument:
    """A source document before chunking."""

    source_path: Path
    title: str
    text: str


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> LoadedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required to load PDF documents. Install with 'pip install pypdf'.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        content = _clean_text(page.extract_text() or "")
        if content:
            pages.append(f"[Page {page_number}]\n{content}")
    return LoadedDocument(path, path.stem, "\n\n".join(pages))


def load_docx(path: Path) -> LoadedDocument:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise ImportError("python-docx is required to load DOCX documents. Install with 'pip install python-docx'.") from exc

    document = DocxDocument(str(path))
    paragraphs = [_clean_text(p.text) for p in document.paragraphs if _clean_text(p.text)]
    table_rows: list[str] = []
    for table in document.tables:
        for row in table.rows:
            cells = [_clean_text(cell.text) for cell in row.cells]
            table_rows.append(" | ".join(cell for cell in cells if cell))
    text = "\n".join(paragraphs + table_rows)
    return LoadedDocument(path, path.stem, text)


def load_document(path: Path) -> LoadedDocument:
    """Load one supported file, rejecting unsupported or empty documents."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF or DOCX.")
    loaded = load_pdf(path) if suffix == ".pdf" else load_docx(path)
    if not loaded.text:
        raise ValueError(f"No extractable text found in {path.name}.")
    return loaded


def load_documents(directory: Path) -> list[LoadedDocument]:
    """Load all supported files in deterministic filename order."""
    documents: list[LoadedDocument] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(load_document(path))
    return documents
