"""Text chunking with stable IDs and source metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from .loaders import LoadedDocument


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


def _stable_id(mi_id: str, source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{mi_id}:{source}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{mi_id}:{source}:{index}:{digest}"


def chunk_document(
    document: LoadedDocument,
    mi_id: str | int = "default",
    max_chars: int = 1200,
    overlap_chars: int = 180,
) -> list[DocumentChunk]:
    """Split a document at paragraph/sentence boundaries while preserving overlap and tagging with tenant MI_ID."""
    mi_id_str = str(mi_id)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document.text) if p.strip()]
    chunks: list[DocumentChunk] = []
    current = ""
    index = 0

    def add_chunk(value: str) -> None:
        nonlocal index
        value = value.strip()
        if not value:
            return
        chunks.append(
            DocumentChunk(
                chunk_id=_stable_id(mi_id_str, document.source_path.name, index, value),
                text=value,
                metadata={
                    "mi_id": mi_id_str,
                    "source": document.source_path.name,
                    "title": document.title,
                    "chunk_index": index,
                },
            )
        )
        index += 1

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        add_chunk(current)
        overlap = current[-overlap_chars:] if overlap_chars else ""
        current = f"{overlap}\n\n{paragraph}".strip()
        while len(current) > max_chars:
            split_at = current.rfind(" ", 0, max_chars)
            split_at = split_at if split_at > max_chars // 2 else max_chars
            add_chunk(current[:split_at])
            current = current[max(0, split_at - overlap_chars) :].strip()

    add_chunk(current)
    return chunks
