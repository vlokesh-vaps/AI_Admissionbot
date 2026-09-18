"""Text chunking using LangChain text splitters with stable IDs and tenant metadata."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunk(Document):
    """A chunk of document text, fully compatible with LangChain Document."""

    def __init__(
        self,
        chunk_id: str = "",
        text: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = dict(metadata) if metadata else {}
        if chunk_id:
            meta.setdefault("chunk_id", chunk_id)
        super().__init__(page_content=text, metadata=meta, id=chunk_id, **kwargs)

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", self.id or "")

    @property
    def text(self) -> str:
        return self.page_content


def _stable_id(mi_id: str, source: str, index: int, text: str, aid_id: int | None = None) -> str:
    identity = aid_id if aid_id is not None else source
    digest = hashlib.sha1(f"{mi_id}:{identity}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{mi_id}:{source}:{index}:{digest}"


def chunk_documents(
    documents: Sequence[Document] | Document,
    mi_id: str | int = "default",
    max_chars: int = 1200,
    overlap_chars: int = 180,
    aid_id: int | None = None,
) -> list[DocumentChunk]:
    """Split documents using LangChain's RecursiveCharacterTextSplitter.

    Each chunk is tagged with tenant MI_ID and a stable chunk ID.
    """
    if isinstance(documents, Document):
        doc_list = [documents]
    else:
        doc_list = list(documents)

    mi_id_str = str(mi_id)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    splits = text_splitter.split_documents(doc_list)

    # Assign stable IDs and tenant metadata to each chunk
    result: list[DocumentChunk] = []
    source_counters: dict[str, int] = {}
    for chunk in splits:
        source = chunk.metadata.get("source", "unknown")
        index = source_counters.get(source, 0)
        source_counters[source] = index + 1

        chunk_id = _stable_id(mi_id_str, source, index, chunk.page_content, aid_id)
        meta = dict(chunk.metadata)
        meta.update({
            "mi_id": mi_id_str,
            "chunk_index": index,
            "chunk_id": chunk_id,
            "is_active": True,
            "active_flag": True,
        })
        if aid_id is not None:
            meta["aid_id"] = aid_id

        doc_chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=chunk.page_content,
            metadata=meta,
        )
        result.append(doc_chunk)

    return result


def chunk_document(
    document: Document,
    mi_id: str | int = "default",
    max_chars: int = 1200,
    overlap_chars: int = 180,
    aid_id: int | None = None,
) -> list[DocumentChunk]:
    """Split a single document using LangChain RecursiveCharacterTextSplitter."""
    return chunk_documents(
        [document],
        mi_id=mi_id,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        aid_id=aid_id,
    )
