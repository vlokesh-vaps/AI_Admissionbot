"""Hybrid keyword and semantic retrieval using LangChain EnsembleRetriever with MI_ID isolation."""

from __future__ import annotations

import re

from collections.abc import Sequence
try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    from langchain_classic.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import models

from src.config import Settings
from src.storage.vector_db import VectorStore


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _normalize(values: Sequence[float]) -> list[float]:
    if len(values) == 0:
        return []
    val_list = [float(v) for v in values]
    low, high = min(val_list), max(val_list)
    if high == low:
        return [1.0 for _ in val_list]
    return [(v - low) / (high - low) for v in val_list]


class HybridRetriever:
    """Retrieve candidates using LangChain's EnsembleRetriever (Qdrant + BM25) with strict tenant isolation."""

    def __init__(self, store: VectorStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self._docs: list[Document] = []
        self.refresh()

    def refresh(self) -> None:
        """Reload all chunks from the vector store."""
        self._docs = self.store.all_chunks()

    def search(
        self,
        query: str,
        mi_id: str | int,
        limit: int | None = None,
    ) -> list[Document]:
        """Search strictly within the specified tenant MI_ID using EnsembleRetriever."""
        if mi_id is None or not str(mi_id).strip():
            raise ValueError("Mandatory multi-tenant parameter 'mi_id' is required for retrieval.")

        mi_id_str = str(mi_id).strip()
        limit = limit or self.settings.retrieval_top_k

        # 1. Create Qdrant retriever with mandatory MI_ID filter
        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.mi_id",
                    match=models.MatchValue(value=mi_id_str),
                )
            ]
        )
        qdrant_retriever = self.store.langchain_store.as_retriever(
            search_kwargs={"k": limit, "filter": tenant_filter}
        )

        # 2. Filter local document cache to current tenant for BM25
        tenant_docs = [
            doc for doc in self._docs
            if doc.metadata.get("mi_id", "").strip() == mi_id_str
        ]

        # If no tenant documents cached, fall back to Qdrant-only
        if not tenant_docs:
            results = qdrant_retriever.invoke(query)
            for i, doc in enumerate(results):
                doc.metadata.setdefault("score", 1.0 - (i / max(len(results), 1)))
                doc.metadata["retriever"] = "semantic"
                doc.metadata.setdefault("mi_id", mi_id_str)
            return results[:limit]

        # 3. Create BM25 retriever from tenant documents
        bm25_retriever = BM25Retriever.from_documents(
            tenant_docs, k=limit, preprocess_func=_tokens
        )

        # 4. Ensemble with configurable weights
        ensemble = EnsembleRetriever(
            retrievers=[qdrant_retriever, bm25_retriever],
            weights=[self.settings.hybrid_vector_weight, self.settings.hybrid_bm25_weight],
        )

        results = ensemble.invoke(query)

        # Assign rank-based scores and tag with retriever info
        for i, doc in enumerate(results):
            doc.metadata.setdefault("score", 1.0 - (i / max(len(results), 1)))
            doc.metadata["retriever"] = "hybrid"
            doc.metadata.setdefault("mi_id", mi_id_str)

        return results[:limit]
