"""Hybrid keyword and semantic retrieval with mandatory multi-tenant MI_ID isolation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(self, corpus: list[list[str]]) -> None:
            self.corpus = corpus

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            scores: list[float] = []
            q_set = set(query_tokens)
            for doc in self.corpus:
                if not doc:
                    scores.append(0.0)
                    continue
                match_count = sum(1 for token in doc if token in q_set)
                scores.append(float(match_count) / len(doc))
            return scores

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
    """Retrieve candidates from semantic (Qdrant) and BM25 indexes with strict tenant isolation."""

    def __init__(self, store: VectorStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self._chunks: list[dict[str, Any]] = []
        self.refresh()

    def refresh(self) -> None:
        """Reload all chunks from the vector store."""
        self._chunks = self.store.all_chunks()

    def search(
        self,
        query: str,
        mi_id: str | int,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search strictly within the specified tenant MI_ID."""
        if mi_id is None or not str(mi_id).strip():
            raise ValueError("Mandatory multi-tenant parameter 'mi_id' is required for retrieval.")

        mi_id_str = str(mi_id).strip()
        limit = limit or self.settings.retrieval_top_k

        # 1. Semantic search with mandatory MI_ID filter in Qdrant
        semantic = self.store.search(query=query, mi_id=mi_id_str, limit=limit * 2)
        semantic_by_id = {item["id"]: item for item in semantic}
        if semantic:
            semantic_scores = _normalize([float(item["score"]) for item in semantic])
            for item, score in zip(semantic, semantic_scores):
                semantic_by_id[item["id"]] = {**item, "score": score}

        # 2. Filter local chunks to current tenant for BM25
        tenant_chunks = [
            chunk for chunk in self._chunks
            if str(chunk.get("mi_id") or chunk.get("metadata", {}).get("mi_id", "")).strip() == mi_id_str
        ]

        # If store has new chunks not yet in cache, include semantic hits in tenant_chunks
        known_ids = {c["id"] for c in tenant_chunks}
        for hit in semantic:
            if hit["id"] not in known_ids:
                tenant_chunks.append(hit)
                known_ids.add(hit["id"])

        if not tenant_chunks:
            return semantic[:limit]

        # 3. BM25 keyword scoring on tenant documents
        tokenized_corpus = [_tokens(chunk["text"]) for chunk in tenant_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = _tokens(query)
        keyword_scores_raw = bm25.get_scores(query_tokens)
        keyword_scores = _normalize(keyword_scores_raw) if any(keyword_scores_raw) else [0.0] * len(tenant_chunks)

        # 4. Fuse scores
        candidates: dict[str, dict[str, Any]] = {}
        for idx, chunk in enumerate(tenant_chunks):
            chunk_id = chunk["id"]
            semantic_score = float(semantic_by_id.get(chunk_id, {}).get("score", 0.0))
            keyword_score = float(keyword_scores[idx]) if idx < len(keyword_scores) else 0.0
            combined = (
                self.settings.hybrid_vector_weight * semantic_score
                + self.settings.hybrid_bm25_weight * keyword_score
            )
            if combined > 0 or semantic_score > 0:
                candidates[chunk_id] = {
                    **chunk,
                    "score": combined,
                    "semantic_score": semantic_score,
                    "bm25_score": keyword_score,
                    "retriever": "hybrid",
                    "mi_id": mi_id_str,
                }

        # Fallback to semantic hits if no hybrid candidates met threshold
        if not candidates and semantic:
            return semantic[:limit]

        return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)[:limit]
