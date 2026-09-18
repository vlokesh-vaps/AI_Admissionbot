"""Candidate reranking with an optional local cross-encoder."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.documents import Document

from src.config import Settings
from src.utils.logging import log_event

logger = logging.getLogger(__name__)


class Reranker:
    """Use a cross-encoder when installed; otherwise use lexical overlap."""

    def __init__(self, settings: Settings) -> None:
        self.model = None
        self._mode = "lexical"
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(settings.reranker_model)
            self._mode = "cross-encoder"
        except Exception as exc:  # pragma: no cover - depends on local model availability
            log_event(logger, logging.WARNING, "reranker_fallback", operation="initialize", mode="lexical", error_type=type(exc).__name__)
        log_event(logger, logging.INFO, "reranker_initialized", operation="initialize", mode=self._mode)

    @property
    def mode(self) -> str:
        """Return the active reranking strategy ('cross-encoder' or 'lexical')."""
        return self._mode

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        text_terms = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        return len(query_terms & text_terms) / max(len(query_terms), 1)

    def rerank(self, query: str, candidates: list[Any], limit: int) -> list[Any]:
        """Rerank candidates (Document or dict) and return the top results."""
        if not candidates:
            return []

        first = candidates[0]
        is_doc = isinstance(first, Document)

        pairs = []
        for item in candidates:
            if isinstance(item, Document):
                text = item.page_content
            elif isinstance(item, dict):
                text = item.get("text") or item.get("page_content", "")
            else:
                text = str(item)
            pairs.append((query, text))

        if self.model is not None:
            scores = self.model.predict(pairs)
        else:
            scores = [self._lexical_score(query, text) for _, text in pairs]

        for item, score in zip(candidates, scores):
            s = float(score)
            if isinstance(item, Document):
                item.metadata["rerank_score"] = s
            elif isinstance(item, dict):
                item["rerank_score"] = s

        def _score(item: Any) -> float:
            if isinstance(item, Document):
                return float(item.metadata.get("rerank_score", 0))
            if isinstance(item, dict):
                return float(item.get("rerank_score", 0))
            return 0.0

        return sorted(candidates, key=_score, reverse=True)[:limit]
