"""Candidate reranking with an optional local cross-encoder."""

from __future__ import annotations

import logging
import re
from typing import Any

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

    def rerank(self, query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self.model is not None:
            pairs = [(query, candidate["text"]) for candidate in candidates]
            scores = self.model.predict(pairs)
            ranked = [
                {**candidate, "rerank_score": float(score)}
                for candidate, score in zip(candidates, scores)
            ]
        else:
            ranked = [
                {**candidate, "rerank_score": self._lexical_score(query, candidate["text"])}
                for candidate in candidates
            ]
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:limit]
