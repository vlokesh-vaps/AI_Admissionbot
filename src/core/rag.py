"""End-to-end retrieval-augmented generation service with multi-tenant isolation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from typing import Any

from openai import OpenAI

from src.config import Settings
from src.core.conversation import ConversationStore
from src.core.prompts import SYSTEM_PROMPT, build_context, build_user_prompt
from src.core.reranker import Reranker
from src.core.retrievers import HybridRetriever
from src.utils.logging import log_event

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 1.0


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: str
    sources: list[dict[str, Any]]
    escalate: bool
    escalation_reason: str | None
    mi_id: str = "default"


class AdmissionRAG:
    """Coordinate retrieval, reranking, conversation memory, and generation."""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: Reranker,
        conversations: ConversationStore,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.conversations = conversations
        self.client = OpenAI(
            api_key=settings.groq_api_key or "missing-groq-key",
            base_url="https://api.groq.com/openai/v1",
            timeout=settings.groq_timeout_seconds,
        )

    @staticmethod
    def _parse_response(content: str) -> tuple[str, bool, str | None]:
        answer_match = re.search(r"ANSWER:\s*(.*?)(?=\nESCALATE:|$)", content, re.S | re.I)
        escalate_match = re.search(r"ESCALATE:\s*(true|false)", content, re.I)
        reason_match = re.search(r"REASON:\s*(.*)$", content, re.S | re.I)
        answer = answer_match.group(1).strip() if answer_match else content.strip()
        escalate = bool(escalate_match and escalate_match.group(1).lower() == "true")
        reason = reason_match.group(1).strip() if reason_match else None
        return answer, escalate, reason if reason and reason.lower() != "none" else None

    @staticmethod
    def _explicit_escalation_reason(question: str) -> str | None:
        normalized = question.lower()
        phrases = (
            "speak to a human",
            "talk to a counselor",
            "contact an agent",
            "admission counselor",
            "human agent",
        )
        if any(phrase in normalized for phrase in phrases):
            return "Applicant explicitly requested an admissions counselor."
        return None

    def _call_groq(self, prompt: str) -> str:
        """Call Groq with timeout and retry for transient errors."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.groq_model,
                    temperature=0.1,
                    max_tokens=700,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                if not response.choices:
                    raise RuntimeError("Groq returned an empty response (no choices).")
                return response.choices[0].message.content or "I could not generate an answer."
            except Exception as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in _TRANSIENT_STATUS_CODES and attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    log_event(logger, logging.WARNING, "llm_retry", operation="generate", status=status, attempt=attempt + 1, max_retries=_MAX_RETRIES, delay_ms=round(delay * 1000))
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError("Groq call failed after retries") from last_exc  # pragma: no cover

    def chat( self, conversation_id: str,
        question: str,
        mi_id: str | int | None = None, ) -> ChatResult:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        target_mi_id = str(mi_id or self.settings.default_mi_id).strip()
        conversation = self.conversations.get_or_create(f"{target_mi_id}:{conversation_id}")
        candidates = self.retriever.search(question, mi_id=target_mi_id)
        ranked = self.reranker.rerank(question, candidates, self.settings.rerank_top_k)
        context = build_context(ranked, self.settings.max_context_chars)
        prompt = build_user_prompt(question, context, conversation.history_text())
        content = self._call_groq(prompt)
        answer, escalate, reason = self._parse_response(content)
        explicit_reason = self._explicit_escalation_reason(question)
        if explicit_reason:
            escalate, reason = True, explicit_reason
        if not ranked and not reason:
            escalate, reason = True, "No supporting knowledge-base context was retrieved."
        conversation.add_message("user", question)
        conversation.add_message("assistant", answer)

        sources = [
            {
                "source": item.get("metadata", {}).get("source"),
                "title": item.get("metadata", {}).get("title"),
                "chunk_index": item.get("metadata", {}).get("chunk_index"),
                "score": round(float(item.get("rerank_score", item.get("score", 0))), 4),
                "mi_id": target_mi_id,
            }
            for item in ranked
        ]
        log_event(
            logger,
            logging.INFO,
            "rag_completed",
            operation="retrieve_generate",
            conversation_id=conversation_id,
            mi_id=target_mi_id,
            retrieved_candidates=len(candidates),
            returned_sources=len(sources),
            escalate=escalate,
        )
        return ChatResult(answer, conversation_id, sources, escalate, reason, mi_id=target_mi_id)
