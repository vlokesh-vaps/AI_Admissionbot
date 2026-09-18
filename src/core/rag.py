"""End-to-end retrieval-augmented generation service with multi-tenant isolation using LangChain."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

from langchain_groq import ChatGroq

from src.config import Settings
from src.core.conversation import ConversationStore
from src.core.prompts import ADMISSION_PROMPT, build_context
from src.core.reranker import Reranker
from src.core.retrievers import HybridRetriever
from src.utils.logging import log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: str
    sources: list[dict[str, Any]]
    escalate: bool
    escalation_reason: str | None
    mi_id: str = "default"


class AdmissionRAG:
    """Coordinate retrieval, reranking, conversation memory, and LLM generation."""

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
        self.llm = ChatGroq(
            api_key=settings.groq_api_key or "missing-groq-key",
            model=settings.groq_model,
            temperature=0.1,
            max_tokens=700,
            max_retries=2,
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

    def _call_llm(self, context: str, history: str, question: str) -> str:
        """Call Groq via LangChain ChatGroq with prompt template."""
        messages = ADMISSION_PROMPT.format_messages(
            context=context or "[No relevant knowledge-base passage was retrieved.]",
            history=history or "[No previous conversation.]",
            question=question,
        )
        response = self.llm.invoke(messages)
        return response.content or "I could not generate an answer."

    def chat(
        self,
        conversation_id: str,
        question: str,
        mi_id: str | int | None = None,
    ) -> ChatResult:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        target_mi_id = str(mi_id or self.settings.default_mi_id).strip()
        conversation = self.conversations.get_or_create(f"{target_mi_id}:{conversation_id}")

        # Retrieve candidates with multi-tenant isolation
        candidates = self.retriever.search(question, mi_id=target_mi_id)

        # Rerank using cross-encoder or lexical fallback
        ranked = self.reranker.rerank(question, candidates, self.settings.rerank_top_k)

        # Build context and call LLM via LangChain
        context = build_context(ranked, self.settings.max_context_chars)
        content = self._call_llm(context, conversation.history_text(), question)

        # Parse structured response
        answer, escalate, reason = self._parse_response(content)

        # Escalation logic
        explicit_reason = self._explicit_escalation_reason(question)
        if explicit_reason:
            escalate, reason = True, explicit_reason
        if not ranked and not reason:
            escalate, reason = True, "No supporting knowledge-base context was retrieved."

        # Update conversation memory
        conversation.add_message("user", question)
        conversation.add_message("assistant", answer)

        sources = [
            {
                "source": doc.metadata.get("source"),
                "title": doc.metadata.get("title"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "score": round(float(doc.metadata.get("rerank_score", doc.metadata.get("score", 0))), 4),
                "mi_id": target_mi_id,
            }
            for doc in ranked
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
