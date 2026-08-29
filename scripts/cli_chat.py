"""Interactive CLI for the admission chatbot."""

from __future__ import annotations

import sys
from uuid import uuid4

from src.config import settings
from src.core.conversation import ConversationStore
from src.core.rag import AdmissionRAG
from src.core.reranker import Reranker
from src.core.retrievers import HybridRetriever
from src.storage.vector_db import VectorStore
from src.utils.logging import configure_logging


def main() -> None:
    configure_logging(settings.log_dir, settings.log_level)
    print("=" * 60)
    print("Admission AI Chatbot CLI")
    print("Type your questions below. Type 'exit' or 'quit' to quit.")
    print("=" * 60)

    store = VectorStore(settings)
    retriever = HybridRetriever(store, settings)
    reranker = Reranker(settings)
    conversations = ConversationStore()
    rag = AdmissionRAG(settings, retriever, reranker, conversations)
    conversation_id = f"cli-{uuid4().hex[:8]}"

    while True:
        try:
            question = input("\nApplicant: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            result = rag.chat(conversation_id, question)
            print(f"\nAssistant:\n{result.answer}")
            if result.sources:
                print("\nSources:")
                for s in result.sources:
                    print(f" - [{s.get('source')}] (Score: {s.get('score')})")
            if result.escalate:
                print(f"\n[Escalation Note]: {result.escalation_reason or 'Escalation recommended'}")
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
