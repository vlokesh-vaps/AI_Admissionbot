"""Conversation state using LangChain ChatMessageHistory."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


@dataclass
class Conversation:
    """Wrapper around LangChain ChatMessageHistory with windowing."""

    conversation_id: str
    history: ChatMessageHistory = field(default_factory=ChatMessageHistory)
    lead_memory: dict[str, str] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        if role == "user":
            self.history.add_user_message(content)
        else:
            self.history.add_ai_message(content)
        # Keep only the last 12 messages
        if len(self.history.messages) > 12:
            self.history.messages = self.history.messages[-12:]

    def history_text(self) -> str:
        return "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in self.history.messages[-6:]
        )


class ConversationStore:
    """Thread-safe process-local store; replace with Redis/database for multi-worker use."""

    def __init__(self) -> None:
        self._items: dict[str, Conversation] = {}
        self._lock = Lock()

    def get_or_create(self, conversation_id: str) -> Conversation:
        with self._lock:
            return self._items.setdefault(conversation_id, Conversation(conversation_id))

    def update_lead(self, conversation_id: str, **fields: str) -> None:
        conversation = self.get_or_create(conversation_id)
        conversation.lead_memory.update({key: value for key, value in fields.items() if value})

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._items.pop(conversation_id, None)