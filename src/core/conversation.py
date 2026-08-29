"""Conversation state and lightweight applicant memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock


@dataclass
class Conversation:
    conversation_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    lead_memory: dict[str, str] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-12:]
        self.updated_at = datetime.now(timezone.utc)

    def history_text(self) -> str:
        return "\n".join(
            f"{message['role'].title()}: {message['content']}" for message in self.messages[-6:]
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