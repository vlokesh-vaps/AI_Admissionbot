"""PostgreSQL repository for saving chat conversations."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.config import Settings
from src.utils.logging import log_event


logger = logging.getLogger(__name__)


class PostgresRepository:
    """Provides conversation persistence to AI_Admission_Conversation without unwanted document workflows."""

    def __init__(self, settings: Settings) -> None:
        self.database_url = settings.database_url
        self.created_by = settings.database_created_by

    def _connect(self) -> Any:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not configured.")
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("psycopg is required when DATABASE_URL is configured.") from exc
        return psycopg.connect(self.database_url)

    def save_message(self, mi_id: int | str, session_id: str, message: Any) -> None:
        """Persist a chat turn (user query and assistant reply) to AI_Admission_Conversation."""
        started = time.perf_counter()
        query = '''
            INSERT INTO "AI_Admission_Conversation"
                ("AIC_Id", "MI_ID", "Session_id", "Message", "CreatedBy", "CreatedDate", "ActiveFlag")
            VALUES (nextval('public."AI_Conversation_AIC_Id_seq"'), %s, %s, %s, %s, CURRENT_TIMESTAMP, true)
        '''
        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise RuntimeError("psycopg is required for PostgreSQL conversation storage.") from exc

        # Ensure mi_id is integer for Postgres bigint column
        try:
            mi_id_int = int(str(mi_id).strip())
        except (ValueError, TypeError):
            mi_id_int = 0

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (mi_id_int, str(session_id), Jsonb(message), self.created_by),
                )
            connection.commit()

        log_event(
            logger,
            logging.INFO,
            "conversation_saved",
            operation="insert",
            table="AI_Admission_Conversation",
            mi_id=mi_id_int,
            session_id=session_id,
            message_roles=[item.get("role") for item in message if isinstance(item, dict)] if isinstance(message, list) else [message.get("role")],
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
