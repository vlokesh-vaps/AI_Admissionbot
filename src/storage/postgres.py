"""Small PostgreSQL repository for ERP documents and chat messages."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.config import Settings
from src.utils.logging import log_event


logger = logging.getLogger(__name__)


class PostgresRepository:
    """Access the two ERP-owned tables without opening a connection at import time."""

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

    def get_document(self, aid_id: int, mi_id: int) -> dict[str, Any] | None:
        started = time.perf_counter()
        query = '''
            SELECT "AID_Id", "MI_ID", "FileName", "FilePath", "ActiveFlag"
            FROM "AI_Admission_Document"
            WHERE "AID_Id" = %s AND "MI_ID" = %s AND "ActiveFlag" = true
        '''
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (aid_id, mi_id))
                row = cursor.fetchone()
                log_event(
                    logger,
                    logging.INFO,
                    "erp_document_lookup",
                    operation="select",
                    table="AI_Admission_Document",
                    aid_id=aid_id,
                    mi_id=mi_id,
                    found=row is not None,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                if row is None:
                    return None
                columns = ["AID_Id", "MI_ID", "FileName", "FilePath", "ActiveFlag"]
                return dict(zip(columns, row))

    def save_message(self, mi_id: int, session_id: str, message: Any) -> None:
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

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (mi_id, session_id, Jsonb(message), self.created_by),
                )
            connection.commit()
        log_event(
            logger,
            logging.INFO,
            "conversation_saved",
            operation="insert",
            table="AI_Admission_Conversation",
            mi_id=mi_id,
            session_id=session_id,
            message_roles=[item.get("role") for item in message if isinstance(item, dict)] if isinstance(message, list) else [message.get("role")],
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
