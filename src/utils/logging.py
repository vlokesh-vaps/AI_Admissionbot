"""Structured logging configuration for admission chatbot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any
from contextvars import ContextVar
from datetime import datetime, timezone


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(value: str | None) -> object:
    """Bind a request ID to the current async context."""
    return _request_id.set(value)


def reset_request_id(token: object) -> None:
    _request_id.reset(token)  # type: ignore[arg-type]


class ReadableFormatter(logging.Formatter):
    """Format structured records as compact, human-readable single lines."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        event = str(getattr(record, "event", record.getMessage())).upper()
        fields: dict[str, Any] = {}
        request_id = _request_id.get()
        if request_id:
            fields["request_id"] = request_id
        if hasattr(record, "structured"):
            fields.update(record.structured)

        values: list[str] = []
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=True, default=str)
            if isinstance(value, bool):
                rendered = str(value).lower()
            elif isinstance(value, (int, float)):
                rendered = str(value)
            else:
                rendered = f'"{str(value).replace(chr(34), chr(92) + chr(34))}"'
            values.append(f"{key}={rendered}")

        line = f"{timestamp} | {record.levelname} | {event}"
        if values:
            line += " | " + " ".join(values)
        if record.exc_info:
            exception_text = str(record.exc_info[1]).replace("\r", " ").replace("\n", " | ")
            if len(exception_text) > 1000:
                exception_text = exception_text[:1000] + "...[truncated]"
            exception_text = exception_text.replace(chr(34), chr(92) + chr(34))
            line += f" | exception_type=\"{record.exc_info[0].__name__}\" exception=\"{exception_text}\""
        return line


def configure_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure console and readable file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "chat.log"

    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on reloads
    root_logger.handlers.clear()

    formatter = ReadableFormatter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Suppress verbose third-party loggers
    for noisy in ("urllib3", "httpx", "httpx2", "httpcore", "qdrant_client", "openai", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


def log_event(logger: logging.Logger, level: int, event_name: str, *, exc_info: Any = None, **kwargs: Any) -> None:
    """Emit a consistently shaped structured event log."""
    extra = {"event": event_name, "structured": kwargs}
    logger.log(level, event_name, extra=extra, exc_info=exc_info)
