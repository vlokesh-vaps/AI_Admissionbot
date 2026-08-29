"""Structured logging configuration for admission chatbot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "structured"):
            log_obj.update(record.structured)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def configure_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure console and JSON-lines file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "chat.log"

    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on reloads
    root_logger.handlers.clear()

    formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Suppress verbose third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "qdrant_client", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


def log_event(logger: logging.Logger, level: int, event_name: str, **kwargs: Any) -> None:
    """Emit a structured event log."""
    extra = {"structured": {"event": event_name, **kwargs}}
    logger.log(level, event_name, extra=extra)
