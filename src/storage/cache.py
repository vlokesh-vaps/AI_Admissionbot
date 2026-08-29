"""JSON file-backed response cache for admission queries."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any

from src.utils.logging import log_event

logger = logging.getLogger(__name__)

_MAX_CACHE_ENTRIES = 10_000


class ResponseCache:
    """A lightweight filesystem JSON cache with TTL support and thread safety."""

    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600, namespace: str = "") -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
        filename = f"chat_cache_{namespace}.json" if namespace else "chat_cache.json"
        self.path = self.cache_dir / filename
        self._lock = Lock()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, conversation_id: str, question: str) -> str:
        return f"{conversation_id.strip()}::{question.strip().lower()}"

    def _load_data(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read cache file %s, initializing empty.", self.path)
            return {}

    def _save_data(self, data: dict[str, Any]) -> None:
        try:
            self._ensure_cache_dir()
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write to cache file %s: %s", self.path, exc)

    def _evict_oldest(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove oldest entries if cache exceeds size limit."""
        if len(data) <= _MAX_CACHE_ENTRIES:
            return data
        sorted_keys = sorted(data, key=lambda k: data[k].get("timestamp", 0))
        excess = len(data) - _MAX_CACHE_ENTRIES
        for key in sorted_keys[:excess]:
            data.pop(key, None)
        logger.info("Cache evicted %d oldest entries (limit=%d)", excess, _MAX_CACHE_ENTRIES)
        return data

    def get(self, conversation_id: str, question: str) -> dict[str, Any] | None:
        """Retrieve a cached answer if present and unexpired."""
        with self._lock:
            data = self._load_data()
            key = self._make_key(conversation_id, question)
            entry = data.get(key)
            if not entry:
                return None
            timestamp = entry.get("timestamp", 0)
            if time.time() - timestamp > self.ttl_seconds:
                # Expired
                data.pop(key, None)
                self._save_data(data)
                return None
            return entry.get("value")

    def set(self, conversation_id: str, question: str, value: dict[str, Any]) -> None:
        """Save a response in the cache with the current timestamp."""
        with self._lock:
            data = self._load_data()
            key = self._make_key(conversation_id, question)
            data[key] = {
                "timestamp": time.time(),
                "value": value,
            }
            data = self._evict_oldest(data)
            self._save_data(data)

    def clear(self) -> None:
        """Clear all entries in this cache."""
        with self._lock:
            if self.path.exists():
                self.path.unlink(missing_ok=True)
