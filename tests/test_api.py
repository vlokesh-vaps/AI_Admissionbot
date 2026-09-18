import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi.testclient import TestClient
    from src.api.app import app
    client = TestClient(app)
except RuntimeError:
    client = None


def test_health_endpoint() -> None:
    if client is None:
        pytest.skip("python-multipart is not installed.")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "indexed_chunks" in data


def test_knowledge_base_status_endpoint() -> None:
    if client is None:
        pytest.skip("python-multipart is not installed.")
    response = client.get("/api/knowledge-base/status")
    assert response.status_code == 200
    data = response.json()
    assert "document_count" in data
    assert "chunk_count" in data
    assert ".pdf" in data["supported_extensions"]


def test_escalation_endpoint() -> None:
    if client is None:
        pytest.skip("python-multipart is not installed.")
    payload = {
        "conversation_id": "test-conv",
        "applicant_name": "John Doe",
        "applicant_email": "john@example.com",
        "note": "Fee concession query",
    }
    response = client.post("/api/escalation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] is True
    assert data["conversation_id"] == "test-conv"


def test_chat_cache_hit_does_not_duplicate_cached_field() -> None:
    if client is None:
        pytest.skip("python-multipart is not installed.")

    cached_response = {
        "conversation_id": "session-cache",
        "answer": "Cached answer",
        "mi_id": "30",
        "sources": [],
        "escalate": False,
        "escalation_reason": None,
        "cached": False,
    }
    with patch("src.api.app.cache.get", return_value=cached_response), patch("src.api.app.erp.save_message") as save_message:
        response = client.post(
            "/api/chat",
            json={
                "conversation_id": "session-cache",
                "question": "Where can I find the online admission form?",
                "mi_id": "30",
            },
        )

    assert response.status_code == 200
    assert response.json()["cached"] is True
    assert response.json()["answer"] == "Cached answer"
    save_message.assert_called_once()
    stored_message = save_message.call_args.args[2]
    assert stored_message == [
        {"role": "user", "content": "Where can I find the online admission form?"},
        {"role": "assistant", "content": "Cached answer"},
    ]
