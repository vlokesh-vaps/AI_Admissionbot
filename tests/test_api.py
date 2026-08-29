import sys
from pathlib import Path
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
