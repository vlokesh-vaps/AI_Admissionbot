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


def test_ui_root() -> None:
    if client is None:
        pytest.skip("python-multipart is not installed.")
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "").lower()
    assert len(response.text) > 0
