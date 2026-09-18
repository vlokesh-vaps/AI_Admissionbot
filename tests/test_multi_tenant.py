"""Tests for multi-tenant isolation with MI_ID filtering and Qdrant storage."""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.core.retrievers import HybridRetriever
from src.ingestion.chunking import chunk_documents
from src.storage.vector_db import VectorStore


class MockEmbeddings(Embeddings):
    """Mock LangChain Embeddings interface for deterministic testing."""

    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            val = float(sum(ord(c) for c in text) % 100) / 100.0
            results.append([val, 1.0 - val, 0.5, 0.25])
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@pytest.fixture
def mock_store(monkeypatch):
    """Create an isolated in-memory Qdrant VectorStore for testing."""
    in_memory_qdrant = QdrantClient(":memory:")
    monkeypatch.setattr("src.storage.vector_db.OllamaEmbeddings", MockEmbeddings)
    store = VectorStore(settings, client=in_memory_qdrant)
    return store


def _make_documents(chunk_id: str, text: str, mi_id: str, source: str, title: str) -> list[Document]:
    """Helper to create a list containing one Document for upsert."""
    return [Document(
        page_content=text,
        metadata={
            "mi_id": mi_id,
            "source": source,
            "title": title,
            "chunk_index": 0,
            "chunk_id": chunk_id,
        },
        id=chunk_id,
    )]


def test_chunking_multi_tenant_metadata():
    """Verify chunk_documents tags all chunks with MI_ID and distinct IDs."""
    doc = Document(
        page_content="Admission criteria for 2026.\n\nFee structure.",
        metadata={"source": "prospectus.pdf", "title": "prospectus"},
    )
    chunks_1001 = chunk_documents([doc], mi_id="1001")
    chunks_1002 = chunk_documents([doc], mi_id="1002")

    assert len(chunks_1001) > 0
    assert chunks_1001[0].metadata["mi_id"] == "1001"
    assert chunks_1002[0].metadata["mi_id"] == "1002"
    assert chunks_1001[0].metadata["chunk_id"] != chunks_1002[0].metadata["chunk_id"]
    assert chunks_1001[0].metadata["chunk_id"].startswith("1001:prospectus.pdf:")
    assert chunks_1002[0].metadata["chunk_id"].startswith("1002:prospectus.pdf:")


def test_qdrant_strict_multi_tenant_search_isolation(mock_store):
    """Verify that tenant 1001 can NEVER retrieve tenant 1002's documents."""
    docs_t1 = _make_documents(
        "1001:doc1:0:abc",
        "Tenant 1001 exclusive scholarship token: ALPHA_1001",
        "1001", "doc1.pdf", "Doc1",
    )
    docs_t2 = _make_documents(
        "1002:doc2:0:xyz",
        "Tenant 1002 exclusive scholarship token: BETA_1002",
        "1002", "doc2.pdf", "Doc2",
    )

    mock_store.upsert(docs_t1 + docs_t2)

    # Search as Tenant 1001
    results_t1 = mock_store.search("scholarship token", mi_id="1001", limit=10)
    assert len(results_t1) == 1
    assert results_t1[0]["metadata"]["mi_id"] == "1001"
    assert "ALPHA_1001" in results_t1[0]["text"]
    assert "BETA_1002" not in results_t1[0]["text"]

    # Search as Tenant 1002
    results_t2 = mock_store.search("scholarship token", mi_id="1002", limit=10)
    assert len(results_t2) == 1
    assert results_t2[0]["metadata"]["mi_id"] == "1002"
    assert "BETA_1002" in results_t2[0]["text"]
    assert "ALPHA_1001" not in results_t2[0]["text"]

    # Search with missing mi_id must raise ValueError
    with pytest.raises(ValueError, match="mi_id"):
        mock_store.search("scholarship token", mi_id="", limit=10)


def test_hybrid_retriever_multi_tenant_isolation(mock_store):
    """Verify HybridRetriever enforces MI_ID across semantic and BM25 search."""
    docs_t1 = _make_documents(
        "1001:doc1:0:abc",
        "Computer Science BTech syllabus at Cambridge Campus.",
        "1001", "cs.pdf", "CS",
    )
    docs_t2 = _make_documents(
        "1002:doc2:0:xyz",
        "Computer Science BTech syllabus at Oxford Campus.",
        "1002", "cs.pdf", "CS",
    )

    mock_store.upsert(docs_t1 + docs_t2)

    retriever = HybridRetriever(mock_store, settings)

    # Query for 1001
    results_1001 = retriever.search("Computer Science BTech syllabus", mi_id="1001")
    assert len(results_1001) > 0
    for doc in results_1001:
        assert doc.metadata.get("mi_id") == "1001"
        assert "Oxford" not in doc.page_content

    # Query for 1002
    results_1002 = retriever.search("Computer Science BTech syllabus", mi_id="1002")
    assert len(results_1002) > 0
    for doc in results_1002:
        assert doc.metadata.get("mi_id") == "1002"
        assert "Cambridge" not in doc.page_content


def test_api_upload_admission_document_endpoint():
    """Test POST /api/ai/admission/document endpoint from .NET ERP."""
    try:
        from fastapi.testclient import TestClient
        from src.api.app import app
        client = TestClient(app)
    except RuntimeError:
        pytest.skip("python-multipart is not yet installed in runtime environment.")

    fake_pdf_content = b"%PDF-1.4 mock pdf content"

    mock_docs = [Document(
        page_content="Sample admission rules.",
        metadata={"source": "student_doc.pdf", "title": "student_doc"},
    )]

    with patch("src.api.app.load_document", return_value=mock_docs) as mock_load, patch("src.api.app.store") as mock_store:
        mock_store.upsert.return_value = 2

        response = client.post(
            "/api/ai/admission/document",
            data={"MI_ID": "1001"},
            files={"File": ("student_doc.pdf", fake_pdf_content, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["mi_id"] == "1001"
        assert data["filename"] == "student_doc.pdf"
        assert data["indexed_chunks"] == 2
        assert "1001" in data["stored_path"]


def test_api_upload_admission_document_validations():
    """Verify validation errors for missing MI_ID and unsupported formats."""
    try:
        from fastapi.testclient import TestClient
        from src.api.app import app
        client = TestClient(app)
    except RuntimeError:
        pytest.skip("python-multipart is not yet installed in runtime environment.")

    fake_content = b"sample content"

    # Missing MI_ID
    res_missing_id = client.post(
        "/api/ai/admission/document",
        data={"MI_ID": "  "},
        files={"File": ("student_doc.pdf", fake_content, "application/pdf")},
    )
    assert res_missing_id.status_code == 422

    # Unsupported format (.txt)
    res_bad_format = client.post(
        "/api/ai/admission/document",
        data={"MI_ID": "1001"},
        files={"File": ("student_doc.txt", fake_content, "text/plain")},
    )
    assert res_bad_format.status_code == 415
    assert "Only PDF and DOCX" in res_bad_format.json()["detail"]
