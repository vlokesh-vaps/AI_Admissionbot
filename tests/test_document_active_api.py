import sys
from pathlib import Path
import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.app import app, store, retriever
from fastapi.testclient import TestClient

client = TestClient(app)


def test_document_active_and_delete_workflow() -> None:
    # 1. Setup sample document chunks in the vector store under mi_id="30"
    filename = "Pre_Admission_Module_Knowledge_Base.pdf"
    mi_id = "30"
    store.delete_document(mi_id, filename)

    docs = [
        Document(
            page_content="The pre-admission fee for B.Tech is Rs. 50,000 payable at registration.",
            metadata={
                "source": filename,
                "mi_id": mi_id,
                "chunk_id": f"{mi_id}:{filename}:1:abc",
                "chunk_index": 1,
                "is_active": True,
                "active_flag": True,
            },
        ),
        Document(
            page_content="Another course M.Tech requires a pre-admission fee of Rs. 60,000.",
            metadata={
                "source": filename,
                "mi_id": mi_id,
                "chunk_id": f"{mi_id}:{filename}:2:def",
                "chunk_index": 2,
                "is_active": True,
                "active_flag": True,
            },
        ),
    ]

    indexed = store.upsert(docs, mi_id=mi_id)
    assert indexed == 2
    retriever.refresh()

    # Verify search initially retrieves the document
    res_active = retriever.search("admission fee", mi_id=mi_id)
    assert len(res_active) > 0
    assert any(filename in d.metadata.get("source", "") for d in res_active)

    # 2. Toggle ActiveFlag to False using ERP casing (MI_ID, FileName, ActiveFlag)
    response = client.post(
        "/api/ai/admission/document/active",
        json={
            "MI_ID": mi_id,
            "FileName": filename,
            "ActiveFlag": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["mi_id"] == mi_id
    assert data["filename"] == filename
    assert data["is_active"] is False
    assert data["updated_chunks"] == 2

    # Verify search now excludes the deactivated document
    res_inactive = retriever.search("admission fee", mi_id=mi_id)
    assert not any(filename in d.metadata.get("source", "") for d in res_inactive)

    # 3. Toggle ActiveFlag back to True using snake_case (mi_id, filename, is_active)
    response = client.post(
        "/api/ai/admission/document/active",
        json={
            "MI_ID": mi_id,
            "FileName": filename,
            "ActiveFlag": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_active"] is True
    assert data["updated_chunks"] == 2

    # Verify search retrieves the document again
    res_reactivated = retriever.search("admission fee", mi_id=mi_id)
    assert any(filename in d.metadata.get("source", "") for d in res_reactivated)

    # 4. Delete document via POST /api/ai/admission/document/delete
    del_response = client.post(
        "/api/ai/admission/document/delete",
        json={
            "MI_ID": mi_id,
            "FileName": filename,
        },
    )
    assert del_response.status_code == 200
    del_data = del_response.json()
    assert del_data["status"] == "success"
    assert del_data["deleted_chunks"] == 2

    # Verify search has 0 results from this file
    res_after_del = retriever.search("admission fee", mi_id=mi_id)
    assert not any(filename in d.metadata.get("source", "") for d in res_after_del)


def test_document_delete_all_for_tenant() -> None:
    filename = "Sample_Rules.pdf"
    mi_id = "30"

    doc = Document(
        page_content="Sample rules content.",
        metadata={
            "source": filename,
            "mi_id": mi_id,
            "chunk_id": f"{mi_id}:{filename}:1:xyz",
            "chunk_index": 1,
            "is_active": True,
        },
    )
    store.upsert([doc], mi_id=mi_id)

    response = client.post(
        "/api/ai/admission/document/delete",
        json={"MI_ID": mi_id, "FileName": "ALL"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["deleted_chunks"] >= 1


def test_auto_create_collection_when_storing_knowledge() -> None:
    # 1. Simulate collection not created by deleting it if it exists
    if store.qdrant_client.collection_exists(store.collection_name):
        store.qdrant_client.delete_collection(store.collection_name)

    assert not store.qdrant_client.collection_exists(store.collection_name)

    # 2. Upsert admission knowledge documents into the non-existent collection
    filename = "Pre_Admission_Module_Knowledge_Base.pdf"
    mi_id = "30"
    doc = Document(
        page_content="Admission is open for undergraduate and postgraduate courses.",
        metadata={
            "source": filename,
            "mi_id": mi_id,
            "chunk_id": f"{mi_id}:{filename}:1:auto",
            "chunk_index": 1,
            "is_active": True,
        },
    )

    indexed = store.upsert([doc], mi_id=mi_id)
    assert indexed == 1

    # 3. Verify collection was automatically created and chunks are stored
    assert store.qdrant_client.collection_exists(store.collection_name)
    assert store.count(mi_id=mi_id) == 1

    # 4. Verify retrieval succeeds
    retriever.refresh()
    results = retriever.search("undergraduate courses", mi_id=mi_id)
    assert len(results) > 0
    assert "undergraduate and postgraduate" in results[0].page_content
