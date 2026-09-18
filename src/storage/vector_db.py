"""Persistent vector-store backed by LangChain QdrantVectorStore with OllamaEmbeddings.

Every search operation strictly requires MI_ID to ensure complete multi-tenant
data isolation across institutions.
"""

from __future__ import annotations

import logging
from typing import Any
import uuid

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src.config import Settings

logger = logging.getLogger(__name__)


def _point_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID string from a chunk identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class VectorStore:
    """Multi-tenant Vector Store using LangChain's QdrantVectorStore and OllamaEmbeddings."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection

        # LangChain embeddings via Ollama
        self.embeddings = OllamaEmbeddings(
            base_url=settings.ollama_host,
            model=settings.ollama_embed_model,
        )

        # Direct Qdrant client for admin operations (reset, count, delete, scroll)
        if client is not None:
            self.qdrant_client = client
        elif settings.qdrant_url:
            try:
                self.qdrant_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=5.0,
                )
                self.qdrant_client.get_collections()
                logger.info("Vector store: Connected to Qdrant Server (%s)", settings.qdrant_url)
            except Exception as exc:
                logger.warning(
                    "Could not connect to Qdrant Server at %s (%s). Using in-memory Qdrant.",
                    settings.qdrant_url,
                    exc,
                )
                self.qdrant_client = QdrantClient(":memory:")
        else:
            self.qdrant_client = QdrantClient(":memory:")

        self._ensure_collection()

        # LangChain vector store for add/search operations
        self.langchain_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            validate_collection_config=False,
            validate_embeddings=False,
        )

    def _ensure_collection(self) -> None:
        """Ensure collection exists and payload index for mi_id is configured."""
        try:
            test_embedding = self.embeddings.embed_query("dimension probe")
            dimension = len(test_embedding)
        except Exception as exc:
            logger.warning("Could not probe embedding dimension (%s); defaulting to 768", exc)
            dimension = 768

        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, distance=COSINE)",
                self.collection_name,
                dimension,
            )

        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.mi_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Index might already exist or not supported on in-memory/embedded
            pass

    def reset(self) -> None:
        """Clear and recreate the Qdrant collection."""
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)
        self._ensure_collection()
        self.langchain_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            validate_collection_config=False,
            validate_embeddings=False,
        )

    def upsert(
        self,
        documents: list[Document],
        mi_id: str | int | None = None,
        batch_size: int = 64,
    ) -> int:
        """Embed and upsert documents tagged with tenant MI_ID."""
        if not documents:
            return 0

        default_mi_id = str(mi_id or self.settings.default_mi_id)
        total = 0

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]

            # Ensure all docs have mi_id and generate stable point IDs
            ids: list[str] = []
            for doc in batch:
                if "mi_id" not in doc.metadata:
                    doc.metadata["mi_id"] = default_mi_id
                chunk_id = doc.metadata.get("chunk_id", doc.id or str(uuid.uuid4()))
                ids.append(_point_uuid(chunk_id))

            self.langchain_store.add_documents(batch, ids=ids)
            total += len(batch)

        return total

    def search(
        self,
        query: str,
        mi_id: str | int,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search vector store enforcing mandatory tenant MI_ID filter.

        Never allows a tenant to retrieve another tenant's documents.
        """
        if mi_id is None or not str(mi_id).strip():
            raise ValueError(
                "Mandatory multi-tenant parameter 'mi_id' is required for search."
            )

        mi_id_str = str(mi_id).strip()
        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.mi_id",
                    match=models.MatchValue(value=mi_id_str),
                )
            ]
        )

        results = self.langchain_store.similarity_search_with_score(
            query, k=limit, filter=tenant_filter
        )

        output: list[dict[str, Any]] = []
        for doc, score in results:
            output.append(
                {
                    "id": doc.metadata.get("chunk_id", doc.id or ""),
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "retriever": "semantic",
                    "mi_id": doc.metadata.get("mi_id", mi_id_str),
                }
            )
        return output

    def all_chunks(self, mi_id: str | int | None = None) -> list[Document]:
        """Retrieve stored documents, optionally filtered by tenant MI_ID."""
        scroll_filter = None
        if mi_id is not None and str(mi_id).strip():
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.mi_id",
                        match=models.MatchValue(value=str(mi_id).strip()),
                    )
                ]
            )

        documents: list[Document] = []
        offset: Any = None
        while True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for record in records:
                payload = record.payload or {}
                metadata = payload.get("metadata", {})
                documents.append(
                    Document(
                        page_content=payload.get("page_content", ""),
                        metadata=metadata,
                        id=metadata.get("chunk_id", str(record.id)),
                    )
                )
            if next_offset is None:
                break
            offset = next_offset

        return documents

    def count(self, mi_id: str | int | None = None) -> int:
        """Count points in Qdrant collection, optionally filtered by MI_ID."""
        count_filter = None
        if mi_id is not None and str(mi_id).strip():
            count_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.mi_id",
                        match=models.MatchValue(value=str(mi_id).strip()),
                    )
                ]
            )
        result = self.qdrant_client.count(
            collection_name=self.collection_name,
            count_filter=count_filter,
        )
        return result.count

    def delete_tenant_documents(self, mi_id: str | int) -> None:
        """Delete all points belonging to a specific tenant MI_ID."""
        mi_id_str = str(mi_id).strip()
        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.mi_id",
                    match=models.MatchValue(value=mi_id_str),
                )
            ]
        )
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=tenant_filter,
        )

    def delete_document(self, mi_id: str | int, filename: str) -> None:
        """Delete all chunks for a specific file belonging to a tenant MI_ID."""
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.mi_id",
                    match=models.MatchValue(value=str(mi_id).strip()),
                ),
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=filename),
                ),
            ]
        )
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=doc_filter,
        )

    def check_health(self) -> dict[str, Any]:
        """Return Qdrant and Ollama connectivity status for health checks."""
        status: dict[str, Any] = {
            "store_backend": "qdrant",
            "collection": self.collection_name,
        }
        try:
            self.embeddings.embed_query("health check")
            status["ollama"] = "connected"
        except Exception as exc:
            status["ollama"] = f"error: {exc}"

        try:
            status["indexed_chunks"] = self.count()
            status["qdrant"] = "connected"
        except Exception as exc:
            status["qdrant"] = f"error: {exc}"
            status["indexed_chunks"] = 0

        return status