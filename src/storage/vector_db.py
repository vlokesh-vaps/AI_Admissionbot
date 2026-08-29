"""Persistent vector-store access backed by Qdrant with Ollama embeddings.

Every search operation strictly requires MI_ID to ensure complete multi-tenant
data isolation across institutions.
"""

from __future__ import annotations

import logging
from typing import Any
import uuid

from ollama import Client, ResponseError
from qdrant_client import QdrantClient, models

from src.config import Settings
from src.ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)


class OllamaEmbeddingService:
    """Adapter around Ollama's embedding endpoint."""

    def __init__(self, host: str, model: str) -> None:
        self.host = host
        self.model = model
        self._dimension: int | None = None
        try:
            self.client = Client(host=host)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {host}. "
                "Ensure Ollama is running: ollama serve"
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embed(model=self.model, input=texts)
        except ResponseError as exc:
            raise RuntimeError(
                f"Ollama embedding failed for model '{self.model}': {exc}. "
                f"Ensure the model is pulled: ollama pull {self.model}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Ollama connection failed at {self.host}: {exc}. "
                "Ensure Ollama is running: ollama serve"
            ) from exc
        embeddings = response.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings.")
        if self._dimension is None and len(embeddings) > 0:
            self._dimension = len(embeddings[0])
        return embeddings

    def get_dimension(self, default: int = 768) -> int:
        """Detect or return embedding dimension."""
        if self._dimension is not None:
            return self._dimension
        try:
            vectors = self.embed(["dimension probe"])
            if vectors and len(vectors[0]) > 0:
                self._dimension = len(vectors[0])
                return self._dimension
        except Exception as exc:
            logger.warning("Could not probe Ollama embedding dimension (%s); defaulting to %d", exc, default)
        return default


def _point_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID string from a chunk identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class VectorStore:
    """Multi-tenant Vector Store backed by Qdrant with Ollama embeddings."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        self.embeddings = OllamaEmbeddingService(
            settings.ollama_host, settings.ollama_embed_model
        )

        if client is not None:
            self.client = client
        elif settings.qdrant_url:
            try:
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=5.0,
                )
                # Quick ping to verify connectivity
                self.client.get_collections()
                logger.info("Vector store: Connected to Qdrant Server (%s)", settings.qdrant_url)
            except Exception as exc:
                logger.warning(
                    "Could not connect to Qdrant Server at %s (%s). Using in-memory Qdrant.",
                    settings.qdrant_url,
                    exc,
                )
                self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure collection exists and payload index for mi_id is configured."""
        dimension = self.embeddings.get_dimension()
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
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
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="mi_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Index might already exist or not supported on in-memory/embedded
            pass

    def reset(self) -> None:
        """Clear and recreate the Qdrant collection."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def upsert(
        self,
        chunks: list[DocumentChunk],
        mi_id: str | int | None = None,
        batch_size: int = 64,
    ) -> int:
        """Embed and upsert document chunks tagged with tenant MI_ID."""
        if not chunks:
            return 0

        default_mi_id = str(mi_id or self.settings.default_mi_id)
        total = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            vectors = self.embeddings.embed(texts)

            points: list[models.PointStruct] = []
            for chunk, vector in zip(batch, vectors):
                chunk_mi_id = str(chunk.metadata.get("mi_id", default_mi_id))
                payload: dict[str, Any] = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "mi_id": chunk_mi_id,
                    "source": chunk.metadata.get("source", ""),
                    "title": chunk.metadata.get("title", ""),
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    **chunk.metadata,
                }
                points.append(
                    models.PointStruct(
                        id=_point_uuid(chunk.chunk_id),
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
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
        query_embedding = self.embeddings.embed([query])[0]

        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="mi_id",
                    match=models.MatchValue(value=mi_id_str),
                )
            ]
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=tenant_filter,
            limit=limit,
            with_payload=True,
        )

        results: list[dict[str, Any]] = []
        for hit in response.points:
            payload = hit.payload or {}
            results.append(
                {
                    "id": payload.get("chunk_id", str(hit.id)),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                    "score": float(hit.score),
                    "retriever": "semantic",
                    "mi_id": payload.get("mi_id", mi_id_str),
                }
            )
        return results

    def all_chunks(self, mi_id: str | int | None = None) -> list[dict[str, Any]]:
        """Retrieve stored chunks, optionally filtered by tenant MI_ID."""
        scroll_filter = None
        if mi_id is not None and str(mi_id).strip():
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="mi_id",
                        match=models.MatchValue(value=str(mi_id).strip()),
                    )
                ]
            )

        chunks: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for record in records:
                payload = record.payload or {}
                chunks.append(
                    {
                        "id": payload.get("chunk_id", str(record.id)),
                        "text": payload.get("text", ""),
                        "metadata": payload,
                        "mi_id": payload.get("mi_id"),
                    }
                )
            if next_offset is None:
                break
            offset = next_offset

        return chunks

    def count(self, mi_id: str | int | None = None) -> int:
        """Count points in Qdrant collection, optionally filtered by MI_ID."""
        count_filter = None
        if mi_id is not None and str(mi_id).strip():
            count_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="mi_id",
                        match=models.MatchValue(value=str(mi_id).strip()),
                    )
                ]
            )
        result = self.client.count(
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
                    key="mi_id",
                    match=models.MatchValue(value=mi_id_str),
                )
            ]
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=tenant_filter,
        )

    def delete_document(self, mi_id: str | int, filename: str) -> None:
        """Delete all chunks for a specific file belonging to a tenant MI_ID."""
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="mi_id",
                    match=models.MatchValue(value=str(mi_id).strip()),
                ),
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=filename),
                ),
            ]
        )
        self.client.delete(
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
            self.embeddings.client.list()
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