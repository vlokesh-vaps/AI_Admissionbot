"""Persistent vector-store backed by LangChain QdrantVectorStore with OllamaEmbeddings.

Every search operation strictly requires MI_ID to ensure complete multi-tenant
data isolation across institutions.
"""

from __future__ import annotations

import logging
from pathlib import Path
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


def _matches_filename(source: str, target: str) -> bool:
    """Check if source filename matches target (supports extensions, stems, wildcards, and case-insensitivity)."""
    source_clean = str(source).strip().lower()
    target_clean = str(target).strip().lower()
    if not target_clean:
        return False
    if target_clean in {"*", "all"}:
        return True
    source_stem = Path(source_clean).stem
    target_stem = Path(target_clean).stem
    return (
        source_clean == target_clean
        or source_clean.endswith(f"_{target_clean}")
        or source_stem == target_stem
        or source_stem.endswith(f"_{target_stem}")
    )


class VectorStore:
    """Multi-tenant Vector Store using LangChain's QdrantVectorStore and OllamaEmbeddings."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        self.using_in_memory = False

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
                if not settings.allow_inmemory_qdrant:
                    raise RuntimeError(
                        f"Qdrant is unavailable at {settings.qdrant_url}; refusing to use ephemeral storage."
                    ) from exc
                logger.warning(
                    "Qdrant unavailable at %s (%s). Explicit in-memory mode is enabled.",
                    settings.qdrant_url,
                    exc,
                )
                self.qdrant_client = QdrantClient(":memory:")
                self.using_in_memory = True
        else:
            if not settings.allow_inmemory_qdrant:
                raise RuntimeError("QDRANT_URL is not configured and in-memory storage is disabled.")
            self.qdrant_client = QdrantClient(":memory:")
            self.using_in_memory = True

        self._ensure_collection()

        # LangChain vector store for add/search operations
        self.langchain_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            validate_collection_config=False,
            validate_embeddings=False,
        )

    def ensure_collection(self) -> bool:
        """Ensure collection exists and payload indexes for mi_id and is_active are configured."""
        try:
            exists = self.qdrant_client.collection_exists(self.collection_name)
        except Exception:
            exists = False

        if not exists:
            try:
                test_embedding = self.embeddings.embed_query("dimension probe")
                dimension = len(test_embedding)
            except Exception as exc:
                logger.warning("Could not probe embedding dimension (%s); defaulting to 768", exc)
                dimension = 768

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
            self.langchain_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                validate_collection_config=False,
                validate_embeddings=False,
            )

        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.mi_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.is_active",
                field_schema=models.PayloadSchemaType.BOOL,
            )
        except Exception:
            pass

        return True

    _ensure_collection = ensure_collection

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

        # Auto-create collection if not created yet
        self.ensure_collection()

        default_mi_id = str(mi_id).strip() if mi_id is not None and str(mi_id).strip() else None
        total = 0

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]

            # Ensure all docs have mi_id and generate stable point IDs
            ids: list[str] = []
            for doc in batch:
                document_mi_id = str(doc.metadata.get("mi_id", "")).strip()
                if not document_mi_id and not default_mi_id:
                    raise ValueError("mi_id is required in every document when no batch tenant is provided.")
                if default_mi_id and document_mi_id and document_mi_id != default_mi_id:
                    raise ValueError("Document tenant does not match the indexing tenant.")
                if not document_mi_id:
                    doc.metadata["mi_id"] = default_mi_id
                chunk_id = doc.metadata.get("chunk_id", doc.id or str(uuid.uuid4()))
                ids.append(_point_uuid(chunk_id))

            self.langchain_store.add_documents(batch, ids=ids)
            total += len(batch)

        return total

    def replace_document(
        self, documents: list[Document], mi_id: str | int, filename: str
    ) -> int:
        """Index a replacement before removing the previous version.

        If parsing or embedding fails, the previous document remains available.
        """
        mi_id_str = str(mi_id).strip()
        if not mi_id_str:
            raise ValueError("mi_id is required when replacing a document.")

        old_ids = self._matching_point_ids(mi_id_str, filename)
        indexed = self.upsert(documents, mi_id=mi_id_str)
        new_ids = {
            _point_uuid(str(doc.metadata.get("chunk_id", doc.id or "")))
            for doc in documents
        }
        stale_ids = [point_id for point_id in old_ids if point_id not in new_ids]
        self._delete_point_ids(stale_ids)
        return indexed

    def _matching_point_ids(self, mi_id: str, filename: str) -> list[Any]:
        if not self.qdrant_client.collection_exists(self.collection_name):
            return []
        matched: list[Any] = []
        offset: Any = None
        while True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=250,
                offset=offset,
                with_payload=True,
            )
            for record in records:
                payload = dict(record.payload or {})
                metadata = dict(payload.get("metadata", {}))
                if str(metadata.get("mi_id") or payload.get("mi_id") or "").strip() != mi_id:
                    continue
                source = str(metadata.get("source") or payload.get("source") or "")
                title = str(metadata.get("title") or payload.get("title") or "")
                if _matches_filename(source, filename) or _matches_filename(title, filename):
                    matched.append(record.id)
            if next_offset is None:
                break
            offset = next_offset
        return matched

    def _delete_point_ids(self, point_ids: list[Any]) -> None:
        for start in range(0, len(point_ids), 500):
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids[start : start + 500]),
            )

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

        if not self.qdrant_client.collection_exists(self.collection_name):
            return []

        mi_id_str = str(mi_id).strip()
        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.mi_id",
                    match=models.MatchValue(value=mi_id_str),
                )
            ],
            must_not=[
                models.FieldCondition(
                    key="metadata.is_active",
                    match=models.MatchValue(value=False),
                )
            ],
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
        if not self.qdrant_client.collection_exists(self.collection_name):
            return []

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
        if not self.qdrant_client.collection_exists(self.collection_name):
            return 0

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

    def delete_tenant_documents(self, mi_id: str | int) -> int:
        """Delete all points belonging to a specific tenant MI_ID."""
        if not self.qdrant_client.collection_exists(self.collection_name):
            return 0

        mi_id_str = str(mi_id).strip()
        matched_point_ids: list[Any] = []
        offset: Any = None
        while True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=250,
                offset=offset,
                with_payload=True,
            )
            for record in records:
                payload = dict(record.payload or {})
                metadata = dict(payload.get("metadata", {}))
                rec_mi_id = str(metadata.get("mi_id") or payload.get("mi_id") or "").strip()
                if rec_mi_id == mi_id_str:
                    matched_point_ids.append(record.id)

            if next_offset is None:
                break
            offset = next_offset

        if matched_point_ids:
            for i in range(0, len(matched_point_ids), 500):
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=matched_point_ids[i : i + 500]),
                )

        return len(matched_point_ids)

    def set_document_active_status(
        self, mi_id: str | int, filename: str, is_active: bool
    ) -> int:
        """Update active status of all chunks matching mi_id and filename."""
        if not self.qdrant_client.collection_exists(self.collection_name):
            return 0

        mi_id_str = str(mi_id).strip()
        target = str(filename or "").strip()

        offset: Any = None
        updated_count = 0

        while True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=250,
                offset=offset,
                with_payload=True,
            )
            for record in records:
                payload = dict(record.payload or {})
                metadata = dict(payload.get("metadata", {}))

                rec_mi_id = str(metadata.get("mi_id") or payload.get("mi_id") or "").strip()
                if rec_mi_id != mi_id_str:
                    continue

                source = str(metadata.get("source") or payload.get("source") or "")
                source_path = str(metadata.get("source_path") or payload.get("source_path") or "")
                title = str(metadata.get("title") or payload.get("title") or "")

                is_target = (
                    not target
                    or target.lower() in {"*", "all"}
                    or _matches_filename(source, target)
                    or (source_path and _matches_filename(Path(source_path).name, target))
                    or (title and _matches_filename(title, target))
                )

                if is_target:
                    metadata["is_active"] = is_active
                    metadata["active_flag"] = is_active
                    payload["metadata"] = metadata
                    payload["is_active"] = is_active
                    payload["active_flag"] = is_active
                    self.qdrant_client.overwrite_payload(
                        collection_name=self.collection_name,
                        payload=payload,
                        points=[record.id],
                    )
                    updated_count += 1

            if next_offset is None:
                break
            offset = next_offset

        return updated_count

    def delete_document(self, mi_id: str | int, filename: str) -> int:
        """Delete all chunks for a specific file belonging to a tenant MI_ID.

        Matches exact filename, stem without extension, case-insensitive, or legacy UUID-prefixed filename.
        Checks both payload['metadata'] and root payload fields.
        If filename is '*' or 'ALL' or empty, deletes all chunks for the tenant.
        """
        if not self.qdrant_client.collection_exists(self.collection_name):
            return 0

        mi_id_str = str(mi_id).strip()
        target = str(filename or "").strip()

        if not target or target.lower() in {"*", "all"}:
            return self.delete_tenant_documents(mi_id_str)

        matched_point_ids: list[Any] = []
        offset: Any = None
        while True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=250,
                offset=offset,
                with_payload=True,
            )
            for record in records:
                payload = dict(record.payload or {})
                metadata = dict(payload.get("metadata", {}))

                rec_mi_id = str(metadata.get("mi_id") or payload.get("mi_id") or "").strip()
                if rec_mi_id != mi_id_str:
                    continue

                source = str(metadata.get("source") or payload.get("source") or "")
                source_path = str(metadata.get("source_path") or payload.get("source_path") or "")
                title = str(metadata.get("title") or payload.get("title") or "")

                if (
                    _matches_filename(source, target)
                    or (source_path and _matches_filename(Path(source_path).name, target))
                    or (title and _matches_filename(title, target))
                ):
                    matched_point_ids.append(record.id)

            if next_offset is None:
                break
            offset = next_offset

        if matched_point_ids:
            for i in range(0, len(matched_point_ids), 500):
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=matched_point_ids[i : i + 500]),
                )

        return len(matched_point_ids)

    def check_health(self) -> dict[str, Any]:
        """Return Qdrant and Ollama connectivity status for health checks."""
        status: dict[str, Any] = {
            "store_backend": "qdrant",
            "collection": self.collection_name,
            "persistent": not self.using_in_memory,
        }
        try:
            self.embeddings.embed_query("health check")
            status["ollama"] = "connected"
        except Exception as exc:
            status["ollama"] = f"error: {exc}"

        try:
            status["indexed_chunks"] = self.count()
            status["qdrant"] = "in_memory" if self.using_in_memory else "connected"
        except Exception as exc:
            status["qdrant"] = f"error: {exc}"
            status["indexed_chunks"] = 0

        return status
