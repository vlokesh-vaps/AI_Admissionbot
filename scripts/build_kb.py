"""Build or rebuild the persistent admission knowledge base."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from src.storage.vector_db import VectorStore
from src.utils.logging import configure_logging, log_event

logger = logging.getLogger(__name__)


def build_knowledge_base(reset: bool = True, mi_id: str | None = None) -> int:
    """Load all source files, chunk them, and persist their embeddings."""
    settings.ensure_directories()
    store = VectorStore(settings)
    if reset:
        if mi_id:
            store.delete_tenant_documents(mi_id)
        else:
            store.reset()

    all_chunks = []
    if mi_id:
        target_dir = settings.tenant_documents_dir(mi_id)
        docs = []
        for p in sorted(target_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.append(load_document(p))
        all_chunks = chunk_documents(docs, mi_id=mi_id)
    else:
        # Load root documents
        root_docs = []
        for p in sorted(settings.documents_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                root_docs.append(load_document(p))
        all_chunks = chunk_documents(root_docs, mi_id=settings.default_mi_id)

        # Load tenant subdirectories
        for tenant_dir in sorted(settings.documents_dir.iterdir()):
            if tenant_dir.is_dir():
                tenant_id = tenant_dir.name
                tenant_docs = []
                for p in sorted(tenant_dir.iterdir()):
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        tenant_docs.append(load_document(p))
                all_chunks.extend(chunk_documents(tenant_docs, mi_id=tenant_id))

    indexed = store.upsert(all_chunks) if all_chunks else 0
    log_event(logger, logging.INFO, "knowledge_base_built", operation="rebuild", chunks=indexed, mi_id=mi_id or "all")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the admission chatbot knowledge base")
    parser.add_argument("--no-reset", action="store_true", help="Keep existing Qdrant records")
    parser.add_argument("--mi-id", type=str, default=None, help="Build for specific tenant MI_ID")
    args = parser.parse_args()
    configure_logging(settings.log_dir, settings.log_level)
    indexed = build_knowledge_base(reset=not args.no_reset, mi_id=args.mi_id)
    print(f"Indexed {indexed} chunks into Qdrant collection '{settings.qdrant_collection}'")


if __name__ == "__main__":
    main()
