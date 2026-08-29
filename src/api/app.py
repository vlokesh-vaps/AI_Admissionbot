"""FastAPI application for the admission chatbot with multi-tenant .NET ERP integration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
import logging

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    EscalationRequest,
    EscalationResponse,
    KnowledgeBaseStatus,
    SourceReference,
    TenantDocumentUploadResponse,
    UploadResponse,
)
from src.config import settings
from src.core.conversation import ConversationStore
from src.core.rag import AdmissionRAG
from src.core.reranker import Reranker
from src.core.retrievers import HybridRetriever
from src.ingestion.chunking import chunk_document
from src.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from src.storage.cache import ResponseCache
from src.storage.vector_db import VectorStore
from src.utils.logging import configure_logging, log_event

configure_logging(settings.log_dir, settings.log_level)
logger = logging.getLogger(__name__)

store = VectorStore(settings)
retriever = HybridRetriever(store, settings)
reranker = Reranker(settings)
conversations = ConversationStore()
cache = ResponseCache(settings.cache_dir, settings.cache_ttl_seconds)
rag = AdmissionRAG(settings, retriever, reranker, conversations)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Log service configuration on startup."""
    log_event(
        logger, logging.INFO, "service_started",
        groq_model=settings.groq_model,
        ollama_host=settings.ollama_host,
        embed_model=settings.ollama_embed_model,
        reranker_mode=reranker.mode,
        vector_store="qdrant",
        qdrant_collection=settings.qdrant_collection,
        cache_path=str(cache.path),
        log_dir=str(settings.log_dir),
        indexed_chunks=store.count(),
    )
    yield


app = FastAPI(
    title="Admission AI Chatbot & Multi-Tenant Ingestion Service",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An internal error occurred."})


def _serialize_result(result: object, cached: bool = False) -> ChatResponse:
    return ChatResponse(
        conversation_id=result.conversation_id,  # type: ignore[attr-defined]
        answer=result.answer,  # type: ignore[attr-defined]
        mi_id=getattr(result, "mi_id", None),
        sources=[SourceReference(**source) for source in result.sources],  # type: ignore[attr-defined]
        escalate=result.escalate,  # type: ignore[attr-defined]
        escalation_reason=result.escalation_reason,  # type: ignore[attr-defined]
        cached=cached,
    )


@app.get("/", response_class=HTMLResponse)
def index_ui() -> HTMLResponse:
    index_file = settings.project_root / "static" / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<!DOCTYPE html><html><body><h1>Admission AI Chatbot</h1></body></html>")


@app.get("/health")
def health() -> dict:
    """Deep health check including Ollama and Qdrant connectivity."""
    store_health = store.check_health()
    return {
        "status": "ok",
        "reranker_mode": reranker.mode,
        **store_health,
    }


@app.post("/api/ai/admission/document", response_model=TenantDocumentUploadResponse)
async def upload_admission_document(
    MI_ID: str = Form(..., description="Unique Tenant Identifier (e.g. 1001)"),
    File: UploadFile = File(..., description="PDF or DOCX document"),
) -> TenantDocumentUploadResponse:
    """Multi-tenant document upload endpoint called by the .NET ERP.

    Processes, chunks, embeds with Ollama, and indexes vector embeddings into
    Qdrant isolated under the specified MI_ID tenant.
    """
    mi_id_clean = str(MI_ID).strip()
    if not mi_id_clean:
        raise HTTPException(status_code=422, detail="MI_ID tenant identifier is required.")

    filename = Path(File.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename provided.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '{suffix}'. Only PDF and DOCX files are supported.",
        )

    data = await File.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum upload size is {settings.max_upload_mb} MB.",
        )

    # Save to tenant isolated folder: data/documents/{MI_ID}/...
    tenant_dir = settings.tenant_documents_dir(mi_id_clean)
    destination = tenant_dir / f"{uuid4().hex}_{filename}"
    destination.write_bytes(data)

    try:
        document = load_document(destination)
        chunks = chunk_document(document, mi_id=mi_id_clean)
        indexed = store.upsert(chunks, mi_id=mi_id_clean)
        retriever.refresh()
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("tenant_document_index_failed", extra={"structured": {"mi_id": mi_id_clean, "filename": filename}})
        raise HTTPException(status_code=422, detail=f"Document could not be indexed: {exc}") from exc

    log_event(
        logger,
        logging.INFO,
        "tenant_document_indexed",
        mi_id=mi_id_clean,
        filename=filename,
        chunks=indexed,
        stored_path=str(destination),
    )

    return TenantDocumentUploadResponse(
        status="success",
        mi_id=mi_id_clean,
        filename=filename,
        stored_path=str(destination),
        indexed_chunks=indexed,
        message="Document successfully processed and indexed into tenant knowledge base.",
    )


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document_legacy(file: UploadFile = File(...)) -> UploadResponse:
    """Legacy single-tenant upload endpoint for backward compatibility."""
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only PDF and DOCX files are supported.")

    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum upload size is {settings.max_upload_mb} MB.",
        )

    destination = settings.documents_dir / f"{uuid4().hex}_{filename}"
    destination.write_bytes(data)
    try:
        document = load_document(destination)
        chunks = chunk_document(document, mi_id=settings.default_mi_id)
        indexed = store.upsert(chunks, mi_id=settings.default_mi_id)
        retriever.refresh()
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("document_index_failed")
        raise HTTPException(status_code=422, detail=f"Document could not be indexed: {exc}") from exc

    log_event(logger, logging.INFO, "document_indexed", filename=filename, chunks=indexed)
    return UploadResponse(
        filename=filename,
        stored_path=str(destination),
        indexed_chunks=indexed,
        mi_id=settings.default_mi_id,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Query the admission chatbot strictly within the applicant's tenant knowledge base."""
    target_mi_id = str(request.mi_id or settings.default_mi_id).strip()
    cache_key = f"{target_mi_id}:{request.conversation_id}"

    cached_value = cache.get(cache_key, request.question)
    if cached_value:
        return ChatResponse(**cached_value, cached=True)

    try:
        result = rag.chat(
            conversation_id=request.conversation_id,
            question=request.question,
            mi_id=target_mi_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("chat_failed", extra={"structured": {"mi_id": target_mi_id}})
        raise HTTPException(status_code=500, detail="Unable to answer the question.") from exc

    response = _serialize_result(result)
    cache.set(cache_key, request.question, response.model_dump())
    return response


@app.post("/api/knowledge-base/rebuild")
def rebuild_knowledge_base() -> dict[str, int]:
    from scripts.build_kb import build_knowledge_base

    indexed = build_knowledge_base(reset=True)
    retriever.refresh()
    return {"indexed_chunks": indexed}


@app.get("/api/knowledge-base/status", response_model=KnowledgeBaseStatus)
def knowledge_base_status(mi_id: str | None = Query(default=None, description="Optional tenant MI_ID filter")) -> KnowledgeBaseStatus:
    if mi_id:
        target_dir = settings.tenant_documents_dir(mi_id)
        documents = [
            path for path in target_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ] if target_dir.exists() else []
        chunk_count = store.count(mi_id=mi_id)
    else:
        documents = [
            path for path in settings.documents_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        chunk_count = store.count()

    return KnowledgeBaseStatus(
        document_count=len(documents),
        chunk_count=chunk_count,
        supported_extensions=sorted(SUPPORTED_EXTENSIONS),
        mi_id=mi_id,
    )


@app.post("/api/escalation", response_model=EscalationResponse)
def escalation(request: EscalationRequest) -> EscalationResponse:
    log_event(
        logger,
        logging.INFO,
        "human_escalation_requested",
        conversation_id=request.conversation_id,
        applicant_name=request.applicant_name,
        applicant_email=request.applicant_email,
        note=request.note,
        mi_id=request.mi_id,
    )
    return EscalationResponse(
        accepted=True,
        conversation_id=request.conversation_id,
        message="Your request has been recorded for an admissions counselor.",
        details={"delivery": "structured application log", "mi_id": request.mi_id},
    )
