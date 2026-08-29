"""FastAPI application for the admission chatbot with multi-tenant .NET ERP integration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
import logging
import time

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
from src.storage.postgres import PostgresRepository
from src.utils.logging import configure_logging, log_event, reset_request_id, set_request_id

configure_logging(settings.log_dir, settings.log_level)
logger = logging.getLogger(__name__)

store = VectorStore(settings)
retriever = HybridRetriever(store, settings)
reranker = Reranker(settings)
conversations = ConversationStore()
cache = ResponseCache(settings.cache_dir, settings.cache_ttl_seconds)
rag = AdmissionRAG(settings, retriever, reranker, conversations)
erp = PostgresRepository(settings)


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


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Attach a correlation ID and emit one concise request completion event."""
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    token = set_request_id(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        log_event(
            logger,
            logging.INFO,
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        reset_request_id(token)

_MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log_event(
        logger,
        logging.ERROR,
        "request_failed",
        request_id=getattr(request.state, "request_id", None),
        method=request.method,
        path=request.url.path,
        error_type=type(exc).__name__,
        exc_info=True,
    )
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


def _save_chat_message(mi_id: str, session_id: str, message: object) -> None:
    if not settings.database_url:
        return
    try:
        erp.save_message(int(mi_id), session_id, message)
    except Exception:
        log_event(logger, logging.ERROR, "conversation_save_failed", operation="insert", table="AI_Admission_Conversation", mi_id=mi_id, session_id=session_id, error_type="database_error", exc_info=True)
        raise RuntimeError("AI_Admission_Conversation insert failed.") from None


def _persist_chat_turn(mi_id: str, session_id: str, question: str, answer: str) -> None:
    """Persist one row containing the user message and assistant reply."""
    _save_chat_message(
        mi_id,
        session_id,
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
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
    File: UploadFile = File(..., description="PDF or DOCX document"),
    MI_ID: str = Form(..., description="Unique Tenant Identifier (e.g. 1001)"),
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
        log_event(logger, logging.ERROR, "document_index_failed", operation="ingest", mi_id=mi_id_clean, filename=filename, error_type=type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Document could not be indexed: {exc}") from exc

    log_event(
        logger,
        logging.INFO,
        "document_indexed",
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Query the admission chatbot strictly within the applicant's tenant knowledge base."""
    target_mi_id = str(request.mi_id or settings.default_mi_id).strip()
    cache_key = f"{target_mi_id}:{request.conversation_id}"
    log_event(
        logger,
        logging.INFO,
        "chat_request",
        operation="answer",
        mi_id=target_mi_id,
        session_id=request.conversation_id,
    )

    cached_value = cache.get(cache_key, request.question)
    if cached_value:
        log_event(
            logger,
            logging.INFO,
            "cache_hit",
            operation="answer",
            mi_id=target_mi_id,
            session_id=request.conversation_id,
            source="cache",
        )
        cached_response = dict(cached_value)
        cached_response["cached"] = True
        response = ChatResponse(**cached_response)
        _persist_chat_turn(target_mi_id, request.conversation_id, request.question, response.answer)
        log_event(
            logger,
            logging.INFO,
            "chat_completed",
            operation="answer",
            mi_id=target_mi_id,
            session_id=request.conversation_id,
            source_count=len(response.sources),
            escalate=response.escalate,
            cached=True,
        )
        return response

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
        log_event(logger, logging.ERROR, "chat_failed", operation="answer", mi_id=target_mi_id, error_type=type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to answer the question.") from exc

    response = _serialize_result(result)
    cache.set(cache_key, request.question, response.model_dump())
    _persist_chat_turn(target_mi_id, request.conversation_id, request.question, response.answer)
    log_event(
        logger,
        logging.INFO,
        "chat_completed",
        operation="answer",
        mi_id=target_mi_id,
        session_id=request.conversation_id,
        source_count=len(response.sources),
        escalate=response.escalate,
        cached=False,
    )
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
        operation="escalation",
        conversation_id=request.conversation_id,
        mi_id=request.mi_id,
    )
    return EscalationResponse(
        accepted=True,
        conversation_id=request.conversation_id,
        message="Your request has been recorded for an admissions counselor.",
        details={"delivery": "structured application log", "mi_id": request.mi_id},
    )
