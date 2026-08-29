"""Request and response schemas for the admission API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload for submitting an applicant admission question."""

    conversation_id: str = Field(default="demo-conversation", description="Conversation ID")
    question: str = Field(..., min_length=1, description="Applicant question")
    mi_id: str | None = Field(default=None, description="Tenant unique identifier (MI_ID)")


class SourceReference(BaseModel):
    """Metadata describing a retrieved document chunk."""

    source: str | None = Field(default=None, description="Source filename")
    title: str | None = Field(default=None, description="Document title")
    chunk_index: int | str | None = Field(default=None, description="Chunk sequence number")
    score: float | None = Field(default=None, description="Relevance / rerank score")
    mi_id: str | None = Field(default=None, description="Tenant MI_ID")


class ChatResponse(BaseModel):
    """Answer returned to the applicant along with source and escalation metadata."""

    conversation_id: str
    answer: str
    mi_id: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    escalate: bool = False
    escalation_reason: str | None = None
    cached: bool = False


class EscalationRequest(BaseModel):
    """Payload for submitting a counselor escalation request."""

    conversation_id: str
    applicant_name: str | None = None
    applicant_email: str | None = None
    note: str | None = None
    mi_id: str | None = None


class EscalationResponse(BaseModel):
    """Confirmation returned when an escalation is recorded."""

    accepted: bool
    conversation_id: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseStatus(BaseModel):
    """Current state of the indexed knowledge base."""

    document_count: int
    chunk_count: int
    supported_extensions: list[str] = Field(default_factory=list)
    mi_id: str | None = None


class UploadResponse(BaseModel):
    """Result returned when a new document is uploaded and indexed."""

    filename: str
    stored_path: str
    indexed_chunks: int
    mi_id: str | None = None


class TenantDocumentUploadResponse(BaseModel):
    """Result returned when a document is uploaded for a specific tenant from the .NET ERP."""

    status: str = Field(default="success", description="Status code or message")
    mi_id: str = Field(..., description="Tenant identifier")
    filename: str = Field(..., description="Uploaded document filename")
    stored_path: str = Field(..., description="Path on server where document is stored")
    indexed_chunks: int = Field(..., description="Number of vector chunks indexed into Qdrant")
    message: str = Field(
        default="Document successfully processed and indexed into tenant knowledge base.",
        description="Informational message",
    )
