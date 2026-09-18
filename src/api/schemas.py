"""Request and response schemas for the admission API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatRequest(BaseModel):
    """Payload for submitting an applicant admission question."""

    conversation_id: str = Field(..., min_length=1, max_length=128, description="Conversation ID")
    question: str = Field(..., min_length=1, max_length=4000, description="Applicant question")
    mi_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$", description="Tenant unique identifier (MI_ID)")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize conversation_id / session_id / Session_id
            if "conversation_id" not in data:
                for k in ("Session_id", "session_id", "Conversation_id", "ConversationId"):
                    if k in data and data[k]:
                        data["conversation_id"] = str(data[k])
                        break
            # Normalize mi_id / MI_ID
            if "mi_id" not in data:
                for k in ("MI_ID", "miId", "tenant_id", "TenantId"):
                    if k in data and data[k] is not None:
                        data["mi_id"] = str(data[k])
                        break
            # Normalize question / Question / message / Message
            if "question" not in data:
                for k in ("Question", "message", "Message", "query", "Query"):
                    if k in data and data[k]:
                        data["question"] = str(data[k])
                        break
        return data


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


class DocumentActiveRequest(BaseModel):
    """Payload to activate or deactivate a document for a tenant."""

    MI_ID: str = Field(..., pattern=r"^[A-Za-z0-9_-]+$", description="Tenant unique identifier (e.g. 30)")
    FileName: str = Field(..., description="Document filename (e.g. Pre_Admission_Module_Knowledge_Base.pdf)")
    ActiveFlag: bool = Field(..., description="Active status flag (true = active, false = inactive)")

    model_config = ConfigDict(extra="forbid")


class DocumentActiveResponse(BaseModel):
    """Result returned after toggling active status of a document."""

    status: str = Field(default="success")
    mi_id: str
    filename: str
    is_active: bool
    updated_chunks: int
    message: str


class DocumentDeleteRequest(BaseModel):
    """Payload to delete a document and its vectors for a tenant."""

    MI_ID: str = Field(..., pattern=r"^[A-Za-z0-9_-]+$", description="Tenant unique identifier (e.g. 30)")
    FileName: str = Field(default="ALL", description="Document filename to delete, or 'ALL' to delete all documents for this tenant")

    model_config = ConfigDict(extra="forbid")


class DocumentDeleteResponse(BaseModel):
    """Result returned after deleting a document."""

    status: str = Field(default="success")
    mi_id: str
    filename: str
    deleted_chunks: int
    message: str
