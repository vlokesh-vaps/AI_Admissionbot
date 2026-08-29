# AI Admission Bot Project Guide

This document explains the project structure, request flow, configuration, and extension points.

## Purpose

The service is a FastAPI admission chatbot with retrieval-augmented generation (RAG). It accepts admission documents from an ERP system, creates searchable embeddings, answers tenant-scoped questions, and records chat messages.

The main components are FastAPI, PostgreSQL, local file storage, Ollama, Qdrant, BM25 retrieval, reranking, and Groq.

## Document Ingestion

The ERP sends a multipart request:

```text
POST /api/ai/admission/document
MI_ID=1001
File=document.pdf
```

The service validates the tenant, validates the uploaded PDF or DOCX, stores it under `data/documents/<MI_ID>/`, extracts text, creates chunks, generates `embeddinggemma:latest` embeddings, and upserts the chunks into Qdrant.

The upload endpoint accepts exactly `File` and `MI_ID`. The uploaded `File` is the only accepted document source. `FilePath` is not used or downloaded. Requests without `File` are rejected.

## Chat Flow

The client sends `conversation_id`, `question`, and `mi_id` to `POST /api/chat`. The service applies the tenant filter to semantic and BM25 retrieval, reranks the results, builds bounded context, calls Groq, and returns the answer with sources and escalation metadata.

User and assistant messages are kept in the in-memory conversation store and are also inserted into `AI_Admission_Conversation` when `DATABASE_URL` is configured. The `Message` column contains JSON text.

Every retrieval operation must include the requested `MI_ID`. This is the primary cross-tenant isolation control.

## Database Tables

The service uses the existing ERP tables and does not create tables automatically.

### `AI_Admission_Document`

The repository reads `AID_Id`, `MI_ID`, `FileName`, `FilePath`, and `ActiveFlag`. Rows are selected by `AID_Id`, `MI_ID`, and `ActiveFlag = true`. The posted file is processed; `FilePath` is not used as a download source.

### `AI_Admission_Conversation`

The service inserts `MI_ID`, `Session_id`, `Message`, `CreatedBy`, `CreatedDate`, and `ActiveFlag`.

Each chat turn creates one row containing both messages. The `Message` value is a JSON array containing only the role and message content:

```json
[
  {
    "role": "user",
    "content": "What is the admission fee?"
  },
  {
    "role": "assistant",
    "content": "The admission fee is ..."
  }
]
```

## Code Map

```text
app.py                    FastAPI server entry point
main.py                   Backward-compatible CLI entry point
src/config.py             Environment-backed settings
src/api/app.py            Routes and request orchestration
src/api/schemas.py        Pydantic request and response models
src/storage/postgres.py   ERP PostgreSQL access
src/storage/vector_db.py  Qdrant and Ollama embedding access
src/storage/cache.py      File-backed response cache
src/ingestion/loaders.py  PDF and DOCX extraction
src/ingestion/chunking.py Stable tenant-tagged chunk creation
src/core/retrievers.py    Hybrid semantic and BM25 retrieval
src/core/reranker.py      Candidate reranking
src/core/rag.py           Generation and escalation logic
src/core/conversation.py  In-memory conversation state
scripts/build_kb.py       Batch knowledge-base rebuild
static/index.html         Browser client
tests/                    Automated tests
```

## Configuration

Copy `.env.example` to `.env` and set real local values. Azure PostgreSQL should use TLS:

```env
DATABASE_URL=postgresql://username:password@host:5432/database?sslmode=require
DATABASE_CREATED_BY=0
OLLAMA_EMBED_MODEL=embeddinggemma:latest
```

Never commit `.env` or place credentials in `.env.example`, source code, README files, or logs.

When changing the embedding model, rebuild the Qdrant knowledge base. Do not mix vectors produced by different embedding models.

## Development Commands

```bash
python main.py                 # Start the API
python main.py chat            # Start the terminal chat
python main.py build-kb        # Rebuild the knowledge base
python -m pytest -q            # Run tests
```

`main.py` is the primary command hub. `app.py`, `scripts/cli_chat.py`, and `scripts/build_kb.py` remain available as compatibility and automation entry points.

The UI is available at `http://localhost:5003`; OpenAPI documentation is available at `http://localhost:5003/docs`.

Tests use isolated or mocked dependencies and do not require the production PostgreSQL, Groq, or Qdrant services.


• mi_id filtering happens in these locations:

  1. API receives mi_id
     src/api/app.py:245

     target_mi_id = str(request.mi_id or settings.default_mi_id).strip()

  2. RAG passes it to the retriever
     src/core/rag.py:119

     candidates = self.retriever.search(question, mi_id=target_mi_id)

  3. Hybrid retriever filters tenant chunks
     src/core/retrievers.py:58

     semantic = self.store.search(
         query=query,
         mi_id=mi_id_str,
         limit=limit * 2,
     )

     BM25 filtering is here:

     src/core/retrievers.py:80

     tenant_chunks = [
         chunk for chunk in self._chunks
         if str(
             chunk.get("mi_id")
             or chunk.get("metadata", {}).get("mi_id", "")
         ).strip() == mi_id_str
     ]

  4. Actual Qdrant/vector database filter
     src/storage/vector_db.py:209

     tenant_filter = models.Filter(
         must=[
             models.FieldCondition(
                 key="mi_id",
                 match=models.MatchValue(value=mi_id_str),
             )
         ]
     )

     It is applied here:

     src/storage/vector_db.py:218

     response = self.client.query_points(
         collection_name=self.collection_name,
         query=query_embedding,
         query_filter=tenant_filter,
         limit=limit,
         with_payload=True,
     )

  5. mi_id is stored in every indexed chunk
     src/ingestion/chunking.py:48

     metadata={
         "mi_id": mi_id_str,
         ...
     }

  So the retrieval flow is:

  Chat request mi_id
      -> app.py
      -> rag.py
      -> retrievers.py
      -> vector_db.py
      -> Qdrant filter key="mi_id"

  Example: a request with mi_id="30" can retrieve only documents indexed with mi_id="30".