# AI Admission Bot & Multi-Tenant Ingestion Service

A high-performance, RAG-powered (Retrieval-Augmented Generation) Admission Assistant chatbot and document ingestion service built with **FastAPI**, **Qdrant Vector Database**, **Ollama Embeddings**, **Groq LLM**, and **Hybrid Vector/BM25 Search**.

Supports seamless integration with **.NET ERP** systems via multi-tenant document upload, enforcing **strict `MI_ID` data isolation** across institutions.

For a developer-oriented explanation of the code, data flow, configuration, and extension points, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

---

## Architecture & Multi-Tenant Isolation

```
┌────────────────────────────────────────────────────────┐
│                   .NET ERP System                      │
│     (Uploads PDF/DOCX with Tenant ID: MI_ID)           │
└───────────────────────────┬────────────────────────────┘
                            │ POST /api/ai/admission/document
                            │ Multipart: MI_ID="1001", File=document.pdf
                            ▼
┌────────────────────────────────────────────────────────┐
│               Python AI Admission Service              │
│                     (Port 5003)                        │
├────────────────────────────────────────────────────────┤
│ 1. Validate File (.pdf/.docx) & Tenant (MI_ID)         │
│ 2. Save to tenant directory: data/documents/{MI_ID}/   │
│ 3. Extract text (PDF/DOCX loaders)                     │
│ 4. Chunk text & tag metadata with MI_ID                │
│ 5. Generate Vector Embeddings via Ollama               │
│ 6. Upsert to Qdrant collection with payload:           │
│    - mi_id: "1001" (Indexed keyword field)             │
│    - source: "document.pdf"                            │
│    - text: chunk text                                  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Qdrant Vector DB                      │
│  Collection: admission_knowledge                       │
│  Payload Index: mi_id (keyword)                        │
│                                                        │
│  MANDATORY SEARCH FILTER:                              │
│  Filter(must=[FieldCondition(key="mi_id", match=1001)])│
│  ==> ZERO CROSS-TENANT DATA LEAKAGE GUARANTEED         │
└────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Multi-Tenant Isolation:** Documents and vector embeddings are partitioned by `MI_ID`. All Qdrant semantic searches and BM25 queries strictly enforce tenant filtering.
- **.NET ERP Integration:** Direct API endpoint `POST /api/ai/admission/document` accepts tenant `MI_ID` and PDF/DOCX files.
- **Ollama Dense Embeddings:** Generates embeddings locally via Ollama (`embeddinggemma:latest` by default; configurable).
- **Qdrant Vector Store:** Fast, scalable vector retrieval with keyword payload indexing on tenant IDs.
- **Hybrid Retrieval System:** Dense vector search combined with sparse BM25 keyword matching and cross-encoder reranking.
- **Interactive Web Interface:** Modern UI served at `/` on port **5003**.
- **Counselor Escalation:** Structured workflow for admission counselor follow-up.
- **High-Speed Caching:** Multi-tenant response cache with configurable TTL.

---

## Project Structure

```
AI_Admissionbot/
├── app.py                      # FastAPI server entry point (Port 5003)
├── main.py                     # CLI chat entry point
├── Dockerfile                  # Container definition (Exposed on 5003)
├── docker-compose.yml          # Admission Bot (5003) + Ollama (11434) + Qdrant (6333)
├── requirements.txt            # Core dependencies
├── requirements-optional.txt   # Optional enhancements
├── .env.example                # Environment variables template
├── .env                        # Local runtime environment
├── data/                       # Local data storage
│   ├── cache/                  # Response cache directory
│   ├── documents/              # Multi-tenant documents: data/documents/{MI_ID}/
│   ├── logs/                   # Structured event logs
│   ├── models/                 # Model cache directory
│   └── qdrant_db/              # Local embedded Qdrant database
├── scripts/
│   ├── build_kb.py             # Batch knowledge base indexing script
│   └── cli_chat.py             # Interactive terminal chat client
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI endpoints (/api/ai/admission/document, /api/chat)
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── core/
│   │   ├── conversation.py     # Tenant-isolated conversation store
│   │   ├── prompts.py          # System prompts and prompt templates
│   │   ├── rag.py              # Core RAG orchestration pipeline
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   └── retrievers.py       # Multi-tenant Hybrid retriever
│   ├── ingestion/
│   │   ├── chunking.py         # Tenant-tagged text chunking
│   │   └── loaders.py          # PDF and DOCX parsers
│   ├── storage/
│   │   ├── cache.py            # Local response cache
│   │   └── vector_db.py        # Qdrant VectorStore with mandatory MI_ID filtering
│   ├── utils/
│   │   └── logging.py          # Structured event logging
│   └── config.py               # Settings and configuration management
├── static/
│   └── index.html              # Frontend Web UI
└── tests/
    ├── conftest.py             # Test fixtures
    ├── test_api.py             # FastAPI endpoint test suite
    ├── test_core.py            # RAG and retriever unit tests
    ├── test_loaders.py         # Document parser tests
    ├── test_multi_tenant.py    # Multi-tenant Qdrant isolation tests
    └── test_ui.py              # Web UI delivery tests
```

---

## Quick Start
### 1. Prerequisites

- **Python 3.10+** (Python 3.12 recommended)
- **Ollama** running locally with the embedding model:
  ```bash
  ollama pull embeddinggemma:latest
  ```
- A **Groq API Key** ([console.groq.com](https://console.groq.com/))

---

### 2. Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   PORT=5003
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_EMBED_MODEL=embeddinggemma:latest
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=admission_knowledge
   DEFAULT_MI_ID=1001
   DATABASE_URL=postgresql://username:password@host:5432/database?sslmode=require
   DATABASE_CREATED_BY=0
   ```

---

### 3. Running the Service on Port 5003

#### Option A: Run directly with the command hub
```bash
python main.py api
```
The API is also the default command:

```bash
python main.py
```
The API and Web UI will be available at:
`http://localhost:5003`

#### Option B: Run with Docker Compose
```bash
docker compose up --build
```
This spins up:
- **Admission Bot API** on port `5003`
- **Ollama Server** on port `11434`
- **Qdrant Vector DB** on port `6333`

When changing `OLLAMA_EMBED_MODEL`, rebuild the knowledge base so all stored vectors use the same embedding model:

```bash
curl -X POST http://localhost:5003/api/knowledge-base/rebuild
```

The current `embeddinggemma:latest` model returns 768-dimensional vectors. Existing Qdrant collections must use the same vector dimension and should be rebuilt after an embedding-model change.

Other workflows:

```bash
python main.py chat
python main.py build-kb
python main.py build-kb --mi-id 1001 --no-reset
```

---

## API Reference

The service listens on `http://localhost:5003` by default. Interactive OpenAPI documentation is available at `/docs`, with the raw schema at `/openapi.json`.

All uploaded files must be PDF or DOCX and are limited to `20 MB` by default. Change the limit with `MAX_UPLOAD_MB`.

### Upload a Tenant Document
Called by the .NET ERP when a user uploads a PDF or DOCX file.

- **Endpoint:** `POST /api/ai/admission/document`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `MI_ID` (Form Text, Required): Unique Tenant Identifier (e.g. `1001`)
  - `File` (Form File, Required): `.pdf` or `.docx` document
- **Example cURL:**
  ```bash
  curl -X POST http://localhost:5003/api/ai/admission/document \
    -F "MI_ID=1001" \
    -F "File=@student_document.pdf"
  ```
- **Sample Response (200 OK):**
  ```json
  {
    "status": "success",
    "mi_id": "1001",
    "filename": "student_document.pdf",
    "stored_path": "data/documents/1001/<generated-id>_student_document.pdf",
    "indexed_chunks": 8,
    "message": "Document successfully processed and indexed into tenant knowledge base."
  }
  ```

Returns `415` for unsupported file types, `413` for files over the configured limit, and `422` when document indexing fails.

The uploaded `File` is the only accepted document source. `FilePath` is retained as ERP metadata and is never downloaded by this service.

The endpoint accepts exactly two multipart fields: `File` and `MI_ID`. Requests without either field are rejected. The ERP must post the document itself as multipart form data.

### Upload a Legacy Document

`POST /api/documents/upload` accepts a single `file` multipart field and indexes it under `DEFAULT_MI_ID`. This route is retained for backward compatibility; new integrations should use the tenant upload route above.

```bash
curl -X POST http://localhost:5003/api/documents/upload \
  -F "file=@student_document.pdf"
```

### Chat with Admission Bot (Tenant-Isolated)
- **Endpoint:** `POST /api/chat`
- **Request Body:**
  ```json
  {
    "conversation_id": "session-123",
    "question": "What is the fee structure for B.Tech Computer Science?",
    "mi_id": "1001"
  }
  ```
- **Example cURL:**
  ```bash
  curl -X POST http://localhost:5003/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"conversation_id\": \"session-123\", \"question\": \"What is the fee?\", \"mi_id\": \"1001\"}"
  ```

The response contains `answer`, `sources`, `escalate`, `escalation_reason`, and a `cached` flag, together with the conversation and tenant IDs. PostgreSQL stores one JSON array per chat turn containing only the user and assistant `role` and `content` values.

### Knowledge Base Status
- **Endpoint:** `GET /api/knowledge-base/status?mi_id=1001`
- **Example cURL:**
  ```bash
  curl "http://localhost:5003/api/knowledge-base/status?mi_id=1001"
  ```

- `GET /api/knowledge-base/status` returns whole-knowledge-base counts.
- `GET /api/knowledge-base/status?mi_id=1001` limits counts to one tenant.
- `POST /api/knowledge-base/rebuild` rebuilds from `data/documents` and returns `{"indexed_chunks": 42}`.

---

### Counselor Escalation
- **Endpoint:** `POST /api/escalation`
- **Request Body:**
  ```json
  {
    "conversation_id": "session-123",
    "applicant_name": "Jane Doe",
    "applicant_email": "jane@example.com",
    "note": "Requesting fee concession details",
    "mi_id": "1001"
  }
  ```

The response confirms whether the request was accepted and identifies the conversation. Requests are recorded in the structured application log.

---

### Health Check
- **Endpoint:** `GET /health`
- **Response:**
  ```json
  {
    "status": "ok",
    "reranker_mode": "lexical",
    "store_backend": "qdrant",
    "collection": "admission_knowledge",
    "ollama": "connected",
    "qdrant": "connected",
    "indexed_chunks": 42
  }
  ```

---

## Running Tests

Run the full automated test suite:
```bash
pytest tests/ -v
```
All multi-tenant isolation, chunking, and core RAG tests run against an isolated in-memory Qdrant instance to guarantee zero cross-tenant data leakage.
