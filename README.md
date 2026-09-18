# AI Admission Bot & Multi-Tenant Ingestion Service (LangChain Powered)

A high-performance, RAG-powered (Retrieval-Augmented Generation) Admission Assistant chatbot and document ingestion service built with the **LangChain Framework**, **FastAPI**, **Qdrant Vector Database**, **Ollama Embeddings**, **Groq LLM (`ChatGroq`)**, and **LangChain Hybrid (`EnsembleRetriever` + `BM25Retriever`) Search**.

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
│ 3. Load via LangChain PyPDFLoader / DOCX parser        │
│ 4. Chunk with LangChain RecursiveCharacterTextSplitter │
│    - Tag metadata: mi_id="1001", stable chunk_id       │
│ 5. Embed with LangChain OllamaEmbeddings               │
│ 6. Upsert to LangChain QdrantVectorStore:              │
│    - metadata.mi_id: "1001" (Indexed keyword field)    │
│    - metadata.source: "document.pdf"                   │
│    - page_content: chunk text                          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Qdrant Vector DB                      │
│  Collection: admission_knowledge                       │
│  Payload Index: metadata.mi_id (keyword)               │
│                                                        │
│  MANDATORY ENSEMBLE RETRIEVER FILTER:                  │
│  - Qdrant Vector Retriever (Filter: metadata.mi_id)    │
│  - BM25Retriever (Tenant documents cache)              │
│  ==> ZERO CROSS-TENANT DATA LEAKAGE GUARANTEED         │
└────────────────────────────────────────────────────────┘
```

---

## Key Features

- **LangChain Core Framework:** Standardized pipeline built on modern LangChain abstractions (`ChatGroq`, `QdrantVectorStore`, `OllamaEmbeddings`, `EnsembleRetriever`, `RecursiveCharacterTextSplitter`, `ChatPromptTemplate`, and `InMemoryChatMessageHistory`).
- **Strict Multi-Tenant Isolation:** Documents and vector embeddings are partitioned by `MI_ID`. All semantic searches and BM25 retrievals strictly enforce institution filtering (`metadata.mi_id`).
- **.NET ERP Integration:** Direct API endpoint `POST /api/ai/admission/document` accepts tenant `MI_ID` and PDF/DOCX files.
- **Ollama Dense Embeddings:** Local embedding generation via `langchain_ollama.OllamaEmbeddings` (`embeddinggemma:latest` by default; configurable).
- **Qdrant Vector Store:** Fast, scalable vector retrieval powered by `langchain_qdrant.QdrantVectorStore` with payload indexing on tenant IDs.
- **Hybrid Ensemble Retrieval:** Weighted ensemble combining `QdrantVectorStore.as_retriever()` and `BM25Retriever.from_documents()` with local cross-encoder reranking.
- **Interactive Web Interface:** Modern UI served at `/` on port **5003**.
- **Counselor Escalation:** Structured detection and workflow for admission counselor follow-up.
- **High-Speed Response Caching:** Tenant-scoped response cache with configurable TTL.
- **Robust Offline Resilience:** Graceful startup even if Ollama is temporarily offline.

---

## Project Structure

```
AI_Admissionbot/
├── app.py                      # FastAPI server entry point (Port 5003)
├── main.py                     # CLI command hub (api, chat, build-kb)
├── Dockerfile                  # Container definition (Exposed on 5003)
├── docker-compose.yml          # Admission Bot (5003) + Ollama (11434); Qdrant may run externally
├── requirements.txt            # Core dependencies (including LangChain packages)
├── .env.example                # Environment variables template
├── .env                        # Local runtime environment
├── data/                       # Local data storage
│   ├── cache/                  # Response cache directory
│   ├── documents/              # Multi-tenant documents: data/documents/{MI_ID}/
│   ├── logs/                   # Structured event logs
│   ├── models/                 # Model cache directory
│   └── qdrant_db/              # Local embedded Qdrant database
├── scripts/
│   ├── build_kb.py             # Batch knowledge base indexing script (LangChain-based)
│   └── cli_chat.py             # Interactive terminal chat client
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI endpoints (/api/ai/admission/document, /api/chat)
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── core/
│   │   ├── conversation.py     # LangChain InMemoryChatMessageHistory conversation store
│   │   ├── prompts.py          # LangChain ChatPromptTemplate definitions
│   │   ├── rag.py              # Core RAG orchestration pipeline (ChatGroq)
│   │   ├── reranker.py         # Cross-encoder / lexical reranker
│   │   └── retrievers.py       # LangChain EnsembleRetriever (Qdrant + BM25)
│   ├── ingestion/
│   │   ├── chunking.py         # LangChain RecursiveCharacterTextSplitter with MI_ID metadata
│   │   └── loaders.py          # LangChain PyPDFLoader and python-docx parsers
│   ├── storage/
│   │   ├── cache.py            # Local response cache
│   │   └── vector_db.py        # LangChain QdrantVectorStore with mandatory MI_ID filtering
│   ├── utils/
│   │   └── logging.py          # Structured event logging
│   └── config.py               # Settings and configuration management
├── static/
│   └── index.html              # Frontend Web UI
└── tests/
    ├── conftest.py             # Test fixtures and environment setup
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
   *(Installs FastAPI, LangChain, langchain-groq, langchain-ollama, langchain-qdrant, langchain-community, rank-bm25, and supporting packages)*

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   PORT=5003
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_EMBED_MODEL=embeddinggemma:latest
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=admission_knowledge
   QDRANT_ALLOW_IN_MEMORY=false
   ```

---

### 3. Running the Service on Port 5003

#### Option A: Run directly with the command hub
```bash
python main.py api
```
Or simply:
```bash
python main.py
```
The API and Web UI will be available at:
`http://localhost:5003`

#### Option B: Run the chatbot and Ollama with Docker Compose
```bash
docker compose up -d --build ollama
docker compose up -d --build --no-deps admission-chatbot
```
This spins up:
- **Admission Bot API** on port `5003`
- **Ollama Server** on port `11434`

Qdrant is a required dependency but may run as an existing external container. The chatbot must be able to resolve and reach the host configured in `QDRANT_URL`:

```env
# Use this when the existing Qdrant container is on the same Docker network.
QDRANT_URL=http://qdrant:6333
QDRANT_ALLOW_IN_MEMORY=false
```

If Qdrant is not on the same Docker network, connect both containers to a shared network or set `QDRANT_URL` to a reachable host address. Do not enable in-memory Qdrant in production; it loses indexed data when the chatbot container restarts.

To start the bundled Qdrant service instead, run:

```bash
docker compose up -d --build
```

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
    "stored_path": "qdrant://admission_knowledge/1001/student_document.pdf",
    "indexed_chunks": 8,
    "message": "Document successfully processed and indexed into Qdrant knowledge base without storing raw files on local disk."
  }
  ```

Returns `415` for unsupported file types, `413` for files over the configured limit, and `422` when document indexing fails.

The uploaded `File` is the only accepted document source. `FilePath` is retained as ERP metadata and is never downloaded by this service.

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

The response contains `answer`, `sources`, `escalate`, `escalation_reason`, and a `cached` flag, together with the conversation and tenant IDs. Vector chunks and responses are scoped strictly to the tenant `mi_id`.

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

All 20 test cases (multi-tenant isolation, chunking, core RAG, document loaders, API endpoints, and web UI) run against an isolated in-memory Qdrant instance to guarantee zero cross-tenant data leakage and 100% test reliability.
