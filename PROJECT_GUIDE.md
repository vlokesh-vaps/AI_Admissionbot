# AI Admission Bot Project Guide (LangChain Architecture)

This document explains the project structure, request flow, LangChain architecture, configuration, and extension points.

## Purpose

The service is a FastAPI admission chatbot with retrieval-augmented generation (RAG) built on the **LangChain framework**. It accepts admission documents from an ERP system, creates searchable embeddings, answers tenant-scoped questions, and records chat messages.

The core technology stack consists of:
- **Web Framework:** FastAPI, Uvicorn
- **AI / LLM Orchestration:** LangChain (`langchain`, `langchain-core`, `langchain-groq`, `langchain-ollama`, `langchain-qdrant`, `langchain-text-splitters`)
- **LLM Provider:** Groq (`ChatGroq` with `llama-3.3-70b-versatile`)
- **Vector Database:** Qdrant (`QdrantVectorStore`)
- **Embeddings:** Ollama (`OllamaEmbeddings` with `embeddinggemma:latest`)
- **Hybrid Retrieval:** LangChain `EnsembleRetriever` combining dense vector search and `BM25Retriever`
- **Reranking:** CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with lexical fallback
- **Database:** PostgreSQL (ERP database integration for chat messages)

---

## Document Ingestion

The ERP sends a multipart request:

```text
POST /api/ai/admission/document
MI_ID=1001
File=document.pdf
```

1. **Validation:** The service validates the tenant `MI_ID` and checks the file extension (`.pdf` or `.docx`).
2. **Storage:** The raw file is saved under `data/documents/<MI_ID>/`.
3. **Loading:** LangChain `PyPDFLoader` (or `python-docx`) extracts text and metadata into a `LoadedDocument(Document)`.
4. **Chunking:** LangChain `RecursiveCharacterTextSplitter` segments the document into `DocumentChunk(Document)` instances tagged with `metadata.mi_id` and a stable, deterministic chunk hash.
5. **Embedding & Storage:** LangChain `QdrantVectorStore` persists document chunks and indexes the `metadata.mi_id` payload field for zero cross-tenant leakage.

The upload endpoint accepts exactly `File` and `MI_ID`. The uploaded `File` is the only accepted document source. `FilePath` is not used or downloaded. Requests without `File` are rejected.

---

## Chat Flow

1. The client sends `conversation_id`, `question`, and `mi_id` to `POST /api/chat`.
2. **Tenant-Filtered Retrieval:** `HybridRetriever` uses LangChain's `EnsembleRetriever`:
   - Dense retrieval via `QdrantVectorStore.as_retriever(search_kwargs={"filter": tenant_filter})`
   - Keyword retrieval via `BM25Retriever.from_documents(tenant_docs)`
3. **Reranking:** `Reranker` scores candidate `Document` objects using a local cross-encoder model or lexical overlap fallback.
4. **Context Construction:** Top reranked passages are formatted into structured citations.
5. **Generation via ChatGroq:** `ChatPromptTemplate` compiles the system prompt, retrieved context, conversation history, and user query into messages for `ChatGroq`.
6. **Structured Parsing & Escalation:** The response is parsed for `ANSWER`, `ESCALATE`, and `REASON`. Counselor escalation is triggered automatically if the query explicitly asks for a counselor or if context is missing.
7. **Memory & Persistence:**
   - Active memory is recorded via `InMemoryChatMessageHistory`.
   - Message turns are archived to the ERP `AI_Admission_Conversation` PostgreSQL table if configured.

Every retrieval operation strictly enforces the requested `MI_ID` to guarantee institution-level multi-tenant isolation.

---

## Database Tables

The service integrates with existing ERP tables and does not create tables automatically.

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

---

## Code Map

```text
app.py                    FastAPI server entry point
main.py                   Unified CLI entry point (api, chat, build-kb)
src/config.py             Environment-backed settings
src/api/app.py            Routes and request orchestration
src/api/schemas.py        Pydantic request and response models
src/storage/postgres.py   ERP PostgreSQL access
src/storage/vector_db.py  LangChain QdrantVectorStore & OllamaEmbeddings access
src/storage/cache.py      File-backed response cache
src/ingestion/loaders.py  LangChain PyPDFLoader and python-docx extraction (LoadedDocument)
src/ingestion/chunking.py LangChain RecursiveCharacterTextSplitter with tenant tagging
src/core/retrievers.py    LangChain EnsembleRetriever (Qdrant + BM25)
src/core/reranker.py      Candidate reranking (Document & dict support)
src/core/rag.py           LangChain ChatGroq generation and escalation logic
src/core/prompts.py       LangChain ChatPromptTemplate prompt construction
src/core/conversation.py  LangChain InMemoryChatMessageHistory conversation state
scripts/build_kb.py       Batch knowledge-base rebuild script
scripts/cli_chat.py       Terminal chat client
static/index.html         Browser client
tests/                    Automated tests (pytest)
```

---

## Multi-Tenant Isolation Flow (`MI_ID`)

Multi-tenant isolation is enforced at every layer of the architecture:

1. **API receives and sanitizes `mi_id`:**
   ```python
   # src/api/app.py
   target_mi_id = str(request.mi_id or settings.default_mi_id).strip()
   ```

2. **RAG pipeline binds conversation and retrieval to `mi_id`:**
   ```python
   # src/core/rag.py
   conversation = self.conversations.get_or_create(f"{target_mi_id}:{conversation_id}")
   candidates = self.retriever.search(question, mi_id=target_mi_id)
   ```

3. **Hybrid retriever filters Qdrant and BM25 to tenant:**
   ```python
   # src/core/retrievers.py
   tenant_filter = models.Filter(
       must=[
           models.FieldCondition(
               key="metadata.mi_id",
               match=models.MatchValue(value=mi_id_str),
           )
       ]
   )
   qdrant_retriever = self.store.langchain_store.as_retriever(
       search_kwargs={"k": limit, "filter": tenant_filter}
   )

   tenant_docs = [
       doc for doc in self._docs
       if doc.metadata.get("mi_id", "").strip() == mi_id_str
   ]
   bm25_retriever = BM25Retriever.from_documents(
       tenant_docs, k=limit, preprocess_func=_tokens
   )

   ensemble = EnsembleRetriever(
       retrievers=[qdrant_retriever, bm25_retriever],
       weights=[self.settings.hybrid_vector_weight, self.settings.hybrid_bm25_weight],
   )
   ```

4. **Qdrant Vector DB enforces indexed payload filter:**
   ```python
   # src/storage/vector_db.py
   self.qdrant_client.create_payload_index(
       collection_name=self.collection_name,
       field_name="metadata.mi_id",
       field_schema=models.PayloadSchemaType.KEYWORD,
   )
   ```

5. **Every indexed chunk is tagged at chunking time:**
   ```python
   # src/ingestion/chunking.py
   chunk.metadata.update({
       "mi_id": mi_id_str,
       "chunk_index": index,
       "chunk_id": chunk_id,
   })
   ```

---

## Configuration

Copy `.env.example` to `.env` and set real local values:

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

Never commit `.env` or place credentials in source code or documentation.

When changing the embedding model, rebuild the Qdrant knowledge base. Do not mix vectors produced by different embedding models.

---

## Development & Testing Commands

```bash
python main.py                 # Start the API on port 5003
python main.py chat            # Start the interactive terminal chat
python main.py build-kb        # Rebuild the knowledge base
pytest tests/ -v               # Run full automated test suite (20/20 passing)
```

The Web UI is available at `http://localhost:5003`; interactive OpenAPI documentation is available at `http://localhost:5003/docs`.