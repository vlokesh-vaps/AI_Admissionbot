"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the admission chatbot."""

    project_root: Path = PROJECT_ROOT
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    log_dir: Path = PROJECT_ROOT / "data" / "logs"

    # Vector DB & Multi-Tenancy (Qdrant in Docker)
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "admission_knowledge")
    default_mi_id: str = os.getenv("DEFAULT_MI_ID", "1001")

    # Embeddings (Ollama) & Generation (Groq)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_timeout_seconds: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))

    # Retrieval & Reranking
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "12"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    hybrid_vector_weight: float = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.65"))
    hybrid_bm25_weight: float = float(os.getenv("HYBRID_BM25_WEIGHT", "0.35"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "18000"))

    # Server, Security & Limits
    api_host: str = os.getenv("HOST", os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = int(os.getenv("PORT", os.getenv("API_PORT", "5003")))
    allowed_origins: tuple[str, ...] = _csv(
        os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://localhost:5003")
    )
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def tenant_documents_dir(self, mi_id: str | int) -> Path:
        """Get or create the tenant-isolated directory for raw uploaded documents."""
        path = self.documents_dir / str(mi_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (
            self.documents_dir,
            self.cache_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
