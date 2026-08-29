import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import settings
from src.core.rag import AdmissionRAG
from src.core.reranker import Reranker
from src.core.retrievers import _normalize
from src.ingestion.chunking import chunk_document
from src.ingestion.loaders import LoadedDocument
from src.storage.cache import ResponseCache


def test_chunk_document_creates_stable_chunks() -> None:
    document = LoadedDocument(Path("guide.docx"), "guide", "First paragraph.\n\nSecond paragraph.")
    chunks = chunk_document(document, mi_id="1001", max_chars=30, overlap_chars=5)
    assert chunks
    assert chunks[0].metadata["source"] == "guide.docx"
    assert chunks[0].metadata["mi_id"] == "1001"
    assert chunks[0].chunk_id.startswith("1001:guide.docx:0:")


def test_normalize_accepts_numpy_scores() -> None:
    assert _normalize(np.array([1.0, 2.0, 3.0])) == [0.0, 0.5, 1.0]


def test_explicit_escalation_detection() -> None:
    assert AdmissionRAG._explicit_escalation_reason("I want to talk to a counselor")
    assert AdmissionRAG._explicit_escalation_reason("What is the fee?") is None


def test_response_parser_extracts_escalation() -> None:
    answer, escalate, reason = AdmissionRAG._parse_response(
        "ANSWER: Please contact the admissions office.\nESCALATE: true\nREASON: Applicant requested a counselor."
    )
    assert answer == "Please contact the admissions office."
    assert escalate is True
    assert reason == "Applicant requested a counselor."


def test_response_parser_removes_contract_fields_from_answer() -> None:
    answer, escalate, reason = AdmissionRAG._parse_response(
        "ANSWER: Certainly, I can help.\nESCALATE: false\nREASON: none"
    )
    assert answer == "Certainly, I can help."
    assert escalate is False
    assert reason is None


def test_cache_round_trip() -> None:
    cache_dir = settings.cache_dir / "test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(cache_dir, ttl_seconds=60)
    value = {"answer": "Admission details", "sources": []}
    cache.set("conversation-1", "What are the fees?", value)
    assert cache.get("conversation-1", "What are the fees?") == value
    cache.path.unlink(missing_ok=True)


def test_cache_namespace_separates_values() -> None:
    cache_dir = settings.cache_dir / "test_cache_namespace"
    cache_dir.mkdir(parents=True, exist_ok=True)
    first = ResponseCache(cache_dir, ttl_seconds=60, namespace="model-a")
    first.set("conversation-1", "What are the fees?", {"answer": "old"})
    second = ResponseCache(cache_dir, ttl_seconds=60, namespace="model-b")
    assert second.get("conversation-1", "What are the fees?") is None
    first.path.unlink(missing_ok=True)


def test_reranker_uses_lexical_fallback_without_local_model() -> None:
    local_settings = replace(
        settings,
        reranker_model="missing-local-reranker",
    )
    reranker = Reranker(local_settings)
    assert reranker.mode == "lexical"
    ranked = reranker.rerank(
        "fee admission",
        [
            {"id": "a", "text": "library hours", "metadata": {}, "score": 0.1},
            {"id": "b", "text": "admission fee details", "metadata": {}, "score": 0.1},
        ],
        limit=1,
    )
    assert ranked[0]["id"] == "b"


def test_response_parser_handles_plain_text() -> None:
    """When Groq returns plain text without ANSWER/ESCALATE markers."""
    answer, escalate, reason = AdmissionRAG._parse_response(
        "The admission fee is Rs. 50,000 per semester."
    )
    assert answer == "The admission fee is Rs. 50,000 per semester."
    assert escalate is False
