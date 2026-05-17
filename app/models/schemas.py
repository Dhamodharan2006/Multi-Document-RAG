"""Pydantic v2 request/response schemas for all API endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Ingestion Schemas ──────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request body for single-document ingestion."""
    source: str = "upload"


class IngestResult(BaseModel):
    """Result of a single document ingestion."""
    doc_id: str
    filename: str
    chunks_created: int
    metadata: dict
    status: str


class BatchIngestResponse(BaseModel):
    """Response for batch ingestion endpoint."""
    results: list[IngestResult]
    total_chunks: int
    failed: list[str]


# ── Document Schemas ───────────────────────────────────────────────

class DocumentMeta(BaseModel):
    """Metadata for an ingested document."""
    doc_id: str
    filename: str
    doc_type: str
    source: str
    pub_year: int | None = None
    chunk_count: int
    ingested_at: str


class DocumentListResponse(BaseModel):
    """Response for listing all documents."""
    documents: list[DocumentMeta]
    total_count: int


class DeleteDocumentResponse(BaseModel):
    """Response for document deletion."""
    doc_id: str
    deleted_chunks: int
    status: str


# ── Query Schemas ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for standard RAG query."""
    question: str
    doc_ids: list[str] | None = None
    mode: Literal["standard", "reasoning"] = "standard"
    top_k: int = Field(default=5, ge=1, le=20)
    year_filter: int | None = None


class SourceChunk(BaseModel):
    """A single source chunk returned with query results."""
    doc_id: str
    filename: str
    text: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    """Response for standard RAG query."""
    answer: str
    sources: list[SourceChunk]
    model_used: str
    latency_ms: int
    relevance_score: float


class CompareRequest(BaseModel):
    """Request body for cross-document comparison."""
    question: str
    doc_ids: list[str] = Field(..., min_length=2)
    aspect: str = "general"


class DocSummary(BaseModel):
    """Summary of a single document used in comparison."""
    doc_id: str
    filename: str
    summary: str


class CompareResponse(BaseModel):
    """Response for cross-document comparison."""
    comparison: str
    doc_summaries: list[DocSummary] = []
    agreements: list[str]
    contradictions: list[str]
    model_used: str
    latency_ms: int


# ── Evaluation Schemas ─────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    """Request body for RAGAS evaluation."""
    questions: list[str] | None = None
    doc_ids: list[str] | None = None
    sample_size: int = Field(default=10, ge=1, le=100)


class RAGASMetrics(BaseModel):
    """RAGAS evaluation metric results."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float
    eval_id: str
    timestamp: str
    question_count: int


class EvaluateResponse(BaseModel):
    """Response for evaluation initiation."""
    eval_id: str
    status: str
    metrics: RAGASMetrics | None = None


# ── Health Schemas ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    qdrant_connected: bool
    groq_connected: bool
    timestamp: str


# ── Internal Schemas ───────────────────────────────────────────────

class QueryRoute(BaseModel):
    """Internal schema for query routing results."""
    doc_scope: list[str] | str = "all"
    mode: Literal["standard", "reasoning"] = "standard"
    requires_comparison: bool = False
    reasoning: str = ""
