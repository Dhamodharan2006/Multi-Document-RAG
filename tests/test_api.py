"""Tests for FastAPI API endpoints using TestClient with mocked services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client with mocked startup."""
    with patch("app.main.get_vector_store") as mock_vs:
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.check_connection.return_value = True
        mock_vs.return_value = mock_store

        from app.main import app
        return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Test the health check endpoint returns proper status."""
    with patch("app.main.get_vector_store") as mock_vs, \
         patch("app.main.get_llm_client") as mock_llm:
        mock_store = MagicMock()
        mock_store.check_connection.return_value = True
        mock_vs.return_value = mock_store

        mock_client = MagicMock()
        mock_client.check_connection.return_value = True
        mock_llm.return_value = mock_client

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "qdrant_connected" in data
        assert "groq_connected" in data
        assert "timestamp" in data


def test_root_endpoint(client: TestClient) -> None:
    """Test the root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Multi-Document RAG System"
    assert data["version"] == "1.0.0"


@patch("app.api.routes.ingest.run_ingestion_pipeline")
def test_ingest_pdf_endpoint(mock_pipeline: MagicMock, client: TestClient) -> None:
    """Test the ingest endpoint with a mock PDF file."""
    from app.models.schemas import IngestResult
    mock_pipeline.return_value = IngestResult(
        doc_id="test-doc-id",
        filename="test.pdf",
        chunks_created=10,
        metadata={"doc_type": "research_paper", "source": "test"},
        status="success",
    )

    import io
    file_content = b"%PDF-1.4 fake pdf content"
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
        data={"source": "test"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "test-doc-id"
    assert data["chunks_created"] == 10
    assert data["status"] == "success"


def test_ingest_rejects_unsupported_format(client: TestClient) -> None:
    """Test that ingest rejects unsupported file formats."""
    import io
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.xlsx", io.BytesIO(b"data"), "application/xlsx")},
        data={"source": "test"},
    )
    assert response.status_code == 400


@patch("app.api.routes.query.run_rag_chain")
def test_query_endpoint(mock_chain: MagicMock, client: TestClient) -> None:
    """Test the query endpoint with a mock RAG chain response."""
    from app.models.schemas import QueryResponse, SourceChunk
    mock_chain.return_value = QueryResponse(
        answer="Machine learning is a subset of AI.",
        sources=[SourceChunk(
            doc_id="doc-1", filename="paper.pdf",
            text="ML text", chunk_index=0, score=0.9,
        )],
        model_used="llama-3.3-70b-versatile",
        latency_ms=500,
        relevance_score=0.85,
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "What is ML?", "mode": "standard", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["model_used"] == "llama-3.3-70b-versatile"


@patch("app.api.routes.query.run_cross_doc_chain")
def test_compare_endpoint(mock_chain: MagicMock, client: TestClient) -> None:
    """Test the compare endpoint with a mock cross-doc chain response."""
    from app.models.schemas import CompareResponse
    mock_chain.return_value = CompareResponse(
        comparison="Paper A uses CNN, Paper B uses RNN.",
        agreements=["Both classify images"],
        contradictions=["Different architectures"],
        model_used="deepseek-r1-distill-llama-70b",
        latency_ms=1200,
    )

    response = client.post(
        "/api/v1/query/compare",
        json={
            "question": "Compare approaches",
            "doc_ids": ["doc-1", "doc-2"],
            "aspect": "methodology",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    assert "agreements" in data
    assert "contradictions" in data


def test_compare_rejects_single_doc(client: TestClient) -> None:
    """Test that compare rejects requests with less than 2 doc_ids."""
    response = client.post(
        "/api/v1/query/compare",
        json={"question": "Compare", "doc_ids": ["doc-1"], "aspect": "general"},
    )
    assert response.status_code == 422  # Pydantic validation


@patch("app.api.routes.evaluate.run_ragas_evaluation")
def test_evaluate_endpoint(mock_eval: MagicMock, client: TestClient) -> None:
    """Test the evaluate endpoint with a mock RAGAS result."""
    from app.models.schemas import RAGASMetrics
    mock_eval.return_value = RAGASMetrics(
        faithfulness=0.85,
        answer_relevancy=0.90,
        context_precision=0.80,
        context_recall=0.75,
        answer_correctness=0.82,
        eval_id="eval-test123",
        timestamp="2024-01-01T00:00:00Z",
        question_count=5,
    )

    response = client.post(
        "/api/v1/evaluate",
        json={"sample_size": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["metrics"]["faithfulness"] == 0.85
