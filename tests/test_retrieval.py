"""Tests for retrieval: embedder, vector store, query router, and self-corrector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.retrieval.query_router import _heuristic_classify


# ── Embedder Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.core.retrieval.embedder._embed_content_sync")
async def test_embedder_returns_correct_dimension(mock_embed: MagicMock) -> None:
    """Test that embedder returns vectors with dimension 3072."""
    # Mock Gemini embedding response
    mock_vector = [0.1] * 3072
    mock_embed.return_value = [mock_vector]

    from app.core.retrieval.embedder import embed_query

    result = await embed_query("test query")
    assert len(result) == 3072
    mock_embed.assert_called_once()


@pytest.mark.asyncio
@patch("app.core.retrieval.embedder._embed_content_sync")
async def test_embedder_batches_chunks(mock_embed: MagicMock) -> None:
    """Test that embedder batches chunks in groups of 20."""
    mock_vector = [0.1] * 3072
    # Return correct number of embeddings for each batch
    mock_embed.side_effect = lambda texts, task: [mock_vector] * len(texts)

    from app.core.retrieval.embedder import embed_chunks

    texts = [f"chunk {i}" for i in range(45)]
    results = await embed_chunks(texts)

    assert len(results) == 45
    assert mock_embed.call_count == 3  # 20 + 20 + 5 = 3 batches


@pytest.mark.asyncio
@patch("app.core.retrieval.embedder._embed_content_sync")
async def test_embed_query_uses_retrieval_query_task(mock_embed: MagicMock) -> None:
    """Test that embed_query uses 'retrieval_query' task type."""
    mock_embed.return_value = [[0.1] * 3072]

    from app.core.retrieval.embedder import embed_query

    await embed_query("test query")
    mock_embed.assert_called_with(["test query"], "retrieval_query")


# ── Vector Store Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.core.retrieval.vector_store.QdrantClient")
async def test_vector_store_upsert_and_search(mock_qdrant_cls: MagicMock) -> None:
    """Test that vector store can upsert chunks and search."""
    mock_client = MagicMock()
    mock_qdrant_cls.return_value = mock_client

    # Mock collections check
    mock_collection = MagicMock()
    mock_collection.name = "academic_papers"
    mock_client.get_collections.return_value.collections = [mock_collection]

    from app.core.retrieval.vector_store import VectorStore

    # Reset singleton for testing
    import app.core.retrieval.vector_store as vs_module
    vs_module._vector_store = None

    store = VectorStore()

    # Test upsert
    chunks = ["chunk 1", "chunk 2"]
    embeddings = [[0.1] * 3072, [0.2] * 3072]
    metadata = {
        "doc_id": "test-123",
        "filename": "test.pdf",
        "doc_type": "research_paper",
        "source": "test",
        "pub_year": 2023,
        "ingested_at": "2024-01-01T00:00:00Z",
    }

    count = await store.upsert_chunks(chunks, embeddings, metadata)
    assert count == 2
    mock_client.upsert.assert_called_once()


# ── Query Router Tests ────────────────────────────────────────────────


def test_query_router_classifies_compare_query() -> None:
    """Test that heuristic classifier detects comparison queries."""
    route = _heuristic_classify("Compare the methodologies used in these two papers")
    assert route.requires_comparison is True
    assert route.mode == "reasoning"


def test_query_router_classifies_standard_query() -> None:
    """Test that heuristic classifier identifies standard queries."""
    route = _heuristic_classify("What dataset was used in this study?")
    assert route.requires_comparison is False
    assert route.mode == "standard"


def test_query_router_classifies_reasoning_query() -> None:
    """Test that heuristic classifier detects reasoning queries."""
    route = _heuristic_classify("Why does the author claim that transformers are better?")
    assert route.mode == "reasoning"


def test_query_router_detects_contrast_keyword() -> None:
    """Test that heuristic classifier detects 'contrast' keyword."""
    route = _heuristic_classify("Contrast the results from experiment A and B")
    assert route.requires_comparison is True


def test_query_router_detects_versus_keyword() -> None:
    """Test that heuristic classifier detects 'versus' keyword."""
    route = _heuristic_classify("Method X versus Method Y performance")
    assert route.requires_comparison is True


# ── Self-Corrector Tests ─────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.core.retrieval.self_corrector.get_llm_client")
async def test_self_corrector_rewrites_bad_query(mock_get_llm: MagicMock) -> None:
    """Test that self-corrector rewrites query when relevance is low."""
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    # Mock generate for rewrite
    mock_llm.generate = AsyncMock(
        return_value="What specific neural network architecture is proposed for NLP tasks?"
    )
    mock_llm.primary_model = "llama-3.3-70b-versatile"

    from app.core.retrieval.self_corrector import rewrite_query

    rewritten = await rewrite_query(
        "tell me about the thing",
        "Query is too vague",
    )

    assert len(rewritten) > 0
    assert rewritten != "tell me about the thing"
    mock_llm.generate.assert_called_once()
