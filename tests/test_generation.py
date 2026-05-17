"""Tests for the generation layer: LLM client, RAG chain, and cross-doc chain."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_llm_client_routes_model_correctly() -> None:
    """Test that LLM client routes to correct model based on question type."""
    with patch("app.core.generation.llm_client.Groq"):
        from app.core.generation.llm_client import LLMClient
        import app.core.generation.llm_client as llm_module
        llm_module._llm_client = None

        client = LLMClient()
        assert client.route_model("standard") == "llama-3.3-70b-versatile"
        assert client.route_model("reasoning") == "deepseek-r1-distill-llama-70b"
        assert client.route_model("compare") == "deepseek-r1-distill-llama-70b"


def test_llm_client_json_extraction() -> None:
    """Test JSON extraction from various response formats."""
    from app.core.generation.llm_client import _extract_json

    result = _extract_json('{"key": "value"}')
    assert result == {"key": "value"}

    result = _extract_json('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}

    result = _extract_json('Here is the result: {"score": 0.85}')
    assert result["score"] == 0.85


def test_llm_client_json_extraction_fails_gracefully() -> None:
    """Test that JSON extraction raises ValueError for invalid input."""
    from app.core.generation.llm_client import _extract_json
    with pytest.raises(ValueError, match="Could not extract valid JSON"):
        _extract_json("This is not JSON at all")


@pytest.mark.asyncio
@patch("app.core.generation.rag_chain.get_llm_client")
@patch("app.core.generation.rag_chain.retrieve_with_self_correction")
@patch("app.core.generation.rag_chain.get_vector_store")
async def test_standard_rag_chain_returns_answer(
    mock_vs: MagicMock, mock_retrieve: MagicMock, mock_llm: MagicMock,
) -> None:
    """Test that RAG chain returns a properly structured response."""
    mock_retrieve.return_value = (
        [{"id": "1", "score": 0.9, "payload": {
            "text": "ML is a subset of AI.", "doc_id": "doc-1",
            "filename": "paper.pdf", "chunk_index": 0,
        }}], 0.85, "What is ML?",
    )
    mock_client = MagicMock()
    mock_client.route_model.return_value = "llama-3.3-70b-versatile"
    mock_client.generate = AsyncMock(return_value="ML is a subset of AI.")
    mock_client.primary_model = "llama-3.3-70b-versatile"
    mock_client.reasoning_model = "deepseek-r1-distill-llama-70b"
    mock_llm.return_value = mock_client
    mock_store = MagicMock()
    mock_store.list_documents = AsyncMock(return_value=[])
    mock_vs.return_value = mock_store

    from app.core.generation.rag_chain import run_rag_chain
    from app.models.schemas import QueryRequest
    request = QueryRequest(question="What is ML?")
    response = await run_rag_chain(request)

    assert response.answer is not None
    assert response.model_used == "llama-3.3-70b-versatile"
    assert len(response.sources) == 1


@pytest.mark.asyncio
@patch("app.core.generation.cross_doc_chain.get_llm_client")
@patch("app.core.generation.cross_doc_chain.get_vector_store")
@patch("app.core.generation.cross_doc_chain.embed_query")
async def test_cross_doc_chain_returns_structured_output(
    mock_embed: MagicMock, mock_vs: MagicMock, mock_llm: MagicMock,
) -> None:
    """Test that cross-doc chain returns comparison with agreements."""
    mock_embed.return_value = [0.1] * 3072
    mock_store = MagicMock()
    mock_store.search = AsyncMock(side_effect=[
        [{"id": "1", "score": 0.9, "payload": {
            "text": "Paper A uses CNN.", "doc_id": "doc-1",
            "filename": "a.pdf", "chunk_index": 0}}],
        [{"id": "2", "score": 0.85, "payload": {
            "text": "Paper B uses RNN.", "doc_id": "doc-2",
            "filename": "b.pdf", "chunk_index": 0}}],
    ])
    mock_vs.return_value = mock_store
    mock_client = MagicMock()
    mock_client.reasoning_model = "deepseek-r1-distill-llama-70b"
    mock_client.generate = AsyncMock(return_value=(
        "Comparison text\n"
        'AGREEMENTS_JSON: ["Both classify"]\n'
        'CONTRADICTIONS_JSON: ["Different arch"]'
    ))
    mock_llm.return_value = mock_client

    from app.core.generation.cross_doc_chain import run_cross_doc_chain
    from app.models.schemas import CompareRequest
    request = CompareRequest(question="Compare", doc_ids=["doc-1", "doc-2"], aspect="methodology")
    response = await run_cross_doc_chain(request)

    assert response.comparison is not None
    assert response.model_used == "deepseek-r1-distill-llama-70b"
    assert mock_store.search.call_count == 2
