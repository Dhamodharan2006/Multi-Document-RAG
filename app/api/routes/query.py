"""Query API endpoints: standard RAG query and cross-document comparison."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.core.generation.cross_doc_chain import run_cross_doc_chain
from app.core.generation.rag_chain import run_rag_chain
from app.models.schemas import (
    CompareRequest,
    CompareResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Execute a standard RAG query.

    Embeds the question, retrieves relevant chunks with self-correction,
    and generates an answer using the appropriate LLM model.

    Args:
        request: QueryRequest with question, optional filters, and mode.

    Returns:
        QueryResponse with answer, sources, model info, latency, and relevance score.
    """
    logger.info(
        "Query request: question='{}', mode={}, top_k={}, doc_ids={}",
        request.question[:60],
        request.mode,
        request.top_k,
        request.doc_ids,
    )

    try:
        response = await run_rag_chain(request)
        return response

    except Exception as e:
        logger.error("Query failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query/compare", response_model=CompareResponse)
async def compare_documents(request: CompareRequest) -> CompareResponse:
    """Execute a cross-document comparison query.

    Retrieves chunks from each document separately and generates
    a structured comparison using the reasoning model.

    Args:
        request: CompareRequest with question, doc_ids (≥2), and aspect.

    Returns:
        CompareResponse with comparison, agreements, contradictions, and metadata.
    """
    logger.info(
        "Compare request: question='{}', doc_ids={}, aspect={}",
        request.question[:60],
        request.doc_ids,
        request.aspect,
    )

    if len(request.doc_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 doc_ids are required for comparison",
        )

    try:
        response = await run_cross_doc_chain(request)
        return response

    except Exception as e:
        logger.error("Comparison failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
