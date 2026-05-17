"""Standard RAG chain: retrieve → self-correct → generate answer."""

from __future__ import annotations

import time

from loguru import logger

from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompt_templates import STANDARD_RAG_PROMPT
from app.core.retrieval.query_router import classify_query
from app.core.retrieval.self_corrector import retrieve_with_self_correction
from app.core.retrieval.vector_store import get_vector_store
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk


async def run_rag_chain(request: QueryRequest) -> QueryResponse:
    """Execute the standard RAG chain for a single question.

    Flow:
    1. Optionally classify query to determine routing
    2. Retrieve chunks with self-correction
    3. Build prompt with context
    4. Generate answer via appropriate LLM model
    5. Return answer with sources, model info, and latency

    Args:
        request: The query request with question, filters, and mode.

    Returns:
        A QueryResponse with the answer, sources, model used, latency, and relevance score.
    """
    start_time = time.perf_counter()
    llm_client = get_llm_client()

    # Determine model based on mode
    model = llm_client.route_model(request.mode)

    # If no doc_ids specified and mode is auto, try to classify
    doc_ids = request.doc_ids
    if doc_ids is None and request.mode == "standard":
        try:
            vector_store = get_vector_store()
            available_docs = await vector_store.list_documents()
            route = await classify_query(request.question, available_docs)

            if route.requires_comparison:
                model = llm_client.reasoning_model

            if isinstance(route.doc_scope, list) and route.doc_scope:
                doc_ids = route.doc_scope
        except Exception as e:
            logger.warning("Query routing failed, searching all docs: {}", str(e))

    # Retrieve with self-correction
    chunks, relevance_score, query_used = await retrieve_with_self_correction(
        question=request.question,
        top_k=request.top_k,
        doc_ids=doc_ids,
        year_filter=request.year_filter,
    )

    # Build context from retrieved chunks
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks):
        payload = chunk.get("payload", {})
        text = payload.get("text", "")
        filename = payload.get("filename", "unknown")
        chunk_index = payload.get("chunk_index", i)

        context_parts.append(
            f"[Source: {filename}, Chunk {chunk_index}]\n{text}"
        )

        sources.append(
            SourceChunk(
                doc_id=payload.get("doc_id", ""),
                filename=filename,
                text=text[:500],  # Truncate for response
                chunk_index=chunk_index,
                score=chunk.get("score", 0.0),
            )
        )

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    # Generate answer
    prompt_messages = STANDARD_RAG_PROMPT.format_messages(
        context=context,
        question=request.question,
    )

    # Convert LangChain messages to a single prompt string
    system_msg = ""
    user_msg = ""
    for msg in prompt_messages:
        if msg.type == "system":
            system_msg = msg.content
        elif msg.type == "human":
            user_msg = msg.content

    answer = await llm_client.generate(
        prompt=user_msg,
        model=model,
        temperature=0.3,
        max_tokens=2048,
        system_message=system_msg,
    )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    logger.info(
        "RAG chain complete: model={}, latency={}ms, sources={}, relevance={:.2f}",
        model,
        latency_ms,
        len(sources),
        relevance_score,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        model_used=model,
        latency_ms=latency_ms,
        relevance_score=relevance_score,
    )
