"""Self-correction module: relevance checking and query rewriting."""

from __future__ import annotations

from loguru import logger

from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompt_templates import (
    QUERY_REWRITE_PROMPT,
    RELEVANCE_CHECK_PROMPT,
)
from app.core.retrieval.embedder import embed_query
from app.core.retrieval.vector_store import get_vector_store


async def check_relevance(
    question: str,
    retrieved_chunks: list[dict],
) -> tuple[float, str]:
    """Check the relevance of retrieved chunks to the question.

    Uses llama-3.3-70b-versatile to score how well the retrieved
    chunks answer the given question.

    Args:
        question: The user's question.
        retrieved_chunks: List of chunk dicts with 'payload' containing 'text'.

    Returns:
        A tuple of (relevance_score: 0.0-1.0, reason: str).
    """
    llm_client = get_llm_client()

    # Build context from chunks
    chunk_texts = []
    for i, chunk in enumerate(retrieved_chunks[:5]):
        text = chunk.get("payload", {}).get("text", "")
        chunk_texts.append(f"[Chunk {i + 1}]: {text[:300]}")

    context = "\n\n".join(chunk_texts)

    prompt = RELEVANCE_CHECK_PROMPT.format(
        question=question,
        context=context,
    )

    try:
        result = await llm_client.generate_json(
            prompt=prompt,
            model=llm_client.primary_model,
        )

        score = float(result.get("relevance_score", 0.5))
        reason = result.get("reason", "No reason provided")

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        logger.info(
            "Relevance check: score={:.2f}, reason={}",
            score,
            reason[:100],
        )
        return score, reason

    except Exception as e:
        logger.warning("Relevance check failed: {}. Defaulting to 0.5", str(e))
        return 0.5, f"Relevance check failed: {str(e)}"


async def rewrite_query(
    original_question: str,
    reason: str,
) -> str:
    """Rewrite a query to improve retrieval results.

    Uses llama-3.3-70b-versatile to generate a more specific,
    academically precise version of the query.

    Args:
        original_question: The original user question.
        reason: The reason the original query had low relevance.

    Returns:
        A rewritten query string.
    """
    llm_client = get_llm_client()

    prompt = QUERY_REWRITE_PROMPT.format(
        question=original_question,
        reason=reason,
    )

    try:
        rewritten = await llm_client.generate(
            prompt=prompt,
            model=llm_client.primary_model,
            temperature=0.3,
            max_tokens=200,
        )

        rewritten = rewritten.strip().strip('"').strip("'")
        logger.info(
            "Query rewritten: '{}' → '{}'",
            original_question[:60],
            rewritten[:60],
        )
        return rewritten

    except Exception as e:
        logger.warning("Query rewrite failed: {}. Using original.", str(e))
        return original_question


async def retrieve_with_self_correction(
    question: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    doc_type: str | None = None,
    year_filter: int | None = None,
) -> tuple[list[dict], float, str]:
    """Retrieve chunks with automatic self-correction.

    Flow:
    1. Embed the query and retrieve top-k chunks
    2. Check relevance score via LLM
    3. If score < 0.5: rewrite query and retrieve again
    4. Return best results

    Args:
        question: The user's question.
        top_k: Number of chunks to retrieve.
        doc_ids: Optional document ID filter.
        doc_type: Optional document type filter.
        year_filter: Optional year filter.

    Returns:
        Tuple of (retrieved_chunks, relevance_score, query_used).
    """
    vector_store = get_vector_store()

    # Step 1: Initial retrieval
    query_vector = await embed_query(question)
    chunks = await vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
        doc_ids=doc_ids,
        doc_type=doc_type,
        year_filter=year_filter,
    )

    if not chunks:
        logger.warning("No chunks retrieved for question: {}", question[:80])
        return [], 0.0, question

    # Step 2: Check relevance
    score, reason = await check_relevance(question, chunks)

    # Step 3: If low relevance, rewrite and retry
    if score < 0.5:
        logger.info(
            "Low relevance ({:.2f}), rewriting query. Reason: {}",
            score,
            reason[:100],
        )
        rewritten = await rewrite_query(question, reason)

        # Re-embed and re-retrieve with the rewritten query
        new_vector = await embed_query(rewritten)
        new_chunks = await vector_store.search(
            query_vector=new_vector,
            top_k=top_k,
            doc_ids=doc_ids,
            doc_type=doc_type,
            year_filter=year_filter,
        )

        if new_chunks:
            # Re-check relevance with new results
            new_score, new_reason = await check_relevance(rewritten, new_chunks)
            logger.info(
                "After rewrite: new_score={:.2f} (was {:.2f})",
                new_score,
                score,
            )
            return new_chunks, new_score, rewritten

    return chunks, score, question
