"""Cross-document reasoning chain for comparing multiple documents."""

from __future__ import annotations

import json
import re
import time

from loguru import logger

from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompt_templates import CROSS_DOC_PROMPT
from app.core.retrieval.embedder import embed_query
from app.core.retrieval.vector_store import get_vector_store
from app.models.schemas import CompareRequest, CompareResponse, DocSummary


async def run_cross_doc_chain(request: CompareRequest) -> CompareResponse:
    """Execute cross-document comparison chain.

    Retrieves chunks from EACH document separately, then combines
    them into a structured comparison prompt for the reasoning model.

    Flow:
    1. Embed the query
    2. Retrieve top chunks from each specified doc_id separately
    3. Format contexts per document
    4. Use deepseek-r1-distill-llama-70b for reasoning
    5. Parse structured output into agreements/contradictions

    Args:
        request: CompareRequest with question, doc_ids (≥2), and aspect.

    Returns:
        CompareResponse with comparison text, agreements, contradictions, and metadata.
    """
    start_time = time.perf_counter()
    llm_client = get_llm_client()
    vector_store = get_vector_store()

    # Use reasoning model for cross-document comparison
    model = llm_client.reasoning_model

    # Step 1: Embed the query
    query_vector = await embed_query(request.question)

    # Step 2: Retrieve chunks from EACH document separately
    doc_contexts: dict[str, list[dict]] = {}
    for doc_id in request.doc_ids:
        chunks = await vector_store.search(
            query_vector=query_vector,
            top_k=5,
            doc_ids=[doc_id],
        )
        doc_contexts[doc_id] = chunks
        logger.debug(
            "Retrieved {} chunks for doc_id={}", len(chunks), doc_id
        )

    # Step 3: Format contexts per document
    formatted_parts = []
    doc_summaries = []

    for doc_id, chunks in doc_contexts.items():
        if not chunks:
            continue

        filename = chunks[0].get("payload", {}).get("filename", "Unknown")
        chunk_texts = []
        for chunk in chunks:
            text = chunk.get("payload", {}).get("text", "")
            chunk_idx = chunk.get("payload", {}).get("chunk_index", 0)
            chunk_texts.append(f"  [Chunk {chunk_idx}]: {text}")

        doc_section = (
            f"=== Document: {filename} (ID: {doc_id}) ===\n"
            + "\n\n".join(chunk_texts)
        )
        formatted_parts.append(doc_section)

        # Create a brief summary placeholder
        doc_summaries.append(
            DocSummary(
                doc_id=doc_id,
                filename=filename,
                summary=f"Excerpts retrieved: {len(chunks)} chunks",
            )
        )

    formatted_doc_contexts = "\n\n" + "\n\n".join(formatted_parts)

    # Step 4: Generate comparison using reasoning model
    prompt_messages = CROSS_DOC_PROMPT.format_messages(
        formatted_doc_contexts=formatted_doc_contexts,
        question=request.question,
        aspect=request.aspect,
    )

    system_msg = ""
    user_msg = ""
    for msg in prompt_messages:
        if msg.type == "system":
            system_msg = msg.content
        elif msg.type == "human":
            user_msg = msg.content

    raw_response = await llm_client.generate(
        prompt=user_msg,
        model=model,
        temperature=0.3,
        max_tokens=3000,
        system_message=system_msg,
    )

    # Step 5: Parse structured output
    agreements, contradictions = _parse_comparison_output(raw_response)

    # Clean response by removing the JSON arrays from the displayed comparison
    comparison = raw_response
    for marker in ["AGREEMENTS_JSON:", "CONTRADICTIONS_JSON:"]:
        idx = comparison.find(marker)
        if idx != -1:
            comparison = comparison[:idx].strip()

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    logger.info(
        "Cross-doc chain complete: model={}, latency={}ms, docs={}, "
        "agreements={}, contradictions={}",
        model,
        latency_ms,
        len(request.doc_ids),
        len(agreements),
        len(contradictions),
    )

    return CompareResponse(
        comparison=comparison,
        doc_summaries=doc_summaries,
        agreements=agreements,
        contradictions=contradictions,
        model_used=model,
        latency_ms=latency_ms,
    )


def _parse_comparison_output(
    text: str,
) -> tuple[list[str], list[str]]:
    """Parse agreements and contradictions JSON arrays from LLM output.

    Looks for AGREEMENTS_JSON and CONTRADICTIONS_JSON markers in the text.

    Args:
        text: The raw LLM response text.

    Returns:
        Tuple of (agreements_list, contradictions_list).
    """
    agreements: list[str] = []
    contradictions: list[str] = []

    try:
        # Extract AGREEMENTS_JSON
        match = re.search(
            r"AGREEMENTS_JSON:\s*(\[.*?\])",
            text,
            re.DOTALL,
        )
        if match:
            agreements = json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        logger.debug("Could not parse AGREEMENTS_JSON from response")

    try:
        # Extract CONTRADICTIONS_JSON
        match = re.search(
            r"CONTRADICTIONS_JSON:\s*(\[.*?\])",
            text,
            re.DOTALL,
        )
        if match:
            contradictions = json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        logger.debug("Could not parse CONTRADICTIONS_JSON from response")

    # Fallback: try to extract from sections if JSON parsing failed
    if not agreements and not contradictions:
        agreements, contradictions = _extract_from_sections(text)

    return agreements, contradictions


def _extract_from_sections(text: str) -> tuple[list[str], list[str]]:
    """Fallback extraction of agreements/contradictions from section headers.

    Args:
        text: The raw LLM response text.

    Returns:
        Tuple of (agreements_list, contradictions_list).
    """
    agreements: list[str] = []
    contradictions: list[str] = []

    # Try to find agreement section
    agree_match = re.search(
        r"(?:Key\s+)?Agreements?:?\s*\n(.*?)(?=(?:Key\s+)?(?:Contradictions?|Differences?|Synthesis|Conclusion|\Z))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if agree_match:
        lines = agree_match.group(1).strip().split("\n")
        for line in lines:
            line = re.sub(r"^[\s\-\*\d\.]+", "", line).strip()
            if line and len(line) > 10:
                agreements.append(line)

    # Try to find contradiction section
    contra_match = re.search(
        r"(?:Key\s+)?(?:Contradictions?|Differences?):?\s*\n(.*?)(?=(?:Synthesis|Conclusion|AGREEMENTS_JSON|\Z))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if contra_match:
        lines = contra_match.group(1).strip().split("\n")
        for line in lines:
            line = re.sub(r"^[\s\-\*\d\.]+", "", line).strip()
            if line and len(line) > 10:
                contradictions.append(line)

    return agreements, contradictions
