"""Query router that classifies queries and decides routing strategy."""

from __future__ import annotations

import json

from loguru import logger

from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompt_templates import QUERY_CLASSIFICATION_PROMPT
from app.models.schemas import QueryRoute


async def classify_query(
    question: str,
    available_docs: list[dict] | None = None,
) -> QueryRoute:
    """Classify a user query to determine routing strategy.

    Analyzes the question to decide:
    - Which documents to search (specific or all)
    - Which LLM model to use (standard or reasoning)
    - Whether cross-document comparison is needed

    Uses llama-3.3-70b-versatile with JSON output for classification.

    Args:
        question: The user's question string.
        available_docs: Optional list of available document metadata dicts.

    Returns:
        A QueryRoute with doc_scope, mode, and requires_comparison fields.
    """
    llm_client = get_llm_client()

    # Build document context for the classifier
    doc_context = ""
    if available_docs:
        doc_list = []
        for doc in available_docs:
            doc_list.append(
                f"- doc_id: {doc.get('doc_id', 'N/A')}, "
                f"filename: {doc.get('filename', 'N/A')}, "
                f"type: {doc.get('doc_type', 'unknown')}"
            )
        doc_context = "\n".join(doc_list)

    prompt = QUERY_CLASSIFICATION_PROMPT.format(
        question=question,
        available_documents=doc_context or "No documents available",
    )

    try:
        result = await llm_client.generate_json(
            prompt=prompt,
            model=llm_client.primary_model,
        )

        route = QueryRoute(
            doc_scope=result.get("mentioned_docs", []) or "all",
            mode=result.get("mode", "standard"),
            requires_comparison=result.get("requires_comparison", False),
            reasoning=result.get("reasoning", ""),
        )

        logger.info(
            "Query classified: mode={}, comparison={}, scope={}",
            route.mode,
            route.requires_comparison,
            route.doc_scope,
        )
        return route

    except Exception as e:
        logger.warning("Query classification failed, using defaults: {}", str(e))
        # Fallback: use simple heuristic
        return _heuristic_classify(question)


def _heuristic_classify(question: str) -> QueryRoute:
    """Fallback heuristic classification when LLM classification fails.

    Uses keyword matching to determine if a question requires
    comparison or reasoning mode.

    Args:
        question: The user's question string.

    Returns:
        A QueryRoute based on keyword analysis.
    """
    q_lower = question.lower()

    comparison_keywords = [
        "compare", "contrast", "difference between", "versus", "vs",
        "differ", "similarities", "agree", "disagree", "contradicts",
    ]
    requires_comparison = any(kw in q_lower for kw in comparison_keywords)

    reasoning_keywords = [
        "why", "how does", "explain", "analyze", "evaluate",
        "what evidence", "implications", "synthesize",
    ]
    needs_reasoning = any(kw in q_lower for kw in reasoning_keywords)

    mode = "reasoning" if (requires_comparison or needs_reasoning) else "standard"

    return QueryRoute(
        doc_scope="all",
        mode=mode,
        requires_comparison=requires_comparison,
        reasoning="heuristic fallback classification",
    )
