"""Shared FastAPI dependencies for dependency injection."""

from __future__ import annotations

from app.core.generation.llm_client import LLMClient, get_llm_client
from app.core.retrieval.vector_store import VectorStore, get_vector_store


async def get_vs() -> VectorStore:
    """Dependency that provides the VectorStore singleton.

    Returns:
        The shared VectorStore instance.
    """
    return get_vector_store()


async def get_llm() -> LLMClient:
    """Dependency that provides the LLMClient singleton.

    Returns:
        The shared LLMClient instance.
    """
    return get_llm_client()
