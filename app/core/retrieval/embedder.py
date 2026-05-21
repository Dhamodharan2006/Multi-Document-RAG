"""Nvidia NIM Embedding API wrapper with batching and retry logic."""

from __future__ import annotations

import asyncio
import httpx

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.utils.helpers import batch_list

# Batch size for embedding requests
_EMBED_BATCH_SIZE = 20

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _embed_content_sync(
    texts: list[str],
    input_type: str,
) -> list[list[float]]:
    """Embed a batch of texts via Nvidia API using httpx.

    Args:
        texts: List of text strings to embed.
        input_type: Either 'passage' or 'query'.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": texts,
        "model": settings.embedding_model,
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": "NONE",
    }
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
    return [item["embedding"] for item in data["data"]]


async def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing using Nvidia API.

    Batches texts into groups of 20 and embeds them with
    input_type='passage'. Includes exponential backoff retry.

    Args:
        texts: List of text chunks to embed.

    Returns:
        List of embedding vectors with dimension 2048.
    """
    logger.info("Embedding {} chunks via Nvidia API", len(texts))
    all_embeddings: list[list[float]] = []

    batches = batch_list(texts, _EMBED_BATCH_SIZE)
    for i, batch in enumerate(batches):
        logger.debug("Embedding batch {}/{} ({} texts)", i + 1, len(batches), len(batch))
        embeddings = await _embed_content_sync(
            batch,
            "passage",
        )
        all_embeddings.extend(embeddings)

    logger.info(
        "Embedding complete: {} vectors of dimension {}",
        len(all_embeddings),
        len(all_embeddings[0]) if all_embeddings else 0,
    )
    return all_embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval using Nvidia API.

    Uses input_type='query' for asymmetric search.

    Args:
        query: The user query text to embed.

    Returns:
        A single embedding vector with dimension 2048.
    """
    logger.debug("Embedding query via Nvidia API: {}...", query[:80])
    embeddings = await _embed_content_sync(
        [query],
        "query",
    )
    return embeddings[0]
