"""Text chunking with recursive character splitting."""

from __future__ import annotations

from loguru import logger

from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting.

    Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting
    that respects paragraph and sentence boundaries.

    Args:
        text: The full text to split into chunks.
        chunk_size: Maximum characters per chunk. Defaults to config value.
        chunk_overlap: Overlap characters between consecutive chunks. Defaults to config value.

    Returns:
        A list of text chunk strings.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)

    logger.info(
        "Chunked text into {} chunks (size={}, overlap={})",
        len(chunks),
        size,
        overlap,
    )
    return chunks
