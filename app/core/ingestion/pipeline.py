"""Full ingestion pipeline: load → tag → chunk → embed → store."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from app.config import settings
from app.core.ingestion.chunker import chunk_text
from app.core.ingestion.loader import load_document
from app.core.ingestion.metadata_tagger import auto_tag
from app.core.retrieval.embedder import embed_chunks
from app.core.retrieval.vector_store import get_vector_store
from app.models.schemas import IngestResult


async def run_ingestion_pipeline(
    file_path: str | Path,
    source: str = "upload",
) -> IngestResult:
    """Execute the complete ingestion pipeline for a single document.

    Pipeline steps:
    1. Load document text from file
    2. Generate metadata tags
    3. Split text into chunks
    4. Embed chunks via Gemini API
    5. Store chunks + embeddings + metadata in Qdrant

    Args:
        file_path: Path to the document file.
        source: Source origin of the document.

    Returns:
        An IngestResult with doc_id, filename, chunk count, metadata, and status.

    Raises:
        Exception: If any pipeline step fails, returns a failed IngestResult.
    """
    path = Path(file_path)
    logger.info("Starting ingestion pipeline for: {}", path.name)

    try:
        # Step 1: Load document
        text = await load_document(path)
        if not text.strip():
            return IngestResult(
                doc_id="",
                filename=path.name,
                chunks_created=0,
                metadata={},
                status="failed: empty document",
            )

        # Step 2: Tag metadata
        metadata = auto_tag(path, source=source)
        doc_id = metadata["doc_id"]

        # Step 3: Chunk text
        chunks = chunk_text(text)
        if not chunks:
            return IngestResult(
                doc_id=doc_id,
                filename=path.name,
                chunks_created=0,
                metadata=metadata,
                status="failed: no chunks produced",
            )

        # Step 4: Embed chunks via Gemini
        embeddings = await embed_chunks(chunks)

        # Step 5: Store in Qdrant
        vector_store = get_vector_store()
        await vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
        )

        logger.info(
            "Ingestion complete: {} → {} chunks stored (doc_id={})",
            path.name,
            len(chunks),
            doc_id,
        )

        return IngestResult(
            doc_id=doc_id,
            filename=path.name,
            chunks_created=len(chunks),
            metadata=metadata,
            status="success",
        )

    except Exception as e:
        logger.error("Ingestion pipeline failed for {}: {}", path.name, str(e))
        return IngestResult(
            doc_id="",
            filename=path.name,
            chunks_created=0,
            metadata={},
            status=f"failed: {str(e)}",
        )
