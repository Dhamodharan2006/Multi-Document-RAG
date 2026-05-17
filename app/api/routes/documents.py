"""Document management API endpoints: list and delete documents."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.core.retrieval.vector_store import get_vector_store
from app.models.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentMeta,
)

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all ingested documents with metadata.

    Fetches unique document IDs and their metadata from Qdrant.

    Returns:
        DocumentListResponse with list of documents and total count.
    """
    logger.info("Listing all documents")

    try:
        vector_store = get_vector_store()
        docs = await vector_store.list_documents()

        documents = [
            DocumentMeta(
                doc_id=d["doc_id"],
                filename=d["filename"],
                doc_type=d["doc_type"],
                source=d["source"],
                pub_year=d.get("pub_year"),
                chunk_count=d["chunk_count"],
                ingested_at=d.get("ingested_at", ""),
            )
            for d in docs
        ]

        return DocumentListResponse(
            documents=documents,
            total_count=len(documents),
        )

    except Exception as e:
        logger.error("Failed to list documents: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(doc_id: str) -> DeleteDocumentResponse:
    """Delete all chunks belonging to a specific document.

    Removes all Qdrant points matching the given doc_id.

    Args:
        doc_id: The document ID to delete.

    Returns:
        DeleteDocumentResponse with doc_id, deleted chunk count, and status.
    """
    logger.info("Delete request for doc_id={}", doc_id)

    try:
        vector_store = get_vector_store()
        deleted_count = await vector_store.delete_by_doc_id(doc_id)

        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {doc_id}",
            )

        return DeleteDocumentResponse(
            doc_id=doc_id,
            deleted_chunks=deleted_count,
            status="deleted",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document {}: {}", doc_id, str(e))
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
