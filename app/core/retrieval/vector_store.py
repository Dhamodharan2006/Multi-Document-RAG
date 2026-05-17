"""Qdrant vector store operations: collection management, upsert, search, delete."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    ScrollRequest,
    VectorParams,
)

from app.config import settings

# Module-level singleton
_vector_store: VectorStore | None = None


class VectorStore:
    """Qdrant vector store wrapper for chunk storage and retrieval.

    Manages collection creation, upserting document chunks with metadata,
    similarity search with filtering, and document deletion.
    """

    def __init__(self) -> None:
        """Initialize Qdrant client connection."""
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.embedding_dimension
        logger.info(
            "Qdrant client initialized: {}:{} collection={}",
            settings.qdrant_host,
            settings.qdrant_port,
            self.collection_name,
        )

    async def ensure_collection(self) -> None:
        """Create the collection if it does not already exist.

        Sets up a vector collection with COSINE distance metric
        and the configured embedding dimension (3072).
        """
        def _create() -> None:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]

            if self.collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '{}' (dim={}, distance=COSINE)",
                    self.collection_name,
                    self.vector_size,
                )
            else:
                logger.info("Qdrant collection '{}' already exists", self.collection_name)

        await asyncio.to_thread(_create)

    async def upsert_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict,
    ) -> int:
        """Upsert document chunks with embeddings and metadata into Qdrant.

        Each chunk is stored as a point with the embedding vector and
        a payload containing the text, document metadata, and chunk index.

        Args:
            chunks: List of text chunk strings.
            embeddings: List of embedding vectors corresponding to chunks.
            metadata: Document-level metadata dict (doc_id, filename, etc.).

        Returns:
            Number of points upserted.
        """
        def _upsert() -> int:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                payload = {
                    "text": chunk,
                    "doc_id": metadata["doc_id"],
                    "filename": metadata["filename"],
                    "doc_type": metadata.get("doc_type", "unknown"),
                    "source": metadata.get("source", "upload"),
                    "pub_year": metadata.get("pub_year"),
                    "chunk_index": i,
                    "ingested_at": metadata.get("ingested_at", ""),
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            # Upsert in batches of 100
            batch_size = 100
            for j in range(0, len(points), batch_size):
                batch = points[j : j + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )

            logger.info(
                "Upserted {} chunks for doc_id={} into Qdrant",
                len(points),
                metadata["doc_id"],
            )
            return len(points)

        return await asyncio.to_thread(_upsert)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        doc_type: str | None = None,
        year_filter: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using vector similarity with optional filters.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results to return.
            doc_ids: Optional list of doc_ids to restrict search to.
            doc_type: Optional document type filter.
            year_filter: Optional publication year filter.

        Returns:
            List of dicts with 'id', 'score', and 'payload' for each match.
        """
        def _search() -> list[dict[str, Any]]:
            filter_conditions = []

            if doc_ids:
                filter_conditions.append(
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=doc_ids),
                    )
                )

            if doc_type:
                filter_conditions.append(
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=doc_type),
                    )
                )

            if year_filter is not None:
                filter_conditions.append(
                    FieldCondition(
                        key="pub_year",
                        match=MatchValue(value=year_filter),
                    )
                )

            search_filter = Filter(must=filter_conditions) if filter_conditions else None

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
            ).points

            logger.info(
                "Qdrant search returned {} results (top_k={}, filters={})",
                len(results),
                top_k,
                bool(filter_conditions),
            )

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in results
            ]

        return await asyncio.to_thread(_search)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all points belonging to a specific document.

        Args:
            doc_id: The document ID whose chunks should be deleted.

        Returns:
            Number of points deleted.
        """
        def _delete() -> int:
            # First count points for this doc
            count_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                ),
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )
            point_ids = [p.id for p in count_result[0]]
            count = len(point_ids)

            if count > 0:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="doc_id",
                                    match=MatchValue(value=doc_id),
                                )
                            ]
                        )
                    ),
                )

            logger.info("Deleted {} chunks for doc_id={}", count, doc_id)
            return count

        return await asyncio.to_thread(_delete)

    async def list_documents(self) -> list[dict[str, Any]]:
        """List all unique documents stored in the collection.

        Scrolls through all points and aggregates by doc_id to
        produce a list of unique documents with their metadata.

        Returns:
            List of dicts with doc metadata and chunk counts.
        """
        def _list() -> list[dict[str, Any]]:
            documents: dict[str, dict[str, Any]] = {}
            offset = None

            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in results:
                    payload = point.payload or {}
                    doc_id = payload.get("doc_id", "unknown")

                    if doc_id not in documents:
                        documents[doc_id] = {
                            "doc_id": doc_id,
                            "filename": payload.get("filename", ""),
                            "doc_type": payload.get("doc_type", "unknown"),
                            "source": payload.get("source", "upload"),
                            "pub_year": payload.get("pub_year"),
                            "ingested_at": payload.get("ingested_at", ""),
                            "chunk_count": 0,
                        }
                    documents[doc_id]["chunk_count"] += 1

                if next_offset is None:
                    break
                offset = next_offset

            logger.info("Listed {} unique documents from Qdrant", len(documents))
            return list(documents.values())

        return await asyncio.to_thread(_list)

    def check_connection(self) -> bool:
        """Check if Qdrant is reachable.

        Returns:
            True if connected, False otherwise.
        """
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


def get_vector_store() -> VectorStore:
    """Get or create the singleton VectorStore instance.

    Returns:
        The shared VectorStore instance.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
