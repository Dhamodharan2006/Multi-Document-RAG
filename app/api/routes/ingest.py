"""Ingestion API endpoints: single and batch document ingestion."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from app.config import settings
from app.core.ingestion.pipeline import run_ingestion_pipeline
from app.models.schemas import BatchIngestResponse, IngestResult
from app.utils.helpers import ensure_directory

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResult)
async def ingest_document(
    file: UploadFile = File(...),
    source: str = Form(default="upload"),
) -> IngestResult:
    """Ingest a single document file.

    Accepts PDF, DOCX, or TXT files via multipart upload.
    Runs the full pipeline: load → tag → chunk → embed → store.

    Args:
        file: The uploaded document file.
        source: Source origin string (e.g., "arxiv", "upload").

    Returns:
        IngestResult with doc_id, filename, chunk count, metadata, and status.
    """
    logger.info("Ingest request: filename={}, source={}", file.filename, source)

    # Validate file type
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in (".pdf", ".docx", ".txt"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: .pdf, .docx, .txt",
            )

    # Save uploaded file
    upload_dir = ensure_directory(settings.upload_dir)
    file_path = upload_dir / (file.filename or "unnamed_file")

    try:
        content = await file.read()
        await asyncio.to_thread(_save_file, file_path, content)
    except Exception as e:
        logger.error("Failed to save uploaded file: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Run ingestion pipeline
    try:
        result = await run_ingestion_pipeline(file_path, source=source)
        if result.status.startswith("failed"):
            raise HTTPException(status_code=422, detail=result.status)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion failed for {}: {}", file.filename, str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    files: list[UploadFile] = File(...),
    source: str = Form(default="upload"),
) -> BatchIngestResponse:
    """Ingest multiple document files concurrently.

    Processes each file through the full ingestion pipeline using
    asyncio.gather for concurrent execution.

    Args:
        files: List of uploaded document files.
        source: Source origin string for all files.

    Returns:
        BatchIngestResponse with results per file, total chunks, and failures.
    """
    logger.info("Batch ingest request: {} files, source={}", len(files), source)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    upload_dir = ensure_directory(settings.upload_dir)

    # Save all files first
    file_paths: list[Path] = []
    for f in files:
        if f.filename:
            ext = Path(f.filename).suffix.lower()
            if ext not in (".pdf", ".docx", ".txt"):
                logger.warning("Skipping unsupported file: {}", f.filename)
                continue

        path = upload_dir / (f.filename or "unnamed_file")
        content = await f.read()
        await asyncio.to_thread(_save_file, path, content)
        file_paths.append(path)

    # Process all files concurrently
    tasks = [run_ingestion_pipeline(fp, source=source) for fp in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successes and failures
    successful: list[IngestResult] = []
    failed: list[str] = []
    total_chunks = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed.append(f"{file_paths[i].name}: {str(result)}")
        elif isinstance(result, IngestResult):
            if result.status == "success":
                successful.append(result)
                total_chunks += result.chunks_created
            else:
                failed.append(f"{result.filename}: {result.status}")

    logger.info(
        "Batch ingest complete: {}/{} succeeded, {} total chunks",
        len(successful),
        len(file_paths),
        total_chunks,
    )

    return BatchIngestResponse(
        results=successful,
        total_chunks=total_chunks,
        failed=failed,
    )


def _save_file(path: Path, content: bytes) -> None:
    """Save file content to disk.

    Args:
        path: Destination file path.
        content: File content bytes.
    """
    with open(path, "wb") as f:
        f.write(content)
