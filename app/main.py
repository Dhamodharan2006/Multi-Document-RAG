"""FastAPI application entry point with CORS, exception handling, and startup events."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.routes import documents, evaluate, ingest, query
from app.config import settings
from app.core.retrieval.vector_store import get_vector_store
from app.models.schemas import HealthResponse
from app.utils.helpers import ensure_directory
from app.utils.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle manager.

    On startup:
    - Configure structured logging
    - Ensure upload directory exists
    - Initialize Qdrant collection

    On shutdown:
    - Log graceful shutdown
    """
    # ── Startup ──
    setup_logging()
    logger.info("Starting Multi-Document RAG System")

    ensure_directory(settings.upload_dir)
    ensure_directory("evaluation/reports")
    ensure_directory("logs")

    # Initialize Qdrant collection
    try:
        vs = get_vector_store()
        await vs.ensure_collection()
        logger.info("Qdrant collection initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize Qdrant: {}", str(e))
        logger.warning("App will start but Qdrant operations will fail until connection is restored")

    yield

    # ── Shutdown ──
    logger.info("Shutting down Multi-Document RAG System")


# ── Create FastAPI App ──────────────────────────────────────────────

app = FastAPI(
    title="Multi-Document RAG System",
    description=(
        "A production-ready Retrieval-Augmented Generation system for "
        "academic research papers with cross-document reasoning and "
        "RAGAS evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ─────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ─────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(evaluate.router)

# ── Exception Handlers ──────────────────────────────────────────────


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with structured error responses.

    Args:
        request: The incoming request.
        exc: The HTTP exception raised.

    Returns:
        JSONResponse with error details and status code.
    """
    logger.warning(
        "HTTP {}: {} | path={}",
        exc.status_code,
        exc.detail,
        request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with safe error responses.

    Never exposes raw stack traces to the client.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        JSONResponse with 500 status and generic error message.
    """
    logger.error(
        "Unhandled exception on {}: {} | {}",
        request.url.path,
        type(exc).__name__,
        str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "An internal server error occurred. Please try again later.",
            "path": str(request.url.path),
        },
    )


# ── Health Check ────────────────────────────────────────────────────


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health status of the application and its dependencies.

    Returns:
        HealthResponse with connection status for Qdrant and Groq.
    """
    from app.core.generation.llm_client import get_llm_client

    # Check Qdrant
    qdrant_ok = False
    try:
        vs = get_vector_store()
        qdrant_ok = vs.check_connection()
    except Exception:
        pass

    # Check Groq
    groq_ok = False
    try:
        llm = get_llm_client()
        groq_ok = llm.check_connection()
    except Exception:
        pass

    status = "healthy" if (qdrant_ok and groq_ok) else "degraded"

    return HealthResponse(
        status=status,
        qdrant_connected=qdrant_ok,
        groq_connected=groq_ok,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Root ────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict:
    """Root endpoint returning API information.

    Returns:
        Dict with API name, version, and documentation URL.
    """
    return {
        "name": "Multi-Document RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
