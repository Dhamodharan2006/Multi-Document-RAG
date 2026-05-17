"""Evaluation API endpoints: run RAGAS evaluation and fetch reports."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from app.core.evaluation.ragas_evaluator import run_ragas_evaluation
from app.core.evaluation.report_generator import load_report
from app.models.schemas import EvaluateRequest, EvaluateResponse, RAGASMetrics

router = APIRouter(prefix="/api/v1", tags=["evaluation"])

# In-memory store for evaluation results
_eval_store: dict[str, dict[str, Any]] = {}


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
) -> EvaluateResponse:
    """Start a RAGAS evaluation run.

    Runs the full RAG pipeline for each question, collects results,
    and computes RAGAS metrics. Can run in background for large datasets.

    Args:
        request: EvaluateRequest with questions, doc_ids, and sample_size.
        background_tasks: FastAPI background task manager.

    Returns:
        EvaluateResponse with eval_id, status, and metrics (if completed synchronously).
    """
    logger.info(
        "Evaluation request: questions={}, sample_size={}",
        len(request.questions) if request.questions else "sample",
        request.sample_size,
    )

    # For small evaluations, run synchronously
    if request.sample_size <= 5:
        try:
            metrics = await run_ragas_evaluation(request)
            _eval_store[metrics.eval_id] = {
                "status": "completed",
                "metrics": metrics,
            }
            return EvaluateResponse(
                eval_id=metrics.eval_id,
                status="completed",
                metrics=metrics,
            )
        except Exception as e:
            logger.error("Evaluation failed: {}", str(e))
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    # For larger evaluations, run in background
    from app.utils.helpers import generate_eval_id

    eval_id = generate_eval_id()
    _eval_store[eval_id] = {"status": "running", "metrics": None}

    async def _run_eval() -> None:
        """Background evaluation task."""
        try:
            metrics = await run_ragas_evaluation(request)
            _eval_store[eval_id] = {
                "status": "completed",
                "metrics": metrics,
            }
        except Exception as e:
            logger.error("Background evaluation failed: {}", str(e))
            _eval_store[eval_id] = {
                "status": "failed",
                "metrics": None,
                "error": str(e),
            }

    # Schedule as background task
    background_tasks.add_task(lambda: asyncio.ensure_future(_run_eval()))

    return EvaluateResponse(
        eval_id=eval_id,
        status="running",
        metrics=None,
    )


@router.get("/evaluate/{eval_id}")
async def get_evaluation_report(eval_id: str) -> dict:
    """Fetch a RAGAS evaluation report by eval_id.

    Checks in-memory store first, then falls back to file-based reports.

    Args:
        eval_id: The evaluation run identifier.

    Returns:
        Full RAGAS report with per-question breakdown.
    """
    logger.info("Fetching evaluation report: eval_id={}", eval_id)

    # Check in-memory store
    if eval_id in _eval_store:
        entry = _eval_store[eval_id]
        result: dict[str, Any] = {
            "eval_id": eval_id,
            "status": entry["status"],
        }
        if entry.get("metrics"):
            result["metrics"] = entry["metrics"].model_dump()
        if entry.get("error"):
            result["error"] = entry["error"]
        return result

    # Check file-based reports
    report = load_report(eval_id)
    if report:
        return report

    raise HTTPException(
        status_code=404,
        detail=f"Evaluation report not found: {eval_id}",
    )
