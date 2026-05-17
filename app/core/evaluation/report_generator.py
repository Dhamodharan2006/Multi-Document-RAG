"""Report generator for RAGAS evaluation results in JSON and CSV formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from loguru import logger

from app.models.schemas import RAGASMetrics


async def generate_report(
    eval_id: str,
    metrics: RAGASMetrics,
    eval_data: dict | None = None,
    output_dir: str | Path = "evaluation/reports",
) -> dict[str, str]:
    """Generate evaluation reports in both JSON and CSV formats.

    Args:
        eval_id: Unique evaluation identifier.
        metrics: The computed RAGAS metrics.
        eval_data: Optional raw evaluation data for per-question breakdown.
        output_dir: Directory to save reports.

    Returns:
        Dict with 'json_path' and 'csv_path' pointing to generated files.
    """
    import asyncio

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"{eval_id}.json"
    csv_path = report_dir / f"{eval_id}.csv"

    # Build JSON report
    report = {
        "eval_id": eval_id,
        "timestamp": metrics.timestamp,
        "question_count": metrics.question_count,
        "aggregate_metrics": {
            "faithfulness": metrics.faithfulness,
            "answer_relevancy": metrics.answer_relevancy,
            "context_precision": metrics.context_precision,
            "context_recall": metrics.context_recall,
            "answer_correctness": metrics.answer_correctness,
        },
    }

    if eval_data:
        report["per_question"] = []
        for i in range(len(eval_data.get("question", []))):
            report["per_question"].append(
                {
                    "question": eval_data["question"][i],
                    "answer": eval_data["answer"][i],
                    "num_contexts": len(eval_data["contexts"][i]),
                    "ground_truth": eval_data.get("ground_truth", [""])[i]
                    if i < len(eval_data.get("ground_truth", []))
                    else "",
                }
            )

    def _write_json() -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _write_csv() -> None:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Score"])
            writer.writerow(["eval_id", eval_id])
            writer.writerow(["timestamp", metrics.timestamp])
            writer.writerow(["question_count", metrics.question_count])
            writer.writerow(["faithfulness", f"{metrics.faithfulness:.4f}"])
            writer.writerow(["answer_relevancy", f"{metrics.answer_relevancy:.4f}"])
            writer.writerow(["context_precision", f"{metrics.context_precision:.4f}"])
            writer.writerow(["context_recall", f"{metrics.context_recall:.4f}"])
            writer.writerow(["answer_correctness", f"{metrics.answer_correctness:.4f}"])

    await asyncio.to_thread(_write_json)
    await asyncio.to_thread(_write_csv)

    logger.info("Reports generated: {} and {}", json_path, csv_path)

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


def load_report(eval_id: str, report_dir: str | Path = "evaluation/reports") -> dict | None:
    """Load an existing evaluation report by eval_id.

    Args:
        eval_id: The evaluation ID to look up.
        report_dir: Directory containing reports.

    Returns:
        The report dict if found, None otherwise.
    """
    report_path = Path(report_dir) / f"{eval_id}.json"

    if not report_path.exists():
        logger.warning("Report not found: {}", report_path)
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        logger.info("Loaded report: {}", report_path)
        return report
    except Exception as e:
        logger.error("Failed to load report {}: {}", eval_id, str(e))
        return None
