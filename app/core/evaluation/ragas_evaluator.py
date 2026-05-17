"""RAGAS evaluation pipeline using Groq LLM as judge and Gemini embeddings."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loguru import logger

from app.config import settings
from app.core.generation.rag_chain import run_rag_chain
from app.core.retrieval.embedder import embed_query
from app.core.retrieval.vector_store import get_vector_store
from app.models.schemas import EvaluateRequest, QueryRequest, RAGASMetrics
from app.utils.helpers import generate_eval_id, utc_now_iso


async def run_ragas_evaluation(
    request: EvaluateRequest,
) -> RAGASMetrics:
    """Execute a full RAGAS evaluation pipeline.

    For each question:
    1. Run the RAG pipeline to get answer + contexts
    2. Collect into RAGAS dataset format
    3. Run RAGAS evaluate() with all 5 metrics
    4. Save report to evaluation/reports/

    Args:
        request: EvaluateRequest with questions, doc_ids, and sample_size.

    Returns:
        RAGASMetrics with all 5 evaluation scores.
    """
    eval_id = generate_eval_id()
    logger.info("Starting RAGAS evaluation: eval_id={}", eval_id)

    # Load questions
    questions = request.questions
    if not questions:
        questions = _load_sample_questions(request.sample_size)

    # Limit to sample_size
    questions = questions[: request.sample_size]

    # Collect RAG results for each question
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    ground_truths = _load_ground_truths()

    for i, question_text in enumerate(questions):
        logger.info("Evaluating question {}/{}: {}", i + 1, len(questions), question_text[:60])

        try:
            # Run RAG pipeline
            query_req = QueryRequest(
                question=question_text,
                doc_ids=request.doc_ids,
                mode="standard",
                top_k=5,
            )
            response = await run_rag_chain(query_req)

            eval_data["question"].append(question_text)
            eval_data["answer"].append(response.answer)
            eval_data["contexts"].append(
                [s.text for s in response.sources]
            )

            # Get ground truth if available
            gt = ground_truths.get(question_text, response.answer)
            eval_data["ground_truth"].append(gt)

        except Exception as e:
            logger.error("Failed to evaluate question '{}': {}", question_text[:40], str(e))
            eval_data["question"].append(question_text)
            eval_data["answer"].append("Error generating answer")
            eval_data["contexts"].append(["No context available"])
            eval_data["ground_truth"].append("N/A")

    # Run RAGAS evaluation
    metrics = await _run_ragas_metrics(eval_data)

    # Build result
    result = RAGASMetrics(
        faithfulness=metrics.get("faithfulness", 0.0),
        answer_relevancy=metrics.get("answer_relevancy", 0.0),
        context_precision=metrics.get("context_precision", 0.0),
        context_recall=metrics.get("context_recall", 0.0),
        answer_correctness=metrics.get("answer_correctness", 0.0),
        eval_id=eval_id,
        timestamp=utc_now_iso(),
        question_count=len(questions),
    )

    # Save report
    await _save_report(eval_id, result, eval_data)

    logger.info(
        "RAGAS evaluation complete: eval_id={}, faithfulness={:.3f}, "
        "relevancy={:.3f}, precision={:.3f}, recall={:.3f}, correctness={:.3f}",
        eval_id,
        result.faithfulness,
        result.answer_relevancy,
        result.context_precision,
        result.context_recall,
        result.answer_correctness,
    )

    return result


async def _run_ragas_metrics(eval_data: dict) -> dict[str, float]:
    """Run RAGAS metrics on the collected evaluation data.

    Uses Groq LLM as judge and Nvidia embeddings.

    Args:
        eval_data: Dict with 'question', 'answer', 'contexts', 'ground_truth' lists.

    Returns:
        Dict with metric name → score mappings.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_groq import ChatGroq
        from langchain_core.embeddings import Embeddings
        import httpx

        # Create RAGAS dataset
        dataset = Dataset.from_dict(eval_data)

        # Configure Groq as judge LLM
        llm = ChatGroq(
            model=settings.primary_model,
            api_key=settings.groq_api_key,
            temperature=0.1,
        )

        class NvidiaEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                url = "https://integrate.api.nvidia.com/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {settings.nvidia_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "input": texts,
                    "model": settings.embedding_model,
                    "input_type": "passage",
                    "encoding_format": "float",
                    "truncate": "NONE",
                }
                with httpx.Client() as client:
                    response = client.post(url, headers=headers, json=payload, timeout=30.0)
                    response.raise_for_status()
                    data = response.json()
                return [item["embedding"] for item in data["data"]]

            def embed_query(self, text: str) -> list[float]:
                url = "https://integrate.api.nvidia.com/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {settings.nvidia_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "input": [text],
                    "model": settings.embedding_model,
                    "input_type": "query",
                    "encoding_format": "float",
                    "truncate": "NONE",
                }
                with httpx.Client() as client:
                    response = client.post(url, headers=headers, json=payload, timeout=30.0)
                    response.raise_for_status()
                    data = response.json()
                return data["data"][0]["embedding"]

        # Configure Nvidia embeddings
        embeddings = NvidiaEmbeddings()

        # Run RAGAS evaluation
        result = await asyncio.to_thread(
            evaluate,
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            ],
            llm=llm,
            embeddings=embeddings,
        )

        return {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
            "context_recall": float(result.get("context_recall", 0.0)),
            "answer_correctness": float(result.get("answer_correctness", 0.0)),
        }

    except Exception as e:
        logger.error("RAGAS evaluation failed: {}", str(e))
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
        }


def _load_sample_questions(sample_size: int = 10) -> list[str]:
    """Load sample questions from the evaluation dataset file.

    Args:
        sample_size: Maximum number of questions to load.

    Returns:
        List of question strings.
    """
    sample_path = Path("evaluation/sample_questions.json")

    if not sample_path.exists():
        logger.warning("Sample questions file not found: {}", sample_path)
        return [
            "What dataset was used in this study?",
            "What are the key findings of the paper?",
            "Summarize the methodology section.",
        ]

    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = [item["question"] for item in data[:sample_size]]
        logger.info("Loaded {} sample questions", len(questions))
        return questions

    except Exception as e:
        logger.error("Failed to load sample questions: {}", str(e))
        return ["What are the main contributions of this research?"]


def _load_ground_truths() -> dict[str, str]:
    """Load ground truth answers from the evaluation dataset file.

    Returns:
        Dict mapping question text to ground truth answer.
    """
    sample_path = Path("evaluation/sample_questions.json")

    if not sample_path.exists():
        return {}

    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            item["question"]: item.get("ground_truth", "")
            for item in data
            if item.get("ground_truth")
        }

    except Exception as e:
        logger.error("Failed to load ground truths: {}", str(e))
        return {}


async def _save_report(
    eval_id: str,
    metrics: RAGASMetrics,
    eval_data: dict,
) -> None:
    """Save evaluation report to JSON file.

    Args:
        eval_id: Unique evaluation identifier.
        metrics: The computed RAGAS metrics.
        eval_data: The raw evaluation data used.
    """
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "eval_id": eval_id,
        "timestamp": metrics.timestamp,
        "metrics": metrics.model_dump(),
        "questions": [],
    }

    for i in range(len(eval_data["question"])):
        report["questions"].append(
            {
                "question": eval_data["question"][i],
                "answer": eval_data["answer"][i],
                "contexts": eval_data["contexts"][i],
                "ground_truth": eval_data["ground_truth"][i],
            }
        )

    report_path = report_dir / f"{eval_id}.json"

    def _write() -> None:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    await asyncio.to_thread(_write)
    logger.info("Evaluation report saved: {}", report_path)
