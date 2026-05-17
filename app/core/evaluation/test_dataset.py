"""Sample test dataset builder for RAGAS evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


def build_sample_dataset() -> list[dict]:
    """Build the sample test dataset with 20 diverse questions.

    Creates questions across 5 categories:
    - Factual (4 questions)
    - Comparative (4 questions)
    - Summarization (4 questions)
    - Multi-hop reasoning (4 questions)
    - Out-of-scope (4 questions)

    Returns:
        List of question dictionaries with id, type, question, ground_truth, and doc_hint.
    """
    dataset = _get_sample_questions()
    logger.info("Built sample dataset with {} questions", len(dataset))
    return dataset


def save_sample_dataset(output_path: str | Path | None = None) -> Path:
    """Save the sample dataset to a JSON file.

    Args:
        output_path: Optional custom path. Defaults to evaluation/sample_questions.json.

    Returns:
        Path to the saved file.
    """
    path = Path(output_path or "evaluation/sample_questions.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_sample_dataset()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info("Sample dataset saved to: {}", path)
    return path


def _get_sample_questions() -> list[dict]:
    """Return the hardcoded list of 20 sample evaluation questions.

    Returns:
        List of question dicts covering all 5 question types.
    """
    return [
        # ── Factual Questions (4) ──
        {
            "id": "q001",
            "type": "factual",
            "question": "What dataset was used in this study?",
            "ground_truth": "The specific dataset used depends on the paper being queried. Common datasets include ImageNet, COCO, SQuAD, and GLUE benchmark.",
            "doc_hint": None,
        },
        {
            "id": "q002",
            "type": "factual",
            "question": "What is the primary evaluation metric reported in the paper?",
            "ground_truth": "The primary evaluation metric varies by paper but commonly includes accuracy, F1-score, BLEU score, or ROUGE score.",
            "doc_hint": None,
        },
        {
            "id": "q003",
            "type": "factual",
            "question": "Who are the authors of this research paper?",
            "ground_truth": "The authors are listed in the paper's header section and vary by document.",
            "doc_hint": None,
        },
        {
            "id": "q004",
            "type": "factual",
            "question": "What baseline models were compared against in the experiments?",
            "ground_truth": "Baseline models typically include state-of-the-art methods from prior work in the same domain.",
            "doc_hint": None,
        },
        # ── Comparative Questions (4) ──
        {
            "id": "q005",
            "type": "comparative",
            "question": "How does the proposed method differ from the baseline approach in terms of architecture?",
            "ground_truth": "The proposed method typically introduces novel architectural components or modifications to existing architectures that differentiate it from baselines.",
            "doc_hint": None,
        },
        {
            "id": "q006",
            "type": "comparative",
            "question": "Compare the computational efficiency of the methods described in these papers.",
            "ground_truth": "Computational efficiency varies based on model size, training time, inference speed, and hardware requirements.",
            "doc_hint": None,
        },
        {
            "id": "q007",
            "type": "comparative",
            "question": "What are the similarities and differences in the training procedures used across these studies?",
            "ground_truth": "Training procedures may share common elements like optimizer choice but differ in learning rates, batch sizes, and data augmentation strategies.",
            "doc_hint": None,
        },
        {
            "id": "q008",
            "type": "comparative",
            "question": "How do the experimental results in paper A contrast with those in paper B?",
            "ground_truth": "Results may show different performance levels on the same benchmarks due to methodological differences.",
            "doc_hint": None,
        },
        # ── Summarization Questions (4) ──
        {
            "id": "q009",
            "type": "summarization",
            "question": "Summarize the key findings of this paper.",
            "ground_truth": "Key findings typically include the main contributions, performance improvements, and novel insights presented in the paper.",
            "doc_hint": None,
        },
        {
            "id": "q010",
            "type": "summarization",
            "question": "Provide a brief overview of the methodology section.",
            "ground_truth": "The methodology section describes the proposed approach, model architecture, training procedure, and evaluation protocol.",
            "doc_hint": None,
        },
        {
            "id": "q011",
            "type": "summarization",
            "question": "What are the main contributions claimed by the authors?",
            "ground_truth": "Main contributions typically include novel methods, improved performance, new datasets, or theoretical insights.",
            "doc_hint": None,
        },
        {
            "id": "q012",
            "type": "summarization",
            "question": "Summarize the related work section and identify the research gap addressed.",
            "ground_truth": "The related work section reviews prior approaches and identifies limitations that the current paper aims to address.",
            "doc_hint": None,
        },
        # ── Multi-hop Reasoning Questions (4) ──
        {
            "id": "q013",
            "type": "multi_hop",
            "question": "What evidence supports the main claim in section 3, and how does it connect to the conclusions?",
            "ground_truth": "Evidence in section 3 typically includes experimental results, ablation studies, or theoretical proofs that support the paper's main thesis.",
            "doc_hint": None,
        },
        {
            "id": "q014",
            "type": "multi_hop",
            "question": "Based on the limitations discussed, what future research directions could address the identified gaps?",
            "ground_truth": "Future directions may include scaling the approach, applying it to new domains, or addressing computational constraints.",
            "doc_hint": None,
        },
        {
            "id": "q015",
            "type": "multi_hop",
            "question": "How do the theoretical assumptions in the paper relate to the empirical results observed?",
            "ground_truth": "Theoretical assumptions should be validated by empirical results, though gaps may exist between theory and practice.",
            "doc_hint": None,
        },
        {
            "id": "q016",
            "type": "multi_hop",
            "question": "If the dataset size were doubled, what impact would you expect on the reported metrics based on the paper's analysis?",
            "ground_truth": "Based on scaling analyses, doubling the dataset typically leads to improved performance with diminishing returns.",
            "doc_hint": None,
        },
        # ── Out-of-Scope Questions (4) ──
        {
            "id": "q017",
            "type": "out_of_scope",
            "question": "What is the current stock price of NVIDIA?",
            "ground_truth": "This information is not available in the provided documents.",
            "doc_hint": None,
        },
        {
            "id": "q018",
            "type": "out_of_scope",
            "question": "What will the weather be like in New York tomorrow?",
            "ground_truth": "This information is not available in the provided documents.",
            "doc_hint": None,
        },
        {
            "id": "q019",
            "type": "out_of_scope",
            "question": "Who won the FIFA World Cup in 2022?",
            "ground_truth": "This information is not available in the provided documents.",
            "doc_hint": None,
        },
        {
            "id": "q020",
            "type": "out_of_scope",
            "question": "How do I cook a perfect risotto?",
            "ground_truth": "This information is not available in the provided documents.",
            "doc_hint": None,
        },
    ]
