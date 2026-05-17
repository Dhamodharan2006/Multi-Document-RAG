"""Automatic metadata extraction and tagging for ingested documents."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from app.models.enums import DocType
from app.utils.helpers import generate_doc_id, get_file_size_kb, utc_now_iso


def auto_tag(file_path: str | Path, source: str = "upload") -> dict:
    """Extract and generate metadata tags for a document.

    Infers document type from filename patterns and extension,
    extracts publication year from filename if present, and
    generates a unique document ID.

    Args:
        file_path: Path to the document file.
        source: Source origin of the document (e.g., "arxiv", "upload", "pubmed").

    Returns:
        A dictionary containing all metadata fields:
            - doc_id: Unique UUID4 identifier
            - filename: Basename of the file
            - doc_type: Inferred document classification
            - source: Origin source string
            - pub_year: Extracted year or None
            - ingested_at: UTC ISO 8601 timestamp
            - file_size_kb: File size in kilobytes
    """
    path = Path(file_path)
    filename = path.name

    doc_id = generate_doc_id()
    doc_type = _infer_doc_type(filename)
    pub_year = _extract_year(filename)
    file_size_kb = get_file_size_kb(path) if path.exists() else 0.0

    metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "doc_type": doc_type,
        "source": source,
        "pub_year": pub_year,
        "ingested_at": utc_now_iso(),
        "file_size_kb": file_size_kb,
    }

    logger.info(
        "Auto-tagged document: {} → type={}, year={}, size={}KB",
        filename,
        doc_type,
        pub_year,
        file_size_kb,
    )
    return metadata


def _infer_doc_type(filename: str) -> str:
    """Infer document type from filename patterns and extension.

    Args:
        filename: The document filename to classify.

    Returns:
        A string matching one of the DocType enum values.
    """
    name_lower = filename.lower()

    # Check for research paper indicators
    paper_patterns = [
        "paper", "arxiv", "ieee", "acm", "journal", "conference",
        "proceedings", "manuscript", "preprint", "publication",
    ]
    if any(pattern in name_lower for pattern in paper_patterns):
        return DocType.RESEARCH_PAPER.value

    # Check for report indicators
    report_patterns = ["report", "survey", "review", "analysis", "whitepaper"]
    if any(pattern in name_lower for pattern in report_patterns):
        return DocType.REPORT.value

    # Check for notes indicators
    notes_patterns = ["notes", "summary", "outline", "draft", "memo"]
    if any(pattern in name_lower for pattern in notes_patterns):
        return DocType.NOTES.value

    # Check for data indicators
    data_patterns = ["data", "dataset", "appendix", "supplement", "table"]
    if any(pattern in name_lower for pattern in data_patterns):
        return DocType.DATA.value

    # Default: classify PDFs as research papers, others as unknown
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return DocType.RESEARCH_PAPER.value

    return DocType.UNKNOWN.value


def _extract_year(filename: str) -> int | None:
    """Extract a 4-digit publication year from the filename.

    Searches for patterns like 2019, 2023, etc. in the filename.

    Args:
        filename: The filename to search for year patterns.

    Returns:
        The extracted year as an integer, or None if no year found.
    """
    match = re.search(r"(20\d{2})", filename)
    if match:
        year = int(match.group(1))
        logger.debug("Extracted year {} from filename: {}", year, filename)
        return year
    return None
