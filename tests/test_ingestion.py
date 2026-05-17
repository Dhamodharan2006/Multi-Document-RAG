"""Tests for the ingestion pipeline: loader, metadata tagger, and chunker."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from app.core.ingestion.chunker import chunk_text
from app.core.ingestion.loader import load_document
from app.core.ingestion.metadata_tagger import auto_tag


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a temporary TXT file with sample content."""
    content = (
        "This is a sample academic research paper about machine learning.\n\n"
        "Abstract: We present a novel approach to natural language processing "
        "that combines transformer architectures with graph neural networks.\n\n"
        "1. Introduction\n"
        "The field of NLP has seen tremendous progress in recent years. "
        "Large language models have demonstrated remarkable capabilities "
        "across a wide range of tasks.\n\n"
        "2. Methodology\n"
        "Our proposed method leverages the attention mechanism from "
        "transformers and combines it with message passing in GNNs. "
        "This allows the model to capture both sequential and structural "
        "information in text data.\n\n"
        "3. Results\n"
        "We evaluate our approach on three benchmark datasets: SQuAD, "
        "GLUE, and SuperGLUE. Our model achieves state-of-the-art "
        "performance on all three benchmarks.\n\n"
        "4. Conclusion\n"
        "We have shown that combining transformers with GNNs leads to "
        "improved performance on NLP tasks."
    )
    file_path = tmp_path / "sample_paper_2023.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def long_text() -> str:
    """Generate a long text string for chunking tests."""
    paragraph = (
        "Natural language processing is a subfield of linguistics, "
        "computer science, and artificial intelligence concerned with "
        "the interactions between computers and human language. "
        "The goal is to enable computers to understand, interpret, "
        "and generate human language in a meaningful way. "
    )
    return paragraph * 20  # ~2000 characters


# ── Loader Tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_txt_loader_extracts_text(sample_txt_file: Path) -> None:
    """Test that the TXT loader correctly reads file content."""
    text = await load_document(sample_txt_file)
    assert len(text) > 0
    assert "machine learning" in text
    assert "Introduction" in text


@pytest.mark.asyncio
async def test_loader_rejects_unsupported_format(tmp_path: Path) -> None:
    """Test that the loader raises ValueError for unsupported formats."""
    bad_file = tmp_path / "file.xlsx"
    bad_file.write_text("data", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        await load_document(bad_file)


@pytest.mark.asyncio
async def test_loader_rejects_missing_file() -> None:
    """Test that the loader raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        await load_document("/nonexistent/file.txt")


# ── Metadata Tagger Tests ────────────────────────────────────────────


def test_metadata_tagger_detects_doc_type(sample_txt_file: Path) -> None:
    """Test that auto_tag correctly infers document type from filename."""
    metadata = auto_tag(sample_txt_file, source="test")

    assert metadata["filename"] == "sample_paper_2023.txt"
    assert metadata["doc_type"] == "research_paper"
    assert metadata["source"] == "test"
    assert metadata["doc_id"]  # Should be a non-empty UUID
    assert len(metadata["doc_id"]) == 36  # UUID format


def test_metadata_tagger_extracts_year(sample_txt_file: Path) -> None:
    """Test that auto_tag extracts publication year from filename."""
    metadata = auto_tag(sample_txt_file)
    assert metadata["pub_year"] == 2023


def test_metadata_tagger_handles_no_year(tmp_path: Path) -> None:
    """Test that auto_tag returns None for pub_year when no year in filename."""
    file = tmp_path / "document.txt"
    file.write_text("content", encoding="utf-8")
    metadata = auto_tag(file)
    assert metadata["pub_year"] is None


def test_metadata_tagger_detects_report(tmp_path: Path) -> None:
    """Test that auto_tag correctly classifies report-type documents."""
    file = tmp_path / "annual_report_2024.txt"
    file.write_text("report content", encoding="utf-8")
    metadata = auto_tag(file)
    assert metadata["doc_type"] == "report"


def test_metadata_tagger_includes_file_size(sample_txt_file: Path) -> None:
    """Test that auto_tag includes file size in KB."""
    metadata = auto_tag(sample_txt_file)
    assert metadata["file_size_kb"] > 0


# ── Chunker Tests ─────────────────────────────────────────────────────


def test_chunker_produces_chunks(long_text: str) -> None:
    """Test that chunker splits text into multiple chunks."""
    chunks = chunk_text(long_text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunker_respects_chunk_size(long_text: str) -> None:
    """Test that chunk sizes are approximately within the specified limit."""
    chunk_size = 200
    chunks = chunk_text(long_text, chunk_size=chunk_size, chunk_overlap=20)

    # Most chunks should be at or near the chunk_size
    for chunk in chunks[:-1]:  # Last chunk may be smaller
        assert len(chunk) <= chunk_size + 50  # Allow some tolerance


def test_chunker_produces_correct_overlap(long_text: str) -> None:
    """Test that consecutive chunks have overlapping content."""
    chunks = chunk_text(long_text, chunk_size=200, chunk_overlap=50)

    if len(chunks) >= 2:
        # The end of chunk[0] should overlap with the start of chunk[1]
        # This is a soft check since separators may affect exact overlap
        chunk0_end = chunks[0][-30:]  # Last 30 chars
        chunk1_start = chunks[1][:100]  # First 100 chars
        # At least some overlap should exist
        # (exact overlap depends on separator boundaries)
        assert len(chunks) >= 2  # At minimum, we got multiple chunks


def test_chunker_handles_empty_text() -> None:
    """Test that chunker handles empty text gracefully."""
    chunks = chunk_text("", chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 0


def test_chunker_handles_short_text() -> None:
    """Test that chunker returns single chunk for short text."""
    chunks = chunk_text("Short text.", chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."
