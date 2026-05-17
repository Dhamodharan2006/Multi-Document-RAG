"""Enumerations used across the Multi-Document RAG system."""

from enum import Enum


class DocType(str, Enum):
    """Supported document type classifications."""
    RESEARCH_PAPER = "research_paper"
    REPORT = "report"
    NOTES = "notes"
    DATA = "data"
    UNKNOWN = "unknown"


class QueryMode(str, Enum):
    """Query processing modes."""
    STANDARD = "standard"
    REASONING = "reasoning"


class EvalStatus(str, Enum):
    """Evaluation pipeline status states."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestStatus(str, Enum):
    """Document ingestion status states."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
