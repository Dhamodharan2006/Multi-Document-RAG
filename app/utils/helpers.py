"""Shared utility functions used across the application."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def generate_doc_id() -> str:
    """Generate a unique document identifier.

    Returns:
        A UUID4 string to uniquely identify a document.
    """
    return str(uuid.uuid4())


def generate_eval_id() -> str:
    """Generate a unique evaluation run identifier.

    Returns:
        A prefixed UUID4 string for evaluation tracking.
    """
    return f"eval-{uuid.uuid4().hex[:12]}"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        ISO 8601 formatted UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def get_file_size_kb(file_path: str | Path) -> float:
    """Calculate file size in kilobytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in KB, rounded to 2 decimal places.
    """
    return round(Path(file_path).stat().st_size / 1024, 2)


def get_file_extension(filename: str) -> str:
    """Extract the lowercase file extension from a filename.

    Args:
        filename: The filename to extract extension from.

    Returns:
        Lowercase file extension without the dot, or empty string.
    """
    ext = Path(filename).suffix.lower()
    return ext.lstrip(".") if ext else ""


def timer_ms(start_time: float) -> int:
    """Calculate elapsed time in milliseconds from a start time.

    Args:
        start_time: The start time from time.perf_counter().

    Returns:
        Elapsed time in milliseconds as an integer.
    """
    return int((time.perf_counter() - start_time) * 1000)


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length with ellipsis.

    Args:
        text: Text to truncate.
        max_length: Maximum character count before truncation.

    Returns:
        Truncated text with '...' appended if it exceeded max_length.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def batch_list(items: list, batch_size: int) -> list[list]:
    """Split a list into batches of the specified size.

    Args:
        items: The list to split into batches.
        batch_size: Maximum number of items per batch.

    Returns:
        A list of lists, each containing at most batch_size items.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
