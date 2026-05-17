"""Document loaders for PDF, DOCX, and TXT files."""

from __future__ import annotations

from pathlib import Path

from loguru import logger


async def load_document(file_path: str | Path) -> str:
    """Load and extract text from a document file.

    Supports PDF, DOCX, and TXT formats. Dispatches to the appropriate
    loader based on file extension.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted text content as a string.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    logger.info("Loading document: {} (type: {})", path.name, ext)

    if ext == ".pdf":
        return await _load_pdf(path)
    elif ext == ".docx":
        return await _load_docx(path)
    elif ext == ".txt":
        return await _load_txt(path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            "Supported formats: .pdf, .docx, .txt"
        )


async def _load_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf.

    Args:
        path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    import asyncio

    def _extract() -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages_text: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        logger.debug("Extracted {} pages from PDF: {}", len(pages_text), path.name)
        return "\n\n".join(pages_text)

    return await asyncio.to_thread(_extract)


async def _load_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx.

    Args:
        path: Path to the DOCX file.

    Returns:
        Concatenated text from all paragraphs.
    """
    import asyncio

    def _extract() -> str:
        from docx import Document

        doc = Document(str(path))
        paragraphs: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        logger.debug(
            "Extracted {} paragraphs from DOCX: {}", len(paragraphs), path.name
        )
        return "\n\n".join(paragraphs)

    return await asyncio.to_thread(_extract)


async def _load_txt(path: Path) -> str:
    """Read text from a plain text file.

    Args:
        path: Path to the TXT file.

    Returns:
        File contents as a string.
    """
    import asyncio

    def _read() -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        logger.debug("Read {} characters from TXT: {}", len(text), path.name)
        return text

    return await asyncio.to_thread(_read)
