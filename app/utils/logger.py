"""Structured logging setup using loguru."""

import sys

from loguru import logger

from app.config import settings


def setup_logging() -> None:
    """Configure loguru with structured format and appropriate log level.

    Removes default handlers and adds a new stderr handler with
    the configured log level and structured format.
    """
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        "logs/app.log",
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=False,
    )

    logger.info("Logging configured at level: {}", settings.log_level)


def get_logger(name: str = __name__) -> logger.__class__:
    """Return a contextualized logger with the given name.

    Args:
        name: The logger context name, typically __name__.

    Returns:
        A loguru logger instance bound with the context name.
    """
    return logger.bind(context=name)
