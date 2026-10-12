"""Logging configuration for SlideShift using rich formatting."""

import logging
import sys
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger(
    name: str = "slideshift",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configures a logger with RichHandler and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if re-called
    if not logger.handlers:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(level)
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


logger = setup_logger()

