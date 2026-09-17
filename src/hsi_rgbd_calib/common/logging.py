"""Logging utilities using Rich.

This module provides a consistent logging setup for the HSI-RGBD calibration toolkit.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


# Custom theme for the toolkit
THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "debug": "dim",
})

# Global console instance
console = Console(theme=THEME)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Set up logging with Rich handler.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to also log to a file.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add Rich handler for console output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(rich_handler)
    
    # Optionally add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def print_banner() -> None:
    """Print the toolkit banner."""
    from hsi_rgbd_calib import __version__
    
    console.print()
    console.print("[bold cyan]HSI-RGBD Calibration Kit[/bold cyan]", justify="center")
    console.print(f"[dim]Version {__version__}[/dim]", justify="center")
    console.print()


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[success][OK][/success] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[error][ERROR][/error] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning][WARN][/warning] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[info][INFO][/info] {message}")
