"""Logging configuration for adam-identification.

Silences noisy third-party loggers (mp_api, pymatgen, emmet) and configures a
clean Rich handler for user-facing output. Called at package import time so
the terminal stays clean when running the CLI or calling identify() from a
script.
"""

from __future__ import annotations

import logging
import warnings

# Third-party libraries that emit verbose output by default.
_NOISY_LOGGERS = [
    "mp_api",
    "pymatgen",
    "emmet",
    "monty",
    "urllib3",
    "httpx",
    "asyncio",
]


def configure_logging(verbose: bool = False) -> None:
    """Configure logging for adam-identification.

    Silences noisy third-party loggers and sets up a Rich console handler
    for the ``adam_identification`` namespace.

    Args:
        verbose: When ``True``, set ``adam_identification`` logger to ``DEBUG``
            level (shows LLM prompts, MP search details, etc.).
    """
    # Suppress noisy dependencies regardless of verbosity setting.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="mp_api")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymatgen")

    # Configure the adam_identification logger with a Rich handler.
    pkg_logger = logging.getLogger("adam_identification")
    if not pkg_logger.handlers:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                show_time=False,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
            pkg_logger.addHandler(handler)
        except ImportError:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            pkg_logger.addHandler(handler)

    pkg_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    pkg_logger.propagate = False
