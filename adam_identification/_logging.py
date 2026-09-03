"""Logging configuration for adam-identification.

Silences noisy third-party loggers (mp_api, pymatgen, emmet) and configures a
clean Rich handler for user-facing output. Called from the CLI so the terminal
stays quiet unless ``--verbose`` is set.

Cross-references:
    ``adam_identification.cli`` — calls :func:`configure_logging`.
"""

from __future__ import annotations

import logging
import os
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

    Silences noisy dependencies. Default CLI output is WARNING (summary panel
    only). ``--verbose`` enables DEBUG logs and Materials Project progress bars.

    Args:
        verbose: When ``True``, set ``adam_identification`` to ``DEBUG``.
    """
    if verbose:
        os.environ.pop("TQDM_DISABLE", None)
    else:
        os.environ["TQDM_DISABLE"] = "1"

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="mp_api")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymatgen")

    pkg_logger = logging.getLogger("adam_identification")
    if not pkg_logger.handlers:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
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

    pkg_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    pkg_logger.propagate = False
