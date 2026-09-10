"""Crash-safe persistence of a :class:`~adam_identification.provenance.trace.Trace`.

Writes ``adam.json`` via ``adam.json.tmp``, ``flush`` + ``fsync``, then
:func:`os.replace`. A leftover ``RUNNING`` file is left as ``RUNNING`` on load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from adam_identification.provenance.trace import Trace


class TraceStore:
    """Atomic persistence of a :class:`~adam_identification.provenance.trace.Trace`.

    Args:
        work_dir: Directory where ``adam.json`` is written. Created if missing.
    """

    _FNAME: str = "adam.json"
    _TMP_FNAME: str = "adam.json.tmp"

    def __init__(self, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        self._path = work_dir / self._FNAME
        self._tmp = work_dir / self._TMP_FNAME
        self._tmp.unlink(missing_ok=True)

    @property
    def path(self) -> Path:
        """Absolute path to ``adam.json``."""
        return self._path

    def save(self, trace: Trace) -> None:
        """Atomically write *trace* to ``adam.json``."""
        payload = trace.model_dump(mode="json")
        text = json.dumps(payload, indent=2) + "\n"
        with open(self._tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(self._tmp, self._path)

    @classmethod
    def load(cls, path: Path) -> Trace:
        """Load a :class:`Trace` from ``adam.json`` without rewriting ``RUNNING``."""
        data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        return Trace.model_validate(data)
