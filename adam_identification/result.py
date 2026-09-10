"""Structured identification output with a full provenance :class:`Trace`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adam_identification.models import Material
from adam_identification.provenance.trace import Trace


@dataclass
class IdentificationResult:
    """Identification output with a unified provenance :class:`Trace`.

    Returned by :func:`~adam_identification.identify` when ``output="result"``.

    Attributes:
        atoms: ``ase.Atoms`` object when ASE is available, else ``None``.
        material: Raw :class:`~adam_identification.models.Material`.
        trace: Full :class:`~adam_identification.provenance.trace.Trace`.
            Access the decision trail via ``trace.identification``.
    """

    atoms: Any
    material: Material
    trace: Trace
