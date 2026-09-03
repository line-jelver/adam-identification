"""Structured identification output with provenance trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adam_identification.models import Material
from adam_identification.trace import IdentificationTrace


@dataclass
class IdentificationResult:
    """Identification output with full provenance trace.

    Returned by :func:`~adam_identification.identify` when ``output="result"``.

    Attributes:
        atoms: ``ase.Atoms`` object when ASE is available, else ``None``.
        material: Raw :class:`~adam_identification.models.Material`.
        trace: Full :class:`~adam_identification.trace.IdentificationTrace`.
    """

    atoms: Any
    material: Material
    trace: IdentificationTrace
