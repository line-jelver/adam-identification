"""Unified material data model for adam-identification.

Defines the :class:`Material` Pydantic model and its nested structure types.
Database clients (Materials Project, PubChem) return ``Material`` instances.
The :func:`~adam_identification.output.to_ase` and
:func:`~adam_identification.output.to_pymatgen` helpers convert these to
standard simulation-library objects.

Three-layer design
------------------
1. :class:`Material` — identity, database IDs, and a typed structure object.
2. :class:`CrystalStructure` / :class:`MoleculeStructure` — geometric
   description used as the DFT workflow input.
3. :class:`MaterialsProjectProperties` — optional physical properties attached
   by the Materials Project client.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class MaterialSource(StrEnum):
    """Records which database a material was fetched from."""

    MATERIALS_PROJECT = "materials_project"
    PUBCHEM = "pubchem"


class AtomicPosition(BaseModel):
    """A single atomic site within a material.

    Crystal positions are fractional (reduced) coordinates.
    Molecule positions are Cartesian coordinates (Å).
    """

    model_config = ConfigDict(frozen=True)

    element: str
    position: tuple[float, float, float]


class CrystalStructure(BaseModel):
    """Geometric description of a periodic solid.

    All coordinates are fractional (reduced). ``unit_cell`` rows are lattice
    vectors in Å.
    """

    model_config = ConfigDict(frozen=True)

    material_type: Literal["crystal"] = "crystal"

    unit_cell: list[list[float]]
    """3×3 matrix; rows are lattice vectors in Å."""
    lattice_parameters: dict[str, float]
    """Keys: a, b, c (Å), alpha, beta, gamma (degrees)."""
    atomic_positions: list[AtomicPosition]
    """Fractional (reduced) coordinates."""

    crystal_system: str
    space_group: str

    nelements: int
    nsites: int


class MoleculeStructure(BaseModel):
    """Geometric description of a finite molecule.

    ``atomic_positions`` are Cartesian coordinates (Å).
    """

    model_config = ConfigDict(frozen=True)

    material_type: Literal["molecule"] = "molecule"

    atomic_positions: list[AtomicPosition] = Field(min_length=1)
    """Cartesian coordinates in Å."""

    smiles: str | None = None
    inchi: str | None = None
    inchi_key: str | None = None

    nelements: int | None = None
    nsites: int | None = None


Structure = Annotated[
    CrystalStructure | MoleculeStructure,
    Field(discriminator="material_type"),
]


class BaseProperties(BaseModel):
    """Common metadata for property sets."""

    model_config = ConfigDict(frozen=True)

    source: MaterialSource
    method: str


class MaterialsProjectProperties(BaseProperties):
    """Bulk properties from the Materials Project (PBE/PBE+U level of theory).

    All fields are ``None`` when not returned by the MP API.
    """

    source: Literal[MaterialSource.MATERIALS_PROJECT] = MaterialSource.MATERIALS_PROJECT
    method: str = "unknown"

    energy_above_hull: float | None = None
    """eV/atom above the convex hull (thermodynamic stability)."""
    formation_energy_per_atom: float | None = None
    """eV/atom formation energy."""
    band_gap: float | None = None
    """DFT band gap in eV (tends to underestimate experiment)."""
    is_metal: bool | None = None
    is_direct_gap: bool | None = None
    magnetic_ordering: str | None = None


AnyProperties = Annotated[
    MaterialsProjectProperties,
    Field(discriminator="source"),
]


class Material(BaseModel):
    """Unified representation of a material returned by the identifier.

    Holds identity information, a typed structure object, and optional
    physical properties. Pass this object to :func:`~adam_identification.output.to_ase`
    or :func:`~adam_identification.output.to_pymatgen` to obtain standard
    simulation-library objects.

    Example::

        from adam_identification import identify
        material = identify("silicon", output="material")

        from adam_identification.output import to_ase, to_pymatgen
        atoms = to_ase(material)           # ase.Atoms
        structure = to_pymatgen(material)  # pymatgen.core.Structure
    """

    name: str
    chemical_formula: str
    source: MaterialSource

    mp_id: str | None = None
    """Materials Project ID, e.g. ``"mp-149"``."""
    pc_cid: int | None = None
    """PubChem Compound ID."""

    structure: Structure
    properties: list[MaterialsProjectProperties] = Field(default_factory=list)

    def get_properties(self, source: MaterialSource) -> MaterialsProjectProperties | None:
        """Return the first property entry matching ``source``, or ``None``.

        Args:
            source: The database source to look up.

        Returns:
            Matching properties object, or ``None`` if not present.
        """
        for p in self.properties:
            if p.source == source:
                return p
        return None

    @property
    def material_type(self) -> Literal["crystal", "molecule"]:
        """Convenience accessor — delegates to ``structure.material_type``."""
        return self.structure.material_type
