"""Unified material data model for adam-identification.

Defines the ``Material`` Pydantic model and its nested structure and properties
models. Database clients (Materials Project, PubChem) return ``Material``
instances. :func:`~adam_identification.output.to_ase` and
:func:`~adam_identification.output.to_pymatgen` convert these to simulation
objects.

Three-layer design
------------------
1. ``Material``         — identity, database IDs, and a typed structure object.
2. ``*Structure``       — geometric description used as DFT workflow input.
                          ``CrystalStructure`` for periodic solids;
                          ``MoleculeStructure`` for finite molecules.
3. ``*Properties``      — physical/chemical properties from a specific database
                          and method. Stored as a list so multiple sources can
                          coexist. Add a new database by subclassing
                          ``BaseProperties`` — ``Material`` itself never changes.

Cross-references
----------------
- ``adam_identification.database.materials_project`` — populates ``CrystalStructure``
  and ``MaterialsProjectProperties``.
- ``adam_identification.database.pubchem``           — populates ``MoleculeStructure``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Source registry ────────────────────────────────────────────────────────────


class MaterialSource(StrEnum):
    """Records which database a material or property set was fetched from."""

    MATERIALS_PROJECT = "materials_project"
    MC3D = "mc3d"
    PUBCHEM = "pubchem"
    PUBCHEM_QC = "pubchem_qc"
    USER_INPUT = "user_input"  # reserved for post-MVP user-supplied structures


# ── Shared structural primitive ────────────────────────────────────────────────


class AtomicPosition(BaseModel):
    """A single atomic site within a material.

    Crystal positions are fractional (reduced) coordinates.
    Molecule positions are Cartesian coordinates (Å).
    """

    model_config = ConfigDict(frozen=True)

    element: str
    position: tuple[float, float, float]


# ── Structure models ───────────────────────────────────────────────────────────


class CrystalStructure(BaseModel):
    """Geometric description of a periodic solid.

    All fields are required — a crystal without a complete unit cell description
    is not a valid input to the DFT workflow.

    Coordinate convention: ``atomic_positions`` are fractional (reduced)
    coordinates. ``unit_cell`` rows are lattice vectors in Å.
    Lattice parameters are stored as raw floats and never rounded —
    comparison tolerance is the responsibility of benchmark code.
    """

    model_config = ConfigDict(frozen=True)

    material_type: Literal["crystal"] = "crystal"

    # Unit cell geometry
    unit_cell: list[list[float]]  # 3×3 matrix; rows = lattice vectors (Å)
    lattice_parameters: dict[str, float]  # keys: a, b, c (Å), alpha, beta, gamma (°)
    atomic_positions: list[AtomicPosition]

    # Symmetry
    crystal_system: str  # e.g. "Cubic", "Hexagonal"
    space_group: str  # e.g. "Fd-3m"

    # Composition counts
    nelements: int
    nsites: int


class MoleculeStructure(BaseModel):
    """Geometric description of a finite molecule.

    ``atomic_positions`` are Cartesian coordinates (Å).
    ``unit_cell`` is always ``None`` — molecules have no periodic cell.

    A molecule without 3D coordinates is not valid workflow input. PubChem
    results with no conformer must be filtered out by the API client before
    constructing ``MoleculeStructure``.
    """

    model_config = ConfigDict(frozen=True)

    material_type: Literal["molecule"] = "molecule"

    atomic_positions: list[AtomicPosition] = Field(min_length=1)

    # Universal chemical identifiers
    smiles: str | None = None
    inchi: str | None = None
    inchi_key: str | None = None

    # Composition
    nelements: int | None = None
    nsites: int | None = None  # total atom count


# Discriminated union used as the ``structure`` field type on ``Material``
Structure = Annotated[
    CrystalStructure | MoleculeStructure,
    Field(discriminator="material_type"),
]


# ── Properties models ──────────────────────────────────────────────────────────


class BaseProperties(BaseModel):
    """Common metadata carried by every properties entry.

    Subclass this to add a new database. ``Material.properties`` is a list of
    ``BaseProperties`` subclasses, so multiple sources can coexist on a single
    material without changing the ``Material`` schema.
    """

    model_config = ConfigDict(frozen=True)

    source: MaterialSource
    method: str  # level of theory or provenance, e.g. "PBE+U", "B3LYP/6-31G*"


class MaterialsProjectProperties(BaseProperties):
    """Bulk properties from the Materials Project (PBE/PBE+U level of theory).

    Applies to crystals only. All energies are DFT-computed at the PBE or
    PBE+U level depending on the element set.

    ``method`` is **aspirational infrastructure** for self-documenting exports
    (e.g. benchmark CSV). The client does **not** resolve DFT run type from the API:
    ``materials/summary`` does not expose ``run_type``; accurate per-entry run
    type (GGA vs GGA+U vs r2SCAN, etc.) lives on ``materials/core`` and would
    require an extra API round-trip per material. The client therefore leaves
    ``method`` at the default ``"unknown"`` unless set elsewhere.

    Fields are ``None`` when MP did not return a value for the material.
    """

    source: Literal[MaterialSource.MATERIALS_PROJECT] = MaterialSource.MATERIALS_PROJECT
    method: str = "unknown"

    # Thermodynamic stability
    energy_above_hull: float | None = None  # eV/atom
    formation_energy_per_atom: float | None = None  # eV/atom

    # Electronic structure
    band_gap: float | None = None  # eV; DFT-computed (tends to underestimate)
    is_metal: bool | None = None
    is_direct_gap: bool | None = None

    # Magnetic
    magnetic_ordering: str | None = None  # e.g. "Ferromagnetic", "Non-magnetic"


class MC3DSourceRecord(BaseModel):
    """One original-database record an MC3D entry was derived from.

    An MC3D entry may cite more than one source record (e.g. the same
    experimental structure catalogued in both ICSD and MPDS). The complete
    ordered list is preserved rather than collapsed to a single label.
    """

    model_config = ConfigDict(frozen=True)

    database: str  # e.g. "icsd", "cod", "mpds"
    record_id: str
    version: str | None = None
    is_theoretical: bool | None = None
    is_high_pressure: bool | None = None
    is_high_temperature: bool | None = None
    # ``None`` means the API did not report the flag — never coerce to False.


class MC3DProvenanceLink(BaseModel):
    """One AiiDA provenance link exposed by an MC3D core record."""

    model_config = ConfigDict(frozen=True)

    label: str  # e.g. "Final relax calculation" (PBEsol-v2) or "Final structure"
    uuid: str
    url: str


class MC3DProperties(BaseProperties):
    """Bulk properties and provenance from Materials Cloud MC3D.

    Applies to crystals only. MC3D structures are DFT-relaxed geometries
    derived from a COD/ICSD/MPDS source record, computed at the ``method``
    methodology (``pbe-v1``, ``pbesol-v1``, or ``pbesol-v2``) — never label an
    MC3D geometry as the original experimental structure.

    Fields are ``None``/empty when MC3D did not return a value for the entry.
    """

    source: Literal[MaterialSource.MC3D] = MaterialSource.MC3D
    method: str  # "pbe-v1" / "pbesol-v1" / "pbesol-v2" — required, no default

    structure_uuid: str
    sources: list[MC3DSourceRecord] = Field(default_factory=list)

    total_energy_ev_per_cell: float | None = None
    cell_volume_ang3: float | None = None
    total_magnetization_mu_b_per_cell: float | None = None
    absolute_magnetization_mu_b_per_cell: float | None = None

    final_scf_uuid: str | None = None
    final_structure_uuid: str | None = None
    provenance_links: list[MC3DProvenanceLink] = Field(default_factory=list)
    record_url: str | None = None
    aiida_node_url: str | None = None


class PubChemQCProperties(BaseProperties):
    """Molecular electronic properties from the PubChemQC dataset.

    Applies to molecules only. All values are computed at the
    B3LYP/6-31G* level of theory using GAMESS (Nakata & Shimazaki,
    J. Chem. Inf. Model. 2017).

    Source: local lookup file generated by
    ``utils/generate_pubchemqc_lookup.py`` from the Hugging Face mirror
    ``molssiai-hub/pubchemqc-pm6``.

    All energies are stored in **eV**. The raw PubChemQC dataset uses
    Hartree; conversion (× 27.2114) is performed by ``PubChemQCClient``
    at ingestion time.
    """

    source: Literal[MaterialSource.PUBCHEM_QC] = MaterialSource.PUBCHEM_QC
    method: str = "B3LYP/6-31G*"

    total_energy: float | None = None  # eV (converted from Hartree at ingestion)
    homo_energy: float | None = None  # eV
    lumo_energy: float | None = None  # eV
    homo_lumo_gap: float | None = None  # eV
    dipole_moment: float | None = None  # Debye


# Discriminated union so Pydantic correctly deserialises subclass fields
# from JSON/dicts (model_validate). Without the discriminator, Pydantic would
# produce bare BaseProperties instances and silently drop subclass-specific fields.
AnyProperties = Annotated[
    MaterialsProjectProperties | MC3DProperties | PubChemQCProperties,
    Field(discriminator="source"),
]


# ── Top-level Material model ───────────────────────────────────────────────────


class Material(BaseModel):
    """Unified representation of a material for the ADaM DFT workflow.

    Holds three kinds of information:

    - **Identity** — human-readable name, formula, source database, and
      database-specific IDs (prefixed fields only).
    - **Structure** — a ``CrystalStructure`` or ``MoleculeStructure`` object
      that fully describes the geometry. This is the direct input to the
      Workflow Engine.
    - **Properties** — a list of ``BaseProperties`` subclass instances, one
      per database/method combination. The list is empty when a material is
      looked up purely for structure (e.g. from PubChem for the workflow).
      Access helpers are provided for common lookup patterns.

    Adding a new database
    ---------------------
    1. Add a value to ``MaterialSource``.
    2. Subclass ``BaseProperties`` with the new fields.
    3. Populate the new properties in the new API client.
    ``Material`` itself does not change.

    Example usage::

        from adam_identification.models import Material, MaterialSource

        # Crystal from Materials Project (structure + properties)
        si: Material = mp_client.get_by_id("mp-149")
        assert isinstance(si.structure, CrystalStructure)
        mp_props = si.get_properties(MaterialSource.MATERIALS_PROJECT)
        assert mp_props is not None
        print(mp_props.band_gap)

        # Molecule from PubChem (structure only — no properties)
        water: Material = pc_client.get_by_name("water")
        assert isinstance(water.structure, MoleculeStructure)
        assert water.properties == []
    """

    # ── Identity ───────────────────────────────────────────────────────────────
    name: str
    chemical_formula: str
    source: MaterialSource  # primary source this instance was fetched from

    # ── Database identifiers (prefixed — source-specific keys only) ────────────
    mp_id: str | None = None  # Materials Project ID, e.g. "mp-149"
    mc3d_id: str | None = None  # MC3D ID, e.g. "mc3d-18058"
    pc_cid: int | None = None  # PubChem Compound ID

    # ── Structure ──────────────────────────────────────────────────────────────
    structure: Structure

    # ── Properties (one entry per database/method; list is often empty) ────────
    properties: list[AnyProperties] = Field(default_factory=list)

    # ── Access helpers ─────────────────────────────────────────────────────────

    def get_properties(self, source: MaterialSource) -> AnyProperties | None:
        """Return the first properties entry matching ``source``, or ``None``.

        Example::

            mp_props = material.get_properties(MaterialSource.MATERIALS_PROJECT)
            if mp_props:
                print(mp_props.band_gap)
        """
        for p in self.properties:
            if p.source == source:
                return p
        return None

    @property
    def material_type(self) -> Literal["crystal", "molecule"]:
        """Convenience accessor — delegates to ``structure.material_type``."""
        return self.structure.material_type

    @property
    def primary_id(self) -> str | None:
        """Return the identifier associated with :attr:`source`.

        Returns ``mp_id`` for Materials Project, ``mc3d_id`` for MC3D, and the
        string form of ``pc_cid`` for PubChem/PubChemQC. ``None`` for sources
        without an applicable database ID, or when the expected ID field was
        never populated.
        """
        if self.source == MaterialSource.MATERIALS_PROJECT:
            return self.mp_id
        if self.source == MaterialSource.MC3D:
            return self.mc3d_id
        if self.source in (MaterialSource.PUBCHEM, MaterialSource.PUBCHEM_QC):
            return str(self.pc_cid) if self.pc_cid is not None else None
        return None
