"""Convert :class:`~adam_identification.models.Material` to ASE or pymatgen objects.

These adapters are the primary user-facing output. Both ``ase`` and ``pymatgen``
are already installed as transitive dependencies of ``mp-api``.

Example::

    from adam_identification import identify
    from adam_identification.output import to_ase, to_pymatgen

    material = identify("silicon", output="material")
    atoms = to_ase(material)            # ase.Atoms with pbc=True
    structure = to_pymatgen(material)   # pymatgen.core.IStructure
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adam_identification.models import CrystalStructure, Material, MoleculeStructure

if TYPE_CHECKING:
    import ase
    import pymatgen.core


def to_ase(material: Material) -> "ase.Atoms":
    """Convert a :class:`~adam_identification.models.Material` to an ``ase.Atoms`` object.

    For crystals: ``pbc=True``, ``cell`` set to the 3×3 lattice matrix,
    ``scaled_positions`` used for fractional coordinates.

    For molecules: ``pbc=False``, ``positions`` in Cartesian Å.

    Args:
        material: Material returned by
            :meth:`~adam_identification.identifier.MaterialIdentifier.identify`.

    Returns:
        ``ase.Atoms`` object.

    Raises:
        ImportError: If ``ase`` is not installed.
    """
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "ase is required for to_ase(). Install with: pip install ase"
        ) from exc

    struct = material.structure
    symbols = [p.element for p in struct.atomic_positions]
    positions = [list(p.position) for p in struct.atomic_positions]

    if isinstance(struct, CrystalStructure):
        return Atoms(
            symbols=symbols,
            scaled_positions=positions,
            cell=struct.unit_cell,
            pbc=True,
        )
    elif isinstance(struct, MoleculeStructure):
        return Atoms(
            symbols=symbols,
            positions=positions,
            pbc=False,
        )
    else:
        raise TypeError(f"Unknown structure type: {type(struct)}")


def to_pymatgen(material: Material) -> "pymatgen.core.IStructure | pymatgen.core.IMolecule":
    """Convert a :class:`~adam_identification.models.Material` to a pymatgen object.

    For crystals: returns a ``pymatgen.core.Structure`` (periodic, with
    fractional coordinates and a ``Lattice``).

    For molecules: returns a ``pymatgen.core.Molecule`` (Cartesian coordinates,
    no periodicity).

    Args:
        material: Material returned by
            :meth:`~adam_identification.identifier.MaterialIdentifier.identify`.

    Returns:
        ``pymatgen.core.Structure`` for crystals,
        ``pymatgen.core.Molecule`` for molecules.

    Raises:
        ImportError: If ``pymatgen`` is not installed.
    """
    try:
        from pymatgen.core import Lattice, Molecule, Structure
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for to_pymatgen(). Install with: pip install pymatgen"
        ) from exc

    struct = material.structure
    species = [p.element for p in struct.atomic_positions]
    coords = [list(p.position) for p in struct.atomic_positions]

    if isinstance(struct, CrystalStructure):
        lattice = Lattice(struct.unit_cell)
        return Structure(lattice, species, coords, coords_are_cartesian=False)
    elif isinstance(struct, MoleculeStructure):
        return Molecule(species, coords)
    else:
        raise TypeError(f"Unknown structure type: {type(struct)}")
