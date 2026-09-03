"""Example: identify a crystal and convert to ASE Atoms / pymatgen Structure.

Prerequisites:
    pip install adam-identification
    export GOOGLE_API_KEY=your_key
    export MATERIALS_PROJECT_API_KEY=your_key

Run:
    python examples/identify_crystal.py
"""

from adam_identification import identify
from adam_identification.output import to_ase, to_pymatgen

MODEL = "gemini-3.1-pro-preview"

queries = [
    "silicon",
    "rutile TiO2",
    "iron (BCC)",
    "alpha-alumina",
]

for query in queries:
    print(f"\nIdentifying: {query!r}")
    try:
        material = identify(query, model=MODEL, output="material")
        struct = material.structure
        print(f"  Formula:     {material.chemical_formula}")
        print(f"  MP ID:       {material.mp_id}")
        print(f"  Space group: {struct.space_group} ({struct.crystal_system})")
        a, b, c = struct.lattice_parameters["a"], struct.lattice_parameters["b"], struct.lattice_parameters["c"]
        print(f"  Lattice:     a={a:.4g}Å  b={b:.4g}Å  c={c:.4g}Å")

        atoms = to_ase(material)
        print(f"  ASE Atoms:   {atoms.symbols} (pbc={atoms.pbc})")

        structure = to_pymatgen(material)
        print(f"  pymatgen:    {structure.formula} in {structure.get_space_group_info()}")
    except Exception as exc:
        print(f"  FAILED: {exc}")
