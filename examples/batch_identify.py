"""Example: high-throughput batch identification.

Identifies multiple crystals and molecules concurrently. Useful for
preparing input structures for DFT workflows at scale.

Prerequisites:
    pip install adam-identification
    export GOOGLE_API_KEY=your_key
    export MATERIALS_PROJECT_API_KEY=your_key

Run:
    python examples/batch_identify.py
"""

from adam_identification import batch_identify
from adam_identification.models import Material

queries = [
    "silicon",
    "water",
    "caffeine",
    "iron (BCC)",
    "sodium chloride",
    "aspirin",
    "graphite",
    "ammonia",
]

print(f"Identifying {len(queries)} materials concurrently (concurrency=4)...\n")

results = batch_identify(
    queries,
    provider="gemini",
    model="gemini-2.5-flash",
    concurrency=4,
    output="ase",
)

for query, result in zip(queries, results):
    if isinstance(result, Exception):
        print(f"  [{query}] FAILED: {result}")
    else:
        atoms = result
        pbc_str = "periodic" if any(atoms.pbc) else "molecule"
        print(f"  [{query}] {atoms.symbols} ({pbc_str}, {len(atoms)} atoms)")
