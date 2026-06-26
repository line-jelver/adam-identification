# adam-identification

Database-grounded LLM agent for atomistic structure identification from natural-language descriptions.

Given a description like `"silicon"`, `"caffeine"`, or `"hexagonal boron nitride"`, the agent:

1. Uses a language model to extract the chemical formula and any disambiguation hints (polymorph, space group, SMILES)
2. Queries the **Materials Project** (crystals) or **PubChem** (molecules) for matching candidates
3. Selects the correct phase or structural isomer
4. Returns an `ase.Atoms` or `pymatgen.Structure` object ready for first-principles calculations

This implements the **database-grounded** (`methodDB`) workflow benchmarked in:

> Line Jelver et al., *"Database-grounded language-model agents for robust atomistic input-structure identification in solids and molecules"*, 2026.
> [[DOI — to be added upon publication]]

---

## Installation

```bash
pip install adam-identification
```

**Python 3.11+ required.**

---

## API keys

The package requires API keys loaded from environment variables or a `.env` file in the working directory:

| Variable | Purpose | Required for |
|---|---|---|
| `MATERIALS_PROJECT_API_KEY` | Materials Project API | Crystal identification |
| `GOOGLE_API_KEY` | Google Gemini | Default LLM provider |
| `OPENAI_API_KEY` | OpenAI | Optional LLM provider |
| `ANTHROPIC_API_KEY` | Anthropic Claude | Optional LLM provider |
| `OPENROUTER_API_KEY` | OpenRouter (DeepSeek, Kimi, Qwen, …) | Optional LLM provider |

Get a Materials Project API key at https://next-gen.materialsproject.org/api

Create a `.env` file:
```
MATERIALS_PROJECT_API_KEY=your_mp_key
GOOGLE_API_KEY=your_google_key
```

---

## Quick start

### Python API

```python
from adam_identification import identify

# Returns an ase.Atoms object (default)
atoms = identify("silicon")
atoms = identify("caffeine")
atoms = identify("hexagonal boron nitride")

# Returns a pymatgen Structure / Molecule
structure = identify("silicon", output="pymatgen")
molecule = identify("caffeine", output="pymatgen")

# Use a specific provider and model
atoms = identify("silicon", provider="openai", model="gpt-5.4-mini")
atoms = identify("water", provider="anthropic", model="claude-haiku-4-5-20251001")
atoms = identify("iron", provider="openrouter", model="deepseek/deepseek-chat-v3-5")

# Access the raw Material object (includes MP properties)
material = identify("silicon", output="material")
print(material.mp_id)                         # e.g. "mp-149"
print(material.structure.space_group)         # "Fd-3m"
print(material.properties[0].band_gap)        # DFT band gap in eV
```

### High-throughput (batch)

```python
from adam_identification import batch_identify

queries = ["silicon", "water", "caffeine", "iron (BCC)", "alpha-alumina"]

results = batch_identify(
    queries,
    provider="gemini",
    model="gemini-2.5-flash",
    concurrency=5,          # simultaneous API calls
    output="ase",           # or "pymatgen" or "material"
)

for query, result in zip(queries, results):
    if isinstance(result, Exception):
        print(f"{query}: FAILED — {result}")
    else:
        print(f"{query}: {result.symbols}")
```

### CLI

```bash
# Single query
adam-identify "silicon"
adam-identify "caffeine"

# With specific provider and model
adam-identify "BCC iron" --provider openai --model gpt-5.4-mini

# Save to file (CIF for crystals, XYZ for molecules)
adam-identify "silicon" --save silicon.cif
adam-identify "caffeine" --save caffeine.xyz

# Batch mode: one query per line in a text file
adam-identify --batch queries.txt --concurrency 5
adam-identify --batch queries.txt --save-dir structures/

# List available providers
adam-identify --list-providers

# Verbose (shows LLM reasoning steps)
adam-identify "silicon" --verbose
```

---

## Supported LLM providers

| Key | Models tested in the paper | Notes |
|---|---|---|
| `gemini` | `gemini-2.5-flash` (recommended), `gemini-2.5-flash-lite`, `gemini-3.1-pro-preview` | Default provider |
| `openai` | `gpt-5.4-mini`, `gpt-5.5` | |
| `anthropic` | `claude-haiku-4-5-20251001`, `claude-sonnet-4-6` | |
| `openrouter` | `deepseek/deepseek-chat-v3-5`, `moonshotai/kimi-k2.6`, `qwen/qwen3.7-max` | Access 300+ models |

The paper shows that **Gemini 2.5 Flash** provides the best cost-performance trade-off for the database-grounded workflow: near-perfect success rates at ~$0.001 per identified structure.

---

## Output objects

| `output=` | Crystal | Molecule |
|---|---|---|
| `"ase"` | `ase.Atoms` with `pbc=True` | `ase.Atoms` with `pbc=False` |
| `"pymatgen"` | `pymatgen.core.Structure` | `pymatgen.core.Molecule` |
| `"material"` | `adam_identification.models.Material` | same |

The `Material` object carries:
- `material.chemical_formula` — reduced formula
- `material.mp_id` — Materials Project ID (crystals)
- `material.pc_cid` — PubChem Compound ID (molecules)
- `material.structure` — `CrystalStructure` or `MoleculeStructure` with coordinates
- `material.properties[0]` — `MaterialsProjectProperties` with band gap, energy above hull, etc.

---

## Error handling

```python
from adam_identification import identify
from adam_identification.exceptions import MaterialNotFoundError, LLMError

try:
    atoms = identify("tungsten trioxide monoclinic")
except MaterialNotFoundError as exc:
    print(f"Not found in database: {exc}")
except LLMError as exc:
    print(f"LLM error (rate limit or auth): {exc}")
```

---

## Citation

If you use this software in scientific work, please cite:

```bibtex
@article{jelver2026identification,
  title   = {Database-grounded language-model agents for robust atomistic
             input-structure identification in solids and molecules},
  author  = {Jelver, Line and {ADaM collaboration}},
  journal = {TBD},
  year    = {2026},
  doi     = {TBD}
}
```

A `CITATION.cff` file is included; GitHub displays a **"Cite this repository"** button automatically.

---

## Development

```bash
git clone https://github.com/line-jelver/adam-identification
cd adam-identification
pip install -e ".[dev]"

# Run tests (no API keys needed — all mocked)
pytest tests/ -v

# Lint and type-check
ruff check adam_identification/
mypy adam_identification/
```

---

## License

MIT — see [LICENSE](LICENSE).
See [NOTICE](NOTICE) for the citation request and third-party attributions.
