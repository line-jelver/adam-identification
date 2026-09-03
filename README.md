# adam-identification

Database-grounded LLM agent for atomistic structure identification from natural-language descriptions.

Given a description like `"silicon"`, `"caffeine"`, or `"rutile TiO2"`, the agent:

1. Uses a language model to extract the chemical formula and any disambiguation hints (polymorph, space group, SMILES)
2. Queries the **Materials Project** (crystals) or **PubChem** (molecules) for matching candidates
3. Selects the correct phase or structural isomer
4. Returns an `ase.Atoms` or `pymatgen.Structure` object ready for first-principles calculations

This implements the **database-grounded** (`methodDB`) workflow benchmarked in:

> Line Jelver et al., *"Database-grounded language-model agents for robust atomistic input-structure identification in solids and molecules"*, 2026.
> [[DOI — to be added upon publication]]

### How it works

```
User query
    │
    ▼
Formula extraction (LLM)
    │ decision = "proceed"         decision = "clarify"
    │                                  └─ ClarificationNeededError
    ▼
  Crystal path                  Molecule path
     │                               │
     │ MP search (≤20)               │ PubChem name search (≤5)
     │ LLM phase selection           │ formula filter
     │   "not_found" → widen (≤50)   │ LLM isomer selection
     │                               │ 3D fetch for winner
     │                               │
     └────────────┬──────────────────┘
                  │ decision = "ambiguous" → AmbiguousIdentificationError
                  │ decision = "select"
                  ▼
              Material
         (ase.Atoms / pymatgen / IdentificationResult)
```

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
| `GOOGLE_API_KEY` | Google Gemini | LLM provider key ``google`` |
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

You must pass a **model slug** (`model=` / `--model`). There is no default.
Smaller or cheaper models often fail to abstain on underspecified queries
(for example they may pick one TiO2 polymorph instead of listing options).

### Python API

```python
from adam_identification import identify

MODEL = "gemini-3.1-pro-preview"  # any model slug your provider accepts

# Returns an ase.Atoms object (default)
atoms = identify("silicon", model=MODEL)
atoms = identify("caffeine", model=MODEL)
atoms = identify("rutile TiO2", model=MODEL)

# Returns a pymatgen Structure / Molecule
structure = identify("silicon", model=MODEL, output="pymatgen")
molecule = identify("caffeine", model=MODEL, output="pymatgen")

# Other providers
atoms = identify("silicon", provider="openai", model="gpt-5.4-mini")
atoms = identify("water", provider="anthropic", model="claude-haiku-4-5-20251001")
atoms = identify("iron (alpha)", provider="openrouter", model="deepseek/deepseek-chat-v3-5")

# Access the raw Material object (includes MP properties)
material = identify("silicon", model=MODEL, output="material")
print(material.mp_id)                         # e.g. "mp-149"
print(material.structure.space_group)         # "Fd-3m"
print(material.properties[0].band_gap)        # DFT band gap in eV

# Full provenance (atoms + Material + IdentificationTrace)
result = identify("silicon", model=MODEL, output="result")
result.atoms
result.material.mp_id
result.trace.outcome
result.trace.needs_review          # True in minimal-interaction when the pick was uncertain
result.trace.suggested_candidates
```

`batch_identify` accepts the same `output` values, including `"result"`.

### Interaction modes

Two prompt modes are supported. Both use the same database-grounded pipeline; they differ in how the model handles underspecified queries.

| Mode | How to enable | Behaviour |
|---|---|---|
| **Standard** (default) | `minimal_interaction=False` | The model may return `ambiguous` and raise `AmbiguousIdentificationError` with candidate options. Used in the paper's **ambiguity / abstention study**. |
| **Minimal interaction** | `minimal_interaction=True` | The model always selects a candidate. Uncertain conventional assumptions are flagged with `trace.needs_review`. Used for headline **Figures 1–3**. |

```python
MODEL = "gemini-3.1-pro-preview"

# Standard: raise and list options when several polymorphs or stackings match
identify("TiO2", model=MODEL)
identify("hexagonal boron nitride", model=MODEL)

# Minimal interaction: pick one and mark the trace for review
result = identify("TiO2", model=MODEL, minimal_interaction=True, output="result")
if result.trace.needs_review:
    print(result.material.mp_id, result.trace.selection_reason)
```

CLI: `adam-identify "TiO2" --model gemini-3.1-pro-preview --minimal-interaction`

Follow-up queries use the same short pattern for crystals and molecules: a conventional name, with the formula appended when the name alone is not unique (`rutile TiO2`, `o-xylene`, `h-BN P6_3/mmc`).

### High-throughput (batch)

```python
from adam_identification import batch_identify

queries = ["silicon", "water", "caffeine", "iron (BCC)", "alpha-alumina"]

results = batch_identify(
    queries,
    provider="google",
    model="gemini-3.1-pro-preview",
    concurrency=5,          # simultaneous API calls
    output="ase",           # or "pymatgen", "material", or "result"
)

for query, result in zip(queries, results):
    if isinstance(result, Exception):
        print(f"{query}: FAILED — {result}")
    else:
        print(f"{query}: {result.symbols}")
```

### CLI

```bash
# --model is required
adam-identify "silicon" --model gemini-3.1-pro-preview
adam-identify "caffeine" --model gemini-3.1-pro-preview

# Other provider
adam-identify "BCC iron" --provider openai --model gpt-5.4-mini

# Save to file (CIF for crystals, XYZ for molecules).
# ASE writes the primitive cell (often P1), not the conventional cell.
adam-identify "silicon" --model gemini-3.1-pro-preview --save silicon.cif
adam-identify "caffeine" --model gemini-3.1-pro-preview --save caffeine.xyz

# Batch mode: one query per line in a text file
adam-identify --batch queries.txt --model gemini-3.1-pro-preview --concurrency 5
adam-identify --batch queries.txt --model gemini-3.1-pro-preview --save-dir structures/

# List available providers
adam-identify --list-providers

# Verbose (DEBUG logs and the full candidate list on ambiguity)
adam-identify "TiO2" --model gemini-3.1-pro-preview --verbose
```

On an ambiguous query the CLI prints a short table of Materials Project or PubChem options (mp_id / space group, or name / CID) plus suggested follow-up queries. Use `--verbose` for the full database list.

---

## Supported LLM providers

The public provider key is **`google`**, not `gemini`. Model slugs stay `gemini-*`. The API key remains `GOOGLE_API_KEY`.

There is no default model: pass `model=` / `--model` for every call.

| Key | Models used in the paper | Notes |
|---|---|---|
| `google` | `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.1-pro-preview` | |
| `openai` | `gpt-5.4-mini`, `gpt-5.5` | |
| `anthropic` | `claude-haiku-4-5-20251001`, `claude-sonnet-4-6` | |
| `openrouter` | `deepseek/deepseek-chat-v3-5`, `moonshotai/kimi-k2.6`, `qwen/qwen3.7-max` | Access 300+ models |

---

## Output objects

| `output=` | Crystal | Molecule |
|---|---|---|
| `"ase"` | `ase.Atoms` with `pbc=True` | `ase.Atoms` with `pbc=False` |
| `"pymatgen"` | `pymatgen.core.Structure` | `pymatgen.core.Molecule` |
| `"material"` | `adam_identification.models.Material` | same |
| `"result"` | `IdentificationResult` (`atoms`, `material`, `trace`) | same |

The `Material` object carries:
- `material.chemical_formula` — reduced formula
- `material.mp_id` — Materials Project ID (crystals)
- `material.pc_cid` — PubChem Compound ID (molecules)
- `material.structure` — `CrystalStructure` or `MoleculeStructure` with coordinates
- `material.properties[0]` — `MaterialsProjectProperties` with band gap, energy above hull, etc.

`IdentificationResult.trace` records extraction, candidate lists, selection decisions, `outcome`, and `needs_review`.

---

## Interactive sessions

`IdentificationSession` wraps `MaterialIdentifier` with cached pipeline state so
that when a query is ambiguous the user can provide one follow-up without repeating
the database lookup.

```python
from adam_identification import IdentificationSession, MaterialIdentifier
from adam_identification.database.materials_project import MaterialsProjectClient
from adam_identification.database.pubchem import PubChemClient
from adam_identification.exceptions import (
    AmbiguousIdentificationError,
    ClarificationNeededError,
)
from adam_identification.llm import get_provider

llm = get_provider("google", "gemini-3.1-pro-preview")
identifier = MaterialIdentifier(
    llm,
    mp_client=MaterialsProjectClient(),
    pubchem_client=PubChemClient(),
)
session = IdentificationSession(identifier)

# --- Ambiguity scenario (selection-stage: crystal polymorphs or molecule isomers) ---
try:
    material = session.identify("boron nitride")
except AmbiguousIdentificationError as exc:
    print(exc.user_message)
    # Show options to the user
    for s in exc.suggested_candidates:
        print(" •", s["suggested_query"])   # e.g. "h-BN P6_3/mmc", "c-BN Fm-3m"
    # User picks one → resume skips the DB lookup and goes straight to selection
    material = session.resume("h-BN P6_3/mmc")

# --- Clarification scenario (extraction-stage: composition unclear) ---
try:
    material = session.identify("that oxide with a bandgap around 3 eV")
except ClarificationNeededError as exc:
    print(exc.user_message)               # "Query does not specify a unique composition."
    for i, q in enumerate(exc.suggested_queries, 1):
        print(f"  {i}. {q}")
    material = session.identify_choice(1) # pick option 1 by number
    # or supply your own follow-up:
    # material = session.identify("ZnO wurtzite")

print(material.mp_id, material.structure.space_group)
```

For headless / benchmark use, call `MaterialIdentifier.identify()` directly — the session layer is optional.

---

## Error handling

```python
from adam_identification import identify
from adam_identification.exceptions import (
    AmbiguousIdentificationError,
    ClarificationNeededError,
    MaterialNotFoundError,
    LLMError,
)

MODEL = "gemini-3.1-pro-preview"
try:
    atoms = identify("tungsten trioxide monoclinic", model=MODEL)
except AmbiguousIdentificationError as exc:
    # exc.user_message: natural-language explanation from the LLM
    # exc.candidates_for_display(): compact table (up to 8 entries)
    # exc.suggested_candidates: [{"index": 0, "suggested_query": "..."}]
    print(exc.user_message)
    for c in exc.candidates_for_display():
        print(c)
except ClarificationNeededError as exc:
    # exc.suggested_queries: list of reliable follow-up query strings
    print(exc.user_message)
    print(exc.suggested_queries)
except MaterialNotFoundError as exc:
    print(f"Not found in database: {exc}")
except LLMError as exc:
    # Sub-types: RateLimitError, AuthenticationError, ProviderUnavailableError
    print(f"LLM error: {exc}")
```

`AmbiguousIdentificationError` always includes a compact candidate table in `str(exc)`,
even without a `user_message`.

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

## Known limitations

- **No default model** — `model=` / `--model` is required for every call.
- **Crystal structures from the Materials Project** are primitive-cell DFT-relaxed geometries (GGA-PBE or HSE06). They may differ from experimental unit cells.
- **Molecule 3D geometries from PubChem** are PubChem-computed conformers (MMFF94); they are not DFT-optimised. A subsequent DFT relaxation is recommended before production calculations.
- **One material per call** — the agent assumes a single-compound query. Multi-component descriptions (e.g. "TiO2 on SiO2 substrate") are treated as a clarification request.
- **LLM non-determinism** — results may vary across model versions and temperature settings. Use `minimal_interaction=False` (standard mode) for benchmarking where reproducible abstention is required.

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
