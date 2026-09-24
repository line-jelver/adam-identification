# MC3D API fixtures

Sanitized JSON fixtures for offline, network-free unit tests of
`adam.database.apis.mc3d.MC3DClient` (see `tests/unit/database/test_mc3d.py`).

No production code reads this directory at runtime. No test in this
repository contacts the live MC3D APIs — these fixtures exist so the client's
request building, response mapping, and error handling can be tested without
network access.

## Live-captured fixtures

Captured **2026-09-23** against the public, unauthenticated
`mc3d-pbesol-v2` endpoints. Only stable schema shape should be asserted from
these files (field names, types, nesting) — not the exact energies, IDs, or
candidate counts, which will drift as the live MC3D database grows.

| File | Captured from | Purpose |
|---|---|---|
| `formula_search_tio2_page1.json` | `GET https://optimade.materialscloud.org/main/mc3d-pbesol-v2/v1/structures?filter=chemical_formula_reduced="O2Ti"&page_limit=5&response_fields=...` | Paginated OPTIMADE formula search (TiO2, reduced formula `O2Ti`). Demonstrates: `data` holds only the current page (5 rows); `meta.data_returned` (20) is the filtered-match count, not the page length; `meta.data_available` (33142) is the whole-database size and must not drive pagination; `links.next` is present and must not be followed automatically. |
| `core_base_mc3d-19249.json` | `GET https://mcxd-api.materialscloud.org/mc3d/pbesol-v2/core_base/mc3d-19249` | Compact per-candidate metadata record: Bravais lattice, Hermann-Mauguin symbol + space-group number, structure UUID, source database record (ICSD, with theoretical/high-pressure/high-temperature flags), and the `"Final relax calculation"` provenance-link label (PBEsol-v2 naming — see architecture-doc Section 4.5; **not** `"Final SCF calculation"`, which is the PBE-v1 label). |
| `hydrate_mc3d-19249_nonorthogonal.json` | `GET https://optimade.materialscloud.org/main/mc3d-pbesol-v2/v1/structures/38e42253-a0b6-499e-adea-2a17e81866b9?response_fields=...` | Selected-structure hydration response for the same `mc3d-19249` entry. This TiO2 phase's primitive cell (space group `Fm-3m`, no. 225) is stored as a face-centered-cubic primitive cell with 60° interaxial angles — **this is the non-orthogonal-lattice fixture**, not a separate capture. Exercises the general `fractional = cartesian @ inverse(lattice)` conversion path (must not assume orthogonal axes) and the "top-level `id` equals `_mcloud_mc3d_id`'s structure UUID" consistency check. |
| `formula_search_no_results.json` | `GET .../structures?filter=chemical_formula_reduced="ArHe"&page_limit=5&...` | Empty-result formula search (`data: []`, `data_returned: 0`) — a formula with zero matching MC3D entries, to test the "no candidates in narrow set" → widen/fallback path without contriving a client-side error. |
| `core_base_404_body.json` | `GET https://mcxd-api.materialscloud.org/mc3d/pbesol-v2/core_base/mc3d-99999999` (HTTP 404) | Real 404 response body (`{"detail": "Entry mc3d-99999999 not found"}`) for a nonexistent `mc3d_id`. Used to test that a 404 maps to a non-retryable, discard-this-candidate outcome (never `MaterialNotFoundError` for a whole-provider failure). |

## Hand-authored (synthetic) fixtures

These cannot be reliably captured live on demand and are deliberately
hand-crafted, derived from the shape of the real fixtures above:

| File | Basis | Purpose |
|---|---|---|
| `hydrate_malformed_structure.json` | Derived from `hydrate_mc3d-19249_nonorthogonal.json` | `nsites: 3` but only 2 `cartesian_site_positions` are present while `species_at_sites` still lists 3 entries — a site-count mismatch. Must be rejected by validation (`len(cartesian_site_positions) == len(species_at_sites) == nsites`) **before** `Material` construction; must never reach the LLM or a calculator input. |
| `transient_503_body.json` | Generic FastAPI-style error body | Represents an HTTP 503 response used to test the transient-retry path (`adam.database.retry`) for MC3D requests (408/425/429/500/502/503/504 retry; 400/404/422 do not). |

## Permitted assertions (contract-test discipline)

Tests using these fixtures — and any future opt-in live contract test — may
assert on stable schema invariants (field presence/types, ID consistency,
lattice vector count, site-count agreement, provenance-link labels) but must
**not** assert on exact energies, exact candidate ordering/count, or the
live database size, since those change over time.
