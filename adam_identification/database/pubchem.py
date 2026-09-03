"""
PubChem REST API client for ADaM.

Maps PubChem PUG REST compound endpoints into the unified ``Material`` model with
``MoleculeStructure`` and ``source=MaterialSource.PUBCHEM``. Used by the Material
Identifier agent when the LLM supplies a SMILES or common name,
or when benchmarks resolve a known CID.

Cross-references:
- ``adam_identification.models`` for ``Material`` / ``MoleculeStructure``.
- ``adam_identification.exceptions`` for typed errors.
"""

from __future__ import annotations

import json
import logging
from typing import Any, NotRequired, TypedDict, cast
from urllib.parse import quote

import httpx

from adam_identification.database.retry import retry_on_transient_error
from adam_identification.exceptions import DatabaseAPIError, MaterialNotFoundError
from adam_identification.models import (
    AtomicPosition,
    Material,
    MaterialSource,
    MoleculeStructure,
)

PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

#: Maximum number of name-search candidates to retrieve for LLM selection.
#: Analogous to ``INITIAL_MAX_RESULTS`` for the crystal path (MP uses 20 for
#: polymorphs; molecules need far fewer because PubChem name results are
#: already highly ranked by relevance).
PUBCHEM_MAX_CANDIDATES: int = 5

#: Wider cap for formula-based (``fastformula``) searches.
#: Unlike name search, formula search returns *all* structural isomers in an
#: unordered set — a formula like C8H10 has >1000 CIDs on PubChem, so we take
#: the top 10 by PubChem internal ranking to ensure common isomers are included.
PUBCHEM_MAX_FORMULA_CANDIDATES: int = 10

logger = logging.getLogger(__name__)


class MoleculeCandidate(TypedDict):
    """Lightweight PubChem compound summary used for LLM candidate selection.

    Fields intentionally exclude 3D coordinates — those are fetched only for
    the winning candidate after LLM selection.

    The ``isomeric_smiles`` and ``inchi_key`` fields expose stereochemistry-
    preserving identifiers so the v2 molecule selection prompt can distinguish
    enantiomers and diastereomers without fetching full 3D records.
    """

    cid: int
    name: str
    formula: str
    smiles: str | None
    isomeric_smiles: NotRequired[str | None]
    inchi_key: NotRequired[str | None]


# IUPAC order, atomic numbers 1–118 (index 0 unused).
_ELEMENT_SYMBOLS: tuple[str | None, ...] = (
    None,
    *(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
        "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
        "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
        "At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
        "Cn Nh Fl Mc Lv Ts Og"
    ).split(),
)

_PROPERTY_FIELDS = (
    "MolecularFormula,CanonicalSMILES,IsomericSMILES,ConnectivitySMILES,"
    "InChI,InChIKey,IUPACName,Title"
)


def _symbol_from_atomic_number(z: int) -> str:
    if z < 1 or z >= len(_ELEMENT_SYMBOLS):
        raise DatabaseAPIError(f"Invalid atomic number from PubChem: {z}")
    return cast(str, _ELEMENT_SYMBOLS[z])


class PubChemClient:
    """Client for the PubChem REST API (molecular identification).

    Supports lookup by SMILES string, common/IUPAC name, or InChIKey.
    Returns ``Material`` instances with ``material_type="molecule"``
    and ``source=MaterialSource.PUBCHEM``.

    No API key required. Uses plain ``httpx`` GET requests.

    Usage::

        client = PubChemClient()
        aspirin = client.get_by_smiles("CC(=O)Oc1ccccc1C(=O)O")  # preferred
        caffeine = client.get_by_name("caffeine")  # fallback

    Benchmark usage (CID known from ground_truth)::

        water = client.get_by_cid(962)

    Args:
        http_client: Optional ``httpx.Client`` (e.g. for tests). If omitted, a
            client with a default timeout is created.
    """

    def __init__(self, *, http_client: httpx.Client | None = None) -> None:
        self._client = http_client if http_client is not None else httpx.Client(timeout=30.0)
        self._owns_client = http_client is None
        self.transient_retries = 0
        """Transient PubChem/HTTP retries consumed since client construction."""

    def close(self) -> None:
        """Close the underlying HTTP client if this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> PubChemClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_by_smiles(self, smiles: str) -> Material:
        """Fetch a molecule by SMILES string.

        **Preferred production method.** SMILES is structurally unambiguous —
        one SMILES resolves to exactly one canonical PubChem entry.

        Uses a POST request so that SMILES containing ``/`` or ``\\`` (E/Z
        stereo-bond notation) are safely sent in the request body rather than
        percent-encoded into the URL path, which PUG-REST rejects.

        Args:
            smiles: SMILES string, e.g. ``"CCO"`` for ethanol.

        Returns:
            A ``Material`` instance with molecular structure and identifiers.

        Raises:
            MaterialNotFoundError: If PubChem returns no match.
            DatabaseAPIError: On HTTP errors or malformed responses.
        """
        normalized = smiles.strip()
        if not normalized:
            raise MaterialNotFoundError("Empty SMILES string was provided.")
        path = "/compound/smiles/cids/JSON"
        data = self._post_json(path, form={"smiles": normalized})
        cid = self._first_cid(data)
        return self._material_from_cid(cid)

    def get_by_name(self, name: str) -> Material:
        """Fetch a molecule by common or IUPAC name.

        Reliable for well-known compounds (water, ethanol, aspirin, caffeine
        etc.). For obscure compounds or names with multiple synonyms, prefer
        ``get_by_smiles`` to avoid CID ambiguity.

        Args:
            name: Common or IUPAC name of the compound.

        Returns:
            A ``Material`` instance with molecular structure and identifiers.

        Raises:
            MaterialNotFoundError: If PubChem returns no match.
            DatabaseAPIError: On HTTP errors or malformed responses.
        """
        normalized = name.strip()
        if not normalized:
            raise MaterialNotFoundError("Empty compound name was provided.")
        path = f"/compound/name/{quote(normalized, safe='')}/cids/JSON"
        cid = self._first_cid(self._request_json(path))
        return self._material_from_cid(cid)

    def get_by_inchi_key(self, inchi_key: str) -> Material:
        """Fetch a molecule by InChIKey (globally unique hash).

        Args:
            inchi_key: Standard InChIKey, e.g. ``"XLYOFNOQVPJJNP-UHFFFAOYSA-N"``.

        Returns:
            A ``Material`` instance with molecular structure and identifiers.

        Raises:
            MaterialNotFoundError: If PubChem returns no match.
            DatabaseAPIError: On HTTP errors or malformed responses.
        """
        normalized = inchi_key.strip()
        if not normalized:
            raise MaterialNotFoundError("Empty InChIKey was provided.")
        path = f"/compound/inchikey/{quote(normalized, safe='')}/cids/JSON"
        cid = self._first_cid(self._request_json(path))
        return self._material_from_cid(cid)

    def get_by_cid(self, pubchem_cid: int) -> Material:
        """Fetch a molecule by its PubChem Compound ID.

        Use when the CID is already known (e.g. from the ground truth file or
        from the multi-candidate selection step). Returns a full ``Material``
        with 3D coordinates.

        Args:
            pubchem_cid: Integer CID, e.g. ``962`` for water.

        Raises:
            MaterialNotFoundError: If the CID does not exist or has no 3D record.
            DatabaseAPIError: On HTTP errors.
        """
        if pubchem_cid <= 0:
            raise MaterialNotFoundError("PubChem CID must be a positive integer.")
        return self._material_from_cid(pubchem_cid)

    def get_molecule_candidates(
        self,
        name: str,
        n: int = PUBCHEM_MAX_CANDIDATES,
    ) -> list[MoleculeCandidate]:
        """Fetch the top-N PubChem candidates by name (no 3D coordinates).

        Returns lightweight :class:`MoleculeCandidate` dicts containing only
        the fields needed for LLM selection — CID, name, formula, and SMILES.
        The 3D structure is **not** fetched; call :meth:`get_by_cid` for the
        winning candidate after LLM selection.

        This is the molecule analogue of the crystal narrow/wide MP search:
        instead of returning one compound, it returns up to N ranked by
        PubChem relevance so an LLM can select the correct structural isomer.

        Args:
            name: Common or IUPAC name to search, e.g. ``"morpholine"``.
            n: Maximum number of candidates to return. Defaults to
                :data:`PUBCHEM_MAX_CANDIDATES` (5).

        Returns:
            List of up to *n* :class:`MoleculeCandidate` dicts, ranked by
            PubChem relevance (most canonical compound first).

        Raises:
            MaterialNotFoundError: If PubChem returns no CIDs for ``name``.
            DatabaseAPIError: On HTTP errors or malformed responses.
        """
        normalized = name.strip()
        if not normalized:
            raise MaterialNotFoundError("Empty compound name was provided.")
        path = f"/compound/name/{quote(normalized, safe='')}/cids/JSON"
        data = self._request_json(path)
        cids = (data.get("IdentifierList", {}).get("CID") or [])[:n]
        if not cids:
            raise MaterialNotFoundError(f"PubChem returned no CIDs for name {name!r}.")
        return self._fetch_candidate_properties(cids)

    def get_molecule_candidates_by_formula(
        self,
        formula: str,
        n: int = PUBCHEM_MAX_FORMULA_CANDIDATES,
    ) -> list[MoleculeCandidate]:
        """Fetch the top-N PubChem candidates by molecular formula (no 3D coordinates).

        Uses the PUG REST ``fastformula`` endpoint — not the ``name`` endpoint.
        Chemical formulae such as ``"C8H10"`` are not PubChem synonyms, so they
        must be queried via ``/compound/fastformula/{formula}/cids`` rather than
        ``/compound/name/{formula}/cids``.

        The default cap is :data:`PUBCHEM_MAX_FORMULA_CANDIDATES` (10), wider
        than the name-search cap, because a formula like ``"C8H10"`` has over
        1 000 isomers in PubChem — taking only 5 risks missing the target.

        Args:
            formula: Hill-system molecular formula, e.g. ``"C8H10"``.
            n: Maximum number of candidates to return. Defaults to
                :data:`PUBCHEM_MAX_FORMULA_CANDIDATES` (10).

        Returns:
            List of up to *n* :class:`MoleculeCandidate` dicts for compounds
            matching the formula, ranked by PubChem.

        Raises:
            MaterialNotFoundError: If PubChem returns no CIDs for ``formula``.
            DatabaseAPIError: On HTTP errors or malformed responses.
        """
        normalized = formula.strip()
        if not normalized:
            raise MaterialNotFoundError("Empty molecular formula was provided.")
        path = f"/compound/fastformula/{quote(normalized, safe='')}/cids/JSON"
        data = self._request_json(path)
        cids = (data.get("IdentifierList", {}).get("CID") or [])[:n]
        if not cids:
            raise MaterialNotFoundError(f"PubChem returned no CIDs for formula {formula!r}.")
        return self._fetch_candidate_properties(cids)

    def _fetch_candidate_properties(self, cids: list[int]) -> list[MoleculeCandidate]:
        """Batch-fetch lightweight properties for a list of CIDs.

        Uses a single PubChem properties endpoint call for all CIDs.

        Args:
            cids: List of PubChem CIDs to fetch. Non-positive sentinel values
                (see :meth:`_first_cid`) are filtered out before the request,
                since a single ``0`` anywhere in the batch causes PubChem to
                reject the *entire* comma-separated request.

        Returns:
            One :class:`MoleculeCandidate` per CID (order matches the
            filtered *cids*).

        Raises:
            MaterialNotFoundError: If no positive CIDs remain after filtering.
        """
        positive_cids = [c for c in cids if c > 0]
        if not positive_cids:
            raise MaterialNotFoundError("No valid (positive) PubChem CIDs to fetch.")
        cid_str = ",".join(str(c) for c in positive_cids)
        path = f"/compound/cid/{cid_str}/property/{_PROPERTY_FIELDS}/JSON"
        data = self._request_json(path)
        rows = data.get("PropertyTable", {}).get("Properties") or []

        candidates: list[MoleculeCandidate] = []
        for row in rows:
            cid = int(row["CID"])
            formula = str(row.get("MolecularFormula") or "")
            smiles = row.get("CanonicalSMILES") or row.get("ConnectivitySMILES")
            isomeric_smiles = row.get("IsomericSMILES") or None
            inchi_key = row.get("InChIKey") or None
            title = row.get("Title")
            iupac = row.get("IUPACName")
            compound_name = str(title or iupac or f"CID-{cid}")
            candidates.append(
                MoleculeCandidate(
                    cid=cid,
                    name=compound_name,
                    formula=formula,
                    smiles=str(smiles) if smiles else None,
                    isomeric_smiles=str(isomeric_smiles) if isomeric_smiles else None,
                    inchi_key=str(inchi_key) if inchi_key else None,
                )
            )
        return candidates

    def _request_json(self, path: str, *, params: dict[str, str] | None = None) -> Any:
        """GET *path* from PUG REST with transient-error retries."""

        def _once() -> Any:
            return self._request_json_once(path, params=params)

        outcome = retry_on_transient_error(_once, label="PubChem")
        self.transient_retries += outcome.transient_retries
        return outcome.value

    def _post_json(self, path: str, *, form: dict[str, str]) -> Any:
        """POST *form* data to *path* on PUG REST with transient-error retries."""

        def _once() -> Any:
            return self._post_json_once(path, form=form)

        outcome = retry_on_transient_error(_once, label="PubChem")
        self.transient_retries += outcome.transient_retries
        return outcome.value

    def _post_json_once(self, path: str, *, form: dict[str, str]) -> Any:
        url = f"{PUG_REST_BASE}{path}"
        try:
            response = self._client.post(url, data=form)
        except httpx.RequestError as error:
            raise DatabaseAPIError(f"PubChem network error for {path}: {error}") from error

        try:
            data = response.json()
        except json.JSONDecodeError as error:
            raise DatabaseAPIError(
                f"PubChem returned non-JSON body (HTTP {response.status_code}) for {path}."
            ) from error

        self._raise_for_pubchem_fault(data, path)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise DatabaseAPIError(
                f"PubChem HTTP {response.status_code} for {path}: {response.text[:500]}"
            ) from error

        return data

    def _request_json_once(self, path: str, *, params: dict[str, str] | None = None) -> Any:
        url = f"{PUG_REST_BASE}{path}"
        try:
            response = self._client.get(url, params=params)
        except httpx.RequestError as error:
            raise DatabaseAPIError(f"PubChem network error for {path}: {error}") from error

        try:
            data = response.json()
        except json.JSONDecodeError as error:
            raise DatabaseAPIError(
                f"PubChem returned non-JSON body (HTTP {response.status_code}) for {path}."
            ) from error

        self._raise_for_pubchem_fault(data, path)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise DatabaseAPIError(
                f"PubChem HTTP {response.status_code} for {path}: {response.text[:500]}"
            ) from error

        return data

    def _raise_for_pubchem_fault(self, data: Any, path: str) -> None:
        if not isinstance(data, dict) or "Fault" not in data:
            return
        fault = data["Fault"]
        code = str(fault.get("Code", ""))
        message = str(fault.get("Message", "unknown error"))
        if "NotFound" in code:
            raise MaterialNotFoundError(f"PubChem: {message} ({path})")
        raise DatabaseAPIError(f"PubChem fault {code}: {message} ({path})")

    def _first_cid(self, data: dict[str, Any]) -> int:
        """Return the first positive CID from a PUG REST identifier list.

        PubChem uses ``0`` as a sentinel for "structure could not be
        standardized" (e.g. an invalid SMILES) rather than omitting it or
        returning a fault — ``{"IdentifierList": {"CID": [0]}}`` is a valid
        200 response. Treat CID 0 (and any other non-positive value) the same
        as "no match" so callers get :class:`MaterialNotFoundError` instead of
        forwarding ``0`` to a downstream endpoint that rejects it.
        """
        cids = data.get("IdentifierList", {}).get("CID") or []
        for cid in cids:
            if int(cid) > 0:
                return int(cid)
        raise MaterialNotFoundError("PubChem returned no valid (positive) CID for this query.")

    def _material_from_cid(self, cid: int) -> Material:
        props = self._fetch_properties(cid)
        compound = self._fetch_3d_compound(cid)
        positions = self._atomic_positions_from_compound(compound)
        if not positions:
            raise MaterialNotFoundError(
                f"No 3D conformer coordinates available for PubChem CID {cid}."
            )

        atoms_z = (compound.get("atoms") or {}).get("element") or []
        nelements = len({int(z) for z in atoms_z})
        nsites = len(atoms_z)

        title = props.get("Title")
        iupac = props.get("IUPACName")
        name = str(title or iupac or f"CID-{cid}")
        formula = str(props["MolecularFormula"])
        smiles = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
        inchi = props.get("InChI")
        inchi_key = props.get("InChIKey")

        structure = MoleculeStructure(
            atomic_positions=positions,
            smiles=str(smiles) if smiles else None,
            inchi=str(inchi) if inchi else None,
            inchi_key=str(inchi_key) if inchi_key else None,
            nelements=nelements,
            nsites=nsites,
        )

        return Material(
            name=name,
            chemical_formula=formula,
            source=MaterialSource.PUBCHEM,
            pc_cid=cid,
            structure=structure,
            properties=[],
        )

    def _fetch_properties(self, cid: int) -> dict[str, Any]:
        path = f"/compound/cid/{cid}/property/{_PROPERTY_FIELDS}/JSON"
        data = self._request_json(path)
        table = data.get("PropertyTable", {}).get("Properties") or []
        if not table:
            raise DatabaseAPIError(f"PubChem returned no properties for CID {cid}.")
        row = table[0]
        if "MolecularFormula" not in row:
            raise DatabaseAPIError(f"PubChem property row missing MolecularFormula for CID {cid}.")
        return cast(dict[str, Any], row)

    def _fetch_3d_compound(self, cid: int) -> dict[str, Any]:
        path = f"/compound/cid/{cid}/JSON"
        data = self._request_json(path, params={"record_type": "3d"})
        compounds = data.get("PC_Compounds") or []
        if not compounds:
            raise MaterialNotFoundError(f"No 3D compound record returned for PubChem CID {cid}.")
        return cast(dict[str, Any], compounds[0])

    def _atomic_positions_from_compound(self, compound: dict[str, Any]) -> list[AtomicPosition]:
        atoms = compound.get("atoms") or {}
        elements = atoms.get("element") or []
        if not elements:
            return []

        for coord_block in compound.get("coords") or []:
            conformers = coord_block.get("conformers") or []
            if not conformers:
                continue
            conf = conformers[0]
            xs = conf.get("x") or []
            ys = conf.get("y") or []
            zs = conf.get("z") or []
            if len(xs) != len(elements) or len(ys) != len(elements) or len(zs) != len(elements):
                continue
            out: list[AtomicPosition] = []
            for z, x, y, zz in zip(elements, xs, ys, zs, strict=True):
                sym = _symbol_from_atomic_number(int(z))
                out.append(
                    AtomicPosition(
                        element=sym,
                        position=(float(x), float(y), float(zz)),
                    )
                )
            return out
        return []
