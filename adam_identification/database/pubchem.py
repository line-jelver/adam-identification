"""PubChem REST API client for adam-identification.

Maps PubChem PUG REST compound endpoints into the unified
:class:`~adam_identification.models.Material` model with
:class:`~adam_identification.models.MoleculeStructure`. No API key required.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict, cast
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
PUBCHEM_MAX_CANDIDATES: int = 5

logger = logging.getLogger(__name__)


class MoleculeCandidate(TypedDict):
    """Lightweight PubChem compound summary used for LLM candidate selection.

    Fields intentionally exclude 3D coordinates — those are fetched only for
    the winning candidate after LLM selection.
    """

    cid: int
    name: str
    formula: str
    smiles: str | None


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
    "MolecularFormula,CanonicalSMILES,ConnectivitySMILES,InChI,InChIKey,IUPACName,Title"
)


def _symbol_from_atomic_number(z: int) -> str:
    if z < 1 or z >= len(_ELEMENT_SYMBOLS):
        raise DatabaseAPIError(f"Invalid atomic number from PubChem: {z}")
    return cast(str, _ELEMENT_SYMBOLS[z])


class PubChemClient:
    """Client for the PubChem REST API (molecular identification).

    Supports lookup by SMILES string, common/IUPAC name, InChIKey, or CID.
    Returns :class:`~adam_identification.models.Material` instances with
    ``material_type="molecule"``. No API key required.

    Usage::

        client = PubChemClient()
        aspirin = client.get_by_smiles("CC(=O)Oc1ccccc1C(=O)O")
        caffeine = client.get_by_name("caffeine")

    Args:
        http_client: Optional ``httpx.Client`` (e.g. for tests). If omitted,
            a client with a 30-second timeout is created.
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

        SMILES is structurally unambiguous — one SMILES resolves to exactly one
        canonical PubChem entry.

        Args:
            smiles: SMILES string, e.g. ``"CCO"`` for ethanol.

        Returns:
            A :class:`~adam_identification.models.Material` with molecular
            structure and identifiers.

        Raises:
            MaterialNotFoundError: If PubChem returns no match.
            DatabaseAPIError: On HTTP errors or malformed responses.
        """
        normalized = smiles.strip()
        if not normalized:
            raise MaterialNotFoundError("Empty SMILES string was provided.")
        path = f"/compound/smiles/{quote(normalized, safe='')}/cids/JSON"
        cid = self._first_cid(self._request_json(path))
        return self._material_from_cid(cid)

    def get_by_name(self, name: str) -> Material:
        """Fetch a molecule by common or IUPAC name.

        Args:
            name: Common or IUPAC name of the compound.

        Returns:
            A :class:`~adam_identification.models.Material` with molecular
            structure and identifiers.

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
        """Fetch a molecule by InChIKey.

        Args:
            inchi_key: Standard InChIKey, e.g.
                ``"XLYOFNOQVPJJNP-UHFFFAOYSA-N"`` for water.

        Returns:
            A :class:`~adam_identification.models.Material` instance.

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

        Returns a full :class:`~adam_identification.models.Material` with 3D
        coordinates.

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
        CID, name, formula, and SMILES. The 3D structure is not fetched; call
        :meth:`get_by_cid` for the winning candidate after selection.

        Args:
            name: Common or IUPAC name to search.
            n: Maximum number of candidates to return.

        Returns:
            List of up to *n* :class:`MoleculeCandidate` dicts, ranked by
            PubChem relevance.

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

    def _fetch_candidate_properties(self, cids: list[int]) -> list[MoleculeCandidate]:
        cid_str = ",".join(str(c) for c in cids)
        path = f"/compound/cid/{cid_str}/property/{_PROPERTY_FIELDS}/JSON"
        data = self._request_json(path)
        rows = data.get("PropertyTable", {}).get("Properties") or []
        candidates: list[MoleculeCandidate] = []
        for row in rows:
            cid = int(row["CID"])
            formula = str(row.get("MolecularFormula") or "")
            smiles = row.get("CanonicalSMILES") or row.get("ConnectivitySMILES")
            title = row.get("Title")
            iupac = row.get("IUPACName")
            compound_name = str(title or iupac or f"CID-{cid}")
            candidates.append(
                MoleculeCandidate(
                    cid=cid,
                    name=compound_name,
                    formula=formula,
                    smiles=str(smiles) if smiles else None,
                )
            )
        return candidates

    def _request_json(self, path: str, *, params: dict[str, str] | None = None) -> Any:
        def _once() -> Any:
            return self._request_json_once(path, params=params)

        outcome = retry_on_transient_error(_once, label="PubChem")
        self.transient_retries += outcome.transient_retries
        return outcome.value

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
        cids = data.get("IdentifierList", {}).get("CID") or []
        if not cids:
            raise MaterialNotFoundError("PubChem returned no CID for this query.")
        return int(cids[0])

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
            raise DatabaseAPIError(
                f"PubChem property row missing MolecularFormula for CID {cid}."
            )
        return cast(dict[str, Any], row)

    def _fetch_3d_compound(self, cid: int) -> dict[str, Any]:
        path = f"/compound/cid/{cid}/JSON"
        data = self._request_json(path, params={"record_type": "3d"})
        compounds = data.get("PC_Compounds") or []
        if not compounds:
            raise MaterialNotFoundError(
                f"No 3D compound record returned for PubChem CID {cid}."
            )
        return cast(dict[str, Any], compounds[0])

    def _atomic_positions_from_compound(
        self, compound: dict[str, Any]
    ) -> list[AtomicPosition]:
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
            if (
                len(xs) != len(elements)
                or len(ys) != len(elements)
                or len(zs) != len(elements)
            ):
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
