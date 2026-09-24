"""Unit tests for the provider-neutral crystal contracts.

Covers ``adam_identification.database.crystal_provider`` (``CrystalCandidate``,
``CrystalSearchResult``, ``CrystalProvider``) and the ``MaterialSource.MC3D`` /
``MC3DProperties`` / ``Material.primary_id`` additions to
``adam_identification.database.models``. No live network access.
"""

from __future__ import annotations

from adam_identification.database.crystal_provider import (
    CrystalCandidate,
    CrystalProvider,
    CrystalSearchResult,
)
from adam_identification.models import (
    AtomicPosition,
    CrystalStructure,
    Material,
    MaterialSource,
    MaterialsProjectProperties,
    MC3DProperties,
    MC3DProvenanceLink,
    MC3DSourceRecord,
    MoleculeStructure,
    PubChemQCProperties,
)


def _crystal_structure() -> CrystalStructure:
    return CrystalStructure(
        unit_cell=[[2.39, 2.39, 0.0], [0.0, 2.39, 2.39], [2.39, 0.0, 2.39]],
        lattice_parameters={
            "a": 3.87,
            "b": 3.87,
            "c": 3.87,
            "alpha": 60.0,
            "beta": 60.0,
            "gamma": 60.0,
        },
        atomic_positions=[
            AtomicPosition(element="Ti", position=(0.0, 0.0, 0.0)),
            AtomicPosition(element="O", position=(0.3, 0.3, 0.3)),
            AtomicPosition(element="O", position=(0.7, 0.7, 0.7)),
        ],
        crystal_system="Cubic",
        space_group="Fm-3m",
        nelements=2,
        nsites=3,
    )


def _mc3d_material() -> Material:
    return Material(
        name="TiO2",
        chemical_formula="TiO2",
        source=MaterialSource.MC3D,
        mc3d_id="mc3d-19249",
        structure=_crystal_structure(),
        properties=[
            MC3DProperties(
                method="pbesol-v2",
                structure_uuid="38e42253-a0b6-499e-adea-2a17e81866b9",
                sources=[
                    MC3DSourceRecord(
                        database="icsd",
                        record_id="189325",
                        version="2017.2",
                        is_theoretical=True,
                        is_high_pressure=False,
                        is_high_temperature=False,
                    )
                ],
                total_energy_ev_per_cell=-2745.824,
                cell_volume_ang3=27.2569,
                total_magnetization_mu_b_per_cell=0.0,
                absolute_magnetization_mu_b_per_cell=0.0,
                provenance_links=[
                    MC3DProvenanceLink(
                        label="Final relax calculation",
                        uuid="c1b7ecec-9702-44b9-90f1-775c2cba19e3",
                        url="https://aiida.materialscloud.org/mc3d-pbesol-v2/api/v4/nodes/c1b7ecec-9702-44b9-90f1-775c2cba19e3",
                    )
                ],
                record_url="https://mc3d.materialscloud.org/?id=mc3d-19249",
            )
        ],
    )


class TestMC3DPropertiesRoundTrip:
    """MC3D materials round-trip through the discriminated ``AnyProperties`` union."""

    def test_round_trip_preserves_mc3d_fields(self) -> None:
        material = _mc3d_material()
        dumped = material.model_dump(mode="json")
        restored = Material.model_validate(dumped)

        assert restored == material
        props = restored.get_properties(MaterialSource.MC3D)
        assert isinstance(props, MC3DProperties)
        assert props.method == "pbesol-v2"
        assert props.structure_uuid == "38e42253-a0b6-499e-adea-2a17e81866b9"
        assert props.sources[0].database == "icsd"
        assert props.provenance_links[0].label == "Final relax calculation"

    def test_round_trip_does_not_collapse_to_base_properties(self) -> None:
        """Without the discriminator this would silently drop subclass fields."""
        material = _mc3d_material()
        restored = Material.model_validate(material.model_dump(mode="json"))
        props = restored.properties[0]
        assert type(props) is MC3DProperties
        assert hasattr(props, "structure_uuid")

    def test_mc3d_properties_null_flags_stay_null(self) -> None:
        """A source flag the API never reported must stay ``None``, not ``False``."""
        props = MC3DProperties(
            method="pbesol-v2",
            structure_uuid="uuid-1",
            sources=[MC3DSourceRecord(database="cod", record_id="1")],
        )
        assert props.sources[0].is_theoretical is None
        assert props.sources[0].is_high_pressure is None


class TestExistingMaterialsStillLoad:
    """Old MP/PubChem serialized ``Material`` payloads must load unchanged."""

    def test_materials_project_material_round_trips(self) -> None:
        material = Material(
            name="Silicon",
            chemical_formula="Si",
            source=MaterialSource.MATERIALS_PROJECT,
            mp_id="mp-149",
            structure=_crystal_structure(),
            properties=[MaterialsProjectProperties(energy_above_hull=0.0, is_metal=False)],
        )
        restored = Material.model_validate(material.model_dump(mode="json"))
        assert restored == material
        props = restored.get_properties(MaterialSource.MATERIALS_PROJECT)
        assert isinstance(props, MaterialsProjectProperties)
        assert props.is_metal is False

    def test_pubchem_qc_molecule_round_trips(self) -> None:
        material = Material(
            name="water",
            chemical_formula="H2O",
            source=MaterialSource.PUBCHEM,
            pc_cid=962,
            structure=MoleculeStructure(
                atomic_positions=[AtomicPosition(element="O", position=(0.0, 0.0, 0.0))],
                smiles="O",
            ),
            properties=[PubChemQCProperties(total_energy=-2081.1)],
        )
        restored = Material.model_validate(material.model_dump(mode="json"))
        assert restored == material
        props = restored.get_properties(MaterialSource.PUBCHEM_QC)
        assert isinstance(props, PubChemQCProperties)
        assert props.total_energy == -2081.1

    def test_pre_mc3d_payload_without_mc3d_id_still_loads(self) -> None:
        """A JSON payload from before ``mc3d_id`` existed must still validate."""
        legacy_payload = {
            "name": "Silicon",
            "chemical_formula": "Si",
            "source": "materials_project",
            "mp_id": "mp-149",
            "structure": _crystal_structure().model_dump(mode="json"),
            "properties": [],
        }
        restored = Material.model_validate(legacy_payload)
        assert restored.mc3d_id is None
        assert restored.mp_id == "mp-149"


class TestPrimaryId:
    """``Material.primary_id`` resolves the correct field per source."""

    def test_materials_project(self) -> None:
        material = Material(
            name="Si",
            chemical_formula="Si",
            source=MaterialSource.MATERIALS_PROJECT,
            mp_id="mp-149",
            structure=_crystal_structure(),
        )
        assert material.primary_id == "mp-149"

    def test_mc3d(self) -> None:
        assert _mc3d_material().primary_id == "mc3d-19249"

    def test_pubchem(self) -> None:
        material = Material(
            name="water",
            chemical_formula="H2O",
            source=MaterialSource.PUBCHEM,
            pc_cid=962,
            structure=MoleculeStructure(
                atomic_positions=[AtomicPosition(element="O", position=(0.0, 0.0, 0.0))],
            ),
        )
        assert material.primary_id == "962"

    def test_pubchem_qc_uses_pc_cid_too(self) -> None:
        material = Material(
            name="water",
            chemical_formula="H2O",
            source=MaterialSource.PUBCHEM_QC,
            pc_cid=962,
            structure=MoleculeStructure(
                atomic_positions=[AtomicPosition(element="O", position=(0.0, 0.0, 0.0))],
            ),
        )
        assert material.primary_id == "962"

    def test_none_when_id_field_missing(self) -> None:
        material = Material(
            name="Si",
            chemical_formula="Si",
            source=MaterialSource.MATERIALS_PROJECT,
            structure=_crystal_structure(),
        )
        assert material.primary_id is None

    def test_user_input_has_no_primary_id(self) -> None:
        material = Material(
            name="custom",
            chemical_formula="Si",
            source=MaterialSource.USER_INPUT,
            structure=_crystal_structure(),
        )
        assert material.primary_id is None


class TestCrystalCandidateContract:
    """``CrystalCandidate``/``CrystalSearchResult`` are minimal, frozen DTOs."""

    def test_candidate_is_frozen(self) -> None:
        candidate = CrystalCandidate(
            source=MaterialSource.MC3D,
            source_id="mc3d-19249",
            structure_ref="38e42253-a0b6-499e-adea-2a17e81866b9",
            formula="TiO2",
            method="pbesol-v2",
            space_group="Fm-3m",
            space_group_number=225,
            crystal_system="Cubic",
            bravais_lattice="cF",
            nsites=3,
            source_database="icsd",
            source_database_id="189325",
            is_theoretical=True,
            is_high_pressure=False,
            is_high_temperature=False,
            total_energy_ev_per_cell=-2745.824,
            cell_volume_ang3=27.2569,
        )
        assert candidate.source_id == "mc3d-19249"
        assert candidate.bravais_lattice == "cF"
        try:
            candidate.source_id = "mc3d-99999"  # type: ignore[misc]
        except Exception as exc:  # pydantic raises ValidationError on frozen models
            assert "frozen" in str(exc).lower() or "immutable" in str(exc).lower()
        else:
            raise AssertionError("CrystalCandidate must be frozen")

    def test_search_result_truncation_shape(self) -> None:
        candidates = [
            CrystalCandidate(
                source=MaterialSource.MC3D,
                source_id=f"mc3d-{i}",
                structure_ref=f"uuid-{i}",
                formula="TiO2",
            )
            for i in range(5)
        ]
        result = CrystalSearchResult(candidates=candidates, total_matches=20, truncated=True)
        assert len(result.candidates) == 5
        assert result.total_matches == 20
        assert result.truncated is True

    def test_candidate_optional_fields_default_to_none(self) -> None:
        candidate = CrystalCandidate(
            source=MaterialSource.MATERIALS_PROJECT,
            source_id="mp-149",
            structure_ref="mp-149",
            formula="Si",
        )
        assert candidate.method is None
        assert candidate.space_group_number is None
        assert candidate.is_theoretical is None


class _FakeProvider:
    """Minimal structural implementation used to check the ``CrystalProvider`` protocol."""

    source = MaterialSource.MC3D

    def search_candidates(self, formula: str, max_results: int) -> CrystalSearchResult:
        return CrystalSearchResult(candidates=[])

    def get_by_id(self, source_id: str) -> Material:
        raise NotImplementedError

    def hydrate(self, candidate: CrystalCandidate) -> Material:
        raise NotImplementedError


def test_crystal_provider_is_runtime_checkable() -> None:
    """Any object with the right method shapes satisfies ``CrystalProvider``."""
    assert isinstance(_FakeProvider(), CrystalProvider)


def test_object_missing_methods_does_not_satisfy_protocol() -> None:
    class _NotAProvider:
        source = MaterialSource.MC3D

    assert not isinstance(_NotAProvider(), CrystalProvider)
