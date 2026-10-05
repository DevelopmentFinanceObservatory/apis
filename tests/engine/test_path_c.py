"""Test standard promote, refine, and identity stability through resubmission."""

from __future__ import annotations

import pytest

from search.engine.errors import EngineInputError
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import (
    _RESUBMIT_MAX_RESOLVED_STAT_VARS,
    BaseIdentity,
    BindingSet,
    DatasetResubmitInput,
    GraphRef,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentitySlot,
    IdentityWindowRange,
    RemovedParts,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import (
    DECLARING_SHAPE,
    GDP_SHAPE,
    PERSON_COUNT_SHAPE,
    destination_entity,
    make_base,
    make_identity,
    origin_entity,
    shape_for,
    subject_entity,
)
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.fixtures import GDP_INDIA_PROVENANCES, FakeGraph, fake_graph_with_provenances


def _identity_request(payload: ResubmitIdentityPayload) -> DatasetResubmitInput:
    """Wrap one identity entry in a dataset-resubmit request."""
    return DatasetResubmitInput(
        entries=[
            ResubmitIdentityEntry(
                variable_text="Health ODA grants to Ethiopia",
                identity=payload.identity,
                base_spec_id=payload.base_spec_id,
                base_identity=payload.base_identity,
                resolved_stat_var_dcids=payload.resolved_stat_var_dcids,
            )
        ]
    )


# Standard dev-finance slots (Health ODA Grants to Ethiopia)
_SCHEME_SLOT = _value_slot(
    "what",
    "DevelopmentFinanceScheme",
    "finance scheme",
    "ODAGrants",
    "Official Development Assistance Grants",
)
_PURPOSE_SLOT = _value_slot(
    "how",
    "DevelopmentFinancePurpose",
    "sector/purpose",
    "DAC/Health",
    "Health (Total)",
)


# ---------------------------------------------------------------------------
# ResubmitIdentityPayload contract: base_spec_id / base_identity / removed
# ---------------------------------------------------------------------------


class TestResubmitIdentityPayloadFields:
    def test_base_spec_id_defaults_none(self):
        inp = ResubmitIdentityPayload(identity=make_identity(shape_for("some_shape")))
        assert inp.base_spec_id is None

    def test_base_identity_defaults_none(self):
        inp = ResubmitIdentityPayload(identity=make_identity(shape_for("some_shape")))
        assert inp.base_identity is None

    def test_removed_defaults_none(self):
        inp = ResubmitIdentityPayload(identity=make_identity(shape_for("some_shape")))
        assert inp.removed is None

    def test_base_spec_id_and_base_identity_round_trip(self):
        base_spec_id, base_identity = make_base(
            DECLARING_SHAPE, entities=[destination_entity("country/ETH")]
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(DECLARING_SHAPE, entities=[destination_entity("country/ETH")]),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
        )
        assert inp.base_spec_id == base_spec_id
        assert inp.base_identity == base_identity

    def test_removed_round_trips(self):
        base_spec_id, base_identity = make_base(shape_for("some_shape"))
        inp = ResubmitIdentityPayload(
            identity=make_identity(shape_for("some_shape")),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            removed=RemovedParts(entities=["destination"]),
        )
        assert inp.removed == RemovedParts(entities=["destination"])

    def test_resolved_stat_var_dcids_accepts_200(self):
        """The default 200-StatVar limit accepts a 200-item request.

        `resolved_stat_var_dcids` is graph state, outside the identity hash.
        """
        inp = ResubmitIdentityPayload(
            identity=make_identity(shape_for("s")), resolved_stat_var_dcids=["sv"] * 200
        )
        assert len(inp.resolved_stat_var_dcids) == 200

    def test_resolved_stat_var_dcids_max_length_over_ceiling_raises(self):
        """Validation rejects more than `_RESUBMIT_MAX_RESOLVED_STAT_VARS` entries."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResubmitIdentityPayload(
                identity=make_identity(shape_for("s")),
                resolved_stat_var_dcids=["sv"] * (_RESUBMIT_MAX_RESOLVED_STAT_VARS + 1),
            )

    def test_legacy_flat_definition_fields_are_rejected(self):
        """The edited definition is accepted only through nested identity."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResubmitIdentityPayload.model_validate(
                {
                    "identity": make_identity(shape_for("s")).model_dump(),
                    "shape": shape_for("s"),
                    "slots": [],
                }
            )

    def test_base_identity_extra_field_forbidden(self):
        """BaseIdentity is a closed schema, same as every other request-side model."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BaseIdentity(shape=shape_for("s"), slots=[], unknown_field="x")

    # These limits match the identity core and reject invalid requests before hashing.
    # Each boundary value is accepted; the next value is rejected.

    def test_identity_binding_set_rejects_refs_over_1024(self):
        from pydantic import ValidationError

        IdentityBindingSet(refs=[f"x{i}" for i in range(1024)])  # at the limit: still valid
        with pytest.raises(ValidationError):
            IdentityBindingSet(refs=[f"x{i}" for i in range(1025)])

    def test_identity_strings_reject_over_1024_code_points(self):
        """Each identity string field enforces the core's 1024-code-point limit."""
        from pydantic import ValidationError

        at, over = "x" * 1024, "x" * 1025

        IdentityBindingValue(ref=at)
        with pytest.raises(ValidationError):
            IdentityBindingValue(ref=over)

        IdentityBindingSet(refs=[at])
        with pytest.raises(ValidationError):
            IdentityBindingSet(refs=[over])

        IdentitySlot(property=at, binding=IdentityBindingValue(ref="v"))
        with pytest.raises(ValidationError):
            IdentitySlot(property=over, binding=IdentityBindingValue(ref="v"))

        # The shape union's own string leaves are IdentityString too --
        # population_type here stands in for the other four schema members,
        # which all share the same annotation.
        BaseIdentity(
            shape={"kind": "statvar_schema", "population_type": at},
            slots=[],
            requested_window={"kind": "all"},
        )
        with pytest.raises(ValidationError):
            BaseIdentity(
                shape={"kind": "statvar_schema", "population_type": over},
                slots=[],
                requested_window={"kind": "all"},
            )

        BaseIdentity(
            shape=shape_for("s"),
            slots=[],
            entities={"destination": [at]},
            requested_window={"kind": "all"},
        )
        with pytest.raises(ValidationError):
            BaseIdentity(
                shape=shape_for("s"),
                slots=[],
                entities={"destination": [over]},
                requested_window={"kind": "all"},
            )

    def test_base_identity_entities_rejects_a_role_over_1024_dcids(self):
        from pydantic import ValidationError

        BaseIdentity(
            shape=shape_for("s"),
            slots=[],
            entities={"destination": [f"country/{i}" for i in range(1024)]},
            requested_window={"kind": "all"},
        )  # at the limit: still valid
        with pytest.raises(ValidationError):
            BaseIdentity(
                shape=shape_for("s"),
                slots=[],
                entities={"destination": [f"country/{i}" for i in range(1025)]},
                requested_window={"kind": "all"},
            )

    def test_base_identity_entities_rejects_more_than_16_roles(self):
        """The identity core accepts at most 16 roles."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BaseIdentity(
                shape=shape_for("s"),
                slots=[],
                entities={f"role{i}": ["x"] for i in range(17)},
                requested_window={"kind": "all"},
            )

    def test_identity_window_range_rejects_year_outside_safe_integer_range(self):
        from pydantic import ValidationError

        IdentityWindowRange(start_year=2**53 - 1)  # at the limit: still valid
        with pytest.raises(ValidationError):
            IdentityWindowRange(start_year=2**53)
        with pytest.raises(ValidationError):
            IdentityWindowRange(end_year=-(2**53))


# ---------------------------------------------------------------------------
# max_candidates ge=2 validation
# ---------------------------------------------------------------------------


class TestMaxCandidatesValidation:
    def test_max_candidates_1_raises_validation_error(self):
        """The minimum candidate count is 2."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResolveOptions(max_candidates=1)

    def test_max_candidates_0_raises_validation_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResolveOptions(max_candidates=0)

    def test_max_candidates_2_is_valid(self):
        opts = ResolveOptions(max_candidates=2)
        assert opts.max_candidates == 2

    def test_max_candidates_clamped_via_resolve(self):
        """A candidate limit above the server ceiling is clamped."""
        from search.engine.config import settings

        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
        request = ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="a resubmitted identity",
                        identity=make_identity(
                            GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities
                        ),
                        base_spec_id=base_spec_id,
                        base_identity=base_identity,
                        resolved_stat_var_dcids=[_GDP_SV_DCID],
                    )
                ]
            ),
            response_options=ResolveOptions(max_candidates=settings().max_candidates + 100),
        )
        result = offline_resolve(request, sdmx=FakeSdmxClient())
        # Should be definite (one SV found), not an error
        assert result.status == "definite"


# ---------------------------------------------------------------------------
# standard promote path
# ---------------------------------------------------------------------------


_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_ENTITY_DCID = "country/IND"

# Ratio-backed shape with `Count_BirthEvent_LiveBirth` as its denominator.
_RATIO_SV_DCID = "Count_MortalityEvent_Infant_AsAFractionOf_Count_BirthEvent_LiveBirth"
_RATIO_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "MortalityEvent",
    "measured_property": "count",
    "stat_type": "measuredValue",
    "measurement_denominator": {"ref": "Count_BirthEvent_LiveBirth"},
}
_RATIO_ENTITY_DCID = "country/KEN"

# The activitySource constraint on this SV: GrossDomesticProduction
_ACTIVITY_SOURCE_SLOT = _value_slot(
    "how",
    "activitySource",
    "activity source",
    "GrossDomesticProduction",
    "Gross Domestic Production",
)


class TestStandardPromote:
    """Pins standard promotion through resubmission."""

    def _make_inp(
        self,
        resolved_stat_var_dcids=None,
        entity_dcid=_ENTITY_DCID,
        slots=None,
        origin_dcids=None,
        base_entity_dcid=None,
    ):
        entities = [subject_entity(entity_dcid)] if entity_dcid else None
        base_entities = [subject_entity(base_entity_dcid or entity_dcid)] if entity_dcid else None
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=base_entities)
        posted_slots = list(slots) if slots is not None else [_ACTIVITY_SOURCE_SLOT]
        identity_entities = [*(entities or [])]
        identity_entities.extend(origin_entity(dcid) for dcid in origin_dcids or [])
        return ResubmitIdentityPayload(
            identity=make_identity(GDP_SHAPE, posted_slots, entities=identity_entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=resolved_stat_var_dcids or [_GDP_SV_DCID],
        )

    def test_definite_from_re_read_anchors(self):
        """Standard promote with valid SV + entity → definite."""
        inp = self._make_inp()
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region.status == "definite"

    def test_shape_anchors_from_graph_not_posted_shape(self):
        """Five-tuple anchors come from re-reading SV arcs, not from the posted shape."""
        inp = self._make_inp()
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        spec = region.specs[0]
        # population_type is the re-read five-tuple anchor
        shape = spec.resolved.shape
        assert shape.population_type and shape.population_type.dcid == "EconomicActivity"
        assert shape.measured_property and shape.measured_property.dcid == "amount"

    def test_extract_skipped_trace(self):
        """Standard promote must record extract step with ran=False."""
        inp = self._make_inp()
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        spec = region.specs[0]
        extract_steps = [s for s in spec.resolved.trace.pipeline_trace if s.step == "extract"]
        assert extract_steps and extract_steps[0].ran is False

    @pytest.mark.parametrize(
        ("shape", "entities"),
        [
            pytest.param(GDP_SHAPE, [], id="standard-shape"),
            pytest.param(shape_for("completely_unknown_shape_xyz"), [], id="unknown-shape"),
            pytest.param(
                shape_for("completely_unknown_shape_xyz"),
                [origin_entity("country/USA")],
                id="unknown-shape-with-origin",
            ),
            pytest.param(
                {"kind": "statvar_schema", "population_type": "doesnt_matter"},
                [origin_entity("country/USA")],
                id="unresolved-shape-with-origin",
            ),
        ],
    )
    def test_missing_base_refuses_before_shape_or_origin_resolution(self, shape, entities):
        """A missing base raises `request_invalid` before shape or origin checks."""
        inp = ResubmitIdentityPayload(identity=make_identity(shape, entities=entities))
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "request_invalid"

    def test_valid_standard_shape_with_origin_dcids_raises_role_incompatible(self):
        """A standard shape rejects origin refs with `role_incompatible`."""
        inp = self._make_inp(origin_dcids=["country/USA"])
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "role_incompatible"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/input/identity/entities/origin"
        assert "economicactivity_amount_measuredvalue_nominal" in str(exc_info.value)

    def test_absent_posted_sv_yields_no_data(self):
        """A base-carried resolved_stat_var_dcids entry absent from the graph →
        no_data."""
        inp = self._make_inp(resolved_stat_var_dcids=["NONEXISTENT_SV_12345"], slots=[])
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region.status == "no_data"

    def test_absent_entity_dcid_refuses_invalid_role_ref(self):
        """A graph-missing `observationAbout` ref raises `invalid_role_ref` (422)."""
        inp = self._make_inp(entity_dcid="NONEXISTENT_ENTITY_XYZ")
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "invalid_role_ref"
        assert exc_info.value.status == 422

    def test_multi_shape_stat_vars_raises(self):
        """A base spanning two five-tuples raises `request_invalid`."""
        inp = self._make_inp(resolved_stat_var_dcids=[_GDP_SV_DCID, _RATIO_SV_DCID], slots=[])
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "request_invalid"

    def test_shape_mismatch_raises(self):
        """A posted shape that differs from the StatVar arcs raises `shape_mismatch`."""
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            entities=[subject_entity(_ENTITY_DCID)],
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(
                shape_for("completely_wrong_shape"), entities=[subject_entity(_ENTITY_DCID)]
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "shape_mismatch"

    def test_unbound_slot_on_zero_constraint_sv_not_flagged(self):
        """An echoed unbound slot is not an edit."""
        # Count_Person carries no constraintProperties whatsoever.
        unbound_slot = _unbound_slot("how", "age", "age")
        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [unbound_slot], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=["Count_Person"],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                materialise_graph=axis_confirming_reader(
                    "Count_Person_EducationalAttainment1StGrade"
                )
            ),
        )
        assert region.status == "definite"

    def test_nested_entity_not_treated_as_refine_edit(self):
        """An identity entity is not a what/how refine constraint."""
        inp = self._make_inp(entity_dcid="country/BRA")
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region.status == "definite"

    def test_posted_where_slot_entity_wins_over_base(self):
        """The posted entity replaces the base entity in the resolved response."""
        # Both entities have data, so check which one resolves.
        inp = self._make_inp(entity_dcid="country/BRA", base_entity_dcid=_ENTITY_DCID)
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region.status == "definite"
        subject_entities = {
            e.ref for e in region.specs[0].resolved.entities if e.role == "observationAbout"
        }
        assert subject_entities == {"country/BRA"}

    def test_no_entity_degrades_to_no_data(self):
        """A standard promote without an entity returns `no_data`."""
        inp = self._make_inp(slots=[_ACTIVITY_SOURCE_SLOT], entity_dcid=None)
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region.status in ("definite", "no_data")

    def test_end_to_end_via_offline_resolve(self):
        """A resubmitted identity promotes through the full resolve path."""
        request = ResolveRequest(input=_identity_request(self._make_inp()))
        result = offline_resolve(request, sdmx=FakeSdmxClient())
        assert result.status == "definite"
        echo = result.query_echo
        assert echo.entry_path == "dataset_resubmit"
        assert echo.extract_skipped is True


# ---------------------------------------------------------------------------
# standard-shape refine: construct-and-confirm sibling path
# ---------------------------------------------------------------------------

_PERSON_FEMALE_SV = "Count_Person_Female"
_PERSON_MALE_SV = "Count_Person_Male"
_PERSON_URBAN_FEMALE_SV = "Count_Person_Urban_Female"
_PERSON_RURAL_FEMALE_SV = "Count_Person_Rural_Female"
_REFINE_ENTITY_DCID = "country/IND"


def _standard_refine_graph() -> FakeGraph:
    """Add synthetic Count_Person_Male data without changing shared fixtures.

    Female, Urban Female, and Rural Female are loaded from the fixture files.
    FakeGraph also needs labels copied from the existing `name` arcs.
    """
    import json
    from pathlib import Path

    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    nodes = json.loads((fixtures_dir / "graph_nodes.json").read_text())
    obs = json.loads((fixtures_dir / "graph_obs.json").read_text())

    # FakeGraph reads top-level labels; copy the existing name-arc values for
    # these two fixture rows, which have null labels.
    for dcid in (_PERSON_URBAN_FEMALE_SV, _PERSON_RURAL_FEMALE_SV):
        if nodes[dcid].get("label") is None:
            name_nodes = nodes[dcid]["arcs"]["name"]["nodes"]
            nodes[dcid]["label"] = name_nodes[-1]["value"]

    def _prop_node(dcid: str, types: list[str]) -> dict:
        return {
            "name": dcid,
            "types": types,
            "dcid": dcid,
            "provenanceId": "dc/base/HumanReadableStatVars",
        }

    nodes[_PERSON_MALE_SV] = {
        "label": "Male population",
        "type": "StatisticalVariable",
        "arcs": {
            "statType": {"nodes": [_prop_node("measuredValue", ["Property"])]},
            "typeOf": {"nodes": [_prop_node("StatisticalVariable", ["Class"])]},
            "description": {
                "nodes": [
                    {
                        "provenanceId": "dc/base/HumanReadableStatVars",
                        "value": "The total number of males in a population.",
                    }
                ]
            },
            "gender": {"nodes": [_prop_node("Male", ["GenderType"])]},
            "constraintProperties": {"nodes": [_prop_node("gender", ["Property"])]},
            "provenance": {"nodes": [_prop_node("dc/base/HumanReadableStatVars", ["Provenance"])]},
            "measuredProperty": {"nodes": [_prop_node("count", ["Property"])]},
            "memberOf": {
                "nodes": [
                    {
                        "name": "Person With Gender = Male",
                        "types": ["StatVarGroup"],
                        "dcid": "dc/g/Person_Gender-Male",
                        "provenanceId": "dc/base/HumanReadableStatVars",
                    }
                ]
            },
            "name": {
                "nodes": [
                    {
                        "provenanceId": "dc/base/HumanReadableStatVars",
                        "value": "Male population",
                    }
                ]
            },
            "populationType": {"nodes": [_prop_node("Person", ["Class"])]},
        },
    }

    _obs = [
        {
            "earliestDate": "2001",
            "latestDate": "2021",
            "obsCount": 3,
            "provenanceId": "dc/base/IndiaCensus_Primary",
        }
    ]
    obs[f"{_PERSON_FEMALE_SV}|{_REFINE_ENTITY_DCID}"] = _obs
    obs[f"{_PERSON_MALE_SV}|{_REFINE_ENTITY_DCID}"] = _obs
    obs[f"{_PERSON_URBAN_FEMALE_SV}|{_REFINE_ENTITY_DCID}"] = _obs
    obs[f"{_PERSON_RURAL_FEMALE_SV}|{_REFINE_ENTITY_DCID}"] = _obs

    return FakeGraph(nodes=nodes, obs=obs)


def _resubmit_refine(inp: ResubmitIdentityPayload):
    """Resubmit `inp` over `_standard_refine_graph`, whose person counts the mint's walk
    reads as the family an edit elects."""
    graph = _standard_refine_graph()
    family = (_PERSON_FEMALE_SV, _PERSON_MALE_SV, _PERSON_URBAN_FEMALE_SV, _PERSON_RURAL_FEMALE_SV)
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(materialise_graph=axis_confirming_reader(*family, graph=graph)),
    )


class TestStandardRefine:
    """Pins standard-shape refine. An edit the posted members do not answer elects the edited
    identity's family."""

    def test_replace_constraint_value_confirms_sibling(self):
        """Changing gender from Female to Male confirms `Count_Person_Male`."""
        edited_slot = _value_slot("how", "gender", "gender", "Male", "Male")
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [edited_slot], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_FEMALE_SV],
        )
        region = _resubmit_refine(inp)
        assert region.status == "definite"
        assert [sv.ref for sv in region.specs[0].resolved.stat_vars.items] == [_PERSON_MALE_SV]

    def test_add_constraint_confirms_sibling(self):
        """Adding an Urban residence constraint confirms `Count_Person_Urban_Female`."""
        added_slot = _value_slot(
            "how",
            "placeOfResidenceClassification",
            "place of residence",
            "Urban",
            "Urban",
        )
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            # gender is the pinned SV's own shape-declared slot key; the
            # schema-level missing-slot-key check requires it present
            # (unbound is fine) alongside the newly-added constraint.
            identity=make_identity(
                PERSON_COUNT_SHAPE,
                [added_slot, _unbound_slot("how", "gender", "gender")],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_FEMALE_SV],
        )
        region = _resubmit_refine(inp)
        assert region.status == "definite"
        assert [sv.ref for sv in region.specs[0].resolved.stat_vars.items] == [
            _PERSON_URBAN_FEMALE_SV
        ]

    def test_sibling_absent_yields_no_data_not_error(self):
        """An absent sibling returns `no_data`, not a request error."""
        edited_slot = _value_slot("what", "gender", "gender", "Nonbinary", "Nonbinary")
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [edited_slot], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_FEMALE_SV],
        )
        region = _resubmit_refine(inp)
        assert region.status == "no_data"
        assert region.no_data_reason == "no_matching_slice"

    def test_refine_on_binding_set_resolves_every_value(self):
        """A two-value residence set resolves both confirmed siblings."""
        binding_set_slot = Slot(
            key=SlotKey(
                axis="how",
                property=GraphRef(
                    dcid="placeOfResidenceClassification", label="place of residence"
                ),
                label="place of residence",
            ),
            binding=BindingSet(
                values=[
                    SlotValue(ref=GraphRef(dcid="Urban", label="Urban"), value_kind="enum_value"),
                    SlotValue(ref=GraphRef(dcid="Rural", label="Rural"), value_kind="enum_value"),
                ]
            ),
        )
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            # The schema requires the existing gender slot, even when unbound.
            identity=make_identity(
                PERSON_COUNT_SHAPE,
                [binding_set_slot, _unbound_slot("how", "gender", "gender")],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_FEMALE_SV],
        )
        region = _resubmit_refine(inp)
        assert region.status == "definite"
        dcids = {sv.ref for sv in region.specs[0].resolved.stat_vars.items}
        assert dcids == {_PERSON_URBAN_FEMALE_SV, _PERSON_RURAL_FEMALE_SV}

    def test_promote_regression_unaffected_by_refine_path(self):
        """An unchanged binding promotes without entering the refine path."""
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(
                PERSON_COUNT_SHAPE, [_unbound_slot("how", "gender", "gender")], entities=entities
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_FEMALE_SV],
        )
        region = _resubmit_refine(inp)
        assert region.status == "definite"
        assert [sv.ref for sv in region.specs[0].resolved.stat_vars.items] == [_PERSON_FEMALE_SV]


# ---------------------------------------------------------------------------
# refine_supported on Shape
# ---------------------------------------------------------------------------


class TestRefineSupported:
    def test_standard_shape_refine_supported_true(self):
        """Standard shapes must have refine_supported=True (construct-and-confirm
        refine is supported, same as named families)."""
        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region.status == "definite"
        assert region.specs[0].resolved.shape.refine_supported is True

    def test_ratio_backed_standard_shape_refine_supported_false(self):
        """Ratio-backed shapes cannot construct sibling StatVars, so refinement is disabled."""
        entities = [subject_entity(_RATIO_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            _RATIO_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            # ageGroupClassification is this SV's only shape-declared what/how
            # slot key; required present (unbound is fine).
            identity=make_identity(
                _RATIO_SHAPE,
                [_unbound_slot("how", "ageGroupClassification", "age group")],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_RATIO_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region.status == "definite"
        assert region.specs[0].resolved.shape.refine_supported is False

    def test_ratio_backed_standard_shape_refine_degrades_to_no_data(self):
        """A ratio-backed edit returns `no_matching_slice` when the family holds no member
        for the edited value."""
        edit_slot = _value_slot(
            "how",
            "ageGroupClassification",
            "age group",
            "Youth",
            "Youth",
        )
        entities = [subject_entity(_RATIO_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            _RATIO_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(_RATIO_SHAPE, [edit_slot], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_RATIO_SV_DCID],
        )
        graph = FakeGraph()
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                materialise_graph=axis_confirming_reader(_RATIO_SV_DCID, graph=graph)
            ),
        )
        assert region.status == "no_data"
        assert region.no_data_reason == "no_matching_slice"


# ---------------------------------------------------------------------------
# spec_id identity floor, end to end via resolve_spec_resubmit
# ---------------------------------------------------------------------------


class TestSpecIdIdentityFloor:
    """`spec_id` includes role-keyed entities and the requested date window."""

    def test_gdp_india_vs_brazil_distinct_spec_ids(self):
        """Different subject entities produce different spec IDs."""
        india_entities = [subject_entity("country/IND")]
        india_base_spec_id, india_base_identity = make_base(
            GDP_SHAPE,
            entities=india_entities,
        )
        brazil_entities = [subject_entity("country/BRA")]
        brazil_base_spec_id, brazil_base_identity = make_base(
            GDP_SHAPE,
            entities=brazil_entities,
        )
        inp_india = ResubmitIdentityPayload(
            identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=india_entities),
            base_spec_id=india_base_spec_id,
            base_identity=india_base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        inp_brazil = ResubmitIdentityPayload(
            identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=brazil_entities),
            base_spec_id=brazil_base_spec_id,
            base_identity=brazil_base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region_india = resolve_spec_resubmit(
            inp=inp_india,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )
        region_brazil = resolve_spec_resubmit(
            inp=inp_brazil,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )
        assert region_india.status == "definite"
        assert region_brazil.status == "definite"
        assert region_india.specs[0].spec_id != region_brazil.specs[0].spec_id

    def test_requested_window_round_trips_stable_despite_narrower_echoed_slot(self):
        """A client carrying identity.requested_window forward gets back the same spec_id."""
        carried_requested_window = TimeWindow(start_year=2010, end_year=2020)
        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            entities=entities,
        )

        inp_first = ResubmitIdentityPayload(
            identity=make_identity(
                GDP_SHAPE,
                [_ACTIVITY_SOURCE_SLOT],
                entities=entities,
                requested_window=carried_requested_window,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        inp_promoted = ResubmitIdentityPayload(
            identity=make_identity(
                GDP_SHAPE,
                [_ACTIVITY_SOURCE_SLOT],
                entities=entities,
                requested_window=carried_requested_window,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region_first = resolve_spec_resubmit(
            inp=inp_first,
            graph=fake_graph_with_provenances(GDP_INDIA_PROVENANCES),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )
        region_promoted = resolve_spec_resubmit(
            inp=inp_promoted,
            graph=fake_graph_with_provenances(GDP_INDIA_PROVENANCES),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )
        assert region_first.status == "definite"
        assert region_promoted.status == "definite"
        assert region_first.specs[0].spec_id == region_promoted.specs[0].spec_id
        assert region_first.specs[0].resolved.trace.requested_window == carried_requested_window
        assert region_promoted.specs[0].resolved.trace.requested_window == carried_requested_window

    def test_when_slot_edit_without_requested_window_yields_different_id_than_original(self):
        """Changing the requested window changes the spec ID."""
        original_window = TimeWindow(start_year=2010, end_year=2020)
        edited_window = TimeWindow(start_year=2015, end_year=2018)
        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            entities=entities,
        )

        inp_original = ResubmitIdentityPayload(
            identity=make_identity(
                GDP_SHAPE,
                [_ACTIVITY_SOURCE_SLOT],
                entities=entities,
                requested_window=original_window,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        inp_edited = ResubmitIdentityPayload(
            identity=make_identity(
                GDP_SHAPE,
                [_ACTIVITY_SOURCE_SLOT],
                entities=entities,
                requested_window=edited_window,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region_original = resolve_spec_resubmit(
            inp=inp_original,
            graph=fake_graph_with_provenances(GDP_INDIA_PROVENANCES),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )
        region_edited = resolve_spec_resubmit(
            inp=inp_edited,
            graph=fake_graph_with_provenances(GDP_INDIA_PROVENANCES),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )

        assert region_original.status == "definite"
        assert region_edited.status == "definite"
        # The edit must NOT collide with the original's id.
        assert region_original.specs[0].spec_id != region_edited.specs[0].spec_id
        # The lineage self-heals: the edited response's own requested_window
        # reflects the NEW window, not the stale original -- so a further
        # promote from here forward (echoing THIS response's requested_window)
        # stays correctly anchored on 2015-2018, not 2010-2020.
        assert region_original.specs[0].resolved.trace.requested_window == original_window
        assert region_edited.specs[0].resolved.trace.requested_window == edited_window

    def test_all_window_identity_is_stable(self):
        """An identity that asks for all dates stays stable across a resubmit."""
        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            entities=entities,
        )
        inp_a = ResubmitIdentityPayload(
            identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        inp_b = ResubmitIdentityPayload(
            identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region_a = resolve_spec_resubmit(
            inp=inp_a, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        region_b = resolve_spec_resubmit(
            inp=inp_b, graph=FakeGraph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
        assert region_a.status == "definite"
        assert region_b.status == "definite"
        assert region_a.specs[0].spec_id == region_b.specs[0].spec_id


# ---------------------------------------------------------------------------
# Reversed identity windows fail validation before hashing
# ---------------------------------------------------------------------------


class TestReversedRequestedWindowRefuses:
    def test_standard_promote_reversed_identity_requested_window_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResubmitIdentityPayload(
                identity=BaseIdentity(
                    shape=GDP_SHAPE,
                    slots=[],
                    entities={"observationAbout": [_ENTITY_DCID]},
                    requested_window={"kind": "range", "start_year": 2020, "end_year": 2015},
                )
            )
