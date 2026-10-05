"""Edge-case tests for the Search contract models."""

import pytest
from pydantic import ValidationError

from search import (
    BindingSet,
    BreadthDim,
    CandidateSet,
    CoverageBare,
    CoverageBreadth,
    CoverageExact,
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    PipelineStep,
    ResolvedEntity,
    ResubmitIdentityPayload,
    Shape,
    SlotKey,
    SlotValue,
    Spec,
    StatVar,
    TimeWindow,
)
from search.engine.spec_build import _resolved_entities, build_spec
from search.engine.spec_id import build_identity_projection, entity_role_key
from search.models import BaseIdentity, Resolution, SentenceToken, SpecObservations
from tests.conftest import base_response, minimal_spec


def _definite_part(spec_id: str, **extra) -> dict:
    return {"status": "definite", "spec": minimal_spec(spec_id), **extra}


class TestResolveResponsePartsAndDerivedStatus:
    """Parts preserve query order; response status is shared or `mixed` according to their values."""

    def test_single_part_status_matches_that_part(self):
        resp = Resolution.model_validate(base_response(parts=[_definite_part("s1")]))
        assert len(resp.parts) == 1
        assert resp.status == "definite"

    def test_multiple_definite_parts_stay_in_query_order(self):
        resp = Resolution.model_validate(
            base_response(parts=[_definite_part("s1"), _definite_part("s2")])
        )
        assert [p.spec.spec_id for p in resp.parts] == ["s1", "s2"]
        assert resp.status == "definite"

    def test_parts_min_length(self):
        with pytest.raises(ValidationError):
            Resolution.model_validate(base_response(parts=[]))

    def test_status_is_mixed_when_parts_disagree(self):
        """Different part statuses produce envelope status `mixed`."""
        resp = Resolution.model_validate(
            base_response(
                parts=[
                    _definite_part("s1"),
                    {"status": "no_data", "no_data": {"reason": "variable_not_resolved"}},
                ]
            )
        )
        assert resp.status == "mixed"

    def test_status_field_on_the_wire_is_ignored_and_recomputed(self):
        """The parts determine status when the posted status disagrees."""
        resp = Resolution.model_validate(
            base_response(status="candidates", parts=[_definite_part("s1")])
        )
        assert resp.status == "definite"


class TestSlotValueKinds:
    def test_time_window_kind(self):
        sv = SlotValue(value_kind="time_window", time_window=TimeWindow(start_year=2015))
        assert sv.value_kind == "time_window"
        assert sv.time_window is not None
        assert sv.time_window.start_year == 2015

    def test_literal_kind(self):
        sv = SlotValue(value_kind="literal", literal="some literal value")
        assert sv.value_kind == "literal"
        assert sv.literal == "some literal value"

    def test_entity_kind_with_ref(self):
        ref = GraphRef(dcid="country/ETH", label="Ethiopia")
        sv = SlotValue(value_kind="entity", ref=ref)
        assert sv.value_kind == "entity"
        assert sv.ref is not None
        assert sv.ref.dcid == "country/ETH"


class TestEntityRole:
    def test_directional_requires_role_and_axis(self):
        role = EntityRoleDirectional(
            kind="directional",
            role=GraphRef(dcid="DevelopmentFinanceRecipient", label="recipient"),
            axis="destination",
        )
        assert role.kind == "directional"
        assert role.axis == "destination"

    def test_directional_missing_axis_raises(self):
        with pytest.raises(ValidationError):
            EntityRoleDirectional.model_validate(
                {"kind": "directional", "role": {"dcid": "Recipient", "label": "recipient"}}
            )

    @pytest.mark.parametrize("legacy_field", ["donor_dcid", "donor_dcids"])
    def test_resubmit_rejects_legacy_donor_fields(self, legacy_field: str):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ResubmitIdentityPayload.model_validate(
                {
                    "identity": {
                        "shape": {"kind": "statvar_schema", "population_type": "Person"},
                        "slots": [],
                        "entities": {},
                        "requested_window": {"kind": "all"},
                    },
                    legacy_field: "country/FRA"
                    if legacy_field == "donor_dcid"
                    else ["country/FRA"],
                }
            )

    @pytest.mark.parametrize("legacy_role", ["provider", "recipient", "from", "to"])
    def test_resolved_entity_rejects_legacy_role_vocabulary(self, legacy_role: str):
        with pytest.raises(ValidationError):
            ResolvedEntity.model_validate({"ref": "country/KEN", "role": legacy_role})

    @pytest.mark.parametrize("legacy_role", ["provider", "recipient"])
    def test_identity_rejects_legacy_role_keys(self, legacy_role: str):
        with pytest.raises(ValidationError, match="unrecognised entity role key"):
            BaseIdentity(
                shape={"kind": "statvar_schema", "population_type": "Person"},
                slots=[],
                entities={legacy_role: ["country/KEN"]},
                requested_window={"kind": "all"},
            )


class TestCoverageArms:
    def test_exact_requires_observation_count(self):
        cov = CoverageExact(kind="exact", has_data=True, observation_count=42)
        assert cov.observation_count == 42

    def test_exact_missing_observation_count_raises(self):
        with pytest.raises(ValidationError):
            CoverageExact.model_validate({"kind": "exact", "has_data": True})

    def test_breadth_requires_non_empty_dimensions(self):
        # empty dimensions should raise (min_length=1)
        with pytest.raises(ValidationError):
            CoverageBreadth.model_validate({"kind": "breadth", "has_data": True, "dimensions": []})

    def test_breadth_with_dimensions_validates(self):
        cov = CoverageBreadth(
            kind="breadth",
            has_data=True,
            dimensions=[BreadthDim(label="donors", count=32)],
        )
        assert len(cov.dimensions) == 1

    def test_bare_requires_only_kind_and_has_data(self):
        cov = CoverageBare(kind="bare", has_data=False)
        assert cov.has_data is False
        assert cov.window is None


class TestSlotKeyPropertyNullable:
    def test_when_axis_no_property(self):
        key = SlotKey(axis="when", label="period")
        assert key.property is None

    def test_how_axis_with_property(self):
        key = SlotKey(
            axis="how",
            property=GraphRef(dcid="DevelopmentFinancePurpose", label="purpose"),
            label="purpose",
        )
        assert key.property is not None
        assert key.property.dcid == "DevelopmentFinancePurpose"


class TestBindingSetMembership:
    def _slot_value(self, dcid: str) -> SlotValue:
        return SlotValue(
            value_kind="enum_value",
            ref=GraphRef(dcid=dcid, label=dcid),
        )

    def test_one_member_set_rejected(self):
        with pytest.raises(ValidationError):
            BindingSet(kind="set", values=[self._slot_value("A")])

    def test_two_member_set_validates(self):
        bs = BindingSet(
            kind="set",
            values=[self._slot_value("A"), self._slot_value("B")],
        )
        assert len(bs.values) == 2


class TestTimeWindow:
    def test_no_bounds_raises(self):
        with pytest.raises(ValidationError):
            TimeWindow()

    def test_start_only_validates(self):
        tw = TimeWindow(start_year=2015)
        assert tw.start_year == 2015
        assert tw.end_year is None

    def test_end_only_validates(self):
        tw = TimeWindow(end_year=2023)
        assert tw.end_year == 2023
        assert tw.start_year is None


class TestCandidateSetMinTwoSpecs:
    def test_one_spec_raises(self):
        with pytest.raises(ValidationError):
            # minimal_spec returns a contract-valid Spec dict that Pydantic coerces.
            CandidateSet(
                ordering="broadest_first",
                max_candidates=5,
                specs=[minimal_spec("s1")],  # ty: ignore[invalid-argument-type]
                truncated=False,
            )

    def test_two_specs_validates(self):
        cs = CandidateSet(
            ordering="broadest_first",
            max_candidates=5,
            # minimal_spec returns a contract-valid Spec dict that Pydantic coerces.
            specs=[minimal_spec("s1"), minimal_spec("s2")],  # ty: ignore[invalid-argument-type]
            truncated=False,
        )
        assert len(cs.specs) == 2


class TestInapplicableSlotsIdentity:
    """`inapplicable_slots` is resolved graph state and does not affect `spec_id`."""

    def test_not_a_member_of_identity_projection_or_base_identity(self):
        shape = {"kind": "statvar_schema", "population_type": "sh1"}
        projection = build_identity_projection(shape, [], entities=None, requested_window=None)
        assert "inapplicable_slots" not in projection
        assert "inapplicable_slots" not in BaseIdentity.model_fields

    def test_specs_differing_only_in_inapplicable_slots_share_spec_id(self):
        shape = Shape(
            shape_id="sh1",
            label="Test Shape",
            population_type=GraphRef(dcid="Pop", label="Pop"),
            measured_property=GraphRef(dcid="Prop", label="Prop"),
            stat_type=GraphRef(dcid="measuredValue", label="measured value"),
            slot_keys=[],
            member_count=1,
        )
        stat_vars = [
            StatVar(
                ref=GraphRef(dcid="sv1", label="SV1"),
                shape_id="sh1",
                slot_values=[],
                constraint_properties=[],
            )
        ]
        coverage = CoverageBare(has_data=True)
        pipeline_trace = [PipelineStep(step="extract", ran=True)]

        spec_without = build_spec(
            shape,
            [],
            stat_vars,
            [],
            coverage,
            pipeline_trace,
            inapplicable_slots=[],
            measure_elected_properties=frozenset(),
        )
        spec_with = build_spec(
            shape,
            [],
            stat_vars,
            [],
            coverage,
            pipeline_trace,
            inapplicable_slots=["SomeProperty"],
            measure_elected_properties=frozenset(),
        )
        assert spec_without.spec_id == spec_with.spec_id
        assert spec_without.resolved.inapplicable_slots == []
        assert spec_with.resolved.inapplicable_slots == ["SomeProperty"]


class TestResolvedObservationsRoundtrip:
    """`observations` belongs to `Spec.resolved`; a top-level copy update would not serialize it."""

    def _spec(self) -> Spec:
        return Spec.model_validate(minimal_spec("s1"))

    def test_nested_update_serialises_observations(self):
        spec = self._spec()
        obs = SpecObservations(series=[], total_obs=0, truncated=False)
        updated = spec.model_copy(
            update={"resolved": spec.resolved.model_copy(update={"observations": obs})}
        )
        dumped = updated.model_dump()
        assert dumped["resolved"]["observations"] is not None
        assert dumped["resolved"]["observations"]["total_obs"] == 0
        assert updated.resolved.observations == obs


class TestResolvedEntityRoleKeys:
    def test_every_role_is_a_valid_key_and_agrees_with_entity_role_key(self):
        entities = [
            Entity(
                ref=GraphRef(dcid="country/ETH", label="Ethiopia"),
                role=EntityRoleSubject(),
            ),
            Entity(
                ref=GraphRef(dcid="country/USA", label="USA"),
                role=EntityRoleDirectional(
                    role=GraphRef(dcid="DonorRole", label="Donor"), axis="origin"
                ),
            ),
            Entity(
                ref=GraphRef(dcid="country/KEN", label="Kenya"),
                role=EntityRoleDirectional(
                    role=GraphRef(dcid="RecipientRole", label="Recipient"), axis="destination"
                ),
            ),
        ]
        resolved = _resolved_entities(entities)
        assert {r.role for r in resolved} == {"observationAbout", "origin", "destination"}
        for source, projected in zip(entities, resolved, strict=True):
            assert projected.role == entity_role_key(source)


class TestIdentityVersionRefusal:
    """An unminted identity version returns `unsupported_schema_version` with status 422."""

    def _client(self):
        from fastapi.testclient import TestClient

        from tests.engine._harness import EMPTY_SDMX
        from tests.engine._search_app import search_app
        from tests.fixtures import FakeGraph, FakeLLM, make_resolver

        graph = FakeGraph()
        app = search_app(graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=EMPTY_SDMX)
        return TestClient(app, raise_server_exceptions=False)

    def test_unminted_identity_version_refuses_422(self):
        client = self._client()
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {"kind": "statvar_schema", "population_type": "sh1"},
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                            },
                            "base_spec_id": "s1",
                            "base_identity": {
                                "identity_version": "spec2",
                                "shape": {"kind": "statvar_schema", "population_type": "sh1"},
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                            },
                        }
                    ],
                }
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["code"] == "unsupported_schema_version"
        assert body["pointer"] == "/input/entries/0/base_identity/identity_version"

    def test_schema_version_refusal_still_points_at_top_level(self):
        """Regression: the schema_version case keeps its own pointer even
        though the check also matches identity_version by loc suffix."""
        client = self._client()
        resp = client.post(
            "/api/search/resolve",
            json={"schema_version": 2, "input": {"kind": "raw_text", "query": "test"}},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["code"] == "unsupported_schema_version"
        assert body["pointer"] == "/schema_version"


class TestSchemaPublishesResolved:
    def test_response_schema_carries_resolved_spec_and_parts(self):
        from fastapi.testclient import TestClient

        from tests.engine._harness import EMPTY_SDMX
        from tests.engine._search_app import search_app
        from tests.fixtures import FakeGraph, FakeLLM, make_resolver

        graph = FakeGraph()
        app = search_app(graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=EMPTY_SDMX)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/search/schema")
        assert resp.status_code == 200
        body = resp.json()
        defs = body["response"]["$defs"]

        assert "ResolvedSpec" in defs
        assert "inapplicable_slots" in defs["ResolvedSpec"]["properties"]

        assert "parts" in defs["Resolution"]["properties"]

        definite_part = defs["DefinitePart"]
        assert "spec" in definite_part["properties"]
        assert "spec" in definite_part["required"]


@pytest.mark.parametrize(
    ("token", "error_type"),
    [
        ({"kind": "note", "text": " A note."}, "literal_error"),
        ({"kind": "text", "text": ".", "note": "headline"}, "extra_forbidden"),
    ],
    ids=["note_kind", "note_key"],
)
def test_a_sentence_token_carries_no_note(token: dict[str, str], error_type: str) -> None:
    """A sentence is heading, byline and departures; a departure states what a note did."""
    with pytest.raises(ValidationError) as caught:
        SentenceToken.model_validate(token)
    assert [error["type"] for error in caught.value.errors()] == [error_type]
