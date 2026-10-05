"""Test widening and reject resubmit seeds that contradict retained base bindings.

An axis posted unbound is dropped from the check. A contradiction returns request_invalid/400.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from search.engine.errors import EngineInputError
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.spec_id import build_spec_identity
from search.models import ResubmitIdentityPayload
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import (
    GDP_SHAPE,
    PERSON_COUNT_SHAPE,
    make_base,
    subject_entity,
    where_slots_for,
)
from tests.engine._builders import (
    binding_set_slot as _set_slot,
)
from tests.engine._builders import (
    unbound_slot as _unbound_slot,
)
from tests.engine._builders import (
    value_slot as _value_slot,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._search_app import search_app
from tests.engine.test_path_c import (
    _PERSON_FEMALE_SV,
    _PERSON_MALE_SV,
    _PERSON_RURAL_FEMALE_SV,
    _PERSON_URBAN_FEMALE_SV,
    _REFINE_ENTITY_DCID,
    _standard_refine_graph,
)
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_ENTITY_DCID = "country/IND"

_UNRESOLVED_SOURCE_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_UnresolvedSource"
_UNRESOLVED_SOURCE_ARCS = {
    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
    "constraintProperties": {"nodes": [{"dcid": "activitySource"}]},
    "activitySource": {"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]},
}


def _graph_with_unresolved_source_sv() -> FakeGraph:
    """Add an unresolved activitySource sibling with the GDP shape and observations."""
    nodes = {
        **_load_json("graph_nodes.json"),
        _UNRESOLVED_SOURCE_SV_DCID: {
            "label": "unresolved-activitySource GDP",
            "arcs": _UNRESOLVED_SOURCE_ARCS,
        },
    }
    real_gdp_key = f"{_GDP_SV_DCID}|{_GDP_ENTITY_DCID}"
    synthetic_key = f"{_UNRESOLVED_SOURCE_SV_DCID}|{_GDP_ENTITY_DCID}"
    default_obs = _load_json("graph_obs.json")
    obs = {**default_obs, synthetic_key: default_obs[real_gdp_key]}
    return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})


_ACTIVITY_SOURCE_BOUND = _value_slot(
    "how",
    "activitySource",
    "activity source",
    "GrossDomesticProduction",
    "Gross Domestic Production",
)
_ACTIVITY_SOURCE_UNBOUND = _unbound_slot("how", "activitySource", "activity source")

_GENDER_FEMALE = _value_slot("how", "gender", "gender", "Female", "Female")
_GENDER_MALE = _value_slot("how", "gender", "gender", "Male", "Male")
_FOLDED_TOTAL_SV = "Count_Person"
_PLACE_URBAN = _value_slot(
    "how", "placeOfResidenceClassification", "place of residence", "Urban", "Urban"
)
_PLACE_RURAL = _value_slot(
    "how", "placeOfResidenceClassification", "place of residence", "Rural", "Rural"
)
_PLACE_UNBOUND = _unbound_slot("how", "placeOfResidenceClassification", "place of residence")


def _edited_identity(shape: dict, slots: list, entities: list):
    return build_spec_identity(shape, slots, entities=entities)[1]


class TestWideningDropsBaseConstraint:
    """An unbound posted axis clears the base's constraint value."""

    def test_unbound_over_bound_base_widens_not_carries_forward(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            slots=[_ACTIVITY_SOURCE_BOUND],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                GDP_SHAPE, [_ACTIVITY_SOURCE_UNBOUND, *where_slots_for(entities)], entities
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite"
        activity_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.dcid == "activitySource"
        )
        assert activity_slot.binding.kind == "unbound", (
            "posting activitySource unbound over a base that bound it to "
            "GrossDomesticProduction must widen, not silently echo the base's "
            f"old value back; got {activity_slot.binding.kind!r}"
        )

    def test_seed_matching_base_and_posted_promotes_normally(self):
        """A seed matching the base and posted value promotes normally."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            slots=[_ACTIVITY_SOURCE_BOUND],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                GDP_SHAPE, [_ACTIVITY_SOURCE_BOUND, *where_slots_for(entities)], entities
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite"
        assert [sv.ref for sv in region.specs[0].resolved.stat_vars.items] == [_GDP_SV_DCID]
        activity_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.dcid == "activitySource"
        )
        assert activity_slot.binding.kind == "value"


class TestSeedContradictsBaseIdentity:
    """Reject seed values outside the base bindings retained by the request."""

    def test_seed_contradicting_bound_slot_refuses_over_http(self):
        """A posted Rural seed contradicting base Urban must return request_invalid/400."""
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[_GENDER_FEMALE, _PLACE_URBAN],
            entities=entities,
        )
        app = search_app(
            graph=_standard_refine_graph(),
            resolver=make_resolver(_standard_refine_graph()),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
        )
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _edited_identity(
                                PERSON_COUNT_SHAPE,
                                [_GENDER_FEMALE, _PLACE_RURAL, *where_slots_for(entities)],
                                entities,
                            ).model_dump(mode="json"),
                            "base_spec_id": base_spec_id,
                            "base_identity": base_identity.model_dump(mode="json"),
                            "resolved_stat_var_dcids": [_PERSON_RURAL_FEMALE_SV],
                        }
                    ],
                }
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "request_invalid"

    def test_seed_contradicting_bound_slot_refuses_direct(self):
        """Direct resolution raises EngineInputError with request_invalid."""
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[_GENDER_FEMALE, _PLACE_URBAN],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [_GENDER_FEMALE, _PLACE_RURAL, *where_slots_for(entities)],
                entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_RURAL_FEMALE_SV],
        )
        try:
            resolve_spec_resubmit(
                inp=inp,
                graph=_standard_refine_graph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
            raised = False
        except EngineInputError as exc:
            raised = True
            assert exc.code == "request_invalid"
        assert raised, "a seed contradicting a bound base slot value must raise EngineInputError"

    def test_seed_differing_only_on_widened_axis_is_accepted(self):
        """A seed lacking the Urban property is accepted when the request widens that axis."""
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[_GENDER_FEMALE, _PLACE_URBAN],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [_GENDER_FEMALE, _PLACE_UNBOUND, *where_slots_for(entities)],
                entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_FEMALE_SV],
        )
        graph = _standard_refine_graph()
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                materialise_graph=axis_confirming_reader(
                    _PERSON_URBAN_FEMALE_SV, _PERSON_RURAL_FEMALE_SV, graph=graph
                )
            ),
        )
        assert region.status == "definite"
        assert [sv.ref for sv in region.specs[0].resolved.stat_vars.items] == [_PERSON_FEMALE_SV]

    def test_seed_matching_base_on_a_multi_axis_shape_promotes(self):
        """A seed matching both bound Person axes promotes without refusal."""
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[_GENDER_FEMALE, _PLACE_URBAN],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [_GENDER_FEMALE, _PLACE_URBAN, *where_slots_for(entities)],
                entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_URBAN_FEMALE_SV],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_standard_refine_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite"
        assert [sv.ref for sv in region.specs[0].resolved.stat_vars.items] == [
            _PERSON_URBAN_FEMALE_SV
        ]


class TestSeedContradictsBaseIdentityUnresolved:
    """An unresolved seed value contradicts a bound base property."""

    def test_seed_with_unresolved_bound_property_refuses(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            slots=[_ACTIVITY_SOURCE_BOUND],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            # The posted value matches the base. The unresolved seed arc can refuse.
            identity=_edited_identity(
                GDP_SHAPE, [_ACTIVITY_SOURCE_BOUND, *where_slots_for(entities)], entities
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_UNRESOLVED_SOURCE_SV_DCID],
        )
        try:
            resolve_spec_resubmit(
                inp=inp,
                graph=_graph_with_unresolved_source_sv(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
            raised = False
        except EngineInputError as exc:
            raised = True
            assert exc.code == "request_invalid"
        assert raised, (
            "a seed whose own value for a base-bound property is present but "
            "unresolved must raise EngineInputError, not pass the gate as if "
            "the seed asserted nothing on that axis"
        )

    def test_seed_with_unresolved_unbound_property_is_irrelevant(self):
        """An unresolved seed property is irrelevant when the base leaves it unbound."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(GDP_SHAPE, [*where_slots_for(entities)], entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_UNRESOLVED_SOURCE_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_graph_with_unresolved_source_sv(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite"


class TestSeedCheckMixedPresenceAndPlurality:
    """Combine each seed's asserted values as a subset. Silent members add no claim.

    Refuse only when the combined values exceed the base binding.
    """

    def test_mixed_presence_seed_matching_representative_value_promotes(self):
        """A folded-total seed's missing gender arc is silent beside the Female member."""
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[_GENDER_FEMALE],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE, [_GENDER_FEMALE, *where_slots_for(entities)], entities
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_FEMALE_SV, _FOLDED_TOTAL_SV],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_standard_refine_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite"

    def test_all_assert_disagreement_seed_matching_base_set_promotes(self):
        """A seed asserting both bound gender values promotes when the base binds both."""
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        gender_set_slot = _set_slot("how", "gender", "gender", ["Female", "Male"])
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[gender_set_slot],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE, [gender_set_slot, *where_slots_for(entities)], entities
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_FEMALE_SV, _PERSON_MALE_SV],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_standard_refine_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite"
        dcids = {sv.ref for sv in region.specs[0].resolved.stat_vars.items}
        assert dcids == {_PERSON_FEMALE_SV, _PERSON_MALE_SV}

    @pytest.mark.parametrize(
        ("base_slots", "identity_slots", "resolved_stat_var_dcids"),
        [
            pytest.param(
                [_GENDER_FEMALE],
                [_GENDER_FEMALE],
                [_PERSON_FEMALE_SV, _PERSON_MALE_SV, _FOLDED_TOTAL_SV],
                id="mixed-presence-padded-disagreeing-member",
            ),
            pytest.param(
                [_GENDER_MALE],
                [_GENDER_FEMALE],
                [_PERSON_FEMALE_SV, _PERSON_MALE_SV, _FOLDED_TOTAL_SV],
                id="mixed-presence-contradicting-representative",
            ),
            pytest.param(
                [_GENDER_FEMALE],
                [_set_slot("how", "gender", "gender", ["Female", "Male"])],
                [_PERSON_FEMALE_SV, _PERSON_MALE_SV],
                id="all-assert-disagreement-contradicting-base-set",
            ),
        ],
    )
    def test_seed_outside_base_identity_refuses(
        self, base_slots, identity_slots, resolved_stat_var_dcids
    ):
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=base_slots,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE, [*identity_slots, *where_slots_for(entities)], entities
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=resolved_stat_var_dcids,
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=_standard_refine_graph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "request_invalid"
