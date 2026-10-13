"""Resubmit identity reconstruction uses all resolved StatVars, preserving
shape keys and untouched bindings after a narrow. A fresh-to-resubmit round
trip also pins the emitted identity and spec id.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_spec_resubmit, resolve_variable
from search.engine.region.state import EngineOptions
from search.engine.retrieval.facts import read_constraints
from search.engine.spec_id import build_spec_identity
from search.models import (
    BindingSet,
    BindingValue,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import (
    GDP_SHAPE,
    PERSON_COUNT_SHAPE,
    make_base,
    make_identity,
    subject_entity,
    where_slots_for,
)
from tests.engine._builders import (
    gref as _gref,
)
from tests.engine._builders import (
    prop_node as _prop_node,
)
from tests.engine._builders import (
    unbound_slot as _unbound_slot,
)
from tests.engine._builders import (
    value_slot as _value_slot,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_ENTITY_DCID = "country/IND"


def _edited_identity(shape, slots, entities):
    return build_spec_identity(shape, slots, entities=entities, open_roles=())[1]


def _differing_siblings_graph() -> FakeGraph:
    """Add India observations so resubmit tests reach identity reconstruction."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    obs = json.loads((fixtures_dir / "graph_obs.json").read_text())
    _obs = [{"earliestDate": "2010", "latestDate": "2020", "obsCount": 4, "importName": "test"}]
    obs["Count_Person_Female|country/IND"] = _obs
    obs["Count_Person_85OrMoreYears_Female|country/IND"] = _obs
    return FakeGraph(obs=obs)


# Rebuild shape keys from all resolved members.


class TestReducedRepresentativeShapeReconstruction:
    """Rebuild shape keys from all resolved members and the posted identity."""

    def test_unedited_resubmit_of_zero_constraint_seed_preserves_spec_id(self):
        """Count_Person's unedited resubmit preserves its unbound age axis and id."""
        entities = [subject_entity(_ENTITY_DCID)]
        unbound_axis = _unbound_slot("how", "age", "age")
        base_spec_id, _base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[unbound_axis],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE, [unbound_axis, *where_slots_for(entities)], entities
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(
                FakeGraph(),
                EngineOptions(
                    materialise_graph=axis_confirming_reader(
                        "Count_Person", "Count_Person_65OrMoreYears"
                    )
                ),
            ),
        )
        assert region.status == "definite"
        assert region.specs[0].spec_id == base_spec_id

    @pytest.mark.parametrize(
        ("axis_property", "axis_label"),
        [
            pytest.param(
                "zzz_never_declared_prop",
                "never declared",
                id="unknown_unbound_axis",
            ),
            pytest.param("industryType", "industry type", id="schema_valid_irrelevant_axis"),
        ],
    )
    def test_irrelevant_axis_now_resolves_unbound_and_inapplicable(self, axis_property, axis_label):
        """An extra unbound axis is inapplicable and changes the identity hash."""
        entities = [subject_entity(_ENTITY_DCID)]
        activity_bound = _value_slot(
            "how",
            "activitySource",
            "activity source",
            "GrossDomesticProduction",
            "Gross Domestic Production",
        )
        shape = GDP_SHAPE
        gdp_sv = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
        base_spec_id, _base_identity = make_base(
            shape,
            slots=[activity_bound],
            entities=entities,
        )

        inp_honest = ResubmitIdentityPayload(
            identity=_edited_identity(
                shape, [activity_bound, *where_slots_for(entities)], entities
            ),
        )
        region_honest = resolve_spec_resubmit(
            inp=inp_honest,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        assert region_honest.status == "definite"
        assert region_honest.specs[0].spec_id == base_spec_id

        irrelevant_axis = _unbound_slot("how", axis_property, axis_label)
        posted_slots = [activity_bound, irrelevant_axis, *where_slots_for(entities)]
        expected_spec_id, _ = build_spec_identity(
            shape, posted_slots, entities=entities, open_roles=()
        )
        inp_tampered = ResubmitIdentityPayload(
            identity=_edited_identity(shape, posted_slots, entities),
        )
        region = resolve_spec_resubmit(
            inp=inp_tampered,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(
                FakeGraph(), EngineOptions(materialise_graph=axis_confirming_reader(gdp_sv))
            ),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        assert spec.spec_id == expected_spec_id
        axis_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == axis_property
        )
        assert axis_slot.binding.kind == "unbound"
        assert axis_property in spec.resolved.inapplicable_slots

    def test_unedited_resubmit_of_differing_constraint_property_siblings_preserves_spec_id(self):
        """The representative asserts gender only. Its sibling also asserts age.

        The resubmit preserves both keys and the id.
        """
        entities = [subject_entity(_ENTITY_DCID)]
        gender_bound = _value_slot("how", "gender", "gender", "Female", "Female")
        age_unbound = _unbound_slot("how", "age", "age")
        base_spec_id, _base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[gender_bound, age_unbound],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [gender_bound, age_unbound, *where_slots_for(entities)],
                entities,
            ),
            # Female is representative and lacks age. The second member asserts it.
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_differing_siblings_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_differing_siblings_graph()),
        )
        assert region.status == "definite"
        assert region.specs[0].spec_id == base_spec_id

    def test_unedited_echo_with_base_declared_axis_is_unbound_and_inapplicable(self):
        """An unedited echo keeps the age slot unbound and inapplicable.

        A widened axis is unbound without the inapplicable flag.
        """
        entities = [subject_entity(_ENTITY_DCID)]
        unbound_axis = _unbound_slot("how", "age", "age")
        base_spec_id, _base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[unbound_axis],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE, [unbound_axis, *where_slots_for(entities)], entities
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(
                FakeGraph(),
                EngineOptions(
                    materialise_graph=axis_confirming_reader(
                        "Count_Person", "Count_Person_65OrMoreYears"
                    )
                ),
            ),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        age_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == "age"
        )
        assert age_slot.binding.kind == "unbound"
        assert "age" in spec.resolved.inapplicable_slots
        assert spec.spec_id == base_spec_id


# Reconstruct bindings after an in-set narrow.

_URBAN_SV = "Count_Person_Urban_Female"  # pre-existing fixture: gender=Female, residence=Urban
_RURAL_SV = "Count_Person_Rural_Female"  # pre-existing fixture: gender=Female, residence=Rural
# Use the value DCID suffix verbatim. Sibling construction matches that token.
_RURAL_ADULT_SV = "Count_Person_Rural_Female_Years18Onwards"


def _narrow_bug_graph(*, rural_adult_has_obs: bool = True) -> FakeGraph:
    """Add a synthetic Rural_Female_Years18Onwards sibling to the graph fixtures.

    Existing Rural and Urban members lack age. Urban asserts it for seed validation.
    With rural_adult_has_obs=False, only the age-silent Rural member has observations.
    """
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    nodes = json.loads((fixtures_dir / "graph_nodes.json").read_text())
    obs = json.loads((fixtures_dir / "graph_obs.json").read_text())

    # FakeGraph reads top-level labels. Copy each fixture's name value.
    for dcid in (_URBAN_SV, _RURAL_SV):
        if nodes[dcid].get("label") is None:
            name_nodes = nodes[dcid]["arcs"]["name"]["nodes"]
            nodes[dcid]["label"] = name_nodes[-1]["value"]

    # Add the age label so grounding preserves the Years18Onwards binding.
    nodes["Years18Onwards"] = {"label": "Years18Onwards", "type": None}

    # Urban asserts age for seed validation. The Rural sibling is age-silent.
    nodes[_URBAN_SV]["arcs"]["age"] = {"nodes": [_prop_node("Years18Onwards", ["Class"])]}
    nodes[_URBAN_SV]["arcs"]["constraintProperties"]["nodes"].append(
        _prop_node("age", ["Property"])
    )

    nodes[_RURAL_ADULT_SV] = {
        "label": "Adult rural female population",
        "type": "StatisticalVariable",
        "arcs": {
            "statType": {"nodes": [_prop_node("measuredValue", ["Property"])]},
            "typeOf": {"nodes": [_prop_node("StatisticalVariable", ["Class"])]},
            "measuredProperty": {"nodes": [_prop_node("count", ["Property"])]},
            "populationType": {"nodes": [_prop_node("Person", ["Class"])]},
            "gender": {"nodes": [_prop_node("Female", ["GenderType"])]},
            "placeOfResidenceClassification": {
                "nodes": [_prop_node("Rural", ["HumanSettlementClassificationEnum"])]
            },
            "age": {"nodes": [_prop_node("Years18Onwards", ["Class"])]},
            "constraintProperties": {
                "nodes": [
                    _prop_node("gender", ["Property"]),
                    _prop_node("placeOfResidenceClassification", ["Property"]),
                    _prop_node("age", ["Property"]),
                ]
            },
            "name": {
                "nodes": [
                    {
                        "provenanceId": "dc/base/HumanReadableStatVars",
                        "value": "Adult rural female population",
                    }
                ]
            },
        },
    }

    _obs = [{"earliestDate": "2001", "latestDate": "2021", "obsCount": 3, "importName": "test"}]
    obs[f"{_URBAN_SV}|{_ENTITY_DCID}"] = _obs
    obs[f"{_RURAL_SV}|{_ENTITY_DCID}"] = _obs
    if rural_adult_has_obs:
        obs[f"{_RURAL_ADULT_SV}|{_ENTITY_DCID}"] = _obs

    return FakeGraph(nodes=nodes, obs=obs)


class TestNarrowLeavesUntouchedAxisIntact:
    """A narrow preserves untouched bindings supported by surviving members."""

    def test_narrow_on_other_axis_leaves_untouched_bound_axis_intact(self):
        entities = [subject_entity(_ENTITY_DCID)]
        age_slot = _value_slot("how", "age", "age", "Years18Onwards")
        gender_slot = _value_slot("how", "gender", "gender", "Female")
        residence_set_slot = Slot(
            key=SlotKey(
                axis="how",
                property=_gref("placeOfResidenceClassification", "place of residence"),
                label="place of residence",
            ),
            binding=BindingSet(
                values=[
                    SlotValue(ref=_gref("Urban"), value_kind="enum_value"),
                    SlotValue(ref=_gref("Rural"), value_kind="enum_value"),
                ]
            ),
        )
        base_identity = make_identity(
            PERSON_COUNT_SHAPE,
            slots=[age_slot, gender_slot, residence_set_slot],
            entities=entities,
        )

        # Narrow residence to Rural and leave age and gender unchanged.
        residence_narrowed_slot = Slot(
            key=residence_set_slot.key,
            binding=BindingValue(value=SlotValue(ref=_gref("Rural"), value_kind="enum_value")),
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [age_slot, gender_slot, residence_narrowed_slot, *where_slots_for(entities)],
                entities,
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_narrow_bug_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_narrow_bug_graph()),
        )
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]

        # Only the Rural_Female_Years18Onwards survivor asserts the retained age binding.
        resolved_svs = [sv.ref.ref for sv in spec.resolved.members]
        assert resolved_svs == [_RURAL_ADULT_SV]

        age_result = next(
            s for s in spec.resolved.slots if s.key.property and s.key.property.ref == "age"
        )
        assert isinstance(age_result.binding, BindingValue)
        assert age_result.binding.value.ref is not None
        assert age_result.binding.value.ref.ref == "Years18Onwards"
        assert "age" not in spec.resolved.inapplicable_slots

        age_identity_slot = next(s for s in spec.definition.identity.slots if s.property == "age")
        base_age_identity_slot = next(s for s in base_identity.slots if s.property == "age")
        assert age_identity_slot == base_age_identity_slot


class TestMixedPresenceSurvivorProbeCannotServeANonAsserter:
    """Availability probes serve members that assert every untouched binding."""

    def test_asserting_sibling_lacks_observations_silent_one_is_still_not_served(self):
        entities = [subject_entity(_ENTITY_DCID)]
        age_slot = _value_slot("how", "age", "age", "Years18Onwards")
        gender_slot = _value_slot("how", "gender", "gender", "Female")
        residence_set_slot = Slot(
            key=SlotKey(
                axis="how",
                property=_gref("placeOfResidenceClassification", "place of residence"),
                label="place of residence",
            ),
            binding=BindingSet(
                values=[
                    SlotValue(ref=_gref("Urban"), value_kind="enum_value"),
                    SlotValue(ref=_gref("Rural"), value_kind="enum_value"),
                ]
            ),
        )
        residence_narrowed_slot = Slot(
            key=residence_set_slot.key,
            binding=BindingValue(value=SlotValue(ref=_gref("Rural"), value_kind="enum_value")),
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [age_slot, gender_slot, residence_narrowed_slot, *where_slots_for(entities)],
                entities,
            ),
        )
        # The sole age-asserting sibling lacks data. Age-silent Rural_Female has data.
        graph = _narrow_bug_graph(rural_adult_has_obs=False)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )

        if region.status == "definite":
            spec = region.specs[0]
            resolved_svs = [sv.ref.ref for sv in spec.resolved.members]
            # A definite response must serve members that support its age binding.
            assert _RURAL_SV not in resolved_svs
            age_result = next(
                s for s in spec.resolved.slots if s.key.property and s.key.property.ref == "age"
            )
            if isinstance(age_result.binding, BindingValue) and age_result.binding.value.ref:
                claimed_age = age_result.binding.value.ref.ref
                served_arcs = graph.node_arcs_batch(resolved_svs)
                asserted_ages = {
                    read_constraints(arcs).get("age")
                    for arcs in served_arcs.values()
                    if arcs is not None
                }
                # Every served member asserting age must match the response value.
                assert asserted_ages <= {claimed_age}
                assert asserted_ages, (
                    "age is claimed bound but no served StatVar asserts age at all"
                )
        else:
            # No confirmable age asserter means the response stays no_data.
            assert region.status == "no_data"


# Refine when no narrowed survivor asserts the untouched binding.

# Use the Years18Onwards DCID suffix. Sibling construction matches value tokens verbatim.
_RURAL_ADULT_STANDARD_SV = "Count_Person_Rural_Female_Years18Onwards"


def _zero_asserter_narrow_graph(*, include_sibling: bool) -> FakeGraph:
    """Build a zero-asserter narrow fixture, optionally adding the fully
    constrained Rural, Female, age=Years18Onwards sibling."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    nodes = json.loads((fixtures_dir / "graph_nodes.json").read_text())
    obs = json.loads((fixtures_dir / "graph_obs.json").read_text())

    for dcid in (_URBAN_SV, _RURAL_SV):
        if nodes[dcid].get("label") is None:
            name_nodes = nodes[dcid]["arcs"]["name"]["nodes"]
            nodes[dcid]["label"] = name_nodes[-1]["value"]

    nodes["Years18Onwards"] = {"label": "Years18Onwards", "type": None}
    nodes[_URBAN_SV]["arcs"]["age"] = {"nodes": [_prop_node("Years18Onwards", ["Class"])]}
    nodes[_URBAN_SV]["arcs"]["constraintProperties"]["nodes"].append(
        _prop_node("age", ["Property"])
    )

    _obs = [{"earliestDate": "2001", "latestDate": "2021", "obsCount": 3, "importName": "test"}]
    obs[f"{_URBAN_SV}|{_ENTITY_DCID}"] = _obs
    obs[f"{_RURAL_SV}|{_ENTITY_DCID}"] = _obs

    if include_sibling:
        nodes[_RURAL_ADULT_STANDARD_SV] = {
            "label": "Adult rural female population",
            "type": "StatisticalVariable",
            "arcs": {
                "statType": {"nodes": [_prop_node("measuredValue", ["Property"])]},
                "typeOf": {"nodes": [_prop_node("StatisticalVariable", ["Class"])]},
                "measuredProperty": {"nodes": [_prop_node("count", ["Property"])]},
                "populationType": {"nodes": [_prop_node("Person", ["Class"])]},
                "gender": {"nodes": [_prop_node("Female", ["GenderType"])]},
                "placeOfResidenceClassification": {
                    "nodes": [_prop_node("Rural", ["HumanSettlementClassificationEnum"])]
                },
                "age": {"nodes": [_prop_node("Years18Onwards", ["Class"])]},
                "constraintProperties": {
                    "nodes": [
                        _prop_node("gender", ["Property"]),
                        _prop_node("placeOfResidenceClassification", ["Property"]),
                        _prop_node("age", ["Property"]),
                    ]
                },
                "name": {
                    "nodes": [
                        {
                            "provenanceId": "dc/base/HumanReadableStatVars",
                            "value": "Adult rural female population",
                        }
                    ]
                },
            },
        }
        obs[f"{_RURAL_ADULT_STANDARD_SV}|{_ENTITY_DCID}"] = _obs

    return FakeGraph(nodes=nodes, obs=obs)


def _elect(inp: ResubmitIdentityPayload, graph: FakeGraph):
    """Resubmit `inp` with the mint's walk reading the person and GDP StatVars `graph`
    holds."""
    candidates = (
        _URBAN_SV,
        _RURAL_SV,
        _RURAL_ADULT_STANDARD_SV,
        "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
    )
    members = [dcid for dcid in candidates if graph.node_arcs(dcid) is not None]
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            graph, EngineOptions(materialise_graph=axis_confirming_reader(*members, graph=graph))
        ),
    )


class TestNarrowWithNoSurvivingAsserterRoutesThroughRefine:
    """A narrow with no surviving member that asserts an untouched bound
    property elects the edited identity's family, answering with its fully constrained
    member or no_data when the family holds none."""

    @staticmethod
    def _base(entities):
        age_slot = _value_slot("how", "age", "age", "Years18Onwards")
        gender_slot = _value_slot("how", "gender", "gender", "Female")
        residence_set_slot = Slot(
            key=SlotKey(
                axis="how",
                property=_gref("placeOfResidenceClassification", "place of residence"),
                label="place of residence",
            ),
            binding=BindingSet(
                values=[
                    SlotValue(ref=_gref("Urban"), value_kind="enum_value"),
                    SlotValue(ref=_gref("Rural"), value_kind="enum_value"),
                ]
            ),
        )
        return make_base(
            PERSON_COUNT_SHAPE,
            slots=[age_slot, gender_slot, residence_set_slot],
            entities=entities,
        )

    @staticmethod
    def _narrow_input(base_spec_id, base_identity, entities) -> ResubmitIdentityPayload:
        age_slot = _value_slot("how", "age", "age", "Years18Onwards")
        gender_slot = _value_slot("how", "gender", "gender", "Female")
        residence_narrowed_slot = Slot(
            key=SlotKey(
                axis="how",
                property=_gref("placeOfResidenceClassification", "place of residence"),
                label="place of residence",
            ),
            binding=BindingValue(value=SlotValue(ref=_gref("Rural"), value_kind="enum_value")),
        )
        return ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [age_slot, gender_slot, residence_narrowed_slot, *where_slots_for(entities)],
                entities,
            ),
        )

    def _no_confirmed_sibling_case(self, case: str) -> tuple[ResubmitIdentityPayload, FakeGraph]:
        if case == "survivor-seeded":
            entities = [subject_entity(_ENTITY_DCID)]
            base_spec_id, base_identity = self._base(entities)
            return (
                self._narrow_input(base_spec_id, base_identity, entities),
                _zero_asserter_narrow_graph(include_sibling=False),
            )

        if case == "edited-constraint":
            slot = _value_slot(
                "how",
                "activitySource",
                "activity source",
                "SomeOtherValue_NotInConstraints",
                "some label",
            )
        else:
            assert case == "added-constraint"
            slot = _value_slot(
                "how",
                "measurementMethod",
                "measurement method",
                "SomeMethod_NotOnSV",
                "some method",
            )

        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
        return (
            ResubmitIdentityPayload(
                identity=_edited_identity(GDP_SHAPE, [slot, *where_slots_for(entities)], entities),
            ),
            FakeGraph(),
        )

    def test_sibling_exists_resolves_to_the_fully_constrained_sibling(self):
        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = self._base(entities)
        inp = self._narrow_input(base_spec_id, base_identity, entities)
        region = _elect(inp, _zero_asserter_narrow_graph(include_sibling=True))
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]
        resolved_svs = [sv.ref.ref for sv in spec.resolved.members]
        assert resolved_svs == [_RURAL_ADULT_STANDARD_SV]

        age_result = next(
            s for s in spec.resolved.slots if s.key.property and s.key.property.ref == "age"
        )
        assert isinstance(age_result.binding, BindingValue)
        assert age_result.binding.value.ref is not None
        assert age_result.binding.value.ref.ref == "Years18Onwards"
        assert "age" not in spec.resolved.inapplicable_slots

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param("edited-constraint", id="edited-constraint"),
            pytest.param("added-constraint", id="added-constraint"),
            pytest.param("survivor-seeded", id="survivor-seeded"),
        ],
    )
    def test_no_confirmed_sibling_yields_no_matching_slice(self, case):
        """An edit whose family holds no member answers `no_matching_slice`."""
        inp, graph = self._no_confirmed_sibling_case(case)
        region = _elect(inp, graph)
        assert region.status == "no_data"
        assert region.no_data_reason == "no_matching_slice"

    def test_minted_spec_round_trips_unedited_without_degrading_to_no_data(self):
        """Echoing the minted narrow unchanged must preserve its id.

        Binding age to members with no age asserter would make the echo look edited.
        """
        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, base_identity = self._base(entities)
        inp = self._narrow_input(base_spec_id, base_identity, entities)
        graph = _zero_asserter_narrow_graph(include_sibling=True)
        region = _elect(inp, graph)
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]

        echo_identity = spec.definition.identity
        echo = ResubmitIdentityPayload(
            identity=echo_identity,
        )
        region2 = _elect(echo, graph)
        assert region2.status == "definite", region2.no_data_reason
        assert region2.specs[0].spec_id == spec.spec_id


# A sparse family can leave one value of an untouched set axis after narrowing.
# This tests binding reconstruction after the assertion gate passes.
_SPARSE_RURAL_MALE_SV = "Count_Person_Rural_Male"


def _sparse_cross_axis_graph() -> FakeGraph:
    """Build the sparse Urban/Female and Rural/Male fixture."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    nodes = json.loads((fixtures_dir / "graph_nodes.json").read_text())
    obs = json.loads((fixtures_dir / "graph_obs.json").read_text())

    if nodes[_URBAN_SV].get("label") is None:
        name_nodes = nodes[_URBAN_SV]["arcs"]["name"]["nodes"]
        nodes[_URBAN_SV]["label"] = name_nodes[-1]["value"]

    nodes[_SPARSE_RURAL_MALE_SV] = {
        "label": "Rural male population",
        "type": "StatisticalVariable",
        "arcs": {
            "statType": {"nodes": [_prop_node("measuredValue", ["Property"])]},
            "typeOf": {"nodes": [_prop_node("StatisticalVariable", ["Class"])]},
            "measuredProperty": {"nodes": [_prop_node("count", ["Property"])]},
            "populationType": {"nodes": [_prop_node("Person", ["Class"])]},
            "gender": {"nodes": [_prop_node("Male", ["GenderType"])]},
            "placeOfResidenceClassification": {
                "nodes": [_prop_node("Rural", ["HumanSettlementClassificationEnum"])]
            },
            "constraintProperties": {
                "nodes": [
                    _prop_node("gender", ["Property"]),
                    _prop_node("placeOfResidenceClassification", ["Property"]),
                ]
            },
            "name": {
                "nodes": [
                    {
                        "provenanceId": "dc/base/HumanReadableStatVars",
                        "value": "Rural male population",
                    }
                ]
            },
        },
    }

    _obs = [{"earliestDate": "2001", "latestDate": "2021", "obsCount": 3, "importName": "test"}]
    obs[f"{_URBAN_SV}|{_ENTITY_DCID}"] = _obs
    obs[f"{_SPARSE_RURAL_MALE_SV}|{_ENTITY_DCID}"] = _obs

    return FakeGraph(nodes=nodes, obs=obs)


class TestASetAxisTheFamilyAssertsInFullStands:
    """A set axis whose every value some family member asserts stands as posted, though
    only some members hold data."""

    def test_a_gender_set_the_family_asserts_in_full_stands_beside_a_narrowed_residence(self):
        entities = [subject_entity(_ENTITY_DCID)]
        residence_set_slot = Slot(
            key=SlotKey(
                axis="how",
                property=_gref("placeOfResidenceClassification", "place of residence"),
                label="place of residence",
            ),
            binding=BindingSet(
                values=[
                    SlotValue(ref=_gref("Urban"), value_kind="enum_value"),
                    SlotValue(ref=_gref("Rural"), value_kind="enum_value"),
                ]
            ),
        )
        gender_set_slot = Slot(
            key=SlotKey(axis="how", property=_gref("gender", "gender"), label="gender"),
            binding=BindingSet(
                values=[
                    SlotValue(ref=_gref("Female"), value_kind="enum_value"),
                    SlotValue(ref=_gref("Male"), value_kind="enum_value"),
                ]
            ),
        )

        # Narrow residence to Urban. Leave the full gender set unchanged.
        residence_narrowed_slot = Slot(
            key=residence_set_slot.key,
            binding=BindingValue(value=SlotValue(ref=_gref("Urban"), value_kind="enum_value")),
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [residence_narrowed_slot, gender_set_slot, *where_slots_for(entities)],
                entities,
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_sparse_cross_axis_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_sparse_cross_axis_graph()),
        )
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]

        resolved_svs = [sv.ref.ref for sv in spec.resolved.members]
        assert resolved_svs == [_URBAN_SV]

        # The family asserts both genders at Urban, the male count holding no data, so the
        # gender set stands as posted.
        gender_result = next(
            s for s in spec.resolved.slots if s.key.property and s.key.property.ref == "gender"
        )
        assert isinstance(gender_result.binding, BindingSet)
        assert {v.ref.ref for v in gender_result.binding.values if v.ref is not None} == {
            "Female",
            "Male",
        }
        assert spec.definition.identity == inp.identity


# Fresh-to-resubmit identity round trip.

_QUERY = "domestic government health spending in Kenya"

_PROP_FIN_SOURCE = "healthFinancingSource"
_VAL_DOMESTIC_GOVT = "DomesticGeneralGovernmentHealthFinancing"
_PROP_FUNCTION = "healthCareFunction"
_VAL_FUNCTION = "HC_6_1"
_PROP_AGE = "ageGroup"
_VAL_AGE_ASSERTED = "AgeGroup_18OrMore"

_SV_DOMESTIC_GOVT = "TEST/HealthSpend_DomesticGovt"
_SV_BY_FUNCTION = "TEST/HealthSpend_ByFunction"
_SV_BY_AGE = "TEST/HealthSpend_ByAge"

_KEN_DCID = "country/KEN"


def _sv_arcs(*, label: str, constraint_prop: str, constraint_value: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "HealthExpenditureEvent"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": constraint_prop}]},
        constraint_prop: {"nodes": [{"dcid": constraint_value}]},
    }


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _SV_DOMESTIC_GOVT: {
        "label": "Domestic general government health expenditure",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            label="Domestic general government health expenditure",
            constraint_prop=_PROP_FIN_SOURCE,
            constraint_value=_VAL_DOMESTIC_GOVT,
        ),
    },
    _SV_BY_FUNCTION: {
        "label": "Health expenditure by function",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            label="Health expenditure by function",
            constraint_prop=_PROP_FUNCTION,
            constraint_value=_VAL_FUNCTION,
        ),
    },
    _SV_BY_AGE: {
        "label": "Health expenditure by age group",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            label="Health expenditure by age group",
            constraint_prop=_PROP_AGE,
            constraint_value=_VAL_AGE_ASSERTED,
        ),
    },
    # A label keeps the selected source binding in value form.
    _VAL_DOMESTIC_GOVT: {"label": "Domestic general government health financing", "type": "Thing"},
}

_DETECT = {
    _QUERY: {
        "svs": [_SV_DOMESTIC_GOVT, _SV_BY_FUNCTION, _SV_BY_AGE],
        "cosine_scores": [0.9, 0.85, 0.8],
    },
}

# All three siblings have Kenya data, so shape discovery includes each constraint axis.
# The highest-scored source member wins the probe.
_OBS = {
    f"{_SV_DOMESTIC_GOVT}|{_KEN_DCID}": [
        {"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}
    ],
    f"{_SV_BY_FUNCTION}|{_KEN_DCID}": [
        {"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}
    ],
    f"{_SV_BY_AGE}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


class _SelectedAxisBindLLM:
    """Bind the source axis. Leave function and age absent for promotion."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_FIN_SOURCE,
                    kind="value",
                    value_dcids=[_VAL_DOMESTIC_GOVT],
                ),
                SlotBindingDraft(
                    axis="how", property_dcid=_PROP_FUNCTION, kind="absent", value_dcids=[]
                ),
                SlotBindingDraft(
                    axis="how", property_dcid=_PROP_AGE, kind="absent", value_dcids=[]
                ),
            ],
            also_plausible=[],
            ask=None,
        ), None


class TestFreshVsResubmitRoundTrip:
    """A fresh spec echoed through resubmit preserves its id and identity."""

    def test_fresh_spec_echoed_through_resubmit_reproduces_id_and_identity(self):
        graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs=_OBS)
        fresh_result = asyncio.run(
            resolve_variable(
                _QUERY,
                entities=["Kenya"],
                date_request=None,  # undated query
                detect_query=_QUERY,
                role_query=_QUERY,
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                llm=_SelectedAxisBindLLM(),
                base_timing={},
                coverage=empty_coverage_snapshot(),
            )
        )
        assert fresh_result.status == "definite"
        fresh_spec = fresh_result.specs[0]
        assert [sv.ref.ref for sv in fresh_spec.resolved.members] == [_SV_DOMESTIC_GOVT]

        base_identity = fresh_spec.definition.identity
        inp = ResubmitIdentityPayload(
            identity=base_identity,
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(
                graph,
                EngineOptions(
                    materialise_graph=axis_confirming_reader(
                        _SV_DOMESTIC_GOVT, _SV_BY_FUNCTION, _SV_BY_AGE, graph=graph
                    )
                ),
            ),
        )
        assert region.status == "definite", region.no_data_reason
        resubmitted_spec = region.specs[0]

        assert resubmitted_spec.spec_id == fresh_spec.spec_id
        assert resubmitted_spec.definition.identity == fresh_spec.definition.identity
