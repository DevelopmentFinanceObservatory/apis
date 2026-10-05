"""Test expansion and summability metadata on resolved specs, resubmits, and nearest-real
results."""

from __future__ import annotations

import json

import pytest
from dfo_spec_kit import identity as spec_identity
from fastapi.testclient import TestClient

import search.engine.region.promote_reread as promote_reread_module
import search.engine.retrieval.bound_values as bound_values_module
from code_index import CodeIndex, DatasetDeclaration
from code_index.query import row_five_tuple
from code_index.shared import get_index
from code_index.types import StatVarRow
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.region.grounding import _suggest_nearest_real
from search.engine.retrieval.bound_values import (
    apply_declaration_version,
    apply_slot_expansions,
    slot_expansions,
)
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import build_spec
from search.models import (
    BaseIdentity,
    BindingUnbound,
    CoverageBare,
    IdentityBindingUnbound,
    IdentityBindingValue,
    IdentitySlot,
    IdentityWindowAll,
    PipelineStep,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    StatVar,
)
from series_coverage.schema import RoleNames
from tests.code_index.aid_identity import education_aid_identity
from tests.engine._builders import bound_place, gref, shape_five_tuple, unbound_slot, value_slot
from tests.engine._fake_coverage import (
    FIXTURE_PROVENANCE,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_HEALTH = "CRSPurposeCode120"  # hierarchical (crsPurpose), real packaged index
_MALARIA_CONTROL = "CRSPurposeCode12262"  # a leaf under _HEALTH
_FLOW_GRANT = "CRSTypeOfFlow10"  # crsTypeOfFlow is flat


def _binding(prop: str, values: list[str], *, kind: str = "value") -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid=prop, kind=kind, value_dcids=values, evidence="mentioned"
    )


# Shared expansion and declaration-version helpers.


def test_apply_slot_expansions_patches_the_matching_slot():
    slot = value_slot("how", "crsPurpose", "crsPurpose", _HEALTH)
    expansions = slot_expansions([_binding("crsPurpose", [_HEALTH])])
    (patched,) = apply_slot_expansions([slot], expansions)
    assert patched.expansion == expansions["crsPurpose"]
    assert _MALARIA_CONTROL in {ref.dcid for ref in patched.expansion.added}


def test_apply_slot_expansions_is_a_no_op_when_expansions_is_empty():
    """An empty expansion map returns the original list object."""
    slots = [value_slot("how", "crsTypeOfFlow", "crsTypeOfFlow", _FLOW_GRANT)]
    result = apply_slot_expansions(slots, {})
    assert result is slots


def test_apply_slot_expansions_leaves_a_non_matching_slot_untouched():
    slot = value_slot("how", "crsTypeOfFlow", "crsTypeOfFlow", _FLOW_GRANT)
    expansions = slot_expansions([_binding("crsPurpose", [_HEALTH])])
    (result,) = apply_slot_expansions([slot], expansions)
    assert result.expansion is None
    assert result is slot  # The helper returns the original Slot object.


def test_apply_slot_expansions_leaves_a_property_less_slot_untouched():
    """Property-less slots remain unchanged."""
    slot = Slot(key=SlotKey(axis="where", property=None, label="where"), binding=BindingUnbound())
    expansions = slot_expansions([_binding("crsPurpose", [_HEALTH])])
    (result,) = apply_slot_expansions([slot], expansions)
    assert result is slot


def test_apply_declaration_version_stamps_the_trace():
    from tests.engine._builders import conjoin_spec, shape_five_tuple

    shape = shape_five_tuple("EconomicActivity", "amount")
    spec = conjoin_spec(shape, "x", sv_dcid="test_sv")
    assert (
        spec.resolved.trace.declaration_version is None
    )  # Initial trace has no declaration version.

    stamped = apply_declaration_version(spec, [_binding("crsPurpose", [_HEALTH])])
    assert stamped.resolved.trace.declaration_version == get_index().stamp.content_hash


def test_apply_declaration_version_is_a_no_op_when_nothing_of_this_kind_happened():
    """A resolution with no expansion has no declaration version."""
    from tests.engine._builders import conjoin_spec, shape_five_tuple

    shape = shape_five_tuple("EconomicActivity", "amount")
    spec = conjoin_spec(shape, "x", sv_dcid="test_sv")

    result = apply_declaration_version(spec, [_binding("gender", ["Female"])])
    assert result is spec
    assert result.resolved.trace.declaration_version is None


def test_apply_declaration_version_stamps_a_plain_spec_the_index_resolves():
    """An identity that binds no code-system property still takes its members from the
    index, where the dataset declarations elect its family."""
    declared = get_index().declarations
    slots = [
        unbound_slot("how", slot["property"], slot["property"])
        if slot["property"] in declared or slot["binding"]["state"] == "unbound"
        else value_slot("how", slot["property"], slot["property"], slot["binding"]["ref"])
        for slot in education_aid_identity()["slots"]
    ]
    shape = shape_five_tuple("FinancialFlow", "amount", "Nominal")
    spec = build_spec(
        shape=shape,
        slots=slots,
        stat_vars=[
            StatVar(
                ref=gref("test_sv"),
                shape_id=shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        pipeline_trace=[PipelineStep(step="extract", ran=True)],
        measure_elected_properties=frozenset(),
    )
    assert spec.resolved.trace.declaration_version is None

    stamped = apply_declaration_version(spec, [])

    assert stamped.resolved.trace.declaration_version == get_index().stamp.content_hash


# Check expansion fields on the resubmit response.

_CRS_SV = "TESTFIXTURE/L5c/FinancingFlow"
_DESTINATION = "country/KEN"


def _crs_arcs(*, crs_purpose: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "crsPurpose"}]},
        "crsPurpose": {"nodes": [{"dcid": crs_purpose}]},
        "observationProperties": {
            "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
        },
    }


def _crs_graph(*, crs_purpose: str) -> FakeGraph:
    return FakeGraph(
        nodes={
            _CRS_SV: {"label": "Test financing flow", "arcs": _crs_arcs(crs_purpose=crs_purpose)},
            _DESTINATION: {"label": "Kenya", "type": "Country"},
        },
        obs={},
        detect={},
        resolve={},
    )


def _crs_sdmx() -> FakeSdmxClient:
    recipient_tuple = (("financingRecipient", _DESTINATION),)
    facet = Facet(
        earliest_date="2015",
        latest_date="2020",
        obs_count=5,
        dates=["2020"],
        values=[1.0],
        facet_id="f1",
    )
    return FakeSdmxClient(observations={_CRS_SV: {recipient_tuple: [facet]}}, availability={})


def _crs_identity(*, crs_purpose_binding: IdentityBindingValue | IdentityBindingUnbound):
    return BaseIdentity(
        shape={
            "kind": "statvar_schema",
            "population_type": "EconomicActivity",
            "measured_property": "amount",
            "stat_type": "measuredValue",
        },
        slots=[IdentitySlot(property="crsPurpose", binding=crs_purpose_binding)],
        entities={"destination": [_DESTINATION]},
        requested_window=IdentityWindowAll(),
    )


def _resolve_response(monkeypatch, *, member_crs_purpose: str, posted_identity: BaseIdentity):
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_CRS_SV, "f1", "country/FRA", _DESTINATION, [2015, 2016, 2017, 2018, 2019])],
        role_names={_CRS_SV: RoleNames(donor="financingProvider", recipient="financingRecipient")},
    )
    graph = _crs_graph(crs_purpose=member_crs_purpose)
    sdmx = _crs_sdmx()
    client = TestClient(
        search_app(graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=sdmx),
        raise_server_exceptions=False,
    )
    # ResubmitIdentityPayload requires a base identity.
    resubmit_input = ResubmitIdentityPayload(
        identity=posted_identity,
        base_spec_id=spec_identity.spec_id_from_value(
            posted_identity.model_dump(exclude_none=True)
        ),
        base_identity=posted_identity,
        resolved_stat_var_dcids=[_CRS_SV],
    )
    return client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "CRS purpose members",
                        **resubmit_input.model_dump(mode="json", exclude_none=True),
                    }
                ],
            }
        },
    )


class TestResubmitDoorWiresBothFields:
    def test_a_hierarchical_binding_carries_expansion_and_declaration_version_on_the_wire(
        self, monkeypatch
    ):
        """The Health parent binding expands to descendants on the wire.

        The member also asserts Health so resubmit seed validation accepts it.
        """
        resp = _resolve_response(
            monkeypatch,
            member_crs_purpose=_HEALTH,
            posted_identity=_crs_identity(crs_purpose_binding=IdentityBindingValue(ref=_HEALTH)),
        )
        assert resp.status_code == 200, resp.text
        part = resp.json()["parts"][0]
        assert part["status"] == "definite"
        spec = part["spec"]

        crs_slot = next(
            s
            for s in spec["resolved"]["slots"]
            if s["key"].get("property") is not None and s["key"]["property"]["dcid"] == "crsPurpose"
        )
        added_dcids = {ref["dcid"] for ref in crs_slot["expansion"]["added"]}
        assert _MALARIA_CONTROL in added_dcids
        assert _HEALTH not in added_dcids  # Expansion metadata lists descendants only.

        assert spec["resolved"]["trace"]["declaration_version"] == get_index().stamp.content_hash

    def test_an_unbound_crs_slot_carries_neither_field(self, monkeypatch):
        """An unbound CRS slot has no expansion or declaration version on the wire."""
        resp = _resolve_response(
            monkeypatch,
            member_crs_purpose=_MALARIA_CONTROL,
            posted_identity=_crs_identity(crs_purpose_binding=IdentityBindingUnbound()),
        )
        assert resp.status_code == 200, resp.text
        part = resp.json()["parts"][0]
        assert part["status"] == "definite"
        spec = part["spec"]

        for slot in spec["resolved"]["slots"]:
            assert slot.get("expansion") is None
        assert spec["resolved"]["trace"].get("declaration_version") is None


# Nearest-real results include metadata for bindings that remain bound.

_NR_POP_TYPE = "TestNRExpandPopType"
_NR_MEAS_PROP = "testNRExpandMeasure"
_NR_STAT_TYPE = "measuredValue"
_NR_PROP_B = "TestNRExpandPropB"
_NR_VAL_B1 = "NRExpandB1"
_NR_VAL_B_OTHER = "NRExpandBOther"
_NR_UNRELATED_PURPOSE = "TestUnrelatedPurpose"  # No index entry exists for this value.
_NR_SV = "TEST/NRExpandSV"
_NR_RECIPIENT = "country/NREXP"


def _nr_sv_arcs(*, crs_purpose: str, prop_b: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "NR Expand SV"}]},
        "populationType": {"nodes": [{"dcid": _NR_POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _NR_MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _NR_STAT_TYPE}]},
        "constraintProperties": {"nodes": [{"dcid": "crsPurpose"}, {"dcid": _NR_PROP_B}]},
        "crsPurpose": {"nodes": [{"dcid": crs_purpose}]},
        _NR_PROP_B: {"nodes": [{"dcid": prop_b}]},
    }


def _nr_shape(*, crs_purpose: str, prop_b: str) -> ShapeDraft:
    sv_arc_facts = {_NR_SV: _nr_sv_arcs(crs_purpose=crs_purpose, prop_b=prop_b)}
    return ShapeDraft(
        shape_id="test_nr_expand_shape",
        label="test nr expand shape",
        pop_type_dcid=_NR_POP_TYPE,
        meas_prop_dcid=_NR_MEAS_PROP,
        stat_type_dcid=_NR_STAT_TYPE,
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="how", property_dcid="crsPurpose", label="CRS Purpose"),
            SlotKeyDraft(axis="what", property_dcid=_NR_PROP_B, label="Prop B"),
        ),
        sv_arc_facts=sv_arc_facts,
        slot_taxonomy={
            "how:crsPurpose": [_HEALTH, _NR_UNRELATED_PURPOSE],
            f"what:{_NR_PROP_B}": [_NR_VAL_B1, _NR_VAL_B_OTHER],
        },
        representative_score=0.95,
    )


def _nr_graph(*, crs_purpose: str, prop_b: str) -> FakeGraph:
    return FakeGraph(
        nodes={
            _NR_RECIPIENT: {"label": "NR Expand Land", "type": "Country"},
            _NR_SV: {
                "label": "NR Expand SV",
                "type": "StatisticalVariable",
                "arcs": _nr_sv_arcs(crs_purpose=crs_purpose, prop_b=prop_b),
            },
        },
        obs={
            f"{_NR_SV}|{_NR_RECIPIENT}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ]
        },
        detect={},
        resolve={},
    )


def _nr_prop_b_binding(values: list[str]) -> SlotBindingDraft:
    return SlotBindingDraft(axis="what", property_dcid=_NR_PROP_B, kind="value", value_dcids=values)


def _nr_crs_purpose_binding(values: list[str]) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid="crsPurpose", kind="value", value_dcids=values
    )


class TestNearestRealWiresBothFields:
    def test_relaxing_a_different_binding_still_carries_the_untouched_crs_slots_expansion(self):
        """Relaxing prop_b leaves Health bound, so expansion and version metadata remain."""
        bindings = [_nr_prop_b_binding([_NR_VAL_B1]), _nr_crs_purpose_binding([_HEALTH])]
        suggestions, warnings, _reads, _presence = _suggest_nearest_real(
            graph=_nr_graph(crs_purpose=_HEALTH, prop_b=_NR_VAL_B_OTHER),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            shape_draft=_nr_shape(crs_purpose=_HEALTH, prop_b=_NR_VAL_B_OTHER),
            bindings=bindings,
            place=bound_place(
                roles={_NR_RECIPIENT: "observationAbout"},
                entities={"observationAbout": [_NR_RECIPIENT]},
                destination_dcid=_NR_RECIPIENT,
            ),
            presence=None,
            date_request=None,
            n_max=1,
            coverage=empty_coverage_snapshot(),
        )

        assert warnings == []
        assert len(suggestions) == 1
        assert suggestions[0].relaxation.property_dcid == _NR_PROP_B  # prop_b is what relaxed
        resolved = suggestions[0].resolved

        crs_slot = next(
            s
            for s in resolved.slots
            if s.key.property is not None and s.key.property.dcid == "crsPurpose"
        )
        assert crs_slot.expansion is not None
        assert _MALARIA_CONTROL in {ref.dcid for ref in crs_slot.expansion.added}
        assert resolved.trace.declaration_version == get_index().stamp.content_hash

    def test_relaxing_the_crs_binding_itself_carries_neither_field(self):
        """Relaxing crsPurpose removes its expansion and declaration-version metadata."""
        bindings = [
            _nr_crs_purpose_binding([_HEALTH]),
            _nr_prop_b_binding([_NR_VAL_B1]),
        ]
        suggestions, warnings, _reads, _presence = _suggest_nearest_real(
            graph=_nr_graph(crs_purpose=_NR_UNRELATED_PURPOSE, prop_b=_NR_VAL_B1),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            shape_draft=_nr_shape(crs_purpose=_NR_UNRELATED_PURPOSE, prop_b=_NR_VAL_B1),
            bindings=bindings,
            place=bound_place(
                roles={_NR_RECIPIENT: "observationAbout"},
                entities={"observationAbout": [_NR_RECIPIENT]},
                destination_dcid=_NR_RECIPIENT,
            ),
            presence=None,
            date_request=None,
            n_max=1,
            coverage=empty_coverage_snapshot(),
        )

        assert warnings == []
        assert len(suggestions) == 1
        assert suggestions[0].relaxation.property_dcid == "crsPurpose"
        resolved = suggestions[0].resolved

        for slot in resolved.slots:
            assert slot.expansion is None
        assert resolved.trace.declaration_version is None


# The dataset the coverage fixture records the member's facets under.
_TEST_PROVENANCE = FIXTURE_PROVENANCE


def _index_carrying_the_fixture_family(*, member_crs_purpose: str) -> CodeIndex:
    """The packaged index plus the fixture StatVar, in a dataset that reads absence as zero.

    The fixture family is one row, so its five-tuple would pin `crsPurpose` and the
    canonical identity would unbind the bound slot (`canonical_identity`). The pin is
    dropped, since a real CRS family holds many purposes."""
    index = get_index()
    row = StatVarRow(
        dcid=_CRS_SV,
        population_type="EconomicActivity",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties={"crsPurpose": member_crs_purpose},
        provenances=(_TEST_PROVENANCE,),
    )
    edited = CodeIndex(
        stamp=index.stamp,
        declarations=tuple(index.declarations.values()),
        codes=index.codes,
        statvars=(*index.statvars, row),
        datasets=(
            *index.datasets.values(),
            DatasetDeclaration(
                provenance=_TEST_PROVENANCE,
                label="Test financing flows",
                synonyms=(),
                opt_in=False,
                absent_observation="zero",
                rank=1,
            ),
        ),
    )
    del edited.dataset_pins[_TEST_PROVENANCE, row_five_tuple(row)]
    return edited


class TestResubmitDoorWiresSummable:
    """The resubmit door serves `summable` on a CRS slot and leaves spec identity alone."""

    @pytest.mark.parametrize(
        ("binding", "member_crs_purpose", "recorded_spec_id"),
        [
            (IdentityBindingValue(ref=_HEALTH), _HEALTH, "spec_bb3b6cb4582ebab1672d76ad80e2ab52"),
            (IdentityBindingUnbound(), _MALARIA_CONTROL, "spec_1059c7ee5ad0e0a209a4a025d1413cbf"),
        ],
        ids=["value", "unbound"],
    )
    def test_a_crs_purpose_slot_reads_summable_and_the_spec_id_is_unchanged(
        self, monkeypatch, binding, member_crs_purpose, recorded_spec_id
    ):
        edited = _index_carrying_the_fixture_family(member_crs_purpose=member_crs_purpose)
        # The identity names the fixture dataset's family, which the resubmit elects again
        # when it re-reads the members.
        for module in (bound_values_module, promote_reread_module):
            monkeypatch.setattr(module, "get_index", lambda: edited)
        resp = _resolve_response(
            monkeypatch,
            member_crs_purpose=member_crs_purpose,
            posted_identity=_crs_identity(crs_purpose_binding=binding),
        )
        assert resp.status_code == 200, resp.text
        spec = resp.json()["parts"][0]["spec"]

        crs_slot = next(
            s
            for s in spec["resolved"]["slots"]
            if s["key"].get("property") is not None and s["key"]["property"]["dcid"] == "crsPurpose"
        )
        assert crs_slot["summable"] is True
        assert spec["spec_id"] == recorded_spec_id
        assert "summable" not in json.dumps(spec["definition"])

    def test_a_slot_of_a_family_the_index_does_not_resolve_reads_not_summable(self, monkeypatch):
        resp = _resolve_response(
            monkeypatch,
            member_crs_purpose=_HEALTH,
            posted_identity=_crs_identity(crs_purpose_binding=IdentityBindingValue(ref=_HEALTH)),
        )
        assert resp.status_code == 200, resp.text
        spec = resp.json()["parts"][0]["spec"]

        assert [slot["summable"] for slot in spec["resolved"]["slots"]] == [False] * len(
            spec["resolved"]["slots"]
        )
