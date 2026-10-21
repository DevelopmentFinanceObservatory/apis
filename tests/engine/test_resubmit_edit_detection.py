"""A named family's elected members follow the posted identity. These tests use
real fixture-index members for the CRS health-purpose family to pin no-op,
window, destination, and rollup/leaf edits without a posted seed.
"""

from __future__ import annotations

from code_index.shared import get_index, get_measure_index
from search.engine.clients.graph import Facet
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.retrieval.bound_values import measure_family_partitions
from search.engine.shape import ShapeDraft, identity_shape_for
from search.models import ResubmitIdentityPayload, TimeWindow
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import destination_entity, make_base
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._index_slots import query_for
from tests.engine.test_candidates_measure_alternatives import _ROLE_NAMES, _ROLE_RECIPIENT
from tests.engine.test_resubmit_alternative_measure_round_trip import _arcs_for_row
from tests.fixtures import FakeGraph

_KEN = "country/KEN"
_ETH = "country/ETH"

_PURPOSE_ROLLUP = "CRSPurposeCode120"  # "health", a real hierarchical parent
_PURPOSE_LEAF = "CRSPurposeCode12220"  # one of its real leaf descendants

_HEALTH_SHAPE = identity_shape_for(
    pop_type_dcid="FinancialFlow",
    meas_prop_dcid="amount",
    stat_type_dcid="measuredValue",
    meas_qual_dcid="Nominal",
    meas_denom_dcid=None,
)

_MEASURE_VALUES = {
    "populationType": "FinancialFlow",
    "measuredProperty": "amount",
    "measurementQualifier": "Nominal",
    "financialFlowType": "Disbursement",
    "nettingBasis": "Gross",
    "financialObligationType": None,
}


def _synthetic_shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="",
        label="",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(),
    )


def _elected_members(purpose: str) -> tuple[str, ...]:
    """Return the fixture-index members for a health-family identity."""
    index = get_index()
    axes = get_measure_index().axes
    bound_values = {
        "crsPurpose": list(index.descendants(purpose)),
        "financialFlowType": ["Disbursement"],
        "nettingBasis": ["Gross"],
    }
    partitions = measure_family_partitions(
        _synthetic_shape(), index, axes, query_for(_synthetic_shape(), bound_values)
    )
    match = next(p for p in partitions if dict(p.values) == _MEASURE_VALUES)
    return match.members


def _member_rows(dcids) -> list:
    index = get_index()
    by_dcid = {row.dcid: row for row in index.statvars}
    rows = [by_dcid[dcid] for dcid in dcids]
    assert rows, "fixture drift: expected real rows for these dcids"
    return rows


def _health_graph(member_rows, *, places: list[str]) -> FakeGraph:
    """Build member nodes and the named purpose rollup needed for grounding."""
    rollup_label = get_index().code_by_dcid(_PURPOSE_ROLLUP).label
    nodes: dict[str, dict] = {_PURPOSE_ROLLUP: {"label": rollup_label}}
    for place in places:
        nodes[place] = {"label": place, "type": "Country"}
    for row in member_rows:
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": _arcs_for_row(row),
        }
    return FakeGraph(nodes=nodes, obs={})


def _sdmx_for(dcids, place_dcid: str) -> FakeSdmxClient:
    observations = {
        sv: {
            ((_ROLE_RECIPIENT, place_dcid),): [
                Facet(
                    earliest_date="2020",
                    latest_date="2020",
                    obs_count=3,
                    dates=["2020"],
                    values=[0.0],
                    facet_id="f1",
                )
            ]
        }
        for sv in dcids
    }
    return FakeSdmxClient(observations=observations, availability={})


def _coverage_for(dcids, place_dcid: str, monkeypatch):
    """Build coverage rows for each dcid at the requested recipient."""
    rows = [(sv, "f1", "country/USA", place_dcid, [2020]) for sv in dcids]
    index = patch_coverage(monkeypatch, rows, role_names=dict.fromkeys(dcids, _ROLE_NAMES))
    return FakeCoverageService(index).snapshot()


def _health_slots(*, purpose: str) -> list:
    return [
        _unbound_slot("how", "crsCooperationModality", "crsCooperationModality"),
        _value_slot("how", "crsPurpose", "crsPurpose", purpose, purpose),
        _unbound_slot("how", "crsTypeOfFinance", "crsTypeOfFinance"),
        _unbound_slot("how", "crsTypeOfFlow", "crsTypeOfFlow"),
        _value_slot(
            "how", "financialFlowType", "financialFlowType", "Disbursement", "Disbursement"
        ),
        _value_slot("how", "nettingBasis", "nettingBasis", "Gross", "Gross"),
    ]


def _health_base(*, purpose: str, place: str = _KEN, window: TimeWindow | None = None):
    return make_base(
        _HEALTH_SHAPE,
        _health_slots(purpose=purpose),
        entities=[destination_entity(place)],
        requested_window=window or TimeWindow(start_year=2015),
    )


class TestNoOpAndCoveredEdits:
    """The production defect itself: a byte-for-byte no-op resubmit posting
    `crsPurpose=CRSPurposeCode120` (the rollup), with no seed at all --
    membership comes entirely from the fixture code index's own election."""

    def test_noop_resubmit_of_the_rollup_bound_identity_resolves_definite(self, monkeypatch):
        _base_spec_id, base_identity = _health_base(purpose=_PURPOSE_ROLLUP)
        member_dcids = _elected_members(_PURPOSE_ROLLUP)
        member_rows = _member_rows(member_dcids)
        graph = _health_graph(member_rows, places=[_KEN])
        sdmx = _sdmx_for(member_dcids, _KEN)
        materialise_graph = axis_confirming_reader(*member_dcids, graph=graph)

        inp = ResubmitIdentityPayload(identity=base_identity)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=sdmx,
            coverage=_coverage_for(member_dcids, _KEN, monkeypatch),
            options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
        )

        assert region.status == "definite", (
            region.no_data_reason if region.status != "definite" else None
        )
        purpose_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.ref == "crsPurpose"
        )
        assert purpose_slot.binding.kind == "value"
        assert purpose_slot.binding.value.ref is not None
        assert purpose_slot.binding.value.ref.ref == _PURPOSE_ROLLUP

    def test_window_only_edit_over_the_same_rollup_binding_resolves(self, monkeypatch):
        """Narrowing the requested window alone, with `crsPurpose` re-posted
        unedited (still the rollup), must not be blocked by a false
        crsPurpose edit."""
        _base_spec_id, _base_identity = _health_base(
            purpose=_PURPOSE_ROLLUP, window=TimeWindow(start_year=2015)
        )
        edited_identity = make_base(
            _HEALTH_SHAPE,
            _health_slots(purpose=_PURPOSE_ROLLUP),
            entities=[destination_entity(_KEN)],
            requested_window=TimeWindow(start_year=2016),
        )[1]
        member_dcids = _elected_members(_PURPOSE_ROLLUP)
        member_rows = _member_rows(member_dcids)
        graph = _health_graph(member_rows, places=[_KEN])
        sdmx = _sdmx_for(member_dcids, _KEN)
        materialise_graph = axis_confirming_reader(*member_dcids, graph=graph)

        inp = ResubmitIdentityPayload(identity=edited_identity)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=sdmx,
            coverage=_coverage_for(member_dcids, _KEN, monkeypatch),
            options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
        )

        assert region.status == "definite", (
            region.no_data_reason if region.status != "definite" else None
        )

    def test_destination_edit_over_the_same_rollup_binding_resolves_bound_to_the_new_place(
        self, monkeypatch
    ):
        """Editing the destination alone (KEN -> ETH), with `crsPurpose`
        re-posted unedited (still the rollup), must not be blocked by a
        false crsPurpose edit, and must resolve bound to ETH."""
        _base_spec_id, _base_identity = _health_base(purpose=_PURPOSE_ROLLUP, place=_KEN)
        edited_identity = make_base(
            _HEALTH_SHAPE,
            _health_slots(purpose=_PURPOSE_ROLLUP),
            entities=[destination_entity(_ETH)],
            requested_window=TimeWindow(start_year=2015),
        )[1]
        member_dcids = _elected_members(_PURPOSE_ROLLUP)
        member_rows = _member_rows(member_dcids)
        graph = _health_graph(member_rows, places=[_KEN, _ETH])
        sdmx = _sdmx_for(member_dcids, _ETH)
        materialise_graph = axis_confirming_reader(*member_dcids, graph=graph)

        inp = ResubmitIdentityPayload(identity=edited_identity)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=sdmx,
            coverage=_coverage_for(member_dcids, _ETH, monkeypatch),
            options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
        )

        assert region.status == "definite", (
            region.no_data_reason if region.status != "definite" else None
        )
        destination_refs = [
            e.ref for e in region.specs[0].resolved.entities if e.role == "destination"
        ]
        assert destination_refs == [_ETH]


class TestRollupLeafEditDetectionBothDirections:
    """Edits elect membership from the posted identity: rollup-to-leaf narrows
    the family, and leaf-to-rollup widens it.
    """

    def test_rollup_to_leaf_edit_is_detected_and_narrows_to_the_leaf_members(self, monkeypatch):
        """Changing the purpose from its rollup to a leaf elects that leaf's
        smaller member set."""
        _base_spec_id, _base_identity = _health_base(purpose=_PURPOSE_ROLLUP)
        edited_identity = make_base(
            _HEALTH_SHAPE,
            _health_slots(purpose=_PURPOSE_LEAF),
            entities=[destination_entity(_KEN)],
            requested_window=TimeWindow(start_year=2015),
        )[1]
        member_dcids = _elected_members(_PURPOSE_LEAF)
        member_rows = _member_rows(member_dcids)
        graph = _health_graph(member_rows, places=[_KEN])
        sdmx = _sdmx_for(member_dcids, _KEN)
        materialise_graph = axis_confirming_reader(*member_dcids, graph=graph)

        inp = ResubmitIdentityPayload(identity=edited_identity)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=sdmx,
            coverage=_coverage_for(member_dcids, _KEN, monkeypatch),
            options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
        )

        assert region.status == "definite", (
            region.no_data_reason if region.status != "definite" else None
        )
        assert {sv.ref.ref for sv in region.specs[0].resolved.members} == set(member_dcids)
        purpose_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.ref == "crsPurpose"
        )
        assert purpose_slot.binding.kind == "value"
        assert purpose_slot.binding.value.ref is not None
        assert purpose_slot.binding.value.ref.ref == _PURPOSE_LEAF

    def test_leaf_to_rollup_edit_is_detected_and_widens_to_the_rollup_members(self, monkeypatch):
        """Changing a leaf to its rollup elects the rollup's full member set,
        then resolves with the rollup binding."""
        _base_spec_id, _base_identity = _health_base(purpose=_PURPOSE_LEAF)
        edited_identity = make_base(
            _HEALTH_SHAPE,
            _health_slots(purpose=_PURPOSE_ROLLUP),
            entities=[destination_entity(_KEN)],
            requested_window=TimeWindow(start_year=2015),
        )[1]
        member_dcids = _elected_members(_PURPOSE_ROLLUP)
        member_rows = _member_rows(member_dcids)
        graph = _health_graph(member_rows, places=[_KEN])
        sdmx = _sdmx_for(member_dcids, _KEN)
        materialise_graph = axis_confirming_reader(*member_dcids, graph=graph)

        inp = ResubmitIdentityPayload(identity=edited_identity)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=sdmx,
            coverage=_coverage_for(member_dcids, _KEN, monkeypatch),
            options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
        )

        assert region.status == "definite", (
            region.no_data_reason if region.status != "definite" else None
        )
        assert {sv.ref.ref for sv in region.specs[0].resolved.members} <= set(member_dcids)
        assert region.specs[0].resolved.stat_vars.member_count == len(member_dcids)
        purpose_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.ref == "crsPurpose"
        )
        assert purpose_slot.binding.kind == "value"
        assert purpose_slot.binding.value.ref is not None
        assert purpose_slot.binding.value.ref.ref == _PURPOSE_ROLLUP
