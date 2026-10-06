"""Issue #968: resubmits preserve posted unbound and set bindings.

The fixture uses two real flow members with ODA and private-finance flow codes.
"""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.clients.graph import Facet
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.shape import identity_shape_for
from search.models import (
    BindingSet,
    BindingUnbound,
    BindingValue,
    GraphRef,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from series_coverage.schema import RoleNames
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import destination_entity, make_identity
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine.test_candidates_measure_alternatives import _ROLE_RECIPIENT
from tests.engine.test_resubmit_alternative_measure_round_trip import _arcs_for_row
from tests.fixtures import FakeGraph

_KEN = "country/KEN"
_CRS_PURPOSE = "CRSPurposeCode12110"
_CRS_MODALITY = "CRSCooperationModalityA02"
_CRS_FINANCE = "CRSTypeOfFinance110"
_ODA_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode12110_Nominal"
)
_PRIVATE_DEV_FINANCE_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow30_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode12110_Nominal"
)


def _shape() -> dict:
    return identity_shape_for(
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )


def _bound_slot(prop: str, value: str) -> Slot:
    return Slot(
        key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(dcid=value, label=value), value_kind="enum_value")
        ),
    )


def _unbound_slot(prop: str) -> Slot:
    return Slot(
        key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
        binding=BindingUnbound(),
    )


def _set_slot(prop: str, *values: str) -> Slot:
    return Slot(
        key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
        binding=BindingSet(
            values=[
                SlotValue(ref=GraphRef(dcid=v, label=v), value_kind="enum_value") for v in values
            ]
        ),
    )


def _member_rows():
    index = get_index()
    rows = [row for row in index.statvars if row.dcid in (_ODA_SV, _PRIVATE_DEV_FINANCE_SV)]
    assert len(rows) == 2, "fixture drift: expected both real Amount/CRSPurposeCode12110 members"
    return rows


def _graph() -> FakeGraph:
    nodes = {_KEN: {"label": "Kenya", "type": "Country"}}
    for row in _member_rows():
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": _arcs_for_row(row),
        }
    return FakeGraph(nodes=nodes, obs={})


def _sdmx():
    observations = {
        dcid: {
            ((_ROLE_RECIPIENT, _KEN),): [
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
        for dcid in (_ODA_SV, _PRIVATE_DEV_FINANCE_SV)
    }
    return FakeSdmxClient(observations=observations, availability={})


def _filter_for(spec, prop: str):
    return next(
        f
        for f in spec.resolved.trace.slot_filters
        if f.key.property is not None and f.key.property.dcid == prop
    )


def _base_slots(flow_slot: Slot) -> list[Slot]:
    return [
        _bound_slot("crsCooperationModality", _CRS_MODALITY),
        _bound_slot("crsPurpose", _CRS_PURPOSE),
        _bound_slot("crsTypeOfFinance", _CRS_FINANCE),
        flow_slot,
        _bound_slot("financialFlowType", "Disbursement"),
        _bound_slot("nettingBasis", "Gross"),
    ]


def _resolve(slots: list[Slot], *, monkeypatch):
    # Facet confirmation uses coverage rows for both SDMX observations.
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(dcid, "f1", "country/USA", _KEN, [2020]) for dcid in (_ODA_SV, _PRIVATE_DEV_FINANCE_SV)],
        role_names={
            dcid: RoleNames(donor="financingProvider", recipient=_ROLE_RECIPIENT)
            for dcid in (_ODA_SV, _PRIVATE_DEV_FINANCE_SV)
        },
    )
    identity = make_identity(_shape(), slots, entities=[destination_entity(_KEN)])
    graph = _graph()
    sdmx = _sdmx()
    seed_dcids = [_ODA_SV, _PRIVATE_DEV_FINANCE_SV]
    materialise_graph = axis_confirming_reader(*seed_dcids, graph=graph)
    inp = ResubmitIdentityPayload(
        identity=identity,
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )


def test_an_unbound_flow_slot_on_a_grants_family_stays_unbound(monkeypatch):
    """A posted unbound flow slot stays open instead of taking the ODA default."""
    region = _resolve(_base_slots(_unbound_slot("crsTypeOfFlow")), monkeypatch=monkeypatch)

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]

    flow_filter = _filter_for(spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "unbound"
    assert flow_filter.provenance != "declared_default"

    finance_filter = _filter_for(spec, "crsTypeOfFinance")
    assert finance_filter.provenance == "binder"


def test_a_posted_flow_type_set_on_a_grants_family_stays_the_posted_set(monkeypatch):
    """A posted set of ODA and private-finance codes remains unchanged."""
    region = _resolve(
        _base_slots(_set_slot("crsTypeOfFlow", "CRSTypeOfFlow10", "CRSTypeOfFlow30")),
        monkeypatch=monkeypatch,
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]

    flow_filter = _filter_for(spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "set"
    assert {ref.dcid for ref in flow_filter.refs} == {"CRSTypeOfFlow10", "CRSTypeOfFlow30"}
    assert flow_filter.provenance != "declared_default"

    finance_filter = _filter_for(spec, "crsTypeOfFinance")
    assert finance_filter.provenance == "binder"
