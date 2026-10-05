"""A named measure-family identity can recover its StatVar without a posted
seed. The real FinancialFlow/grantEquivalent fixture also pins shape mismatch,
empty election, and the standard-shape refusal.
"""

from __future__ import annotations

import pytest

from code_index import CodeIndex
from code_index.shared import get_index
from search.engine.errors import EngineInputError
from search.engine.region import promote_reread, resolve_spec_resubmit
from search.models import BaseIdentity, ResubmitIdentityPayload
from tests.engine._base_identity import GDP_SHAPE, make_base, make_identity, subject_entity
from tests.engine._builders import value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_ENTITY_DCID = "country/KEN"

_B_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "FinancialFlow",
    "measured_property": "grantEquivalent",
    "stat_type": "measuredValue",
    "measurement_qualifier": "Nominal",
}
_B_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)
# The posted what/how slots: every constraint property _B_SV's own graph node
# declares below, financialFlowType included. It appears on both the base
# and the posted identity below, satisfying identity_edit.check_slots_declared's
# posted-covers-base requirement.
_CONSTRAINT_SLOTS = [
    value_slot("how", "crsTypeOfFlow", "crsTypeOfFlow", "CRSTypeOfFlow10"),
    value_slot("how", "crsTypeOfFinance", "crsTypeOfFinance", "CRSTypeOfFinance110"),
    value_slot(
        "how", "crsCooperationModality", "crsCooperationModality", "CRSCooperationModalityA01"
    ),
    value_slot("how", "crsPurpose", "crsPurpose", "CRSPurposeCode51010"),
    value_slot("how", "financialFlowType", "financialFlowType", "Disbursement"),
]
_CONSTRAINT_PROPS = (
    "financialFlowType",
    "crsTypeOfFlow",
    "crsTypeOfFinance",
    "crsCooperationModality",
    "crsPurpose",
)
_CONSTRAINT_VALUES = {
    "financialFlowType": "Disbursement",
    "crsTypeOfFlow": "CRSTypeOfFlow10",
    "crsTypeOfFinance": "CRSTypeOfFinance110",
    "crsCooperationModality": "CRSCooperationModalityA01",
    "crsPurpose": "CRSPurposeCode51010",
}


def _sv_arcs(
    *, measured_property: str = "grantEquivalent", stat_type: str = "measuredValue"
) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": _B_SV}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": measured_property}]},
        "statType": {"nodes": [{"dcid": stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in _CONSTRAINT_PROPS]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in _CONSTRAINT_VALUES.items()},
    }


def _graph(*, sv_arcs: dict | None = None) -> FakeGraph:
    nodes = {
        _ENTITY_DCID: {"label": "Kenya", "type": "Country"},
        _B_SV: {
            "label": "test grant equivalent named-family member",
            "type": "StatisticalVariable",
            "arcs": sv_arcs if sv_arcs is not None else _sv_arcs(),
        },
        **{dcid: {"label": dcid} for dcid in (*_CONSTRAINT_PROPS, *_CONSTRAINT_VALUES.values())},
    }
    obs = {
        f"{_B_SV}|{_ENTITY_DCID}": [
            # No `provenanceId`: the member's facet is its code-index dataset's
            # (`fixtures.provenance.facet_provenance`), the one a resubmit reads it from.
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
        ]
    }
    return FakeGraph(nodes=nodes, obs=obs)


def _b_identity(slots: list = _CONSTRAINT_SLOTS) -> tuple[str, BaseIdentity]:
    entities = [subject_entity(_ENTITY_DCID)]
    return make_base(_B_SHAPE, slots, entities=entities)


def test_named_family_round_trip_with_no_seed():
    """Recover the named-family member without a posted seed."""
    spec_id, identity = _b_identity()
    inp = ResubmitIdentityPayload(identity=identity, base_spec_id=spec_id, base_identity=identity)
    region = resolve_spec_resubmit(
        inp=inp, graph=_graph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
    )
    assert region.status == "definite"
    assert [sv.ref for sv in region.specs[0].resolved.stat_vars.items] == [_B_SV]


def test_standard_catch_all_shape_without_seed_still_refuses():
    """A catch-all identity without a seed still refuses at the seed pointer."""
    entities = [subject_entity(_ENTITY_DCID)]
    spec_id, identity = make_base(GDP_SHAPE, entities=entities)
    inp = ResubmitIdentityPayload(
        identity=make_identity(GDP_SHAPE, entities=entities),
        base_spec_id=spec_id,
        base_identity=identity,
    )
    with pytest.raises(EngineInputError) as exc_info:
        resolve_spec_resubmit(
            inp=inp, graph=_graph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
    assert exc_info.value.code == "request_invalid"
    assert exc_info.value.pointer == "/input/resolved_stat_var_dcids"


def test_recovered_member_disagreeing_with_posted_shape_raises_shape_mismatch():
    """Recovered arcs that disagree with the posted five-tuple raise
    `shape_mismatch`."""
    spec_id, identity = _b_identity()
    inp = ResubmitIdentityPayload(identity=identity, base_spec_id=spec_id, base_identity=identity)
    graph = _graph(sv_arcs=_sv_arcs(stat_type="medianValue"))
    with pytest.raises(EngineInputError) as exc_info:
        resolve_spec_resubmit(
            inp=inp, graph=graph, sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
        )
    assert exc_info.value.code == "shape_mismatch"


def test_named_family_with_no_recoverable_members_refuses_as_no_data():
    """A constraint combination with no indexed member returns no_data."""
    bogus_slots = [
        value_slot("how", "crsTypeOfFlow", "crsTypeOfFlow", "CRSTypeOfFlow10"),
        value_slot("how", "crsTypeOfFinance", "crsTypeOfFinance", "CRSTypeOfFinance110"),
        value_slot(
            "how",
            "crsCooperationModality",
            "crsCooperationModality",
            "CRSCooperationModalityA01",
        ),
        value_slot("how", "crsPurpose", "crsPurpose", "NoSuchPurposeCode"),
        value_slot("how", "financialFlowType", "financialFlowType", "Disbursement"),
    ]
    spec_id, identity = _b_identity(bogus_slots)
    inp = ResubmitIdentityPayload(identity=identity, base_spec_id=spec_id, base_identity=identity)
    region = resolve_spec_resubmit(
        inp=inp, graph=_graph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
    )
    assert region.status == "no_data"


def test_named_family_with_an_empty_election_answers_no_data_beside_a_posted_seed(monkeypatch):
    """If election returns no rows, the posted seed does not stand in for the family. The
    test patches only recovery's index; validation sees the real family."""
    real_index = get_index()
    empty_statvars_index = CodeIndex(
        stamp=real_index.stamp,
        declarations=real_index.declarations.values(),
        codes=real_index.codes,
        statvars=(),
        datasets=real_index.datasets.values(),
    )
    monkeypatch.setattr(promote_reread, "get_index", lambda: empty_statvars_index)
    spec_id, identity = _b_identity()
    inp = ResubmitIdentityPayload(
        identity=identity,
        base_spec_id=spec_id,
        base_identity=identity,
        resolved_stat_var_dcids=[_B_SV],
    )
    region = resolve_spec_resubmit(
        inp=inp, graph=_graph(), sdmx=FakeSdmxClient(), coverage=empty_coverage_snapshot()
    )
    assert region.status == "no_data"
    assert region.no_data_reason == "variable_not_resolved"
