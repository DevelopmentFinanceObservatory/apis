"""Tests resubmission with measure axes absent from the selected grantEquivalent family.

The padded identity keeps both axes unbound and lists them in `inapplicable_slots`.
grantEquivalent publishes no alternatives, so the test pads the elected identity itself.
"""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.clients.graph import Facet
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.spec_id import mint_identity
from search.models import IdentityBindingUnbound, IdentitySlot, ResubmitIdentityPayload
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_candidates_measure_alternatives import (
    _EXTRA_SV,
    _ROLE_RECIPIENT,
    _ROLES,
    _run,
    _spec_for,
)
from tests.fixtures import FakeGraph

_KEN = "country/KEN"


def _arcs_for_row(row) -> dict:
    """Build a role-bearing StatVar's arcs from its packaged-index row."""
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": row.dcid}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": (
            {"nodes": [{"dcid": row.measurement_qualifier}]}
            if row.measurement_qualifier
            else {"nodes": []}
        ),
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        "observationProperties": {"nodes": [{"dcid": p} for p in _ROLES]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def _real_nominal_family() -> list:
    """Return packaged `grantEquivalent`/`Nominal` rows for the CRS purpose and flow type."""
    index = get_index()
    return [
        row
        for row in index.statvars
        if row.population_type == "FinancialFlow"
        and row.measured_property == "grantEquivalent"
        and row.measurement_qualifier == "Nominal"
        and row.constraint_properties.get("crsPurpose") == "CRSPurposeCode51010"
        and row.constraint_properties.get("financialFlowType") == "Disbursement"
    ]


def _family_graph(member_rows) -> FakeGraph:
    nodes = {_KEN: {"label": "Kenya", "type": "Country"}}
    for row in member_rows:
        nodes[row.dcid] = {
            "label": row.dcid,
            "type": "StatisticalVariable",
            "arcs": _arcs_for_row(row),
        }
    return FakeGraph(nodes=nodes, obs={})


def _sibling_sdmx(sibling_sv: str) -> FakeSdmxClient:
    observations = {
        sibling_sv: {
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
    }
    return FakeSdmxClient(observations=observations, availability={})


def test_an_identity_padded_with_never_asserted_measure_axes_resolves_with_inapplicable_slots(
    monkeypatch,
):
    """Both padded axes stay unbound and appear in `inapplicable_slots`.

    The fixture has no observations, so the result is `no_data` with `no_observations`.
    """
    result = _run(monkeypatch)
    elected_spec = _spec_for(result.specs, _EXTRA_SV)
    elected_identity = elected_spec.definition.identity

    padded_slots = [
        *elected_identity.slots,
        IdentitySlot(property="financialObligationType", binding=IdentityBindingUnbound()),
        IdentitySlot(property="nettingBasis", binding=IdentityBindingUnbound()),
    ]
    padded_projection = elected_identity.model_dump(exclude_none=True)
    padded_projection["slots"] = [s.model_dump(exclude_none=True) for s in padded_slots]
    padded_spec_id, padded_identity = mint_identity(padded_projection)

    member_rows = _real_nominal_family()
    graph = _family_graph(member_rows)
    materialise_graph = axis_confirming_reader(*(row.dcid for row in member_rows), graph=graph)

    inp = ResubmitIdentityPayload(
        identity=padded_identity, base_spec_id=padded_spec_id, base_identity=padded_identity
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(materialise_graph=materialise_graph),
    )
    assert region.status == "no_data"
    assert region.no_data_reason == "no_observations"
    spec = region.flagged_match[0]
    assert spec.spec_id == padded_spec_id
    for padded_prop in ("financialObligationType", "nettingBasis"):
        padded_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.dcid == padded_prop
        )
        assert padded_slot.binding.kind == "unbound"
        assert padded_prop in spec.resolved.inapplicable_slots
