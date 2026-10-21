"""When served StatVars omit `measurementQualifier`, no measure partition matches.

The dev graph serves re-minted CRS StatVars this way, while the fixture index declares Nominal.

The tests return separate candidates, and the no-data fallback refuses an unlabelled measure. The
fixture uses real CRS siblings but reads their missing qualifier from graph arcs.
"""

from __future__ import annotations

import asyncio

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import ELECTED_SHAPE_NO_DATA
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "grant equivalent flows or gross disbursements to Kenya"
_KEN_DCID = "country/KEN"

# Elected FinancialFlow/grantEquivalent shape.
_RECALLED_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)
_PROP_PURPOSE = "crsPurpose"
# The model binds crsPurpose to the recalled member's value, so no sibling needs construction.
_PURPOSE_VALUE = "CRSPurposeCode11110"
_ROLE_RECIPIENT = "financingRecipient"

# Alternate FinancialFlow/amount measure sibling.
_ALT_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)


def _sv_arcs_no_measurement_qualifier(dcid: str) -> dict:
    """Build fixture-index arcs without `measurementQualifier`."""
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == dcid)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV: {
        "label": "test recalled sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_no_measurement_qualifier(_RECALLED_SV),
    },
    _ALT_SV: {
        "label": "test amount sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_no_measurement_qualifier(_ALT_SV),
    },
}

_DETECT = {
    _QUERY: {
        "svs": [_RECALLED_SV, _ALT_SV],
        "cosine_scores": [0.9, 0.85],
    }
}

_SDMX_OBS = {
    _RECALLED_SV: {
        ((_ROLE_RECIPIENT, _KEN_DCID),): [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=3,
                dates=["2020"],
                values=[0.0],
                facet_id="f1",
            )
        ]
    },
    # The alternate uses its recalled member as the construct-and-confirm representative.
    _ALT_SV: {
        ((_ROLE_RECIPIENT, _KEN_DCID),): [
            Facet(
                earliest_date="2020",
                latest_date="2021",
                obs_count=5,
                dates=["2020", "2021"],
                values=[10.0, 12.0],
                facet_id="f2",
            )
        ]
    },
}

# The elected member has no rows. Its measure sibling has data.
_SDMX_OBS_ELECTED_HAS_NO_DATA = {
    _ALT_SV: _SDMX_OBS[_ALT_SV],
}

# Both shapes are role-bearing. Kenya binds the destination. No assertion uses the donor dcid.
_ROLE_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_COVERAGE_ROWS = [
    (_RECALLED_SV, "f1", "country/USA", _KEN_DCID, [2018, 2019, 2020]),
    (_ALT_SV, "f2", "country/USA", _KEN_DCID, [2020, 2021]),
]
# Coverage contains only the measure sibling.
_COVERAGE_ROWS_ELECTED_HAS_NO_DATA = [
    (_ALT_SV, "f2", "country/USA", _KEN_DCID, [2020, 2021]),
]


class _ElectRecalledAlsoPlausibleAltLLM:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_PURPOSE,
                    kind="value",
                    value_dcids=[_PURPOSE_VALUE],
                )
            ],
            also_plausible=[1],
            ask=None,
        ), None


def _run(monkeypatch, obs: dict = _SDMX_OBS, coverage_rows: list = _COVERAGE_ROWS):
    role_names = {row[0]: _ROLE_NAMES for row in coverage_rows}
    _auto_coverage_index_1 = patch_coverage(monkeypatch, coverage_rows, role_names=role_names)
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs={})
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            entity_roles=door_roles_for_test_query(_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations=obs, availability={}),
            llm=_ElectRecalledAlsoPlausibleAltLLM(),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )


def test_no_measure_partition_neither_candidate_folds(monkeypatch):
    """Missing qualifier arcs leave both candidates separate with no alternative measures."""
    result = _run(monkeypatch)
    assert result.status == "candidates"
    assert len(result.specs) == 2
    served_svs = {sv.ref.ref for spec in result.specs for sv in spec.resolved.members}
    assert served_svs == {_RECALLED_SV, _ALT_SV}
    for spec in result.specs:
        assert not spec.resolved.alternative_measures


def test_s8_with_no_measure_partition_refuses_honestly(monkeypatch):
    """A sibling without a measure label cannot replace the data-less elected shape."""
    result = _run(
        monkeypatch,
        obs=_SDMX_OBS_ELECTED_HAS_NO_DATA,
        coverage_rows=_COVERAGE_ROWS_ELECTED_HAS_NO_DATA,
    )
    assert result.status == "no_data"
    assert result.gate == "only_alternate_has_data"
    assert not any(w.code == ELECTED_SHAPE_NO_DATA for w in result.warnings)
