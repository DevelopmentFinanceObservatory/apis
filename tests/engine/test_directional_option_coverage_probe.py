"""Role-bearing slot options omit coverage because flat probes lose donor-recipient pairs,
while their facet options come from the coverage index's role pairs.

Subject-place queries retain coverage."""

from __future__ import annotations

import asyncio

from materialise.graph import Facet as MaterialiseFacet
from search.engine.clients.graph import Facet
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

_SV_GRANT = "TESTFIXTURE/A2a/Flow_Grant"
_SV_LOAN = "TESTFIXTURE/A2a/Flow_Loan"
_KEN = "country/KEN"
_NOR = "country/NOR"
_USA = "country/USA"
_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
_SOURCE = "dc/base/TestSource"


def _coverage(monkeypatch):
    """Mirrors the SDMX fixture's Norway- and USA-donor rows at Kenya and their provenance."""
    rows = [
        (sv, facet_id, donor, _KEN, [2020])
        for sv in (_SV_GRANT, _SV_LOAN)
        for facet_id, donor in ((f"{sv}-nor", _NOR), (f"{sv}-usa", _USA))
    ]
    index = patch_coverage(
        monkeypatch,
        rows,
        role_names=dict.fromkeys((_SV_GRANT, _SV_LOAN), _ROLE_NAMES),
        facet_metadata={row[1]: FacetMetadata(provenance_id=_SOURCE) for row in rows},
    )
    return FakeCoverageService(index).snapshot()


def _facet(*, obs_count: int = 3, facet_id: str = "f1") -> Facet:
    return Facet(
        earliest_date="2015",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
        provenance_id="dc/base/TestSource",
    )


def _flow_arcs(*, modality: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
        "constraintProperties": {"nodes": [{"dcid": "modality"}]},
        "modality": {"nodes": [{"dcid": modality}]},
    }


class _FixedBindLLM:
    """Leaves modality open so both members appear as slot options."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[]), None


def test_directional_single_place_gets_no_slot_options_coverage(monkeypatch):
    """A recipient-only query leaves Grant and Loan coverage unset.

    Role-bearing shapes skip slot coverage probes. Facet options list the provenance of
    every creditor's rows at Kenya, the open axis."""
    role_query = "aid to Kenya"
    entities = ["Kenya"]
    resolve = {"Kenya": _KEN, "Norway": _NOR, "USA": _USA}
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
            _SV_GRANT: {"label": "Grant flow", "arcs": _flow_arcs(modality="Grant")},
            _SV_LOAN: {"label": "Loan flow", "arcs": _flow_arcs(modality="Loan")},
            "Grant": {"label": "Grant"},
            "Loan": {"label": "Loan"},
            "modality": {"label": "modality"},
        },
        detect={role_query: {"svs": [_SV_GRANT, _SV_LOAN], "cosine_scores": [0.9, 0.85]}},
        resolve=resolve,
        obs={
            f"{_SV_GRANT}|{_KEN}": [
                {
                    "earliestDate": "2015",
                    "latestDate": "2020",
                    "obsCount": 4,
                    "provenanceId": "dc/base/TestSource",
                }
            ],
            f"{_SV_LOAN}|{_KEN}": [
                {
                    "earliestDate": "2015",
                    "latestDate": "2020",
                    "obsCount": 2,
                    "provenanceId": "dc/base/TestSource",
                }
            ],
        },
    )
    sdmx = FakeSdmxClient(
        observations={
            _SV_GRANT: {
                (("creditor", _NOR), ("debtor", _KEN)): [_facet()],
                (("creditor", _USA), ("debtor", _KEN)): [_facet()],
            },
            _SV_LOAN: {
                (("creditor", _NOR), ("debtor", _KEN)): [_facet()],
                (("creditor", _USA), ("debtor", _KEN)): [_facet()],
            },
        }
    )
    materialise_graph = FakeGraphReader(
        facets={
            (_SV_GRANT, _KEN): [
                MaterialiseFacet(
                    facet_id="f1",
                    provenance_id="dc/base/TestSource",
                    measurement_method=None,
                    unit=None,
                    obs_count=4,
                    earliest_date="2015",
                    latest_date="2020",
                )
            ],
            (_SV_LOAN, _KEN): [
                MaterialiseFacet(
                    facet_id="f2",
                    provenance_id="dc/base/TestSource",
                    measurement_method=None,
                    unit=None,
                    obs_count=2,
                    earliest_date="2015",
                    latest_date="2020",
                )
            ],
        }
    )
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=entities,
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=_coverage(monkeypatch),
            options=EngineOptions(
                include_option_coverage=True, materialise_graph=materialise_graph
            ),
        )
    )
    assert result.status == "definite", result
    spec = result.specs[0]
    modality_opts = next(
        so
        for so in spec.resolved.slot_options
        if so.key.property and so.key.property.ref == "modality"
    )
    assert all(o.coverage is None for o in modality_opts.options)
    assert [
        (fo.axis, [o.value.ref.ref for o in fo.options if o.value.ref is not None], fo.truncated)
        for fo in spec.resolved.facet_options
    ] == [("provenance", [_SOURCE], False)]
