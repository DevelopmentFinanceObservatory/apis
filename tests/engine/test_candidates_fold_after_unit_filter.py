"""The bound-unit filter must run before measure-sibling folding. The elected grantEquivalent member
has no conversion path to USD, while its amount sibling already uses USD. Filtering first lets the
sibling survive and answer the request.
"""

from __future__ import annotations

import asyncio

from code_index.shared import get_index
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.state import EngineOptions
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    door_roles_for_test_query,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.fixtures.provenance import index_provenance
from tests.materialise.conftest import FakeGraphReader

_QUERY = "grant equivalent flows or gross disbursements to Kenya"
_KEN_DCID = "country/KEN"

# Elected role-bearing grantEquivalent shape.
_POP_TYPE = "FinancialFlow"
_MEAS_PROP = "grantEquivalent"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"

_RECALLED_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)
_EXTRA_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)
_PROP_PURPOSE = "crsPurpose"
_EXTRA_PURPOSE = "CRSPurposeCode51010"
_ROLES = ("financingProvider", "financingRecipient")
_ROLE_RECIPIENT = "financingRecipient"

# Plausible amount sibling of the elected shape.
_ALT_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)

# Each member's facets are its code-index dataset's, the only source an answer serves it from.
_EXTRA_PROVENANCE = index_provenance(_EXTRA_SV)
_ALT_PROVENANCE = index_provenance(_ALT_SV)

# The retained member's KES facet has no declared conversion to USD.
_UNCONVERTIBLE_CURRENCY = "CurrencyKES"
# The sibling already has a native USD facet.
_USD_CURRENCY = "CurrencyUSD"


def _sv_arcs_from_real_row(dcid: str) -> dict:
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == dcid)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV: {
        "label": "test recalled sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_RECALLED_SV),
    },
    # Its real fixture-index row declares the flow roles, and the
    # declared-role refresh reads its own graph node, so the synthetic
    # node carries the same observationProperties arc.
    _EXTRA_SV: {
        "label": "test index-only sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_EXTRA_SV),
    },
    _EXTRA_PURPOSE: {"label": "CRS purpose 51010"},
    _ALT_SV: {
        "label": "test amount sibling",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_ALT_SV),
    },
}

_DETECT = {
    _QUERY: {
        "svs": [_RECALLED_SV, _ALT_SV],
        "cosine_scores": [0.9, 0.85],
    }
}

# Only _EXTRA_SV survives the crsPurpose bind. Its search-side facets feed unit elections and need
# provenance ids for the provenance axis.
_SDMX_OBS = {
    _EXTRA_SV: {
        ((_ROLE_RECIPIENT, _KEN_DCID),): [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=3,
                dates=["2020"],
                values=[0.0],
                facet_id="f1",
                unit=_UNCONVERTIBLE_CURRENCY,
                provenance_id=_EXTRA_PROVENANCE,
            )
        ]
    },
    _ALT_SV: {
        ((_ROLE_RECIPIENT, _KEN_DCID),): [
            Facet(
                earliest_date="2020",
                latest_date="2021",
                obs_count=5,
                dates=["2020", "2021"],
                values=[10.0, 12.0],
                facet_id="f2",
                unit=_USD_CURRENCY,
                provenance_id=_ALT_PROVENANCE,
            )
        ]
    },
}


class _ElectGrantEquivalentAlsoPlausibleAmountLLM:
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
                    value_dcids=[_EXTRA_PURPOSE],
                )
            ],
            also_plausible=[1],
            ask=None,
        ), None


def _materialise_clients():
    """Build probe clients. Only the amount sibling has a USD-satisfying facet. The preview
    binds Kenya and reads the provider open, so the double serves each provider's rows."""
    fixtures = {
        _EXTRA_SV: ObsFacetFixture(
            facet_id="fx",
            points=(("2020", 5.0),),
            provenance_id=_EXTRA_PROVENANCE,
            unit=_UNCONVERTIBLE_CURRENCY,
        ),
        _ALT_SV: ObsFacetFixture(
            facet_id="fy",
            points=(("2020", 7.0),),
            provenance_id=_ALT_PROVENANCE,
            unit=_USD_CURRENCY,
        ),
    }
    materialise_graph_base, materialise_sdmx, _requests = build_materialise_double(
        {
            (sv, ((_ROLES[0], "country/USA"), (_ROLE_RECIPIENT, _KEN_DCID))): [fixture]
            for sv, fixture in fixtures.items()
        },
    )
    materialise_graph = FakeGraphReader(
        facets=materialise_graph_base._facets,
        arcs={
            _EXTRA_SV: {"measurementQualifier": (_MEAS_QUAL,)},
            _ALT_SV: {"measurementQualifier": (_MEAS_QUAL,)},
            _USD_CURRENCY: {"typeOf": ("CurrencyUnit",)},
        },
    )
    return materialise_graph, materialise_sdmx


_ROLE_NAMES = RoleNames(donor=_ROLES[0], recipient=_ROLES[1])


def _run(monkeypatch):
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_EXTRA_SV, "f1", "country/USA", _KEN_DCID, [2018, 2019, 2020]),
            (_ALT_SV, "f2", "country/USA", _KEN_DCID, [2020, 2021]),
        ],
        role_names={_EXTRA_SV: _ROLE_NAMES, _ALT_SV: _ROLE_NAMES},
        # The retained member lacks USD conversion. The amount sibling has a native USD facet.
        facet_metadata={
            "f1": FacetMetadata(provenance_id=_EXTRA_PROVENANCE, unit=_UNCONVERTIBLE_CURRENCY),
            "f2": FacetMetadata(provenance_id=_ALT_PROVENANCE, unit=_USD_CURRENCY),
        },
    )
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs={})
    materialise_graph, materialise_sdmx = _materialise_clients()
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
            sdmx=FakeSdmxClient(observations=_SDMX_OBS, availability={}),
            llm=_ElectGrantEquivalentAlsoPlausibleAmountLLM(),
            base_timing={},
            unit_text="USD",
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            options=EngineOptions(
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ),
        )
    )


def test_the_unit_incompatible_survivor_is_dropped_before_the_fold_can_hide_its_sibling(
    monkeypatch,
):
    """Filter the KES member before folding so the USD amount sibling can answer."""
    result = _run(monkeypatch)
    assert result.status == "definite", result
    assert len(result.specs) == 1
    survivor = result.specs[0]
    assert any(sv.ref.ref == _ALT_SV for sv in survivor.resolved.members), (
        f"expected the unit-compatible sibling {_ALT_SV!r} to answer, "
        f"got {[sv.ref.ref for sv in survivor.resolved.members]!r}"
    )


def _bound_ref(identity, property_dcid: str) -> str | None:
    """Return a bound slot's dcid, or `None` when the slot is absent or unbound."""
    for slot in identity.slots:
        if slot.property == property_dcid and slot.binding.state == "value":
            return slot.binding.ref
    return None


def test_the_excluded_survivor_never_rides_along_as_an_alternative_measure(monkeypatch):
    """The surviving USD spec carries no crsPurpose binding from the excluded KES candidate."""
    result = _run(monkeypatch)
    survivor = result.specs[0]
    alternatives = survivor.resolved.alternative_measures or []
    for alt in alternatives:
        assert _bound_ref(alt.definition.identity, _PROP_PURPOSE) != _EXTRA_PURPOSE, (
            f"expected no alternative_measures entry to carry the excluded "
            f"candidate's own bound purpose {_EXTRA_PURPOSE!r}: {alt!r}"
        )


def test_lone_survivor_carries_its_own_member_population(monkeypatch):
    """A lone survivor carries its own exact member population into the result."""
    result = _run(monkeypatch)
    assert result.status == "definite", result
    survivor = result.specs[0]
    assert any(sv.ref.ref == _ALT_SV for sv in survivor.resolved.members)

    assert result.member_population is not None
    assert result.member_population.admitted_dcids == (_ALT_SV,)
    assert result.member_population.exact is True
