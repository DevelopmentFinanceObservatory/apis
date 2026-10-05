"""Facet options under a request-form unit are judged with the units the decision elected.

One StatVar at Kenya carries a euro facet under a provenance that declares the DAC recipe,
and a pound facet under another provenance with no conversion recipe. A request for current
US dollars elects the euro facet, which converts, so pinning the first provenance beside that
request is admitted; the pound facet holds no elected unit, so pinning the second is refused.
A first resolve, a candidate and a resubmit each build their option pins through
`convertibility.option_pins`, and each offers the first provenance alone. Testing the request
literally against the euro facet would withhold the first; dropping the request would offer
the second.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from materialise.facets import FacetSelection
from search.engine.region.candidate_options import _candidate_facet_probe
from search.engine.region.facet_probe import FacetProbe
from search.engine.region.option_coverage import _facet_options
from search.engine.region.phase_answer import _answer_facet_probe
from search.engine.region.promote_finalize import _promote_facet_probe
from search.models import IdentityFacetValueRequest, IdentityUnitRequest
from tests.engine._fake_coverage import build_coverage_index
from tests.fixtures import FakeGraph

_SV = "TESTFIXTURE/OptionPins/Expenditure"
_KEN = "country/KEN"
_PROVENANCE = "dc/base/OptionPinsSrc"
_OTHER_PROVENANCE = "dc/base/OptionPinsOtherSrc"
_REQUEST = IdentityUnitRequest(currency="USD", prices="current")
_FACETS = {"unit": IdentityFacetValueRequest(request=_REQUEST)}
_ELECTED = {
    _SV: (
        {
            "unit": FacetSelection(
                axis="unit",
                ref="CurrencyEUR",
                basis="bound",
                default_policy=None,
                default_policy_version=None,
                selection_reason=None,
                dispatch_facet_id="f-eur",
            )
        },
    )
}
_GRAPH = FakeGraph(
    obs={
        f"{_SV}|{_KEN}": [
            {
                "facetId": "f-eur",
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "dates": ["2020"],
                "provenanceId": _PROVENANCE,
                "unit": "CurrencyEUR",
            },
            {
                "facetId": "f-gbp",
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "dates": ["2020"],
                "provenanceId": _OTHER_PROVENANCE,
                "unit": "CurrencyGBP",
            },
        ]
    },
    currency_units={"CurrencyEUR": "EUR", "CurrencyUSD": "USD", "CurrencyGBP": "GBP"},
)
_ENTITIES = {"observationAbout": [_KEN]}


def _offered_provenances(probe: FacetProbe) -> list[str | None]:
    options = _facet_options(
        probe.sv_dcids,
        probe.probe_entities,
        years=probe.years,
        graph=_GRAPH,
        coverage=build_coverage_index([], role_names={}),
        pins=probe.pins,
        deadline=None,
    )
    return [
        o.value.ref.dcid if o.value.ref is not None else None
        for fo in options
        if fo.axis == "provenance"
        for o in fo.options
    ]


def test_a_first_resolve_offers_only_the_provenance_its_elected_unit_admits():
    state = SimpleNamespace(
        unit_request=_REQUEST,
        bound_unit_outcome=object(),
        unit_elected_selections=_ELECTED,
    )
    mat_result = SimpleNamespace(sv_dcids=[_SV], preview_years=None, member_facts=None)

    probe = _answer_facet_probe(cast(Any, state), cast(Any, mat_result), _ENTITIES)

    assert _offered_provenances(probe) == [_PROVENANCE]


def test_a_candidate_offers_only_the_provenance_its_elected_unit_admits():
    spec = SimpleNamespace(
        spec_id="spec_c", definition=SimpleNamespace(identity=SimpleNamespace(facets=_FACETS))
    )
    grounded = SimpleNamespace(
        cand_obs_entities={"spec_c": ([_SV], _ENTITIES)},
        cand_probe_entities={"spec_c": _ENTITIES},
        cand_plans={},
        cand_elected={"spec_c": _ELECTED},
        cand_scopes={},
    )

    probe = _candidate_facet_probe(
        cast(Any, SimpleNamespace(date_request=None)), cast(Any, grounded), cast(Any, spec)
    )

    assert _offered_provenances(probe) == [_PROVENANCE]


def test_a_resubmit_offers_only_the_provenance_its_elected_unit_admits():
    state = SimpleNamespace(
        probe_role_entities=_ENTITIES,
        inp=SimpleNamespace(identity=SimpleNamespace(facets=_FACETS)),
        elected_unit=_ELECTED,
    )
    mat_result = SimpleNamespace(sv_dcids=[_SV], preview_years=None, member_facts=None)

    probe = _promote_facet_probe(cast(Any, state), cast(Any, mat_result))

    assert _offered_provenances(probe) == [_PROVENANCE]
