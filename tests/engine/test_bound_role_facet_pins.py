"""A facet pin on a role-bearing spec with every role bound is judged by `pin_admitted`.

The debtor-only flow below binds its one role to Kenya, so no role axis is discovered. Two
StatVars answer: both carry a recorded series with no method, and only the first carries an
`Estimated` projection. A resubmit that pins a value only one StatVar carries is refused, a
pin both carry is served, and the facet options a pinned spec offers are the values a pin
beside that pin serves.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.clients.graph import Facet
from search.engine.errors import EngineInputError
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_spec_resubmit
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.models import (
    BaseIdentity,
    IdentityBindingValue,
    IdentityFacetValueAbsent,
    ResubmitIdentityPayload,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_SV_PROJECTED = "TESTFIXTURE/BoundRoleFacetPins/Projected"
_SV_RECORDED = "TESTFIXTURE/BoundRoleFacetPins/Recorded"
_KEN = "country/KEN"
_WB = "country/WBX"
_IDS = "source/IDS"
_OTHER = "source/Other"
_QUERY = "debt service"

# Facet id -> (StatVar, provenance, method, unit, years).
_FACETS = {
    "p-rec": (_SV_PROJECTED, _IDS, None, "USD", [2020, 2021]),
    "p-est": (_SV_PROJECTED, _IDS, "Estimated", "USD", [2025, 2026]),
    "p-survey": (_SV_PROJECTED, _OTHER, "Survey", "USD", [2020]),
    "r-rec": (_SV_RECORDED, _IDS, None, "USD", [2020, 2021]),
    "r-survey": (_SV_RECORDED, _OTHER, "Survey", "USD", [2021]),
}


def _arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "debtor"}]},
    }


def _coverage() -> FakeCoverageService:
    return FakeCoverageService(
        build_coverage_index(
            [(sv, facet_id, _WB, _KEN, years) for facet_id, (sv, *_, years) in _FACETS.items()],
            role_names={sv: RoleNames(donor=None, recipient="debtor") for sv in _svs()},
            facet_metadata={
                facet_id: FacetMetadata(provenance_id=prov, measurement_method=method, unit=unit)
                for facet_id, (_, prov, method, unit, _) in _FACETS.items()
            },
        )
    )


def _svs() -> list[str]:
    return [_SV_PROJECTED, _SV_RECORDED]


def _sdmx() -> FakeSdmxClient:
    observations: dict = {}
    for facet_id, (sv, prov, method, unit, years) in _FACETS.items():
        observations.setdefault(sv, {}).setdefault((("debtor", _KEN),), []).append(
            Facet(
                earliest_date=str(years[0]),
                latest_date=str(years[-1]),
                obs_count=len(years),
                dates=[str(y) for y in years],
                values=[1.0] * len(years),
                facet_id=facet_id,
                provenance_id=prov,
                measurement_method=method,
                unit=unit,
            )
        )
    return FakeSdmxClient(observations=observations, availability={})


def _graph() -> FakeGraph:
    return FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            **{sv: {"label": sv.rsplit("/", 1)[-1], "arcs": _arcs()} for sv in _svs()},
            _IDS: {"label": "IDS"},
            _OTHER: {"label": "Other"},
        },
        detect={_QUERY: {"svs": _svs(), "cosine_scores": [0.9, 0.9]}},
        resolve={"Kenya": _KEN},
        obs={},
    )


class _FixedBindLLM:
    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[]), None


def _fresh():
    graph = _graph()
    result = asyncio.run(
        resolve_variable(
            "debt",
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=_sdmx(),
            base_timing={},
            coverage=_coverage().snapshot(),
        )
    )
    assert result.status == "definite", result.no_data_reason
    spec = result.specs[0]
    assert sorted(sv.ref.ref for sv in spec.resolved.members) == sorted(_svs())
    return spec


def _resubmit(facets: dict, *, include_option_coverage: bool = False):
    fresh = _fresh()
    echo = ResubmitIdentityPayload(
        identity=BaseIdentity.model_validate(
            {**fresh.definition.identity.model_dump(mode="python"), "facets": facets}
        ),
    )
    return resolve_spec_resubmit(
        inp=echo,
        graph=_graph(),
        sdmx=_sdmx(),
        coverage=_coverage().snapshot(),
        options=walk_options(
            _graph(), EngineOptions(include_option_coverage=include_option_coverage)
        ),
    )


def test_a_pin_only_one_statvar_carries_is_refused():
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit({"measurementMethod": IdentityBindingValue(ref="Estimated")})

    assert exc_info.value.code == "facet_not_available"
    assert exc_info.value.pointer == "/input/identity/facets/measurementMethod"
    assert "Estimated" in str(exc_info.value)


@pytest.mark.parametrize(
    "facets",
    [
        {"measurementMethod": IdentityFacetValueAbsent(ref=None)},
        {"measurementMethod": IdentityBindingValue(ref="Survey")},
        {"provenance": IdentityBindingValue(ref=_OTHER)},
    ],
    ids=["null-method", "survey", "other-provenance"],
)
def test_a_pin_every_statvar_carries_is_served(facets):
    result = _resubmit(facets)

    assert result.status == "definite", result.no_data_reason
    assert sorted(sv.ref.ref for sv in result.spec.resolved.members) == sorted(_svs())


def test_a_null_method_pin_constrains_the_provenance_pin_beside_it():
    """The other provenance serves only Survey facets, so beside a pin on the facets with no
    method it has nothing to serve. The refusal points at the provenance the client must
    change, names both axes, and offers provenances rather than methods."""
    with pytest.raises(EngineInputError) as exc_info:
        _resubmit(
            {
                "provenance": IdentityBindingValue(ref=_OTHER),
                "measurementMethod": IdentityFacetValueAbsent(ref=None),
            }
        )

    refusal = exc_info.value
    assert refusal.code == "facet_not_available"
    assert refusal.pointer == "/input/identity/facets/provenance"
    assert "provenance" in str(refusal)
    assert "measurementMethod" in str(refusal)
    assert [a.ref for a in refusal.alternatives or []] == [_IDS, _OTHER]


def _offered(result) -> dict[str, list[str | None]]:
    return {
        fo.axis: [o.value.ref.ref if o.value.ref is not None else None for o in fo.options]
        for fo in result.spec.resolved.facet_options
    }


def test_options_on_a_pinned_spec_are_admitted_beside_its_pin():
    """With provenance pinned to IDS, the method choices are the IDS facets every StatVar
    carries: the recorded series alone, so the method axis is omitted. Survey appears only
    under the other provenance, and Estimated only on one StatVar. The provenance axis lists
    both sources, each pinnable in place of IDS."""
    result = _resubmit({"provenance": IdentityBindingValue(ref=_IDS)}, include_option_coverage=True)

    assert result.status == "definite", result.no_data_reason
    assert _offered(result) == {"provenance": [_IDS, _OTHER], "unit": ["USD"]}


def test_options_on_a_method_pin_follow_that_pin():
    """With the method pinned to Survey, only the other provenance serves it on both
    StatVars, and the method axis lists the values pinnable in its place."""
    result = _resubmit(
        {"measurementMethod": IdentityBindingValue(ref="Survey")}, include_option_coverage=True
    )

    assert result.status == "definite", result.no_data_reason
    assert _offered(result) == {
        "provenance": [_OTHER],
        "measurementMethod": [None, "Survey"],
        "unit": ["USD"],
    }
