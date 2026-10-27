"""A unit request elects one group across a Spec's two StatVars, as a mint of the Spec does.
SourceA holds a US dollar facet and SourceB a euro one, under their own sources, so each
key serves one member and the requested currency decides which: a dollar request serves
SourceA natively and leaves SourceB unserved. A pound request, which neither holds, serves
SourceA too: the members tie on every count and SourceA's provenance sorts first. The
fixture covers buffered, streamed, and availability-check paths.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.rate_tables import RateTableCache
from search.engine.concurrency import scoped_concurrency
from search.engine.region import promote_finalize, promote_ground, resolve_spec_resubmit
from search.engine.region.codes import (
    DATE_COVERAGE_PARTIAL,
    ENTITY_LABEL_UNRESOLVED,
    MEMBERSHIP_UNVERIFIED,
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_PROBE_FAILED,
)
from search.engine.region.convertibility import elected_unit_refs
from search.engine.region.state import EngineOptions
from search.engine.unit.conversion_catalogue import (
    ConversionCatalogue,
    RecipeTables,
    bound_conversion_catalogue,
)
from search.models import (
    IdentityBindingValue,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    ResubmitIdentityPayload,
    Warning,
)
from search.observations import OBSERVATIONS_PREVIEW_TOO_LARGE
from tests.engine._base_identity import make_identity, subject_entity
from tests.engine._builders import binding_set_slot as _binding_set_slot
from tests.engine._builders import value_slot
from tests.engine._conjoin_fixtures import member_departures
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
    walk_options,
    with_member_arcs,
)
from tests.fixtures import FakeGraph
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_ENTITY = "country/KEN"
_UGANDA = "country/UGA"
_SHAPE: dict = {
    "kind": "statvar_schema",
    # A population type the DAC recipe converts, which the materialise graph's arcs state.
    "population_type": "FinancialFlow",
    "measured_property": "testConvertMultiExpenditure",
    "stat_type": "measuredValue",
    "measurement_qualifier": "Nominal",
}
_SV_A = "Test_ConvertMulti_SourceA"
_SV_B = "Test_ConvertMulti_SourceB"
_PROP = "financingSource"
_VAL_A = "SourceA"
_VAL_B = "SourceB"
_PROVENANCE_A = "dc/base/ConvertMultiSrcA"
_PROVENANCE_B = "dc/base/ConvertMultiSrcB"
_RECIPE = "ConversionRecipeDAC"
_YEAR = "2020"
_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_FIVE_TUPLE_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
    "measuredProperty": {"nodes": [{"dcid": "testConvertMultiExpenditure"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
}


def _sv_arcs(value: str) -> dict:
    return {
        **_FIVE_TUPLE_ARCS,
        "constraintProperties": {"nodes": [{"dcid": _PROP}]},
        _PROP: {"nodes": [{"dcid": value}]},
    }


_NODES: dict = {
    _SV_A: {
        "label": "Convert multi SourceA",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(_VAL_A),
    },
    _SV_B: {
        "label": "Convert multi SourceB",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(_VAL_B),
    },
    _ENTITY: {"label": "Kenya", "type": "Country"},
    _UGANDA: {"label": "Uganda", "type": "Country"},
}

_HOW_SET_SLOT = _binding_set_slot("how", _PROP, "financing source", [_VAL_A, _VAL_B])


# Search coverage reads are separate from materialisation's conversion reads.
def _obs(years: tuple[str, ...], places: tuple[str, ...] = (_ENTITY,)) -> dict:
    def facet(facet_id: str, provenance: str, unit: str) -> dict:
        return {
            "facetId": facet_id,
            "earliestDate": years[0],
            "latestDate": years[-1],
            "obsCount": len(years),
            "provenanceId": provenance,
            "unit": unit,
        }

    return {
        key: value
        for place in places
        for key, value in (
            (f"{_SV_A}|{place}", [facet("f_a", _PROVENANCE_A, "CurrencyUSD")]),
            (f"{_SV_B}|{place}", [facet("f_b", _PROVENANCE_B, "CurrencyEUR")]),
        )
    }


_USD_FIXTURE = ObsFacetFixture(
    facet_id="f_a", points=(), provenance_id=_PROVENANCE_A, unit="CurrencyUSD"
)
_EUR_FIXTURE = ObsFacetFixture(
    facet_id="f_b", points=(), provenance_id=_PROVENANCE_B, unit="CurrencyEUR"
)


_RATES = {"CurrencyUSD": {int(_YEAR): 1.0}, "CurrencyEUR": {int(_YEAR): 0.9}}


def _sdmx_router(
    observations_by_sv: dict[str, str], *, rates: dict[str, dict[int, float]] = _RATES
) -> MaterialiseSdmxClient:
    """Serve exchange-rate rows, USD at 1.0 and EUR at 0.9 by default, beside each StatVar's body."""

    def observations(request: httpx.Request) -> httpx.Response:
        params = dict(httpx.QueryParams(request.url.query.decode()))
        var = params.get("c[variableMeasured]", "")
        body = observations_by_sv.get(var, _csv_header(["observationAbout"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(
        recipe_table_bodies(_RECIPE, rates=rates),
        fallback=observations,
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


def _recipe_arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    """The DAC recipe on `_PROVENANCE_A` and `_PROVENANCE_B`, provenances in no dataset, and
    its three currencies."""
    arcs: dict[str, dict[str, tuple[str, ...]]] = {}
    for provenance in (_PROVENANCE_A, _PROVENANCE_B):
        arcs |= conversion_arcs(
            recipe_id=_RECIPE,
            provenance_id=provenance,
            currencies=["CurrencyUSD", "CurrencyEUR", "CurrencyGBP"],
        )
        arcs[provenance]["isPartOf"] = ()
    return arcs


def _run(
    *,
    stream_observations: bool = False,
    include_observations: bool | None = None,
    rates: dict[str, dict[int, float]] = _RATES,
    years: tuple[str, ...] = (_YEAR,),
    currency: str = "USD",
    places: tuple[str, ...] = (_ENTITY,),
):
    graph = FakeGraph(nodes=_NODES, obs=_obs(years, places))
    materialise_graph = FakeGraphReader(
        arcs={
            _SV_A: dict(CONVERTIBLE_STAT_VAR_ARCS),
            _SV_B: dict(CONVERTIBLE_STAT_VAR_ARCS),
            _PROVENANCE_A: {},
            **_recipe_arcs(),
        },
        facets={
            **{(_SV_A, place): [_facet_from_fixture(_USD_FIXTURE)] for place in places},
            **{(_SV_B, place): [_facet_from_fixture(_EUR_FIXTURE)] for place in places},
        },
    )
    materialise_sdmx = _sdmx_router(
        {
            _SV_A: _csv_header(["observationAbout"])
            + "".join(
                _csv_row(
                    stat_var=_SV_A,
                    role_dcids=[place],
                    fixture=_USD_FIXTURE,
                    date_str=year,
                    value=100.0,
                )
                for place in places
                for year in years
            ),
            _SV_B: _csv_header(["observationAbout"])
            + "".join(
                _csv_row(
                    stat_var=_SV_B,
                    role_dcids=[place],
                    fixture=_EUR_FIXTURE,
                    date_str=year,
                    value=50.0,
                )
                for place in places
                for year in years
            ),
        },
        rates=rates,
    )
    entities = [subject_entity(place) for place in places]
    request = IdentityUnitRequest(currency=currency, prices="current")
    base_identity = make_identity(_SHAPE, slots=[_HOW_SET_SLOT], entities=entities)
    inp = ResubmitIdentityPayload(
        identity=base_identity.model_copy(
            update={"facets": {"unit": IdentityFacetValueRequest(request=request)}}
        ),
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            graph,
            EngineOptions(
                include_observations=(
                    not stream_observations
                    if include_observations is None
                    else include_observations
                ),
                stream_observations=stream_observations,
                materialise_graph=with_member_arcs(materialise_graph, graph),
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ),
        ),
    )


_GROUND_WARNING = Warning(
    code=ENTITY_LABEL_UNRESOLVED,
    severity="warn",
    message="no ->name label found for entity dcid(s) country/KEN; presented with a dcid-echo label.",
)


@contextmanager
def _grounding_warns() -> Iterator[None]:
    """Make grounding emit `_GROUND_WARNING`, as it does for an unlabelled entity.

    The fixture's entity must stay labelled because the resubmit refuses an unlabelled ref
    before grounding, so the warning is appended to the real grounding result."""
    real = promote_ground._ground_answer

    def _ground_answer(*args, **kwargs):
        *rest, warnings, labels = real(*args, **kwargs)
        return (*rest, [*warnings, _GROUND_WARNING], labels)

    with patch.object(promote_ground, "_ground_answer", _ground_answer):
        yield


_OBS_ENTITIES_WARNING = Warning(
    code=OBSERVATIONS_PROBE_FAILED,
    severity="warn",
    message="no observation entity is placed for the shape's role.",
)


@contextmanager
def _obs_entities_warns() -> Iterator[None]:
    """Make grounding's observation-entities derivation emit `_OBS_ENTITIES_WARNING`.

    The real role-to-entity map is kept, so only the warning is added, as for an unplaced
    role."""
    real = promote_ground._observation_entities

    def _observation_entities(*args, **kwargs):
        entities, _ = real(*args, **kwargs)
        return entities, _OBS_ENTITIES_WARNING

    with patch.object(promote_ground, "_observation_entities", _observation_entities):
        yield


def _left_out_of_requested(warning: Warning) -> tuple[int, int]:
    """The series counts an `observations_no_matching_source` warning states: those the
    served source holds nothing for, and those requested."""
    counts = re.match(r"(\d+) of (\d+) ", warning.message)
    assert counts is not None, warning
    return int(counts[1]), int(counts[2])


def _names_the_left_out_series(warnings: tuple[Warning, ...]) -> None:
    """The one warning naming SourceB's series: the resubmit's source slot carries the served
    provenance, SourceA's, under which SourceB holds nothing, so 1 of the 2 requested series
    has no observations."""
    (warning,) = (w for w in warnings if w.code == OBSERVATIONS_NO_MATCHING_SOURCE)
    assert _left_out_of_requested(warning) == (1, 2)


def _run_relaunch(region):
    """Run the streamed observations relaunch closure for a resubmit result."""

    async def _await() -> tuple:
        async with scoped_concurrency():
            task = region.observations_relaunch(region.specs[0].spec_id)
            return await task

    return asyncio.run(_await())


def test_a_dollar_request_serves_the_dollar_member_and_leaves_the_euro_one():
    """A US dollar request elects SourceA's native dollar facet, the group's, and SourceB,
    which holds only euros, is unserved: it has no elected unit and no series, and the part
    names it as left out of SourceA's provenance, on the set binding naming the members."""
    captured: list[object] = []
    real_build_observations = promote_finalize._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("elected"))
        return real_build_observations(*args, **kwargs)

    with patch.object(promote_finalize, "_build_observations", _spy):
        region = _run()

    assert region.status == "definite", f"expected definite, got {region.status!r}"
    assert captured, "expected the include_observations fetch to call _build_observations"
    elected = captured[0]
    assert elected is not None
    assert elected_unit_refs(elected) == {_SV_A: frozenset({"CurrencyUSD"})}

    spec = region.specs[0]
    assert spec.resolved.observations is not None
    (series,) = spec.resolved.observations.series
    assert series.stat_var.ref == _SV_A
    assert series.conversion is None
    assert [p.value for p in series.points] == [100.0]
    assert member_departures(region) == [(f"how:{_PROP}", "Convert multi SourceB", _PROVENANCE_A)]


_GBP_RATES = {**_RATES, "CurrencyGBP": {int(_YEAR): 0.8}}


def test_streamed_relaunch_with_a_bound_request_unit_carries_converted_points():
    """The streamed relaunch converts with the same rate cache as the buffered path. A pound
    request serves SourceA (neither member holds pounds, and SourceA's provenance sorts first
    on a tie), whose dollars convert at 0.8 pounds to the dollar."""
    region = _run(stream_observations=True, currency="GBP", rates=_GBP_RATES)
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    assert region.specs[0].resolved.observations is None
    assert region.observations_relaunch is not None

    events, warns = _run_relaunch(region)
    _names_the_left_out_series(warns)
    assert {w.code for w in warns} == {OBSERVATIONS_NO_MATCHING_SOURCE}
    assert events, "expected at least one streamed observations event"
    (series,) = events[0].observations.series
    assert series.stat_var.ref == _SV_A
    assert series.conversion is not None
    assert [p.value for p in series.points] == [pytest.approx(100.0 * 0.8)]


def test_facet_refusal_receives_the_served_members_elected_units():
    """Facet refusal receives the elected unit of each member the group serves: SourceA's
    dollars under a dollar request, and nothing for SourceB, which the group leaves out."""
    captured: list[object] = []
    import search.engine.facet_refusals as facet_refusals_module

    real = facet_refusals_module.facet_refusal

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("elected_units"))
        return real(*args, **kwargs)

    with patch.object(facet_refusals_module, "facet_refusal", _spy):
        # _phase_ground_build imports facet_refusal locally (deferred import),
        # binding the module attribute at call time, so patching the module
        # attribute here is observed.
        region = _run()

    assert region.status == "definite", f"expected definite, got {region.status!r}"
    assert captured, "expected the availability check to call facet_refusal"
    assert captured[0] == {_SV_A: frozenset({"CurrencyUSD"})}


_RECORDED_USD = ObsFacetFixture(
    facet_id="f_recorded", points=(), provenance_id=_PROVENANCE_B, unit="CurrencyUSD"
)
_ESTIMATED_EUR = ObsFacetFixture(
    facet_id="f_estimated",
    points=(),
    provenance_id=_PROVENANCE_B,
    measurement_method="Estimated",
    unit="CurrencyEUR",
)


def test_a_null_method_pin_decides_the_unit_among_the_facets_with_no_method():
    """One StatVar carries a recorded US dollar series with no method and an Estimated euro
    series, under a provenance that declares the DAC recipe. A resubmit pinning the method to
    the facets that declare none and requesting euros elects the recorded series and converts
    it. Deciding with the method open would elect the native euro facet instead."""
    graph = FakeGraph(
        nodes=_NODES,
        obs={
            f"{_SV_A}|{_ENTITY}": [
                {
                    "facetId": "f_recorded",
                    "earliestDate": _YEAR,
                    "latestDate": _YEAR,
                    "obsCount": 1,
                    "provenanceId": _PROVENANCE_B,
                    "unit": "CurrencyUSD",
                },
                {
                    "facetId": "f_estimated",
                    "earliestDate": _YEAR,
                    "latestDate": _YEAR,
                    "obsCount": 1,
                    "provenanceId": _PROVENANCE_B,
                    "measurementMethod": "Estimated",
                    "unit": "CurrencyEUR",
                },
            ]
        },
    )
    materialise_graph = FakeGraphReader(
        arcs={_SV_A: dict(CONVERTIBLE_STAT_VAR_ARCS), **_recipe_arcs()},
        facets={
            (_SV_A, _ENTITY): [
                _facet_from_fixture(_RECORDED_USD),
                _facet_from_fixture(_ESTIMATED_EUR),
            ]
        },
    )
    materialise_sdmx = _sdmx_router(
        {
            _SV_A: _csv_header(["observationAbout"])
            + _csv_row(
                stat_var=_SV_A,
                role_dcids=[_ENTITY],
                fixture=_RECORDED_USD,
                date_str=_YEAR,
                value=100.0,
            )
            + _csv_row(
                stat_var=_SV_A,
                role_dcids=[_ENTITY],
                fixture=_ESTIMATED_EUR,
                date_str=_YEAR,
                value=200.0,
            ),
        }
    )
    slot = value_slot("how", _PROP, "financing source", _VAL_A)
    base_identity = make_identity(_SHAPE, slots=[slot], entities=[subject_entity(_ENTITY)])
    inp = ResubmitIdentityPayload(
        identity=base_identity.model_copy(
            update={
                "facets": {
                    "unit": IdentityFacetValueRequest(
                        request=IdentityUnitRequest(currency="EUR", prices="current")
                    ),
                    "measurementMethod": IdentityFacetValueAbsent(ref=None),
                }
            }
        ),
    )
    captured: list[object] = []
    real_build_observations = promote_finalize._build_observations

    def _spy(*args, **kwargs):
        captured.append(kwargs.get("elected"))
        return real_build_observations(*args, **kwargs)

    with patch.object(promote_finalize, "_build_observations", _spy):
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(
                graph,
                EngineOptions(
                    include_observations=True,
                    materialise_graph=with_member_arcs(materialise_graph, graph),
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=RateTableCache(),
                ),
            ),
        )

    assert region.status == "definite", f"expected definite, got {region.status!r}"
    assert captured and captured[0] is not None
    assert elected_unit_refs(captured[0]) == {_SV_A: frozenset({"CurrencyUSD"})}
    series = region.specs[0].resolved.observations.series
    assert [p.value for s in series for p in s.points] == [pytest.approx(100.0 * 0.9)]


def _resubmit_source_b(extra_facets: dict):
    """Resubmit SourceB alone, whose only facet is euros under the recipe provenance, asking
    for current US dollars, with any other facet pins."""
    graph = FakeGraph(
        nodes=_NODES, obs={f"{_SV_B}|{_ENTITY}": _obs((_YEAR,))[f"{_SV_B}|{_ENTITY}"]}
    )
    materialise_graph = FakeGraphReader(
        arcs={_SV_B: dict(CONVERTIBLE_STAT_VAR_ARCS), **_recipe_arcs()},
        facets={(_SV_B, _ENTITY): [_facet_from_fixture(_EUR_FIXTURE)]},
    )
    materialise_sdmx = _sdmx_router(
        {
            _SV_B: _csv_header(["observationAbout"])
            + _csv_row(
                stat_var=_SV_B,
                role_dcids=[_ENTITY],
                fixture=_EUR_FIXTURE,
                date_str=_YEAR,
                value=50.0,
            ),
        }
    )
    slot = value_slot("how", _PROP, "financing source", _VAL_B)
    base_identity = make_identity(_SHAPE, slots=[slot], entities=[subject_entity(_ENTITY)])
    request = IdentityUnitRequest(currency="USD", prices="current")
    inp = ResubmitIdentityPayload(
        identity=base_identity.model_copy(
            update={"facets": {"unit": IdentityFacetValueRequest(request=request), **extra_facets}}
        ),
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            graph,
            EngineOptions(
                include_observations=True,
                include_option_coverage=True,
                materialise_graph=with_member_arcs(materialise_graph, graph),
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ),
        ),
    )


def test_the_provenance_offered_beside_a_converting_unit_request_serves_when_pinned():
    """The decision elects the euro facet for a current US dollar request, so the facet
    options offer its provenance, and pinning that provenance beside the request answers."""
    region = _resubmit_source_b({})

    assert region.status == "definite", f"expected definite, got {region.status!r}"
    offered = {
        fo.axis: [o.value.ref.ref if o.value.ref is not None else None for o in fo.options]
        for fo in region.specs[0].resolved.facet_options
    }
    assert offered.get("provenance") == [_PROVENANCE_B]

    pinned = _resubmit_source_b({"provenance": IdentityBindingValue(ref=_PROVENANCE_B)})

    assert pinned.status == "definite", f"expected definite, got {pinned.status!r}"


@pytest.mark.parametrize("include_observations", [True, False])
def test_statvar_whose_conversion_fails_refuses_rate_coverage_gap(include_observations):
    """A pound request serves SourceA (neither member holds pounds, and SourceA's provenance
    sorts first on a tie), and with no pound rate its conversion fails `rate_missing`. The
    resubmit's gate finds the gap from its narrowed plan before any read and refuses as a
    fresh request does, with the cause, with or without observations requested."""
    region = _run(include_observations=include_observations, currency="GBP")
    assert region.status == "no_data", (
        f"expected no_data, got {region.status!r} (warnings={region.warnings!r})"
    )
    assert region.no_data_reason == "rate_coverage_gap"
    assert region.cause is not None
    assert (region.cause.code, region.cause.entities) == ("rate_missing", [f"CurrencyGBP@{_YEAR}"])
    # The refusal holds no Spec, so the fetch's spec-stamped warning stays off it, as on a
    # fresh request.
    assert _beside_membership(region) == ()


@pytest.mark.parametrize(("include_observations", "fetches"), [(True, 1), (False, 0)])
def test_the_resubmit_gate_fetches_nothing(include_observations, fetches):
    """The bound-unit gate decides from the narrowed plan, so a resubmit fetches the preview
    once when it is requested and not at all otherwise."""
    real = promote_finalize._build_observations
    calls: list[object] = []

    def _spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    with patch.object(promote_finalize, "_build_observations", _spy):
        region = _run(include_observations=include_observations)

    assert region.status == "definite", f"expected definite, got {region.status!r}"
    assert len(calls) == fetches


def test_an_oversized_preview_answers_definite_with_the_preview_omitted(search_settings):
    """SourceA's two places, the pairs a dollar request serves, over a cap of one: the
    narrowed plan confirms the unit, so the resubmit answers definite and its preview is
    omitted as too large, as it is without a unit."""
    search_settings(obs_preview_max_pairs=1)
    region = _run(include_observations=True, places=(_ENTITY, _UGANDA))
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    observations = region.specs[0].resolved.observations
    assert observations is not None
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PREVIEW_TOO_LARGE


# A fresh request refuses at the bound-unit gate with its accumulated warnings, ground warnings
# included and unstamped because the refusal holds no Spec (`phase_answer.py`'s
# `_gather_definite_coverage`). For this fixture that is exactly the ground warning.
_FRESH_REFUSAL_WARNINGS = (_GROUND_WARNING,)


def _beside_membership(region) -> tuple[Warning, ...]:
    """`region`'s warnings other than `membership_unverified`, which every answer over this
    fixture carries first, on a fresh request and a resubmit alike: its members come from
    the posted StatVars rather than a confirmed walk of their family."""
    return tuple(w for w in region.warnings if w.code != MEMBERSHIP_UNVERIFIED)


@pytest.mark.parametrize("include_observations", [True, False])
def test_bound_unit_no_data_refusal_carries_unstamped_ground_warnings(include_observations):
    """The no_data refusal, a pound request with no pound rate, carries the ground warning
    without a spec id, as a fresh request does for the same evidence."""
    with _grounding_warns():
        region = _run(include_observations=include_observations, currency="GBP")
    assert region.status == "no_data"
    assert region.no_data_reason == "rate_coverage_gap"
    assert region.specs == ()
    assert _beside_membership(region) == _FRESH_REFUSAL_WARNINGS


def test_definite_result_stamps_ground_warnings_with_its_spec():
    """Past the gate the ground warning joins the definite result stamped with its Spec."""
    with _grounding_warns():
        region = _run(include_observations=True)
    assert region.status == "definite"
    stamped = _GROUND_WARNING.model_copy(update={"spec_id": region.specs[0].spec_id})
    assert stamped in region.warnings
    assert _GROUND_WARNING not in region.warnings


def test_bound_unit_no_data_refusal_orders_ground_warning_before_obs_entities_warning():
    """The refusal, a pound request with no pound rate, carries the unstamped ground warning,
    then the observation-entities warning, as a fresh request orders them."""
    with _grounding_warns(), _obs_entities_warns():
        region = _run(include_observations=True, currency="GBP")
    assert region.status == "no_data"
    assert region.no_data_reason == "rate_coverage_gap"
    assert _beside_membership(region) == (_GROUND_WARNING, _OBS_ENTITIES_WARNING)


def test_definite_result_carries_obs_entities_warning_once_in_definite_warnings_order():
    """The definite result holds the observation-entities warning once, ahead of the ground
    warning stamped with its spec id."""
    with _grounding_warns(), _obs_entities_warns():
        region = _run(include_observations=True)
    assert region.status == "definite"
    stamped = _GROUND_WARNING.model_copy(update={"spec_id": region.specs[0].spec_id})
    assert region.warnings.count(_OBS_ENTITIES_WARNING) == 1
    _names_the_left_out_series(region.warnings)
    assert tuple(
        w for w in _beside_membership(region) if w.code != OBSERVATIONS_NO_MATCHING_SOURCE
    ) == (_OBS_ENTITIES_WARNING, stamped)


def _catalogue(rate_years: dict[str, frozenset[int]]) -> ConversionCatalogue:
    """The DAC recipe on `_PROVENANCE_A` and `_PROVENANCE_B`, with rates in `rate_years`."""
    return ConversionCatalogue(
        recipes={_PROVENANCE_A: _RECIPE, _PROVENANCE_B: _RECIPE},
        tables={
            _RECIPE: RecipeTables(
                rate_years=rate_years,
                deflator_years={},
                base_year_min=2019,
                base_year_max=2020,
            )
        },
        national_currencies={},
        default_base_year=2020,
    )


def test_a_rate_gap_shortens_the_period_to_the_years_the_rates_cover():
    """Both series run 2019 to 2020. A pound request serves SourceA (neither member holds
    pounds, and SourceA's provenance sorts first on a tie), and the pound has a rate in 2020
    alone, so converting SourceA's dollars reads 2020. The identity's window, the served
    years and the departure all say so, where without the loaded tables the answer refuses
    rate_coverage_gap."""
    rates = {"CurrencyUSD": {2019: 1.0, 2020: 1.0}, "CurrencyGBP": {2020: 0.8}}
    years = ("2019", "2020")
    with bound_conversion_catalogue(
        _catalogue({"USD": frozenset({2019, 2020}), "GBP": frozenset({2020})})
    ):
        region = _run(include_observations=True, rates=rates, years=years, currency="GBP")

    assert region.status == "definite", f"expected definite, got {region.status!r}"
    spec = region.specs[0]
    window = spec.definition.identity.requested_window
    assert (window.kind, window.start_year, window.end_year) == ("range", 2020, 2020)
    assert {
        s.stat_var.ref: [p.date for p in s.points] for s in spec.resolved.observations.series
    } == {_SV_A: ["2020"]}
    assert DATE_COVERAGE_PARTIAL in [w.code for w in region.warnings]
    (fact,) = [f for f in region.departure_facts.values() if f.code == "year_unavailable"]
    assert (fact.data_years, fact.converted_years, fact.limited_by, fact.shown) == (
        (2019, 2020),
        (2020, 2020),
        "exchange_rates",
        "2020",
    )

    unbound = _run(include_observations=True, rates=rates, years=years, currency="GBP")
    assert unbound.no_data_reason == "rate_coverage_gap"
