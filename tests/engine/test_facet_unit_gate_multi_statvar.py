"""Checks per-StatVar unit availability through parsed and raw-text resolution.

A `kind="set"` binding groups two StatVars in one spec. Each needs a satisfying facet, judged
from the fetch path's availability evidence rather than the size-capped preview.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import httpx

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.facets import FacetSelection, UnitRequest
from materialise.graph import SubjectPage
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.errors import GraphInfraError
from search.engine.region import phase_answer as phase_answer_module
from search.engine.region import resolve_variable
from search.engine.region.bound_unit_gate import bound_unit_gate_reason, plan_unit_evidence
from search.engine.region.codes import (
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_CLIENTS_UNAVAILABLE,
    OBSERVATIONS_SERIES_TRUNCATED,
    OBSERVATIONS_TIMED_OUT,
)
from search.engine.region.state import EngineOptions
from search.models import IdentityUnitRequest
from search.observations import OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED
from series_coverage.schema import FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    door_roles_for_test_query,
    materialise_double_from_obs,
    preview_and_plan,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_RECIPE = "ConversionRecipeDAC"
_SOURCE_PROVENANCE = "dc/base/Src"

_PROP = "financingSource"
_VAL_A = "SourceA"
_VAL_B = "SourceB"

_FIVE_TUPLE_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "populationType": {"nodes": [{"dcid": "TestUnitGateActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testUnitGateExpenditure"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    # A bound unit request runs region/convertibility.py's own decision,
    # which needs check_nominal to pass for a native `{USD,current}` match --
    # declared here (not only on materialise_graph's own copy) so `graph`'s
    # own re-derived five-tuple (region/candidates.py's enumerate_members,
    # via _group_preview_sv_dcids) agrees with materialise_graph's, rather
    # than the two graphs disagreeing on this family's own qualifier.
    "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
}


def _sv_arcs(label: str, value: str) -> dict:
    return {
        **_FIVE_TUPLE_ARCS,
        "name": {"nodes": [{"value": label}]},
        "constraintProperties": {"nodes": [{"dcid": _PROP}]},
        _PROP: {"nodes": [{"dcid": value}]},
    }


def _std_arcs(label: str, *, meas_prop: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "TestUnitGateActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


_SV_A = "Test_UnitGate_SourceA"
_SV_B = "Test_UnitGate_SourceB"
_SV_C = "Test_UnitGate_Alt"
_QUERY = "unit gate multi statvar test variable in Kenya"

_NODES = {
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _SV_A: {
        "label": "Unit gate SourceA",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs("Unit gate SourceA", _VAL_A),
    },
    _SV_B: {
        "label": "Unit gate SourceB",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs("Unit gate SourceB", _VAL_B),
    },
    _SV_C: {
        "label": "Unit gate alternate",
        "type": "StatisticalVariable",
        "arcs": _std_arcs("Unit gate alternate", meas_prop="testUnitGateAlt"),
    },
}


def _member_arcs(node_arcs: dict) -> dict[str, tuple[str, ...]]:
    """This fixture's own `{"prop": {"nodes": [{"dcid"|"value": ...}, ...]}}`
    node-arc shape (as posted to `FakeGraph(nodes=...)` above), flattened into
    the `{property: (dcid_or_value, ...)}` tuple form
    `statvar_members.MemberArcReader` (and `FakeGraphReader`'s own `arcs=`)
    expect -- the SAME real graph state `graph` and `materialise_graph` both
    read, materialise_double_from_obs' own facets-only constructor just
    doesn't populate this half of it."""
    return {
        prop: tuple(n.get("dcid") or n.get("value") for n in entry["nodes"])
        for prop, entry in node_arcs.items()
    }


def _set_bind_llm() -> object:
    class _SetBindLLM:
        async def generate_structured(self, *, prompt, system, schema):
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid=_PROP,
                        kind="set",
                        value_dcids=[_VAL_A, _VAL_B],
                    )
                ],
                also_plausible=None,
                ask=None,
            ), None

    return _SetBindLLM()


def _obs_facet(*, unit: str, value: float, provenance_id: str = _SOURCE_PROVENANCE) -> dict:
    return {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": provenance_id,
        "unit": unit,
        "dates": ["2020"],
        "values": [value],
    }


_USD_ONLY = {"CurrencyUSD": {2020: 1.0}}


def _sdmx_with_rate_anchor(
    obs: dict, rates: dict[str, dict[int, float]] = _USD_ONLY
) -> MaterialiseSdmxClient:
    """Serves the observations in `obs` and an exchange-rate table holding `rates`, by
    default only `CurrencyUSD`, whose absent `CurrencyEUR` row is a rate gap for SV_B.
    """
    bodies: dict[str, str] = {}
    for key, facet_dicts in obs.items():
        sv, entity = key.split("|", 1)
        body = bodies.get(sv, _csv_header(["observationAbout"]))
        for i, facet_dict in enumerate(facet_dicts):
            fixture = ObsFacetFixture(
                facet_id=facet_dict.get("facetId") or f"f{i}",
                unit=facet_dict.get("unit"),
                provenance_id=facet_dict.get("provenanceId"),
            )
            body += _csv_row(
                stat_var=sv,
                role_dcids=[entity],
                fixture=fixture,
                date_str=facet_dict["dates"][0],
                value=facet_dict["values"][0],
            )
        bodies[sv] = body

    def observations(request: httpx.Request) -> httpx.Response:
        sv = request.url.params.get("c[variableMeasured]", "")
        body = bodies.get(sv, _csv_header(["observationAbout"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(
        recipe_table_bodies(_RECIPE, rates=rates),
        fallback=observations,
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


def _run(**kwargs):
    """Resolve the test query over `obs` (`_resolution`)."""
    return asyncio.run(_resolution(**kwargs))


def _resolution(
    *,
    obs: dict,
    unit_text: str | None,
    llm=None,
    also_plausible_svs: dict | None = None,
    include_option_coverage: bool = False,
    sdmx_obs: dict | None = None,
    rates: dict[str, dict[int, float]] = _USD_ONLY,
    include_observations: bool = True,
    stream_observations: bool = False,
    nodes: dict = _NODES,
):
    """The resolve of the test query over `obs` and `nodes`; `sdmx_obs` overrides the rows the
    SDMX double serves and `rates` the exchange-rate table."""
    detect_svs = [_SV_A, _SV_B]
    cosine_scores = [0.95, 0.9]
    if also_plausible_svs is not None:
        detect_svs = [*detect_svs, *also_plausible_svs]
        cosine_scores = [*cosine_scores, *[0.85] * len(also_plausible_svs)]
    graph = FakeGraph(
        nodes=nodes,
        detect={_QUERY: {"svs": detect_svs, "cosine_scores": cosine_scores}},
        resolve={"Kenya": "country/KEN"},
        obs=obs,
    )
    materialise_graph, _unused_sdmx = materialise_double_from_obs(obs)
    materialise_sdmx = _sdmx_with_rate_anchor(obs if sdmx_obs is None else sdmx_obs, rates)
    # Nominal facets and a DAC conversion recipe let unit planning reach the availability gate.
    materialise_graph._arcs = {
        _SV_A: {**_member_arcs(_NODES[_SV_A]["arcs"]), **CONVERTIBLE_STAT_VAR_ARCS},
        _SV_B: {**_member_arcs(_NODES[_SV_B]["arcs"]), **CONVERTIBLE_STAT_VAR_ARCS},
        **conversion_arcs(
            recipe_id=_RECIPE,
            provenance_id=_SOURCE_PROVENANCE,
            currencies=["CurrencyUSD", "CurrencyEUR"],
        ),
    }
    materialise_graph._subjects = {
        ("testUnitGateExpenditure", "measuredProperty"): SubjectPage(
            dcids=(_SV_A, _SV_B), truncated=False
        ),
    }
    return resolve_variable(
        _QUERY,
        entities=["Kenya"],
        date_request=None,
        detect_query=_QUERY,
        role_query=_QUERY,
        pac=True,
        graph=graph,
        resolver=make_resolver(graph),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        llm=llm if llm is not None else _set_bind_llm(),
        base_timing={},
        unit_text=unit_text,
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            include_observations=include_observations,
            stream_observations=stream_observations,
            include_option_coverage=include_option_coverage,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        ),
    )


def test_single_shape_rate_gap_refuses_rate_coverage_gap():
    """SV_B's only facet is EUR, which the recipe converts to USD, and the rate table holds
    no EUR rate for 2020. The gate finds the gap from the plan before any read and refuses
    with the cause the conversion would raise."""
    obs = {
        f"{_SV_A}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=10.0)],
        f"{_SV_B}|country/KEN": [_obs_facet(unit="CurrencyEUR", value=5.0)],
    }
    result = _run(obs=obs, unit_text="US dollars")
    assert result.status == "no_data"
    assert result.no_data_reason == "rate_coverage_gap"
    assert result.cause is not None
    assert (result.cause.code, result.cause.entities) == ("rate_missing", ["CurrencyEUR@2020"])


def test_a_covered_conversion_answers_definite_without_an_observation_read(monkeypatch):
    """With a 2020 euro rate SV_B's euro facet converts, so the gate passes from the plan
    and the rate table alone; an answer without a preview reads no observation."""

    def _no_read(*args, **kwargs):
        raise AssertionError("the bound-unit gate must not read observations")

    monkeypatch.setattr(phase_answer_module, "_build_observations", _no_read)
    obs = {
        f"{_SV_A}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=10.0)],
        f"{_SV_B}|country/KEN": [_obs_facet(unit="CurrencyEUR", value=5.0)],
    }
    result = _run(
        obs=obs,
        unit_text="US dollars",
        rates={"CurrencyUSD": {2020: 1.0}, "CurrencyEUR": {2020: 0.9}},
        include_observations=False,
    )
    assert result.status == "definite", result
    assert {sv.ref.dcid for sv in result.specs[0].resolved.members} == {_SV_A, _SV_B}


def test_a_unit_gate_refusal_never_probes_coverage(monkeypatch):
    """The unit gate returns before coverage probes.

    Both probe entry points raise if reached.
    """

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("standard_option_coverage must not be reached")

    monkeypatch.setattr(
        "search.engine.region.option_coverage.standard_option_coverage", _fail_if_called
    )

    def _fail_if_facet_called(*args, **kwargs):
        raise AssertionError("facet_options probe must not be reached")

    monkeypatch.setattr(
        "search.engine.region.option_coverage._facet_options", _fail_if_facet_called
    )

    obs = {
        f"{_SV_A}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=10.0)],
        f"{_SV_B}|country/KEN": [_obs_facet(unit="CurrencyEUR", value=5.0)],
    }
    result = _run(obs=obs, unit_text="US dollars", include_option_coverage=True)
    assert result.status == "no_data"
    assert result.no_data_reason == "rate_coverage_gap"


def test_single_shape_full_coverage_answers_definite_with_all_series():
    """Both StatVars satisfy the USD request and contribute a series."""
    obs = {
        f"{_SV_A}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=10.0)],
        f"{_SV_B}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=20.0)],
    }
    result = _run(obs=obs, unit_text="US dollars")
    assert result.status == "definite", f"expected definite, got {result.status!r}"
    spec = result.specs[0]
    assert {sv.ref.dcid for sv in spec.resolved.members} == {_SV_A, _SV_B}
    assert spec.resolved.observations is not None
    assert {s.stat_var.ref for s in spec.resolved.observations.series} == {_SV_A, _SV_B}


def test_single_shape_empty_read_answers_from_the_plan():
    """The coverage index holds SV_B's USD facet at Kenya, but its read returns no rows. The
    gate decides from the plan, which serves both StatVars, so the answer is definite and
    its preview holds SV_A's series alone, as an answer without a unit request does."""
    obs = {
        f"{_SV_A}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=10.0)],
        f"{_SV_B}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=20.0)],
    }
    result = _run(
        obs=obs,
        unit_text="US dollars",
        sdmx_obs={f"{_SV_A}|country/KEN": obs[f"{_SV_A}|country/KEN"]},
    )
    assert result.status == "definite", (
        f"expected definite, got {result.status!r} (no_data_reason={result.no_data_reason!r})"
    )
    spec = result.specs[0]
    assert {sv.ref.dcid for sv in spec.resolved.members} == {_SV_A, _SV_B}
    assert spec.resolved.observations is not None
    assert {s.stat_var.ref for s in spec.resolved.observations.series if s.total_obs} == {_SV_A}


class _SetPlusAltBindLLM:
    """Elects the SET group (SV_A/SV_B, index 0) and flags the plain
    aggregate alt shape (SV_C, index 1) as also_plausible -- the candidates
    fan-out this drop test exercises."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP,
                    kind="set",
                    value_dcids=[_VAL_A, _VAL_B],
                )
            ],
            also_plausible=[1],
            ask=None,
        ), None


def test_candidate_with_partial_coverage_is_dropped_leaving_the_full_alternate():
    """The per-StatVar filter drops the partial USD set and keeps the fully covered SV_C alternate."""
    obs = {
        f"{_SV_A}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=10.0)],
        f"{_SV_B}|country/KEN": [_obs_facet(unit="CurrencyEUR", value=5.0)],
        f"{_SV_C}|country/KEN": [_obs_facet(unit="CurrencyUSD", value=20.0)],
    }
    result = _run(
        obs=obs,
        unit_text="US dollars",
        llm=_SetPlusAltBindLLM(),
        also_plausible_svs=[_SV_C],
    )
    assert result.status == "definite", (
        f"expected the partial-coverage SET candidate to be dropped, collapsing to a "
        f"lone-survivor definite; got status={result.status!r}"
    )
    spec = result.specs[0]
    assert {sv.ref.dcid for sv in spec.resolved.members} == {_SV_C}


# A role-bearing bulk scope: the role-pair product exceeds `obs_preview_max_pairs`, so the
# preview ships at most `role_bulk_series_max` pairs. Availability comes from the coverage index.

_ROLE_SV_A = "Test_UnitGateRole_SourceA"
_ROLE_SV_B = "Test_UnitGateRole_SourceB"
_ROLE_QUERY = "unit gate role aid from Norway to Kenya and Uganda"
_NOR = "country/NOR"
_KEN = "country/KEN"
_UGA = "country/UGA"
_RECIPIENTS = (_KEN, _UGA)
_ROLE_PROVENANCE = "dc/base/UnitGateRoleSrc"
_ROLE_FACET = "FRoleUSD"
_EUR_RATES = {"CurrencyUSD": {2020: 1.0}, "CurrencyEUR": {2020: 0.9}}


def _role_sv_arcs(label: str, value: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "testUnitGateRoleAmount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
        "constraintProperties": {"nodes": [{"dcid": _PROP}]},
        _PROP: {"nodes": [{"dcid": value}]},
    }


_ROLE_NODES = {
    _NOR: {"label": "Norway", "type": "Country"},
    _KEN: {"label": "Kenya", "type": "Country"},
    _UGA: {"label": "Uganda", "type": "Country"},
    _ROLE_SV_A: {
        "label": "Unit gate role SourceA",
        "type": "StatisticalVariable",
        "arcs": _role_sv_arcs("Unit gate role SourceA", _VAL_A),
    },
    _ROLE_SV_B: {
        "label": "Unit gate role SourceB",
        "type": "StatisticalVariable",
        "arcs": _role_sv_arcs("Unit gate role SourceB", _VAL_B),
    },
}


@dataclass(frozen=True)
class _RoleFacet:
    """A USD facet holding one point a year in `years` at every recipient."""

    facet_id: str = _ROLE_FACET
    provenance_id: str = _ROLE_PROVENANCE
    years: tuple[int, ...] = (2020,)


type _RoleFacets = Mapping[str, Sequence[_RoleFacet]]


def _one_usd_facet(covered_svs: tuple[str, ...]) -> _RoleFacets:
    return {sv: (_RoleFacet(),) for sv in covered_svs}


def _role_coverage(
    monkeypatch, covered_svs: tuple[str, ...], facets: _RoleFacets | None = None
) -> CoverageSnapshot:
    """Coverage confirming each StatVar in `covered_svs` at both recipients under its
    `facets`, by default one USD facet."""
    facets = facets if facets is not None else _one_usd_facet(covered_svs)
    index = patch_coverage(
        monkeypatch,
        [
            (sv, f.facet_id, _NOR, r, list(f.years))
            for sv in covered_svs
            for f in facets[sv]
            for r in _RECIPIENTS
        ],
        role_names={
            sv: RoleNames(donor="provider", recipient="recipient")
            for sv in (_ROLE_SV_A, _ROLE_SV_B)
        },
        facet_metadata={
            f.facet_id: FacetMetadata(provenance_id=f.provenance_id, unit="CurrencyUSD")
            for per_sv in facets.values()
            for f in per_sv
        },
    )
    return FakeCoverageService(index).snapshot()


def _role_sdmx(
    covered_svs: tuple[str, ...],
    *,
    raise_series: BaseException | None = None,
    facets: _RoleFacets | None = None,
) -> FakeSdmxClient:
    """The bulk series read: a 1.0 point per covered pair in each year of its `facets`, by
    default one 2020 USD point, or `raise_series`."""
    facets = facets if facets is not None else _one_usd_facet(covered_svs)
    return FakeSdmxClient(
        raise_series=raise_series,
        observations={
            sv: {
                (("provider", _NOR), ("recipient", r)): [
                    Facet(
                        earliest_date=str(f.years[0]),
                        latest_date=str(f.years[-1]),
                        obs_count=len(f.years),
                        dates=tuple(str(year) for year in f.years),
                        values=(1.0,) * len(f.years),
                        facet_id=f.facet_id,
                        provenance_id=f.provenance_id,
                        unit="CurrencyUSD",
                    )
                    for f in facets[sv]
                ]
                for r in _RECIPIENTS
            }
            for sv in covered_svs
        },
    )


def _run_role_bulk(
    monkeypatch,
    *,
    unit_text: str,
    rates: dict[str, dict[int, float]],
    raise_series: BaseException | None = None,
    facets: _RoleFacets | None = None,
    recipe_provenances: Sequence[str] = (_ROLE_PROVENANCE,),
):
    """Resolve a two-StatVar role-bearing ask over Norway to Kenya and Uganda.

    Coverage confirms both StatVars at both recipients under their `facets`, by default one
    native USD facet, and the materialise reads serve the same pairs. Data under each of
    `recipe_provenances` converts with the DAC recipe. The caller lowers the pair caps.
    """
    covered_svs = (_ROLE_SV_A, _ROLE_SV_B)
    facets = facets if facets is not None else _one_usd_facet(covered_svs)
    coverage = _role_coverage(monkeypatch, covered_svs, facets)
    body = _csv_header(["provider", "recipient"]) + "".join(
        _csv_row(
            stat_var=sv,
            role_dcids=[_NOR, r],
            fixture=ObsFacetFixture(
                facet_id=f.facet_id,
                points=tuple((str(year), 1.0) for year in f.years),
                provenance_id=f.provenance_id,
                unit="CurrencyUSD",
            ),
            date_str=str(year),
            value=1.0,
        )
        for sv in covered_svs
        for f in facets[sv]
        for year in f.years
        for r in _RECIPIENTS
    )

    def observations(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(recipe_table_bodies(_RECIPE, rates=rates), fallback=observations)
    materialise_graph = FakeGraphReader(
        arcs={
            _ROLE_SV_A: _member_arcs(_ROLE_NODES[_ROLE_SV_A]["arcs"]),
            _ROLE_SV_B: _member_arcs(_ROLE_NODES[_ROLE_SV_B]["arcs"]),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_ROLE_PROVENANCE,
                currencies=["CurrencyUSD", "CurrencyEUR"],
            ),
            **{p: {"conversionRecipe": (_RECIPE,)} for p in recipe_provenances},
        },
        subjects={
            ("testUnitGateRoleAmount", "measuredProperty"): SubjectPage(
                dcids=(_ROLE_SV_A, _ROLE_SV_B), truncated=False
            ),
        },
    )
    graph = FakeGraph(
        nodes=_ROLE_NODES,
        detect={_ROLE_QUERY: {"svs": [_ROLE_SV_A, _ROLE_SV_B], "cosine_scores": [0.95, 0.9]}},
        resolve={"Norway": _NOR, "Kenya": _KEN, "Uganda": _UGA},
        obs={},
    )
    entities = ["Norway", "Kenya", "Uganda"]
    return asyncio.run(
        resolve_variable(
            "unit gate role aid",
            entities=entities,
            date_request=None,
            detect_query=_ROLE_QUERY,
            role_query=_ROLE_QUERY,
            entity_roles=door_roles_for_test_query(_ROLE_QUERY, entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=_role_sdmx(covered_svs, raise_series=raise_series, facets=facets),
            llm=_set_bind_llm(),
            base_timing={},
            unit_text=unit_text,
            coverage=coverage,
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=server.client(base_url=MATERIALISE_BASE_URL),
                rate_cache=RateTableCache(),
            ),
        )
    )


def test_role_bulk_scope_answers_definite_when_every_statvar_is_confirmed(
    monkeypatch, search_settings
):
    """The preview shows one pair of SV_A, yet coverage confirms both StatVars: definite."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=1)
    result = _run_role_bulk(monkeypatch, unit_text="US dollars", rates=_EUR_RATES)
    assert result.status == "definite", (
        f"expected definite, got {result.status!r} (no_data_reason={result.no_data_reason!r})"
    )
    spec = result.specs[0]
    assert {sv.ref.dcid for sv in spec.resolved.members} == {_ROLE_SV_A, _ROLE_SV_B}
    observations = spec.resolved.observations
    assert observations is not None
    assert {s.stat_var.ref for s in observations.series} == {_ROLE_SV_A}
    assert any(w.code == OBSERVATIONS_SERIES_TRUNCATED for w in result.warnings)


def test_role_bulk_display_read_failure_answers_from_coverage(monkeypatch, search_settings):
    """Coverage confirms both StatVars, so a failed or timed-out series read is a display failure:
    the answer is definite with the preview omitted and the failure's warning, not a 503."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=1)
    for exc, code in (
        (GraphInfraError("simulated SDMX failure"), OBSERVATIONS_BULK_FETCH_FAILED),
        (TimeoutError("simulated shared-deadline crossing"), OBSERVATIONS_TIMED_OUT),
    ):
        result = _run_role_bulk(
            monkeypatch, unit_text="US dollars", rates=_EUR_RATES, raise_series=exc
        )
        assert result.status == "definite", (
            f"{code}: expected definite, got {result.status!r} "
            f"(no_data_reason={result.no_data_reason!r})"
        )
        observations = result.specs[0].resolved.observations
        assert observations is not None
        assert observations.series == []
        assert observations.omitted is not None
        assert observations.omitted.reason == code
        assert any(w.code == code for w in result.warnings)


def test_converted_role_bulk_scope_previews_converted_points(monkeypatch, search_settings):
    """A converted unit over a bulk role read converts the points it shows. One 2020 dollar
    at 0.9 euros per dollar is 0.9 euros, stated in euros with its recipe."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=1)
    result = _run_role_bulk(monkeypatch, unit_text="euros", rates=_EUR_RATES)
    assert result.status == "definite", (
        f"expected definite, got {result.status!r} (no_data_reason={result.no_data_reason!r})"
    )
    spec = result.specs[0]
    assert {sv.ref.dcid for sv in spec.resolved.members} == {_ROLE_SV_A, _ROLE_SV_B}
    observations = spec.resolved.observations
    assert observations is not None
    assert observations.omitted is None
    (series,) = observations.series
    assert [(point.date, point.value) for point in series.points] == [("2020", 0.9)]
    assert series.facet["unit"].ref == "CurrencyEUR"
    assert series.conversion is not None
    assert series.conversion.recipe_id == _RECIPE
    assert not any(w.code == OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED for w in result.warnings)


_SHARED_PROVENANCE = "dc/base/UnitGateSharedSrc"

# Each StatVar holds its own facet under the recipe's provenance over 2019 and 2020, and both
# hold a shared facet under another provenance in 2020. Each StatVar on its own ranks its own
# facet first, on its years; merged across both, the shared facet covers four pairs to each
# own facet's two, so the bulk read serves the shared facet.
_SPLIT_FACETS = {
    sv: (
        _RoleFacet(facet_id=f"FRoleOwn{i}", years=(2019, 2020)),
        _RoleFacet("FRoleShared", _SHARED_PROVENANCE),
    )
    for i, sv in enumerate((_ROLE_SV_A, _ROLE_SV_B))
}


def test_role_bulk_converts_every_statvar_under_the_provenance_the_read_serves(
    monkeypatch, search_settings
):
    """Each StatVar's unit decision ranks its own facet, but the bulk read serves the shared
    one, whose provenance has a recipe. The gate checks the shared facet's rows, so the
    conversion converts them under the shared provenance: both StatVars convert and show,
    none left out for an election under another provenance."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=4)
    result = _run_role_bulk(
        monkeypatch,
        unit_text="euros",
        rates=_EUR_RATES,
        facets=_SPLIT_FACETS,
        recipe_provenances=(_SHARED_PROVENANCE,),
    )
    assert result.status == "definite", (
        f"expected definite, got {result.status!r} (no_data_reason={result.no_data_reason!r})"
    )
    assert not any(w.code == OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED for w in result.warnings)
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert observations.omitted is None
    assert sorted(
        (s.stat_var.ref, s.facet["provenance"].ref, s.facet["unit"].ref)
        for s in observations.series
    ) == sorted(
        (sv, _SHARED_PROVENANCE, "CurrencyEUR")
        for sv in (_ROLE_SV_A, _ROLE_SV_B)
        for _ in _RECIPIENTS
    )


def test_a_role_bulk_conversion_names_the_statvar_it_leaves_out(monkeypatch, search_settings):
    """SV_B declares no Nominal qualifier, so its conversion refuses: SV_A's pairs convert and
    show, and a warning names SV_B under `observations_converted_scope_unsupported`."""
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=4)
    covered_svs = (_ROLE_SV_A, _ROLE_SV_B)
    unit = FacetSelection(
        axis="unit",
        ref="CurrencyUSD",
        basis="deployment_fallback",
        default_policy=None,
        default_policy_version=None,
        selection_reason="most_observations",
        dispatch_facet_id=_ROLE_FACET,
        request=UnitRequest(currency="EUR", prices="current"),
    )
    source = FacetSelection(
        axis="provenance",
        ref=_ROLE_PROVENANCE,
        basis="deployment_fallback",
        default_policy=None,
        default_policy_version=None,
        selection_reason=None,
        dispatch_facet_id=_ROLE_FACET,
    )
    sv_b_arcs = _member_arcs(_ROLE_NODES[_ROLE_SV_B]["arcs"])
    del sv_b_arcs["measurementQualifier"]
    server = RateTableServer(recipe_table_bodies(_RECIPE, rates=_EUR_RATES))
    (observations, warn), _plan = preview_and_plan(
        list(covered_svs),
        {"provider": [_NOR], "recipient": list(_RECIPIENTS)},
        graph=FakeGraph(nodes=_ROLE_NODES, obs={}),
        materialise_graph=FakeGraphReader(
            arcs={
                _ROLE_SV_A: _member_arcs(_ROLE_NODES[_ROLE_SV_A]["arcs"]),
                _ROLE_SV_B: sv_b_arcs,
                **conversion_arcs(
                    recipe_id=_RECIPE,
                    provenance_id=_ROLE_PROVENANCE,
                    currencies=["CurrencyUSD", "CurrencyEUR"],
                ),
            }
        ),
        materialise_sdmx=server.client(base_url=MATERIALISE_BASE_URL),
        rate_cache=RateTableCache(),
        sdmx=_role_sdmx(covered_svs),
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        elected={sv: ({"unit": unit, "provenance": source},) for sv in covered_svs},
        coverage=_role_coverage(monkeypatch, covered_svs),
    )

    assert observations.omitted is None
    assert {s.stat_var.ref for s in observations.series} == {_ROLE_SV_A}
    assert [[(p.date, p.value) for p in s.points] for s in observations.series] == [
        [("2020", 0.9)]
    ] * len(_RECIPIENTS)
    assert warn is not None
    assert warn.code == OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED
    assert _ROLE_SV_B in warn.message


# In the pipeline the unit election reads the same coverage index, so a StatVar the index lacks
# never reaches the gate. These tests feed the gate a real bulk fetch over such an index.


def _usd_election(*, converted: bool) -> dict[str, tuple[dict[str, FacetSelection], ...]]:
    """Both StatVars elect the USD facet, natively or as the source of a euro conversion."""
    selection = FacetSelection(
        axis="unit",
        ref="CurrencyUSD",
        basis="deployment_fallback",
        default_policy=None,
        default_policy_version=None,
        selection_reason="most_observations" if converted else "native_currency_preferred",
        dispatch_facet_id=_ROLE_FACET,
        request=UnitRequest(currency="EUR" if converted else "USD", prices="current"),
    )
    return {sv: ({"unit": selection},) for sv in (_ROLE_SV_A, _ROLE_SV_B)}


def _role_bulk_gate(monkeypatch, *, covered_svs: tuple[str, ...], converted: bool):
    """Run the role-bearing bulk fetch for both StatVars, then the single-shape gate."""
    obs_entities = {"provider": [_NOR], "recipient": list(_RECIPIENTS)}
    sdmx = _role_sdmx(covered_svs)
    preview, plan = preview_and_plan(
        [_ROLE_SV_A, _ROLE_SV_B],
        obs_entities,
        graph=FakeGraph(nodes=_ROLE_NODES, obs={}),
        materialise_graph=None,
        materialise_sdmx=None,
        sdmx=sdmx,
        unit_request=IdentityUnitRequest(currency="EUR" if converted else "USD", prices="current"),
        elected=_usd_election(converted=converted),
        coverage=_role_coverage(monkeypatch, covered_svs),
    )
    reason = bound_unit_gate_reason(plan, obs_entities, [_ROLE_SV_A, _ROLE_SV_B])
    return preview[0], plan_unit_evidence(plan), reason, sdmx


def test_role_bulk_gate_refuses_a_statvar_absent_from_coverage(monkeypatch, search_settings):
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=1)
    _obs, evidence, reason, _sdmx = _role_bulk_gate(
        monkeypatch, covered_svs=(_ROLE_SV_A,), converted=False
    )
    assert evidence.confirmed == {_ROLE_SV_A}
    assert reason == "facet_not_available"


def test_converted_role_bulk_gate_confirms_from_the_native_index_without_a_read(
    monkeypatch, search_settings
):
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=1)
    observations, evidence, reason, sdmx = _role_bulk_gate(
        monkeypatch, covered_svs=(_ROLE_SV_A, _ROLE_SV_B), converted=True
    )
    # Without the materialise clients the preview cannot convert, so it reads nothing.
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_CLIENTS_UNAVAILABLE
    assert evidence.confirmed == {_ROLE_SV_A, _ROLE_SV_B}
    assert reason is None
    assert sdmx.series_calls == []


def test_converted_role_bulk_gate_refuses_a_statvar_with_no_native_pairs(
    monkeypatch, search_settings
):
    search_settings(obs_preview_max_pairs=1, role_bulk_series_max=1)
    _obs, evidence, reason, _sdmx = _role_bulk_gate(
        monkeypatch, covered_svs=(_ROLE_SV_A,), converted=True
    )
    assert evidence.confirmed == {_ROLE_SV_A}
    assert reason == "facet_not_available"
