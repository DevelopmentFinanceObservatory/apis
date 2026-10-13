"""End-to-end tests for per-candidate unit election, conversion, and facet-probe reuse.
USD and GBP candidates retain their own native units under one USD request."""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import replace

import httpx
import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import member_conversion as member_conversion_module
from materialise.errors import (
    DatasetDisagreementError,
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MembersNotHeldError,
    OverCeilingError,
    UpstreamVersionChangedError,
)
from materialise.facets import FacetSelection, subject_probe
from materialise.rate_tables import RateTableCache, RateTableInvalidError
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import NonCurrencyUnitError, RateCoverageGapError, RecipeNotDeclaredError
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.errors import EngineInputError
from search.engine.quantity_read import QuantityOutput
from search.engine.region import candidate_grounding as candidate_grounding_module
from search.engine.region import candidate_unit_election as candidate_unit_election_module
from search.engine.region import convertibility as convertibility_module
from search.engine.region import phase_election as phase_election_module
from search.engine.region import promote_ground as promote_ground_module
from search.engine.region import resolve_spec_resubmit
from search.engine.region.candidate_unit_election import (
    CandidateUnitElections,
    _candidate_unit_elections,
)
from search.engine.region.convertibility import UnitCoverage
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.engine.spec_id import mint_identity
from search.models import (
    GraphRef,
    IdentityUnitRequest,
    ResolvedQualifier,
    ResubmitIdentityPayload,
)
from tests.engine._fake_coverage import build_coverage_index, empty_coverage_snapshot
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
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_SV_A = "TESTFIXTURE/CandUnitElection/AmountA"
_SV_B = "TESTFIXTURE/CandUnitElection/AmountB"
_KEN = "country/KEN"
_QUERY = "candidate unit election amount in Kenya"

_PROV_A = "dc/base/CandUnitElectionSrcA"
_PROV_B = "dc/base/CandUnitElectionSrcB"
_RECIPE = "ConversionRecipeDAC"
_TABLE_PROVENANCE = RECIPE_BODIES[_RECIPE].table_provenance


def _sv_arcs(*, label: str, meas_prop: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        # The flow population and qualifier the DAC recipe converts, as materialise reads them.
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
    }


_NODES = {
    _KEN: {"label": "Kenya", "type": "Country"},
    _SV_A: {"label": "Amount A", "arcs": _sv_arcs(label="Amount A", meas_prop="candUnitElectionA")},
    _SV_B: {"label": "Amount B", "arcs": _sv_arcs(label="Amount B", meas_prop="candUnitElectionB")},
}


class _FixedBindLLM:
    """Elects the first shape and marks the second as also plausible, or with `alone`
    elects the first shape by itself."""

    def __init__(self, *, alone: bool = False) -> None:
        self.also_plausible = [] if alone else [1]

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=self.also_plausible, ask=None
        ), None


_USD_FIXTURE_A = ObsFacetFixture(
    facet_id="pv_a", points=(("2020", 100.0),), provenance_id=_PROV_A, unit="CurrencyUSD"
)
_GBP_FIXTURE_B = ObsFacetFixture(
    facet_id="pv_b", points=(("2023", 50.0),), provenance_id=_PROV_B, unit="CurrencyGBP"
)


_RATES = {"CurrencyUSD": {2020: 1.0, 2023: 1.0}, "CurrencyGBP": {2020: 0.8, 2023: 0.8}}


def _materialise_sdmx(
    *, b: ObsFacetFixture = _GBP_FIXTURE_B, rates: dict[str, dict[int, float]] = _RATES
) -> MaterialiseSdmxClient:
    """Serve each StatVar's rows, `_SV_B`'s from `b`, beside the DAC rate table of `rates`."""
    fetch_bodies = {
        _SV_A: _csv_header(["observationAbout"])
        + _csv_row(
            stat_var=_SV_A, role_dcids=[_KEN], fixture=_USD_FIXTURE_A, date_str="2020", value=100.0
        ),
        _SV_B: _csv_header(["observationAbout"])
        + "".join(
            _csv_row(stat_var=_SV_B, role_dcids=[_KEN], fixture=b, date_str=date, value=value)
            for date, value in b.points
        ),
    }

    def observations(request: httpx.Request) -> httpx.Response:
        sv = request.url.params.get("c[variableMeasured]", "")
        body = fetch_bodies.get(sv, "")
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(recipe_table_bodies(_RECIPE, rates=rates), fallback=observations)
    return server.client(base_url=MATERIALISE_BASE_URL)


def _materialise_graph(b: ObsFacetFixture = _GBP_FIXTURE_B) -> FakeGraphReader:
    """Provides real facets so both candidates survive the full resolve."""
    return FakeGraphReader(
        arcs={
            _SV_A: dict(CONVERTIBLE_STAT_VAR_ARCS),
            _SV_B: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROV_B,
                currencies=["CurrencyUSD", "CurrencyGBP"],
            ),
        },
        facets={
            (_SV_A, _KEN): [_facet_from_fixture(_USD_FIXTURE_A)],
            (_SV_B, _KEN): [_facet_from_fixture(b)],
        },
    )


def _materialise_graph_no_facets() -> FakeGraphReader:
    """Leaves facets empty so direct election tests can detect any fresh probe."""
    return FakeGraphReader(
        arcs={
            _SV_A: dict(CONVERTIBLE_STAT_VAR_ARCS),
            _SV_B: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROV_B,
                currencies=["CurrencyUSD", "CurrencyGBP"],
            ),
        },
    )


class _SummaryReadGraph(FakeGraph):
    """Serves each facet as the live graph does: with its dates to a dated read alone."""

    def observation_facets(self, *, stat_var, entity, needs_dates=False):
        facets = super().observation_facets(
            stat_var=stat_var, entity=entity, needs_dates=needs_dates
        )
        return facets if needs_dates else [replace(f, dates=(), values=()) for f in facets]


def _graph(
    *,
    b: ObsFacetFixture = _GBP_FIXTURE_B,
    svs: tuple[str, ...] = (_SV_A, _SV_B),
    summary_reads: bool = False,
) -> FakeGraph:
    """The Search graph detecting `svs` for `_QUERY`, with `_SV_B`'s facet from `b`."""
    graph = _SummaryReadGraph if summary_reads else FakeGraph
    return graph(
        nodes=_NODES,
        detect={_QUERY: {"svs": list(svs), "cosine_scores": [0.9, 0.85][: len(svs)]}},
        resolve={"Kenya": _KEN},
        obs={
            f"{_SV_A}|{_KEN}": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "facetId": "pv_a",
                    "provenanceId": _PROV_A,
                    "unit": "CurrencyUSD",
                    "dates": ["2020"],
                    "values": [100.0],
                }
            ],
            f"{_SV_B}|{_KEN}": [
                {
                    "earliestDate": b.points[0][0],
                    "latestDate": b.points[-1][0],
                    "obsCount": len(b.points),
                    "facetId": b.facet_id,
                    "provenanceId": b.provenance_id,
                    "unit": b.unit,
                    "dates": [date for date, _value in b.points],
                    "values": [value for _date, value in b.points],
                }
            ],
        },
    )


def _run(*, materialise_graph: FakeGraphReader | None = None):
    return asyncio.run(
        resolve_variable(
            "amount",
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=_graph(),
            resolver=make_resolver(_graph()),
            llm=_FixedBindLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            base_timing={},
            unit_text="USD",
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph
                if materialise_graph is not None
                else _materialise_graph(),
                materialise_sdmx=_materialise_sdmx(),
                rate_cache=RateTableCache(),
            ),
        )
    )


def _spec_by_sv(result, sv: str):
    for spec in result.specs:
        if any(item.ref.ref == sv for item in spec.resolved.members):
            return spec
    raise AssertionError(f"{sv!r} not found in any surviving candidate spec")


def _date_range_ends(result) -> dict[str, str | None]:
    return {
        sv.ref.ref: (sv.data_date_range.end if sv.data_date_range is not None else None)
        for spec in result.specs
        for sv in spec.resolved.members
    }


def test_candidates_each_judged_by_its_own_election_not_a_pooled_union():
    """USD and GBP candidates keep their own elected units and 2020/2023 dates.
    A literal-only election would drop the GBP candidate despite its conversion recipe."""
    result = _run()
    assert result.status == "candidates", result
    assert len(result.specs) == 2

    spec_a = _spec_by_sv(result, _SV_A)
    spec_b = _spec_by_sv(result, _SV_B)
    assert spec_a.resolved.coverage.has_data is True
    assert spec_b.resolved.coverage.has_data is True

    ends = _date_range_ends(result)
    assert ends[_SV_A] == "2020", ends
    assert ends[_SV_B] == "2023", ends


def test_a_literal_only_election_would_have_dropped_the_converting_candidate(monkeypatch):
    """A literal USD-only election drops _SV_B, whose native GBP facet needs conversion."""

    def _literal_only(sv_dcids, *, probes, unit_request, **_kwargs):
        elected: dict = {}
        for sv in sv_dcids:
            for facet in probes[sv].candidates if probes and sv in probes else ():
                if facet.unit == f"Currency{unit_request.currency}":
                    elected[sv] = (
                        {
                            "unit": FacetSelection(
                                axis="unit",
                                ref=facet.unit,
                                basis="deployment_fallback",
                                default_policy=None,
                                default_policy_version=None,
                                selection_reason="native_currency_preferred",
                                dispatch_facet_id=facet.facet_id or "f0",
                            )
                        },
                    )
        return CandidateUnitElections(elected=elected, refusals={})

    monkeypatch.setattr(candidate_grounding_module, "_candidate_unit_elections", _literal_only)
    result = _run()
    survivors = {item.ref.ref for spec in result.specs for item in spec.resolved.members}
    assert _SV_B not in survivors, (
        f"expected the literal-only perturbation to drop {_SV_B!r} (its own native GBP "
        f"facet is never a literal USD match); got status={result.status!r}, "
        f"survivors={survivors!r}"
    )
    assert _SV_A in survivors, survivors


# A search-side Facet (clients.graph.Facet) for each stat_var's candidate,
# the shape a read plan's probe carries (`read_plan.series_probes`), handed to
# _candidate_unit_elections as `probes`. facet_id is set, since
# materialise.facets.select_facet's dispatchable filter excludes any
# candidate whose facet_id is None (this file's `_graph()` obs dicts carry
# their own "facetId" entries for the same purpose, for the full-resolve
# path above).
_SEARCH_FACET_A = Facet(
    earliest_date="2020",
    latest_date="2020",
    obs_count=1,
    dates=("2020",),
    values=(100.0,),
    facet_id="pv_a",
    provenance_id=_PROV_A,
    unit="CurrencyUSD",
)
_SEARCH_FACET_B = Facet(
    earliest_date="2023",
    latest_date="2023",
    obs_count=1,
    dates=("2023",),
    values=(50.0,),
    facet_id="pv_b",
    provenance_id=_PROV_B,
    unit="CurrencyGBP",
)


_UNIT_REQUEST_USD = IdentityUnitRequest(currency="USD", prices="current")
_ROLE_BINDINGS_KEN = {"observationAbout": [_KEN]}
# Both this file's _candidate_unit_elections calls declare
# ("observationAbout",), subject-only, so `coverage` is never read on that
# path -- an empty index, never consulted, satisfies the required argument.
_SUBJECT_ONLY_COVERAGE = build_coverage_index([], role_names={})


def _decide_one(*, materialise_graph: FakeGraphReader, sv: str, facet: Facet, memo: dict):
    """Calls candidate unit election directly to isolate its facet-probe count."""
    return _candidate_unit_elections(
        [sv],
        unit_request=_UNIT_REQUEST_USD,
        role_bindings=_ROLE_BINDINGS_KEN,
        declared_roles=("observationAbout",),
        probes={sv: subject_probe({_KEN: [facet]}, [_KEN])},
        years=None,
        graph=materialise_graph,
        sdmx=_materialise_sdmx(),
        coverage=_SUBJECT_ONLY_COVERAGE,
        memo=memo,
        deadline=None,
    )


def test_candidate_unit_election_reuses_precomputed_facets_with_no_probe():
    """The plan's probes decide both candidates with zero `variable_facets_batch` calls.
    The empty-facets double makes any unexpected probe visible."""
    materialise_graph = _materialise_graph_no_facets()
    elected_a = _decide_one(
        materialise_graph=materialise_graph, sv=_SV_A, facet=_SEARCH_FACET_A, memo={}
    )
    elected_b = _decide_one(
        materialise_graph=materialise_graph, sv=_SV_B, facet=_SEARCH_FACET_B, memo={}
    )
    assert elected_a.elected[_SV_A][0]["unit"].ref == "CurrencyUSD"
    assert elected_b.elected[_SV_B][0]["unit"].ref == "CurrencyGBP"
    probes = materialise_graph.calls["variable_facets_batch"]
    assert probes == 0, (
        f"expected the precomputed facets to absorb both candidates' own decisions with "
        f"no fresh probe; variable_facets_batch was called {probes} time(s)"
    )


def test_candidate_unit_election_memo_miss_still_makes_no_probe():
    """A fresh memo still makes zero probes because the plan's probes supply the election."""
    materialise_graph = _materialise_graph_no_facets()
    first = _decide_one(
        materialise_graph=materialise_graph, sv=_SV_A, facet=_SEARCH_FACET_A, memo={}
    )
    second = _decide_one(
        materialise_graph=materialise_graph, sv=_SV_A, facet=_SEARCH_FACET_A, memo={}
    )
    assert first.elected[_SV_A][0]["unit"].ref == "CurrencyUSD"
    assert second.elected[_SV_A][0]["unit"].ref == "CurrencyUSD"
    probes = materialise_graph.calls["variable_facets_batch"]
    assert probes == 0, (
        f"expected the memo-always-miss perturbation to still make 0 probes (precomputed "
        f"facets alone already absorb them); got {probes}"
    )


def test_candidates_sharing_a_statvar_and_bindings_elect_from_their_own_probes():
    """Two candidates give SV_A the same bindings but different read plans, one a USD facet
    and one a GBP facet. Sharing one memo, each elects from its own probe."""
    materialise_graph = _materialise_graph_no_facets()
    memo: dict = {}
    usd = _decide_one(
        materialise_graph=materialise_graph, sv=_SV_A, facet=_SEARCH_FACET_A, memo=memo
    )
    gbp = _decide_one(
        materialise_graph=materialise_graph, sv=_SV_A, facet=_SEARCH_FACET_B, memo=memo
    )
    assert usd.elected[_SV_A][0]["unit"].ref == "CurrencyUSD"
    assert gbp.elected[_SV_A][0]["unit"].ref == "CurrencyGBP"
    assert len(memo) == 2


def test_a_candidate_with_no_plan_probe_falls_back_to_a_real_probe():
    """A candidate whose answer carries no plan probe makes one real probe."""
    materialise_graph = _materialise_graph_no_facets()
    elected = _candidate_unit_elections(
        [_SV_A],
        unit_request=_UNIT_REQUEST_USD,
        role_bindings=_ROLE_BINDINGS_KEN,
        declared_roles=("observationAbout",),
        probes=None,
        years=None,
        graph=materialise_graph,
        sdmx=_materialise_sdmx(),
        coverage=_SUBJECT_ONLY_COVERAGE,
        memo={},
        deadline=None,
    )
    assert materialise_graph.calls["variable_facets_batch"] == 1
    # No candidate carries any facts in this double (no `facets=` at all),
    # so the fallback probe finds nothing to elect from, consistent with
    # the guard that an unknown election elects nothing and records the refusal. The assertion
    # covers this double's empty case, distinct from what a real
    # role-bearing probe would find.
    assert elected.elected == {}
    assert isinstance(elected.refusals[_SV_A], DefaultFacetUndefinedError)


def _decide_raising(monkeypatch, exc: Exception, memo: dict):
    def _raise(*args, **kwargs):
        raise exc

    monkeypatch.setattr(candidate_unit_election_module, "decide_part_unit", _raise)
    return _decide_one(
        materialise_graph=_materialise_graph_no_facets(),
        sv=_SV_A,
        facet=_SEARCH_FACET_A,
        memo=memo,
    )


def test_definite_unit_refusal_omits_the_candidate_statvar(monkeypatch):
    """A refusal that proves the StatVar cannot serve the unit omits it from `elected`,
    returns it in `refusals`, and memoises it so a memo hit returns the same refusal."""
    refusal = FacetNotAvailableError("no USD facet")
    memo: dict = {}
    first = _decide_raising(monkeypatch, refusal, memo)
    assert first == CandidateUnitElections(elected={}, refusals={_SV_A: refusal})
    assert list(memo.values()) == [refusal]
    assert _decide_raising(monkeypatch, AssertionError("memo miss"), memo) == first


@pytest.mark.parametrize(
    "exc",
    [
        UpstreamVersionChangedError("rate table changed version during its load"),
        OverCeilingError(ceiling_kind="candidate_window", count=500, cap=500),
        MembersNotHeldError(stat_vars=[_SV_A]),
        DatasetDisagreementError("drifted", dataset=_PROV_A, stat_var=_SV_A),
    ],
    ids=["upstream_version_changed", "over_ceiling", "members_not_held", "dataset_disagreement"],
)
def test_unestablished_unit_election_raises_rather_than_omitting(monkeypatch, exc):
    """An election that fails for an infrastructure or ceiling reason leaves the StatVar unknown.
    Omitting it would let the candidate gate count it inside a complete negative, so it raises,
    as the elected shape's `decide_part_unit` does, and memoises nothing."""
    memo: dict = {}
    with pytest.raises(type(exc)):
        _decide_raising(monkeypatch, exc, memo)
    assert memo == {}


def _dataset_disagreement(sv: str) -> DatasetDisagreementError:
    return DatasetDisagreementError(
        "the member's facets disagree with the elected dataset", dataset=_PROV_A, stat_var=sv
    )


def test_an_alternates_dataset_disagreement_drops_it_and_the_elected_answer_stands(
    monkeypatch,
):
    """An alternate whose unit election raises `DatasetDisagreementError` is dropped, as an
    alternate whose mint raises it is (`candidates._materialise_alt_shapes`). The elected
    shape answers definite on its own instead of the request failing."""
    _refuse_by_statvar(monkeypatch, {_SV_B: _dataset_disagreement(_SV_B)})
    result = _run()
    assert result.status == "definite", result
    assert [item.ref.ref for spec in result.specs for item in spec.resolved.members] == [_SV_A]


def test_the_elected_shapes_dataset_disagreement_still_fails_the_request(monkeypatch):
    """The elected shape's own disagreement propagates: there is no answer to keep."""
    _refuse_by_statvar(monkeypatch, {_SV_A: _dataset_disagreement(_SV_A)})
    with pytest.raises(DatasetDisagreementError):
        _run()


def _refuse_by_statvar(monkeypatch, refusals: dict[str, Exception]) -> None:
    """Make each named StatVar's candidate election raise its refusal; others elect normally."""
    real = candidate_unit_election_module.decide_part_unit

    def _decide(sv_dcids, **kwargs):
        # The candidate election decides one StatVar per call.
        (sv,) = sv_dcids
        exc = refusals.get(sv)
        if exc is not None:
            raise exc
        return real(sv_dcids, **kwargs)

    monkeypatch.setattr(candidate_unit_election_module, "decide_part_unit", _decide)


def _recipe_refusal() -> RecipeNotDeclaredError:
    return RecipeNotDeclaredError("no conversionRecipe declared", cause="recipe_undeclared")


def _rate_gap_refusal() -> RateCoverageGapError:
    return RateCoverageGapError("no rate for KES", cause="rate_missing", entities=["KES"])


def test_recipe_refusal_on_every_candidate_answers_recipe_not_declared(monkeypatch):
    """A StatVar whose election raises `RecipeNotDeclaredError` holds data the request cannot
    convert. With no candidate left, the candidates door answers as the single-shape door does:
    `recipe_not_declared` with its cause."""
    _refuse_by_statvar(monkeypatch, {_SV_A: _recipe_refusal(), _SV_B: _recipe_refusal()})
    result = _run()
    assert result.status == "unsupported", result
    assert result.unsupported is not None
    assert result.unsupported.code == "recipe_not_declared"
    assert result.unsupported.pointer == "/identity/facets/unit"
    assert result.unsupported.cause is not None
    assert result.unsupported.cause.code == "recipe_undeclared"


def test_mixed_refusals_answer_with_the_first_candidates_refusal(monkeypatch):
    """Candidates refused for different reasons answer with the first candidate's refusal, and
    the elected shape is first. A rate gap there answers `rate_coverage_gap` with its cause."""
    _refuse_by_statvar(monkeypatch, {_SV_A: _rate_gap_refusal(), _SV_B: _recipe_refusal()})
    result = _run()
    assert result.status == "no_data", result
    assert result.no_data_reason == "rate_coverage_gap"
    assert result.cause is not None
    assert result.cause.code == "rate_missing"
    assert result.cause.entities == ["KES"]
    assert [ref.ref for ref in result.elected_stat_vars or []] == [_SV_A]


def test_rate_gap_from_a_later_candidate_names_that_candidates_statvars(monkeypatch):
    """A rate gap passed on from the next candidate describes that candidate. The no-data
    answer names its StatVar, as the single-shape door does for the shape that refused."""
    _refuse_by_statvar(
        monkeypatch, {_SV_A: FacetNotAvailableError("no facet"), _SV_B: _rate_gap_refusal()}
    )
    result = _run()
    assert result.status == "no_data", result
    assert result.no_data_reason == "rate_coverage_gap"
    assert result.cause is not None
    assert result.cause.code == "rate_missing"
    assert result.elected_stat_vars == [GraphRef(ref=_SV_B, label="Amount B")]


@pytest.mark.parametrize(
    ("refusal", "status", "code"),
    [
        (_recipe_refusal(), "unsupported", "recipe_not_declared"),
        (
            NonCurrencyUnitError("USDollar is not a currency", unit="USDollar"),
            "unsupported",
            "facet_binding_unsupported",
        ),
    ],
    ids=["recipe", "non_currency"],
)
def test_conversion_refusal_outranks_a_missing_facet(monkeypatch, refusal, status, code):
    """A missing facet on the elected shape passes the answer to the next candidate's conversion
    refusal, which says data exists that the request cannot convert. The definition names the
    refusing candidate's shape with the unit request, and resubmitting it through the
    single-shape door meets the same refusal."""
    _refuse_by_statvar(monkeypatch, {_SV_A: FacetNotAvailableError("no facet"), _SV_B: refusal})
    result = _run()
    assert result.status == status, result
    assert result.unsupported is not None
    assert result.unsupported.code == code
    identity = result.unsupported.definition["identity"]
    assert isinstance(identity, dict)
    assert identity["shape"]["measured_property"] == "candUnitElectionB", identity
    assert identity["facets"]["unit"] == {
        "state": "value",
        "request": {"currency": "USD", "prices": "current"},
    }

    # The single-shape door elects through `decide_part_unit`. Refuse there as the candidate
    # election did: the elected StatVar lacks a facet, the refusing StatVar cannot convert.
    decided: list[list[str]] = []

    def _decide_part(sv_dcids, **_kwargs):
        decided.append(list(sv_dcids))
        raise refusal if _SV_B in sv_dcids else FacetNotAvailableError("no facet")

    monkeypatch.setattr(promote_ground_module, "decide_part_unit", _decide_part)
    _spec_id, minted = mint_identity(identity)
    with pytest.raises(EngineInputError) as exc_info:
        resolve_spec_resubmit(
            inp=ResubmitIdentityPayload(identity=minted),
            graph=_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(
                _graph(),
                EngineOptions(
                    include_observations=True,
                    materialise_graph=with_member_arcs(_materialise_graph(), _graph()),
                    materialise_sdmx=_materialise_sdmx(),
                    rate_cache=RateTableCache(),
                ),
            ),
        )
    assert decided == [[_SV_B]]
    assert exc_info.value.code == code


def test_missing_facets_on_every_candidate_still_answer_facet_not_available(monkeypatch):
    """Control: with no conversion refusal, the empty survivor list answers `facet_not_available`."""
    _refuse_by_statvar(
        monkeypatch,
        {_SV_A: FacetNotAvailableError("no facet"), _SV_B: FacetNotAvailableError("no facet")},
    )
    result = _run()
    assert result.status == "no_data", result
    assert result.no_data_reason == "facet_not_available"


def test_candidate_unit_election_with_no_bound_entity_makes_no_graph_read():
    """A candidate with no scope member and no destination has empty role bindings. Its
    election elects nothing before any probe, so it reads no countries under Earth."""
    materialise_graph = FakeGraphReader(child_places={("Earth", "Country"): [_KEN]})
    memo: dict = {}
    elected = _candidate_unit_elections(
        [_SV_A],
        unit_request=_UNIT_REQUEST_USD,
        role_bindings={},
        declared_roles=("observationAbout",),
        probes={},
        years=None,
        graph=materialise_graph,
        sdmx=_materialise_sdmx(),
        coverage=_SUBJECT_ONLY_COVERAGE,
        memo=memo,
        deadline=None,
    )
    assert elected == CandidateUnitElections(elected={}, refusals={})
    assert memo == {}
    assert set(materialise_graph.calls.values()) == {0}


_SPARSE_GBP_B = ObsFacetFixture(
    facet_id="pv_b",
    points=(("2021", 40.0), ("2023", 50.0)),
    provenance_id=_PROV_B,
    unit="CurrencyGBP",
)
_SPARSE_RATES = {
    "CurrencyUSD": {2020: 1.0, 2021: 1.0, 2023: 1.0},
    "CurrencyGBP": {2020: 0.8, 2021: 0.8, 2023: 0.8},
}
"""No 2022 rate, a year inside `_SV_B`'s span that its series holds no row in."""


def _sparse_options(*, include_observations: bool) -> EngineOptions:
    return EngineOptions(
        include_observations=include_observations,
        materialise_graph=_materialise_graph(_SPARSE_GBP_B),
        materialise_sdmx=_materialise_sdmx(b=_SPARSE_GBP_B, rates=_SPARSE_RATES),
        rate_cache=RateTableCache(),
    )


def _sparse_resolve(*, svs: tuple[str, ...], include_observations: bool):
    """A fresh dateless USD request whose `_SV_B` series holds 2021 and 2023 alone, over a
    graph whose undated facet reads state only each facet's span."""
    graph = _graph(b=_SPARSE_GBP_B, svs=svs, summary_reads=True)
    return asyncio.run(
        resolve_variable(
            "amount",
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(alone=len(svs) == 1),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            base_timing={},
            unit_text="USD",
            coverage=empty_coverage_snapshot(),
            options=_sparse_options(include_observations=include_observations),
        )
    )


def _points(spec) -> list[tuple[str, float | None]]:
    observations = spec.resolved.observations
    if observations is None:
        return []
    return [(p.date, p.value) for s in observations.series for p in s.points]


@pytest.mark.parametrize("include_observations", [False, True])
def test_a_sparse_series_answers_definite_on_the_fresh_and_resubmit_doors(include_observations):
    """`_SV_B`'s facet states its 2021 to 2023 span alone, and the rate table holds no 2022
    rate. The bound-unit gate reads the series' dates before refusing, so the fresh answer
    and the resubmit of its definition both answer definite with the same converted points,
    and a mint of those rows needs no 2022 rate."""
    fresh = _sparse_resolve(svs=(_SV_B,), include_observations=include_observations)
    assert fresh.status == "definite", fresh
    (spec,) = fresh.specs
    identity = spec.definition.identity
    graph = _graph(b=_SPARSE_GBP_B, svs=(_SV_B,), summary_reads=True)
    options = _sparse_options(include_observations=include_observations)
    resubmit = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=dataclasses.replace(
            options, materialise_graph=with_member_arcs(options.materialise_graph, graph)
        ),
    )
    assert resubmit.status == "definite", resubmit
    (resubmitted,) = resubmit.specs
    expected = [("2023", pytest.approx(50.0 / 0.8)), ("2021", pytest.approx(40.0 / 0.8))]
    for answered in (spec, resubmitted):
        assert sorted(_points(answered), reverse=True) == (expected if include_observations else [])


def test_a_sparse_candidate_survives_the_candidates_door():
    """The same sparse series as the second candidate: the gate reads its dates before
    refusing, so it stays beside the first candidate."""
    result = _sparse_resolve(svs=(_SV_A, _SV_B), include_observations=False)
    assert result.status == "candidates", result
    survivors = {item.ref.ref for spec in result.specs for item in spec.resolved.members}
    assert survivors == {_SV_A, _SV_B}


def _cover_with_omission(monkeypatch, *, refused: str | None = None) -> None:
    """Have each candidate's conversion find `_SV_B`'s omitting two regions, `refused`'s
    refusing a rate, and any other candidate's converting in full."""
    omitting = UnitCoverage(
        omitted=frozenset((_SV_B, "observationAbout", e) for e in ("africaRegion", "southOfSahara"))
    )

    def _convertible(mat, filtered, _elected, **_kwargs):
        (sv,) = mat.sv_dcids
        if sv == refused:
            return filtered, UnitCoverage(refusal=(sv, _rate_gap_refusal()))
        return filtered, omitting if sv == _SV_B else UnitCoverage()

    monkeypatch.setattr(candidate_grounding_module, "convertible_answer", _convertible)


def _omitted_warnings(result) -> list[str | None]:
    return [w.spec_id for w in result.warnings if w.code == "currency_unmapped_omitted"]


def test_a_candidate_set_warns_once_for_the_candidate_whose_conversion_omits_entities(
    monkeypatch,
):
    _cover_with_omission(monkeypatch)
    result = _run()
    assert result.status == "candidates", result
    assert _omitted_warnings(result) == [_spec_by_sv(result, _SV_B).spec_id]


def test_a_definite_survivor_carries_its_omission_warning(monkeypatch):
    """The first candidate refuses a rate, so the second answers alone and carries the
    warning its conversion raised."""
    _cover_with_omission(monkeypatch, refused=_SV_A)
    result = _run()
    assert result.status == "definite", result
    (spec,) = result.specs
    assert _omitted_warnings(result) == [spec.spec_id]


def _leave_a_qualifier_unbound(monkeypatch) -> None:
    """Have the fresh request's query qualifier stay unbound, so it answers
    `qualifier_not_bound` with its grounded Spec as the flagged match."""
    monkeypatch.setattr(
        phase_election_module,
        "_resolve_query_qualifiers",
        lambda _state: [ResolvedQualifier(text="girls", named=[], disposition="unbound")],
    )


def test_a_carve_out_states_the_plan_its_conversion_never_read(monkeypatch):
    """An unbound qualifier answers before the bound-unit gate, so no conversion runs and the
    flagged Spec states the plan as filtered, Kenya's rows included, however a conversion
    would have omitted them."""
    _leave_a_qualifier_unbound(monkeypatch)
    calls: list[object] = []

    def _omitting_kenya(plan, *_args, **_kwargs) -> UnitCoverage:
        calls.append(plan)
        return UnitCoverage(omitted=frozenset({(_SV_B, "observationAbout", _KEN)}))

    monkeypatch.setattr(convertibility_module, "unit_coverage", _omitting_kenya)

    result = _sparse_resolve(svs=(_SV_B,), include_observations=False)

    assert (result.status, result.no_data_reason) == ("no_data", "qualifier_not_bound")
    (flagged,) = result.flagged_match
    assert flagged.resolved.coverage.has_data is True
    assert [sv.data_date_range.end for sv in flagged.resolved.members] == ["2023"]
    assert calls == []


def test_a_rate_table_that_fails_to_load_leaves_a_carve_out_answer_as_it_was(monkeypatch):
    """The conversion's rate table would fail to load, but the unbound qualifier answers
    before any conversion reads it."""
    _leave_a_qualifier_unbound(monkeypatch)

    def _failing(*_args, **_kwargs):
        raise RateTableInvalidError("the exchange-rate table failed to load")

    monkeypatch.setattr(member_conversion_module, "_load_tables", _failing)

    result = _sparse_resolve(svs=(_SV_B,), include_observations=False)

    assert (result.status, result.no_data_reason) == ("no_data", "qualifier_not_bound")
