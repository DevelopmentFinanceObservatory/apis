"""Test role-index narrowing across units, sources, counts, and discovered scopes."""

from __future__ import annotations

import asyncio

import httpx

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.rate_tables import RateTableCache
from materialise.roles import axis_for_role
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.models import TimeWindow
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import CoverageRow, FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    build_materialise_double,
    sdmx_double,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_SV_DCID = "TESTFIXTURE/RoleScopeNarrowingE2E/Flow"
_ORIGIN_ONLY_SV_DCID = "TESTFIXTURE/RoleScopeNarrowingE2E/OriginOnlyFlow"
_KEN = "country/KEN"
_NOR = "country/NOR"
_TZA = "country/TZA"
_FRA = "country/FRA"
_DEU = "country/DEU"


def _facet(
    *,
    facet_id: str,
    unit: str | None = None,
    provenance_id: str | None = None,
    latest_date: str = "2020",
) -> Facet:
    return Facet(
        earliest_date="2015",
        latest_date=latest_date,
        obs_count=1,
        dates=[latest_date],
        values=[0.0],
        facet_id=facet_id,
        unit=unit,
        provenance_id=provenance_id,
    )


def _declaring_arcs() -> dict:
    """Build a bilateral flow shape that discovers recipients when no place is named."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in ("creditor", "debtor")]},
    }


def _origin_only_declaring_arcs() -> dict:
    """Build an origin-only shape that discovers providers when no place is named."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "creditor"}]},
    }


_UNBOUND_DONOR = "TESTFIXTURE/RoleScopeNarrowingE2E/UnboundDonor"
_UNBOUND_RECIPIENT = "TESTFIXTURE/RoleScopeNarrowingE2E/UnboundRecipient"


def _coverage_seed_for(
    role_name: str,
    members: list[str],
    svs: list[str],
    *,
    facet_id: str = "f1",
    member_facet_ids: dict[str, str] | None = None,
    years: list[int] | None = None,
) -> tuple[list[CoverageRow], dict[str, RoleNames]]:
    """Build one row per SV/member with the opposite axis unbound.

    Years default to 2020; facet ids can vary by member.
    """
    discover_axis = axis_for_role(role_name)
    assert discover_axis is not None, f"{role_name!r} has no flow axis"
    opposite = _UNBOUND_DONOR if discover_axis == "destination" else _UNBOUND_RECIPIENT
    member_facet_ids = member_facet_ids or {}
    row_years = years if years is not None else [2020]
    rows: list[CoverageRow] = [
        (
            sv,
            member_facet_ids.get(member, facet_id),
            *((opposite, member) if discover_axis == "destination" else (member, opposite)),
            row_years,
        )
        for sv in svs
        for member in members
    ]
    role_names = {
        sv: (
            RoleNames(donor="creditor", recipient=role_name)
            if discover_axis == "destination"
            else RoleNames(donor=role_name, recipient="debtor")
        )
        for sv in svs
    }
    return rows, role_names


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _run(
    *,
    resolve: dict[str, str],
    label_dcids: list[str],
    sdmx: FakeSdmxClient,
    unit_text: str | None = None,
    date_request: DateRequest | None = None,
    sv_dcid: str = _SV_DCID,
    sv_arcs: dict | None = None,
    materialise_graph=None,
    materialise_sdmx=None,
    monkeypatch=None,
    coverage_rows: list[CoverageRow] | None = None,
    coverage_role_names: dict[str, RoleNames] | None = None,
    coverage_facet_metadata: dict[str, FacetMetadata] | None = None,
    include_observations: bool = False,
):
    if coverage_rows is not None:
        assert monkeypatch is not None, "coverage_rows needs monkeypatch to seed it"
        _auto_coverage_index_1 = patch_coverage(
            monkeypatch,
            coverage_rows,
            role_names=coverage_role_names or {},
            facet_metadata=coverage_facet_metadata,
        )
    nodes = {
        **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
        sv_dcid: {
            "label": "Test flow",
            "arcs": sv_arcs if sv_arcs is not None else _declaring_arcs(),
        },
        **{dcid: {"label": dcid.rsplit("/", 1)[-1]} for dcid in label_dcids},
    }
    graph = FakeGraph(
        nodes=nodes,
        detect={"aid disbursed": {"svs": [sv_dcid], "cosine_scores": [0.9]}},
        resolve=resolve,
        obs={},
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=[],
            date_request=date_request,
            detect_query="aid disbursed",
            role_query="aid disbursed",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_steps=[],
            base_timing={},
            unit_text=unit_text,
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            options=EngineOptions(
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
                include_observations=include_observations,
            ),
        )
    )


def test_unit_binding_narrows_which_sources_are_cited(monkeypatch, search_settings):
    """USD keeps Kenya's tuple and drops Norway's; pair cap 1 selects bulk preview."""
    search_settings(obs_preview_max_pairs=1)
    rows: list[CoverageRow] = [
        (_SV_DCID, "fA", _UNBOUND_DONOR, _KEN, [2021]),
        (_SV_DCID, "fB", _UNBOUND_DONOR, _NOR, [2021]),
    ]
    role_names = {_SV_DCID: RoleNames(donor="creditor", recipient="debtor")}
    facet_metadata = {
        "fA": FacetMetadata(unit="CurrencyUSD", provenance_id="source/A"),
        "fB": FacetMetadata(unit="CurrencyEUR", provenance_id="source/B"),
    }
    observations = {
        _SV_DCID: {
            (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [
                _facet(
                    facet_id="fA", unit="CurrencyUSD", provenance_id="source/A", latest_date="2021"
                )
            ],
            (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [
                _facet(
                    facet_id="fB", unit="CurrencyEUR", provenance_id="source/B", latest_date="2021"
                )
            ],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    # Conversion uses a separate materialise client with a fixed native-USD response.
    # This test measures source narrowing.
    materialise_body = _csv_header(["debtor"]) + _csv_row(
        stat_var=_SV_DCID,
        role_dcids=[_KEN],
        fixture=ObsFacetFixture(
            facet_id="fA", points=(("2021", 1.0),), provenance_id="source/A", unit="CurrencyUSD"
        ),
        date_str="2021",
        value=1.0,
    )
    materialise_graph = FakeGraphReader(
        arcs={
            _SV_DCID: {"measurementQualifier": ("Nominal",)},
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
    )
    materialise_sdmx = sdmx_double(csv_body=materialise_body)
    result = _run(
        resolve={"Kenya": _KEN, "Norway": _NOR},
        label_dcids=["source/A", "source/B"],
        sdmx=sdmx,
        unit_text="USD",
        date_request=DateRequest(window=None, latest=True),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
        coverage_facet_metadata=facet_metadata,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    resolved_source_dcids = [r.dcid for r in spec.resolved.trace.resolved_sources]
    assert sorted(resolved_source_dcids) == ["source/A"]
    # The narrowed role index excludes source/B from citations.
    assert "source/B" not in resolved_source_dcids
    dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
    assert dims["sources"] == 1
    assert dims["recipients"] == 1
    assert spec.resolved.coverage.window == TimeWindow(start_year=2021, end_year=2021)
    when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
    assert when_slot.binding.value.time_window == TimeWindow(start_year=2021, end_year=2021)
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == [_KEN]


def test_a_facet_without_provenance_is_read_by_its_facet_id_and_cites_nothing(monkeypatch):
    """A facet without a provenance serves each pair: the preview binds the provenance axis to
    the null ref and reads the facet by its id, one request per pair. Coverage counts both
    recipients and no source, and nothing is cited."""
    rows, role_names = _coverage_seed_for("debtor", [_KEN, _NOR], [_SV_DCID])
    observations = {
        _SV_DCID: {
            (("debtor", _KEN),): [_facet(facet_id="fA")],
            (("debtor", _NOR),): [_facet(facet_id="fB")],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})
    _graph, materialise_sdmx, requests = build_materialise_double(
        {
            (_SV_DCID, (("debtor", member), ("creditor", _UNBOUND_DONOR))): [
                ObsFacetFixture(facet_id="f1", points=(("2020", 1.0),))
            ]
            for member in (_KEN, _NOR)
        }
    )

    result = _run(
        resolve={"Kenya": _KEN, "Norway": _NOR},
        label_dcids=[],
        sdmx=sdmx,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
        coverage_facet_metadata={"f1": FacetMetadata()},
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=materialise_sdmx,
        include_observations=True,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert spec.resolved.trace.resolved_sources == []
    assert spec.resolved.coverage.has_data is True
    dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
    assert (dims["recipients"], dims["sources"]) == (2, 0)
    delivered = spec.resolved.observations
    assert delivered is not None and delivered.omitted is None
    assert sum(len(series.points) for series in delivered.series) == 2
    assert [request.url.params.get("c[facetId]") for request in requests] == ["f1", "f1"]


def test_sources_and_recipients_are_independent_counts(monkeypatch):
    """Three recipients sharing one provenance count as 3 recipients and 1 source."""
    rows, role_names = _coverage_seed_for("debtor", [_KEN, _NOR, _TZA], [_SV_DCID])
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _run(
        resolve={"Kenya": _KEN, "Norway": _NOR, "Tanzania": _TZA},
        label_dcids=["source/A"],
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
        coverage_facet_metadata={"f1": FacetMetadata(provenance_id="source/A")},
        sdmx=sdmx,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
    assert dims["recipients"] == 3
    assert dims["sources"] == 1


def test_unit_binding_recounts_a_provider_axis_scope(monkeypatch, search_settings):
    """A USD binding narrows an origin-only scope to FRA and source/A.

    Provider and source counts both become 1; no destination axis exists.
    """
    search_settings(obs_preview_max_pairs=1)
    rows, role_names = _coverage_seed_for(
        "creditor",
        [_FRA, _DEU],
        [_ORIGIN_ONLY_SV_DCID],
        member_facet_ids={_FRA: "fA", _DEU: "fB"},
    )
    facet_metadata = {
        "fA": FacetMetadata(unit="CurrencyUSD", provenance_id="source/A"),
        "fB": FacetMetadata(unit="CurrencyEUR", provenance_id="source/B"),
    }
    observations = {
        _ORIGIN_ONLY_SV_DCID: {
            (("creditor", _FRA),): [
                _facet(facet_id="fA", unit="CurrencyUSD", provenance_id="source/A")
            ],
            (("creditor", _DEU),): [
                _facet(facet_id="fB", unit="CurrencyEUR", provenance_id="source/B")
            ],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    # Conversion uses a separate materialise client with a fixed native-USD response.
    # This test measures provider and source narrowing.
    materialise_body = _csv_header(["creditor"]) + _csv_row(
        stat_var=_ORIGIN_ONLY_SV_DCID,
        role_dcids=[_FRA],
        fixture=ObsFacetFixture(
            facet_id="fA", points=(("2020", 1.0),), provenance_id="source/A", unit="CurrencyUSD"
        ),
        date_str="2020",
        value=1.0,
    )
    materialise_graph = FakeGraphReader(
        arcs={
            _ORIGIN_ONLY_SV_DCID: {"measurementQualifier": ("Nominal",)},
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
    )
    materialise_sdmx = sdmx_double(csv_body=materialise_body)
    result = _run(
        resolve={"France": _FRA, "Germany": _DEU},
        label_dcids=["source/A", "source/B"],
        sdmx=sdmx,
        unit_text="USD",
        sv_dcid=_ORIGIN_ONLY_SV_DCID,
        sv_arcs=_origin_only_declaring_arcs(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        monkeypatch=monkeypatch,
        coverage_rows=rows,
        coverage_role_names=role_names,
        coverage_facet_metadata=facet_metadata,
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert sorted(r.dcid for r in spec.resolved.trace.resolved_sources) == ["source/A"]
    dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
    assert dims["providers"] == 1
    assert dims["sources"] == 1
    assert spec.resolved.scope is not None
    assert [m.ref for m in spec.resolved.scope.members] == [_FRA]


# Each candidate narrows its own discovered scope.

_SV_CAND_A = "TESTFIXTURE/RoleScopeNarrowingE2E/CandFlowA"
_SV_CAND_B = "TESTFIXTURE/RoleScopeNarrowingE2E/CandFlowB"


def _declaring_arcs_alt_meas_prop() -> dict:
    """Build a second bilateral shape with a distinct measuredProperty."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amountAlt"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in ("creditor", "debtor")]},
    }


class _CandidatesBindLLM:
    """Elect the first shape and flag the second as also plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
        ), None


def test_candidates_each_narrow_their_own_scope_to_the_surviving_role_index(monkeypatch):
    """Two candidates discover KEN/USD/source-A and NOR/EUR/source-B.

    A USD request narrows each candidate to Kenya.
    Pair cap 1 triggers bulk preview and separate unit probes.
    """
    _bilateral_facets = {
        (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [
            _facet(facet_id="fA", unit="CurrencyUSD", provenance_id="source/A", latest_date="2021")
        ],
        (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [
            _facet(facet_id="fB", unit="CurrencyEUR", provenance_id="source/B", latest_date="2021")
        ],
    }
    observations = {_SV_CAND_A: _bilateral_facets, _SV_CAND_B: _bilateral_facets}
    sdmx = FakeSdmxClient(observations=observations, availability={})
    rows, role_names = _coverage_seed_for(
        "debtor",
        [_KEN, _NOR],
        [_SV_CAND_A, _SV_CAND_B],
        member_facet_ids={_KEN: "fA", _NOR: "fB"},
        years=[2021],
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        rows,
        role_names=role_names,
        facet_metadata={
            "fA": FacetMetadata(unit="CurrencyUSD", provenance_id="source/A"),
            "fB": FacetMetadata(unit="CurrencyEUR", provenance_id="source/B"),
        },
    )
    materialise_graph, _ignored_sdmx, _requests = build_materialise_double(
        {
            (sv_dcid, (("debtor", _KEN),)): [
                ObsFacetFixture(
                    facet_id="fA",
                    points=(("2021", 1.0),),
                    unit="CurrencyUSD",
                    provenance_id="source/A",
                )
            ]
            for sv_dcid in (_SV_CAND_A, _SV_CAND_B)
        }
        | {
            (sv_dcid, (("debtor", _NOR),)): [
                ObsFacetFixture(
                    facet_id="fB",
                    points=(("2023", 1.0),),
                    unit="CurrencyEUR",
                    provenance_id="source/B",
                )
            ]
            for sv_dcid in (_SV_CAND_A, _SV_CAND_B)
        }
    )
    # USD election validates the CurrencyUSD type. EUR has no matching declaration.
    materialise_graph._arcs = {"CurrencyUSD": {"typeOf": ("CurrencyUnit",)}}
    # The joint scope probe uses the KEN/USD body. Per-recipient reads retain each candidate facet.
    # Every body carries both role columns, as the service returns them.
    _fetch_bodies = {
        (_SV_CAND_A, _KEN): _csv_header(["creditor", "debtor"])
        + _csv_row(
            stat_var=_SV_CAND_A,
            role_dcids=[_UNBOUND_DONOR, _KEN],
            fixture=ObsFacetFixture(
                facet_id="fA", points=(("2021", 1.0),), unit="CurrencyUSD", provenance_id="source/A"
            ),
            date_str="2021",
            value=1.0,
        ),
        (_SV_CAND_A, _NOR): _csv_header(["creditor", "debtor"])
        + _csv_row(
            stat_var=_SV_CAND_A,
            role_dcids=[_UNBOUND_DONOR, _NOR],
            fixture=ObsFacetFixture(
                facet_id="fB", points=(("2023", 1.0),), unit="CurrencyEUR", provenance_id="source/B"
            ),
            date_str="2023",
            value=1.0,
        ),
        (_SV_CAND_B, _KEN): _csv_header(["creditor", "debtor"])
        + _csv_row(
            stat_var=_SV_CAND_B,
            role_dcids=[_UNBOUND_DONOR, _KEN],
            fixture=ObsFacetFixture(
                facet_id="fA", points=(("2021", 1.0),), unit="CurrencyUSD", provenance_id="source/A"
            ),
            date_str="2021",
            value=1.0,
        ),
        (_SV_CAND_B, _NOR): _csv_header(["creditor", "debtor"])
        + _csv_row(
            stat_var=_SV_CAND_B,
            role_dcids=[_UNBOUND_DONOR, _NOR],
            fixture=ObsFacetFixture(
                facet_id="fB", points=(("2023", 1.0),), unit="CurrencyEUR", provenance_id="source/B"
            ),
            date_str="2023",
            value=1.0,
        ),
    }
    _decision_bodies = {sv: _fetch_bodies[(sv, _KEN)] for sv in (_SV_CAND_A, _SV_CAND_B)}

    def _materialise_sdmx_handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        sv = params.get("c[variableMeasured]", "")
        debtor_param = params.get("c[debtor]", "")
        debtor_dcids = debtor_param.split(",") if debtor_param else []
        if len(debtor_dcids) == 1:
            body = _fetch_bodies.get((sv, debtor_dcids[0]), _csv_header(["creditor", "debtor"]))
        else:
            body = _decision_bodies.get(sv, _csv_header(["creditor", "debtor"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    materialise_sdmx = MaterialiseSdmxClient(
        base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(_materialise_sdmx_handler)
    )
    graph = FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _NOR: {"label": "Norway", "type": "Country"},
            _SV_CAND_A: {"label": "Test flow A", "arcs": _declaring_arcs()},
            _SV_CAND_B: {"label": "Test flow B", "arcs": _declaring_arcs_alt_meas_prop()},
        },
        detect={"aid disbursed": {"svs": [_SV_CAND_A, _SV_CAND_B], "cosine_scores": [0.9, 0.85]}},
        resolve={"Kenya": _KEN, "Norway": _NOR},
        obs={},
    )

    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=[],
            date_request=DateRequest(window=None, latest=True),
            detect_query="aid disbursed",
            role_query="aid disbursed",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_CandidatesBindLLM(),
            sdmx=sdmx,
            base_steps=[],
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

    assert result.status == "candidates", result
    assert len(result.specs) == 2
    for spec in result.specs:
        dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
        assert dims["recipients"] == 1
        assert spec.resolved.coverage.window == TimeWindow(start_year=2021, end_year=2021)
        when_slot = next(slot for slot in spec.resolved.slots if slot.key.axis == "when")
        assert when_slot.binding.value.time_window == TimeWindow(start_year=2021, end_year=2021)
        assert spec.resolved.scope is not None
        assert [m.ref for m in spec.resolved.scope.members] == [_KEN]
