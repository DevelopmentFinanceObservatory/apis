"""Facet options for role-bearing StatVars come from the coverage index.

The options read the bound role pairs' coverage rows within the window, and pinning an
offered value through the request's facet binding serves only the facets carrying it. A
facet that declares no measurement method is the ref-less option (`None` below), pinned by
a null ref.
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter

from search.engine.facet_refusals import candidate_facet_refusal, role_index_facets
from search.engine.region.option_coverage import _facet_options
from search.engine.region.resubmit_request import _facet_filter_from_bindings
from search.models import (
    FacetOptions,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    TimeWindow,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    build_observations_planned,
)
from tests.engine._role_candidates import role_candidate
from tests.fixtures import FakeGraph

_SV = "TESTFIXTURE/RoleFacetOptions/DebtService"
_SV_RECORDED_ONLY = "TESTFIXTURE/RoleFacetOptions/DebtServiceRecordedOnly"
_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
_WB = "country/WBX"
_BANK = "country/BNX"
_KEN = "country/KEN"
_NGA = "country/NGA"
_PROVENANCE = "dc/base/TestIDS"
_OTHER_PROVENANCE = "dc/base/TestOther"
_USD = "USDollar"

_RECORDED_YEARS = range(2020, 2025)
_ESTIMATED_YEARS = range(2025, 2031)

# One IDS-like pair (World Bank to Kenya) with a recorded facet that declares no method and
# an Estimated projection facet with more points, so materialise's default serves the
# projection unless a pin steers it. A second creditor to Kenya reports under its own method and
# provenance, and a Nigeria-only facet shows a pair the probe never binds. A sibling StatVar
# carries only a recorded series, with no projection.
_ROWS = [
    (_SV, "rec", _WB, _KEN, list(_RECORDED_YEARS)),
    (_SV, "est", _WB, _KEN, list(_ESTIMATED_YEARS)),
    (_SV, "bank", _BANK, _KEN, [2012, 2013, 2014]),
    (_SV, "nga", _WB, _NGA, [2020]),
    (_SV_RECORDED_ONLY, "rec2", _WB, _KEN, list(range(2020, 2031))),
]
_METADATA = {
    "rec": FacetMetadata(provenance_id=_PROVENANCE, unit=_USD),
    "est": FacetMetadata(provenance_id=_PROVENANCE, measurement_method="Estimated", unit=_USD),
    "bank": FacetMetadata(
        provenance_id=_OTHER_PROVENANCE, measurement_method="BankReported", unit="Euro"
    ),
    "nga": FacetMetadata(provenance_id=_PROVENANCE, measurement_method="NigeriaOnly", unit=_USD),
    "rec2": FacetMetadata(provenance_id=_PROVENANCE, unit=_USD),
}
_PAIR = {"creditor": [_WB], "debtor": [_KEN]}
_OPEN_CREDITOR = {"debtor": [_KEN]}


def _index():
    return build_coverage_index(
        _ROWS,
        role_names={_SV: _ROLE_NAMES, _SV_RECORDED_ONLY: _ROLE_NAMES},
        facet_metadata=_METADATA,
    )


def _graph_without_facet_reads(monkeypatch: pytest.MonkeyPatch) -> FakeGraph:
    """A graph that fails any observation-facet read, so a passing test proves the
    role-bearing path never makes one."""
    graph = FakeGraph()

    def _no_facet_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("role-bearing facet options must not read graph facets")

    monkeypatch.setattr(graph, "observation_facets_batch", _no_facet_read)
    return graph


def _refs(options: list[FacetOptions]) -> dict[str, list[str | None]]:
    return {
        fo.axis: [o.value.ref.dcid if o.value.ref is not None else None for o in fo.options]
        for fo in options
    }


def test_named_pair_offers_only_that_pairs_facets(monkeypatch):
    options = _facet_options(
        [_SV],
        _PAIR,
        years=None,
        graph=_graph_without_facet_reads(monkeypatch),
        coverage=_index(),
        deadline=None,
    )

    assert _refs(options) == {
        "provenance": [_PROVENANCE],
        "measurementMethod": ["Estimated", None],
        "unit": [_USD],
    }
    assert all(fo.source == "observed" and not fo.truncated for fo in options)


def test_open_creditor_axis_offers_every_creditors_facets_at_the_bound_debtor(monkeypatch):
    """Ranked by observation count: 11 World Bank points to Kenya (6 Estimated, 5 recorded
    with no method) against 3 from the bank."""
    options = _facet_options(
        [_SV],
        _OPEN_CREDITOR,
        years=None,
        graph=_graph_without_facet_reads(monkeypatch),
        coverage=_index(),
        deadline=None,
    )

    assert _refs(options) == {
        "provenance": [_PROVENANCE, _OTHER_PROVENANCE],
        "measurementMethod": ["Estimated", None, "BankReported"],
        "unit": [_USD, "Euro"],
    }


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (TimeWindow(start_year=2015, end_year=2024), {"provenance": [_PROVENANCE], "unit": [_USD]}),
        (
            TimeWindow(start_year=2027, end_year=2027),
            {"provenance": [_PROVENANCE], "measurementMethod": ["Estimated"], "unit": [_USD]},
        ),
    ],
)
def test_a_value_with_no_observation_in_the_window_is_absent(monkeypatch, window, expected):
    """Inside 2015-2024 only the recorded facet has points, so the method axis holds no
    declared value and is omitted."""
    options = _facet_options(
        [_SV],
        _PAIR,
        years=window,
        graph=_graph_without_facet_reads(monkeypatch),
        coverage=_index(),
        deadline=None,
    )

    assert _refs(options) == expected


def test_ranking_counts_in_window_points(monkeypatch):
    """Inside 2012-2020 the bank's three points outrank the World Bank's one, though the World
    Bank leads over the whole series."""
    options = _facet_options(
        [_SV],
        _OPEN_CREDITOR,
        years=TimeWindow(start_year=2012, end_year=2020),
        graph=_graph_without_facet_reads(monkeypatch),
        coverage=_index(),
        deadline=None,
    )

    assert _refs(options)["provenance"] == [_OTHER_PROVENANCE, _PROVENANCE]


def test_the_refless_option_ranks_last_among_equal_counts(monkeypatch):
    """Inside 2020-2029 the recorded series with no method and the Estimated projection each
    hold five points at the World Bank to Kenya pair, so the ref-less option follows."""
    options = _facet_options(
        [_SV],
        _PAIR,
        years=TimeWindow(start_year=2020, end_year=2029),
        graph=_graph_without_facet_reads(monkeypatch),
        coverage=_index(),
        deadline=None,
    )

    assert _refs(options)["measurementMethod"] == ["Estimated", None]


def _served_years(facet_filter, *, window: TimeWindow) -> set[str]:
    """Serve the World Bank to Kenya pair through materialise's per-pair preview."""
    points = {
        "rec": tuple((str(y), 1.0) for y in _RECORDED_YEARS),
        "est": tuple((str(y), 2.0) for y in _ESTIMATED_YEARS),
    }
    materialise_graph, materialise_sdmx, _requests = build_materialise_double(
        {
            (_SV, (("creditor", _WB), ("debtor", _KEN))): [
                ObsFacetFixture(
                    facet_id=facet_id,
                    points=points[facet_id],
                    provenance_id=_METADATA[facet_id].provenance_id,
                    measurement_method=_METADATA[facet_id].measurement_method,
                    unit=_METADATA[facet_id].unit,
                )
                for facet_id in ("rec", "est")
            ]
        }
    )
    observations, _warn = build_observations_planned(
        [_SV],
        _PAIR,
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        window=window,
        facet_filter=facet_filter,
        coverage=FakeCoverageService(_index()).snapshot(),
    )
    return {p.date for s in observations.series for p in s.points}


def test_pinning_each_offered_value_serves_only_its_facets(monkeypatch):
    """Every offered value, posted back as the request's facet binding, serves the facet it
    names and nothing else. The ref-less method option goes back as `"ref": null`."""
    window = TimeWindow(start_year=2022, end_year=2030)
    graph = _graph_without_facet_reads(monkeypatch)
    options = _facet_options(
        [_SV], _PAIR, years=window, graph=graph, coverage=_index(), deadline=None
    )
    binding = TypeAdapter(IdentityFacetBinding)

    served: dict[tuple[str, str | None], set[str]] = {}
    for fo in options:
        for option in fo.options:
            wire_option = json.loads(option.model_dump_json(exclude_none=True))
            ref = wire_option["value"]["ref"]["dcid"] if "ref" in wire_option["value"] else None
            pin = binding.validate_json(json.dumps({"state": "value", "ref": ref}))
            assert json.loads(binding.dump_json(pin, exclude_none=True)) == {
                "state": "value",
                "ref": ref,
            }
            source_dcids = [ref] if fo.axis == "provenance" and ref is not None else []
            facet_filter = _facet_filter_from_bindings(
                graph,
                source_dcids,
                {fo.axis: pin} if fo.axis != "provenance" else None,
                deadline=None,
            )
            served[(fo.axis, ref)] = _served_years(facet_filter, window=window)

    years = {"rec": set(_RECORDED_YEARS), "est": set(_ESTIMATED_YEARS)}
    attribute = {"provenance": "provenance_id", "measurementMethod": "measurement_method"}
    for (axis, ref), served_years in served.items():
        carrying = [f for f in years if getattr(_METADATA[f], attribute.get(axis, "unit")) == ref]
        allowed = {str(y) for f in carrying for y in years[f] if 2022 <= y <= 2030}
        assert served_years, (axis, ref)
        assert served_years <= allowed, (axis, ref)
    assert served[("measurementMethod", "Estimated")] == {str(y) for y in _ESTIMATED_YEARS}
    assert served[("measurementMethod", None)] == {"2022", "2023", "2024"}


def _discovered_role_index(svs: list[str], *, window: TimeWindow) -> dict:
    """The role index a discovered creditor scope at Kenya holds: each StatVar's facets with
    a row at the bound debtor inside `window`, one candidate per facet."""
    index: dict = {}
    for sv, facet_id, creditor, debtor, years in _ROWS:
        in_window = [y for y in years if window.start_year <= y <= window.end_year]
        if sv not in svs or debtor != _KEN or not in_window:
            continue
        metadata = _METADATA[facet_id]
        index.setdefault(sv, []).append(
            role_candidate(
                facet_id=facet_id,
                role_tuples={(("creditor", creditor), ("debtor", debtor)): len(in_window)},
                latest_date=str(max(in_window)),
                provenance_id=metadata.provenance_id,
                measurement_method=metadata.measurement_method,
                unit=metadata.unit,
            )
        )
    return index


def test_a_value_only_some_statvars_carry_is_withheld(monkeypatch):
    """Only one of the two StatVars carries the Estimated projection, so pinning it is
    refused and it is not offered. The recorded series, with no method, is then the only
    method value, and a lone ref-less option is no choice, so the method axis is omitted."""
    options = _facet_options(
        [_SV, _SV_RECORDED_ONLY],
        _OPEN_CREDITOR,
        years=TimeWindow(start_year=2022, end_year=2030),
        graph=_graph_without_facet_reads(monkeypatch),
        coverage=_index(),
        deadline=None,
    )

    assert _refs(options) == {"provenance": [_PROVENANCE], "unit": [_USD]}


def test_an_option_is_offered_exactly_when_its_pin_is_not_refused(monkeypatch):
    """Every value either StatVar carries at Kenya in the window, pinned alone, is refused by
    the discovered scope's refusal exactly when the options withhold it. The one pinnable
    value left out is the recorded series' missing method: with Estimated withheld it would
    be the lone method option, which offers no choice."""
    window = TimeWindow(start_year=2012, end_year=2030)
    svs = [_SV, _SV_RECORDED_ONLY]
    graph = _graph_without_facet_reads(monkeypatch)
    offered = {
        (axis, ref)
        for axis, refs in _refs(
            _facet_options(
                svs, _OPEN_CREDITOR, years=window, graph=graph, coverage=_index(), deadline=None
            )
        ).items()
        for ref in refs
    }
    role_index = _discovered_role_index(svs, window=window)
    carried = {
        (axis, getattr(candidate, attribute))
        for candidates in role_index.values()
        for candidate in candidates
        for axis, attribute in (
            ("provenance", "provenance_id"),
            ("measurementMethod", "measurement_method"),
            ("unit", "unit"),
        )
    }

    pinnable = {
        (axis, ref)
        for axis, ref in carried
        if candidate_facet_refusal(
            graph,
            role_index_facets(role_index),
            {
                axis: IdentityFacetValueAbsent(ref=None)
                if ref is None
                else IdentityBindingValue(ref=ref)
            },
            deadline=None,
        )
        is None
    }
    assert carried - pinnable == {
        ("provenance", _OTHER_PROVENANCE),
        ("measurementMethod", "Estimated"),
        ("measurementMethod", "BankReported"),
        ("unit", "Euro"),
    }
    assert offered == pinnable - {("measurementMethod", None)}
    assert offered == {("provenance", _PROVENANCE), ("unit", _USD)}
