"""A mint of a definition whose window names no end year reads to the current year and is
judged against what the definition asked: data that ends before the current year falls
short of nothing, so the manifest's window has origin `query` and carries no
`coverage_below_scope`. A stated bound the data falls short of is still a clamp.
"""

from __future__ import annotations

import httpx
import pytest

from datasets_api.kinds.spec_id import WindowOverCap, check_window_span
from datasets_api.shape import StatVarShape
from dc_sdmx import SdmxClient
from materialise import window as window_module
from materialise.graph import Facet
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames, YearHorizon
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR = "MalariaGrantDisbursements"
_PLACE = "country/KEN"
_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _client(years: range):
    """A mint door over one series holding a point in each of `years`."""
    facet = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=len(years),
        earliest_date=str(years[0]),
        latest_date=str(years[-1]),
    )
    graph = FakeMintGraphReader(
        arcs={
            _STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        },
        facets={(_STAT_VAR, _PLACE): [facet]},
        policies={},
        labels={"OECD_DAC2A": "OECD DAC2a", _PLACE: "Kenya"},
    )
    body = _CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_PLACE},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},1000,,facet1\r\n"
        for year in years
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )
    coverage = build_coverage_index(
        [(_STAT_VAR, "facet1", ABSENT_ENTITY, _PLACE, list(years))],
        role_names={_STAT_VAR: RoleNames(donor=None, recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(base_url="https://datacommons.one.org/core/api", transport=transport),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=FakeCoverageService(coverage),
    )


def _post(years: range, window: dict[str, object]) -> httpx.Response:
    return _client(years).post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [f"dcid:{_STAT_VAR}"],
                "scope": {
                    "entities": {"recipient": [_PLACE]},
                    "facets": {
                        "provenance": {"ref": "OECD_DAC2A"},
                        "measurementMethod": {"ref": "GrantEquivalent"},
                        "unit": {"ref": "USD"},
                    },
                    "window": window,
                },
            },
            "options": {"bundle": False},
        },
    )


def _mint(years: range, window: dict[str, object]) -> dict:
    response = _post(years, window)
    assert response.status_code == 200, response.text
    return response.json()


_SINCE_2016 = {"kind": "range", "start_year": 2016}


@pytest.mark.parametrize(
    ("years", "window", "applied", "warned"),
    [
        (range(2016, 2022), _SINCE_2016, (2016, 2021, "query"), False),
        (range(2016, 2027), _SINCE_2016, (2016, 2026, "query"), False),
        (range(2016, 2031), _SINCE_2016, (2016, 2026, "query"), False),
        (range(2018, 2022), _SINCE_2016, (2018, 2021, "coverage_clamp"), True),
        (
            range(2016, 2022),
            {"kind": "range", "start_year": 2016, "end_year": 2026},
            (2016, 2021, "coverage_clamp"),
            True,
        ),
    ],
    ids=[
        "open_end_data_ends_before_the_clock",
        "open_end_data_ends_at_the_clock",
        "open_end_data_ends_after_the_clock",
        "open_end_data_starts_after_the_stated_start",
        "closed_end_data_ends_before_it",
    ],
)
def test_a_mint_judges_its_window_against_the_years_the_definition_states(
    monkeypatch: pytest.MonkeyPatch,
    years: range,
    window: dict[str, object],
    applied: tuple[int, int, str],
    warned: bool,
) -> None:
    """Under a 2026 clock "since 2016" reads 2016 to 2026. Data ending in 2021, in 2026 or
    in 2030 answers the whole of it: the manifest's window ends at the last year served,
    2026 at the latest, with origin `query` and no `coverage_below_scope`. The start year
    the definition states and a stated end year are held to the data: a series starting in
    2018, and a closed 2016 to 2026 over data ending in 2021, are clamps and warn."""
    monkeypatch.setattr(window_module, "current_year", lambda: 2026)

    manifest = _mint(years, window)

    served = manifest["coverage"]["window"]
    assert (served["start_year"], served["end_year"], served["origin"]) == applied
    codes = {warning["code"] for warning in manifest["warnings"]}
    assert ("coverage_below_scope" in codes) == warned
    assert manifest["coverage"]["rows"] == applied[1] - applied[0] + 1
    assert manifest["definition"]["identity"]["requested_window"] == window


def test_a_new_calendar_year_leaves_an_open_ended_dataset_whose_data_has_ended_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Since 2016" over data ending in 2021 mints the same content under a 2026 and a 2027
    clock: the year the range ends at moves, and the window served and the rows do not."""
    minted = []
    for clock in (2026, 2027):
        monkeypatch.setattr(window_module, "current_year", lambda clock=clock: clock)
        minted.append(_mint(range(2016, 2022), _SINCE_2016))

    assert minted[0]["coverage"]["window"] == minted[1]["coverage"]["window"]
    assert minted[0]["content_version"] == minted[1]["content_version"]


def test_a_role_bearing_mint_refuses_a_window_its_role_read_cannot_list(
    monkeypatch: pytest.MonkeyPatch, search_settings
) -> None:
    """A role-bearing window is admitted by the rule Search's role routes read its years by
    (`search.scope_window.role_read_years`): floored at 1960, capped at the coverage
    horizon's last year (2023 here), and at most `scope_max_window_years` long. Over the cap
    the mint refuses `request_invalid` at the window before any build, where Search's resubmit
    of the minted identity would refuse `scope_too_broad`; at the cap it mints."""
    monkeypatch.setattr(window_module, "current_year", lambda: 2026)
    search_settings(scope_max_window_years=5)
    years = range(2016, 2022)

    refused = _post(years, {"kind": "range", "start_year": 1900, "end_year": 2020})

    assert refused.status_code == 400, refused.text
    problem = refused.json()
    assert (problem["code"], problem["pointer"]) == ("request_invalid", "/input/scope/window")
    assert (
        problem["detail"]
        == "A window reads at most 5 years of data; 1900 to 2020 reads 1960 to 2020, which is "
        "61 years."
    )
    minted = _mint(years, {"kind": "range", "start_year": 2016, "end_year": 2020})
    assert minted["coverage"]["rows"] == 5


_HORIZON = YearHorizon(epoch_year=1960, mask_words=1)


def _shape(*roles: str) -> StatVarShape:
    return StatVarShape(
        stat_var="dcid:Exemplar",
        declared_roles=roles,
        slots=(),
        measured_property=None,
        population_type=None,
    )


@pytest.mark.parametrize(
    ("window", "roles", "refused"),
    [
        ({"kind": "range", "start_year": 1900, "end_year": 2100}, ("recipient",), True),
        ({"kind": "range", "end_year": 2100}, ("donor", "recipient"), True),
        ({"kind": "range", "start_year": 1900, "end_year": 2100}, ("observationAbout",), False),
        ({"kind": "range", "start_year": 1950, "end_year": 1964}, ("recipient",), False),
        ({"kind": "range", "start_year": 2030, "end_year": 2100}, ("recipient",), False),
        ({"kind": "all"}, ("recipient",), False),
    ],
    ids=[
        "role_window_over_the_cap",
        "open_start_floors_at_1960",
        "subject_only_reads_every_year",
        "floored_window_within_the_cap",
        "window_past_the_horizon_holds_no_year",
        "all_years",
    ],
)
def test_check_window_span_counts_the_years_a_role_read_lists(
    search_settings, window: dict[str, object], roles: tuple[str, ...], refused: bool
) -> None:
    """Only a role-bearing range is counted, over the floored and horizon-capped years.
    A range left with no year is no window-length refusal: Search reads it as no data."""
    search_settings(scope_max_window_years=10)

    def check() -> None:
        check_window_span(
            window, shapes=(_shape(*roles),), horizon=_HORIZON, pointer="/input/scope/window"
        )

    if refused:
        with pytest.raises(WindowOverCap):
            check()
    else:
        check()


@pytest.mark.parametrize(
    ("window", "message"),
    [
        (
            {"kind": "range", "start_year": 1960, "end_year": 2010},
            "A window reads at most 10 years of data; 1960 to 2010 is 51 years.",
        ),
        (
            {"kind": "range", "start_year": 0, "end_year": 3000},
            "A window reads at most 10 years of data; 0 to 3000 reads 1960 to 2023, which is "
            "64 years.",
        ),
        (
            {"kind": "range", "end_year": 2000},
            "A window reads at most 10 years of data; up to 2000 reads 1960 to 2000, which is "
            "41 years.",
        ),
        (
            {"kind": "range", "start_year": 1990},
            "A window reads at most 10 years of data; from 1990 reads 1990 to 2015, which is "
            "26 years.",
        ),
    ],
    ids=["within_floor_and_horizon", "past_both", "open_start", "open_end_at_the_clock"],
)
def test_a_window_refusal_names_the_years_it_reads_beside_the_years_asked_when_they_differ(
    monkeypatch: pytest.MonkeyPatch, search_settings, window: dict[str, object], message: str
) -> None:
    """The refusal states the cap and the years the read lists, floored at 1960 and capped at
    the 2023 horizon, and names the years asked beside them only where a bound moved. Under a
    2015 clock an open end reads to 2015."""
    monkeypatch.setattr(window_module, "current_year", lambda: 2015)
    search_settings(scope_max_window_years=10)
    with pytest.raises(WindowOverCap) as refused:
        check_window_span(
            window, shapes=(_shape("recipient"),), horizon=_HORIZON, pointer="/input/scope/window"
        )
    assert str(refused.value) == message
