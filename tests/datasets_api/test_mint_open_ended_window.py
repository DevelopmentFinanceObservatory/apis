"""A mint of a definition whose window names no end year reads to the current year and is
judged against what the definition asked: data that ends before the current year falls
short of nothing, so the manifest's window has origin `query` and carries no
`coverage_below_scope`. A stated bound the data falls short of is still a clamp.
"""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import window as window_module
from materialise.graph import Facet
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
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


def _mint(years: range, window: dict[str, object]) -> dict:
    response = _client(years).post(
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
