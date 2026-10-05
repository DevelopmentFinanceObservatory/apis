"""Live SDMX smoke tests, skipped unless `DFO_SDMX_LIVE_BASE` is set.
Use a Cloud Run API root ending in `/core/api`; set `DFO_SDMX_LIVE_TOKEN` behind a WAF.
`DFO_SDMX_LIVE_REQUIRED` makes collection fail when the base URL is unset.
"""

from __future__ import annotations

import os

import pytest

from dc_sdmx.client import SdmxClient

_LIVE_BASE = os.environ.get("DFO_SDMX_LIVE_BASE")

if not _LIVE_BASE and os.environ.get("DFO_SDMX_LIVE_REQUIRED"):
    raise RuntimeError(
        "DFO_SDMX_LIVE_REQUIRED is set but DFO_SDMX_LIVE_BASE is empty -- this "
        "suite would skip every test and report green over zero live calls"
    )

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _LIVE_BASE, reason="set DFO_SDMX_LIVE_BASE to run live SDMX smoke tests"
    ),
]

# Engine base queries need this count; an authenticated deployment returning no rows is unusable.
_PROBE_VARIABLE = "Count_Person"
_PROBE_ENTITY = "country/USA"
_PROBE_ROLE = "observationAbout"


@pytest.fixture
def live_client():
    client = SdmxClient(base_url=_LIVE_BASE, token=os.environ.get("DFO_SDMX_LIVE_TOKEN"))
    try:
        yield client
    finally:
        client.close()


def test_live_data_query_returns_observations(live_client):
    result = live_client.data(variables=[_PROBE_VARIABLE], entities={_PROBE_ROLE: [_PROBE_ENTITY]})
    assert _PROBE_ROLE in result.roles
    assert result.rows
    for row in result.rows:
        assert row.variable == _PROBE_VARIABLE
        assert row.role_values.get(_PROBE_ROLE) == _PROBE_ENTITY
    assert any(row.value_as_float() is not None for row in result.rows)


def test_live_availability_query_returns_filtered_member(live_client):
    result = live_client.availability(
        variables=[_PROBE_VARIABLE],
        component=_PROBE_ROLE,
        filters={_PROBE_ROLE: [_PROBE_ENTITY]},
    )
    assert _PROBE_ENTITY in result.components.get(_PROBE_ROLE, ())


def test_live_availability_query_time_period_two_year_list(live_client):
    # The 2020-2021 OR-list covers years USA has population data.
    result = live_client.availability(
        variables=[_PROBE_VARIABLE],
        component=_PROBE_ROLE,
        filters={_PROBE_ROLE: [_PROBE_ENTITY]},
        time_period=["2020", "2021"],
    )
    assert _PROBE_ENTITY in result.components.get(_PROBE_ROLE, ())


def test_live_data_time_period_latest_returns_fewer_rows(live_client):
    # USA's history has multiple observations; LATEST returns one per facet series.
    unfiltered = live_client.data(
        variables=[_PROBE_VARIABLE], entities={_PROBE_ROLE: [_PROBE_ENTITY]}
    )
    latest = live_client.data(
        variables=[_PROBE_VARIABLE],
        entities={_PROBE_ROLE: [_PROBE_ENTITY]},
        time_period="LATEST",
    )
    assert latest.rows
    assert len(latest.rows) < len(unfiltered.rows)
