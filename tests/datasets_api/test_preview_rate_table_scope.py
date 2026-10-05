"""A multi-member preview converts every member against one version of each rate table, even when
the table's provenance refreshes between members."""

from __future__ import annotations

import pytest

from datasets_api.preview import preview_requests
from datasets_api.revision import SentinelRevisionReader
from materialise.facets import FacetBinding, UnitRequest
from materialise.rate_tables import RateTableCache
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService
from tests.materialise.test_conversion_generation import (
    _BARE_STAT_VAR,
    _BARE_STAT_VAR_B,
    _STAT_VAR,
    _STAT_VAR_B,
    _coverage_for,
    _exchange_data_requests,
    _refreshing_fixture,
    _request,
)

from ._app_factories import mint_deps
from ._artifact_doubles import FakeArtifactStore
from .conftest import FakeDatasetStore


def test_preview_members_share_one_table_when_the_refresh_date_moves_between_loads():
    """Each member's call reads the table once. Without one request-scoped view the second
    member would load the refreshed table and convert at 0.8 in place of 0.9."""
    sdmx, requests, graph = _refreshing_fixture()
    eur_request = FacetBinding(state="request", request=UnitRequest("EUR", "current"))
    store = FakeDatasetStore()
    deps = mint_deps(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        rate_cache=RateTableCache(),
        coverage=FakeCoverageService(
            _coverage_for([(_BARE_STAT_VAR, "country/KEN"), (_BARE_STAT_VAR_B, "country/KEN")])
        ),
    )

    envelope = preview_requests(
        [
            _request(stat_var=_STAT_VAR, unit=eur_request),
            _request(stat_var=_STAT_VAR_B, unit=eur_request),
        ],
        deps=deps,
        limit=10,
    )

    values = [
        member.sample.rows[0][member.sample.columns.index("value")] for member in envelope.members
    ]
    assert values == [pytest.approx(90.0), pytest.approx(180.0)]
    assert len(_exchange_data_requests(requests)) == 1
    assert graph.refresh_reads == 2
