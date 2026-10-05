"""A multi-member preview in each recipient's own currency refuses `currency_unmapped` as its
collection mint does: only when no member serves a row."""

from __future__ import annotations

import pytest

from datasets_api.preview import preview_requests
from datasets_api.revision import SentinelRevisionReader
from materialise.rate_tables import RateTableCache
from materialise.recipes import RateCoverageGapError
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService
from tests.materialise.test_conversion_finish import (
    _BARE_STAT_VAR,
    _BARE_STAT_VAR_B,
    _destination_currency_inputs,
    _graph,
)

from ._app_factories import mint_deps
from ._artifact_doubles import FakeArtifactStore
from .conftest import FakeDatasetStore


def _preview(first: list[str], second: list[str]):
    """Preview the first StatVar at the `first` recipients beside the second at `second`."""
    inputs = _destination_currency_inputs({_BARE_STAT_VAR: first, _BARE_STAT_VAR_B: second})
    store = FakeDatasetStore()
    deps = mint_deps(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(inputs.arcs),
        sdmx=inputs.sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        rate_cache=RateTableCache(),
        coverage=FakeCoverageService(inputs.coverage),
    )
    return preview_requests(inputs.requests, deps=deps, limit=10)


def test_a_member_whose_rows_are_all_omitted_previews_empty_beside_a_served_member():
    kenya, region = _preview(["country/KEN"], ["africaRegion"]).members
    assert (kenya.total_rows, region.total_rows) == (1, 0)
    assert [w.code for w in region.warnings if w.code == "currency_unmapped_omitted"] == [
        "currency_unmapped_omitted"
    ]


def test_a_preview_whose_every_member_is_omitted_refuses_naming_them_all():
    with pytest.raises(RateCoverageGapError) as exc_info:
        _preview(["africaRegion"], ["southOfSahara"])
    assert (exc_info.value.cause, exc_info.value.entities) == (
        "currency_unmapped",
        ("africaRegion", "southOfSahara"),
    )
