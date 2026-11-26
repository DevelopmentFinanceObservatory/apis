"""The year before a range's partial year, which a mint reads only to count its reported share
(`generation._row_pass_mask`, `generation._share_totals`): the served frame, its warnings and
its ceiling exclude those rows. The rows' source marks 2022 partial."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from materialise import MaterialisationResult, materialise_group, materialise_spec
from materialise.enrichment import EnrichmentLookup
from materialise.facets import FacetBinding, UnitRequest
from materialise.rate_tables import RateTableCache
from materialise.reported_share import reported_share
from materialise.window import RequestedWindow
from origin_catalogue import bound_catalogue
from tests.origin_catalogue.fake_catalogue import documenting

from .test_conversion_finish import (
    _BARE_STAT_VAR,
    _EXCHANGE_VAR,
    _MEMBER_PROVENANCE,
    _ROLE_HEADER,
    _STAT_VAR,
    _SUBJECT_HEADER,
    _base_arcs,
    _coverage_for,
    _graph,
    _rate_row,
    _request,
    _role_row,
    _router_sdmx,
)

_KEN = "country/KEN"
_REGION = "southOfSahara"
_WINDOW = RequestedWindow(
    kind="range", start_year=2022, end_year=2022, include=frozenset({"partial"})
)


@pytest.fixture(autouse=True)
def _marked_2022() -> Iterator[None]:
    with bound_catalogue(documenting({_MEMBER_PROVENANCE: {"partialObservationDate": ("2022",)}})):
        yield


def _minted(
    entry: str, *, region_years: tuple[str, ...], ceiling: int = 1_000
) -> MaterialisationResult:
    """A 2022 range mint into each recipient's own currency of Kenya's 2021 and 2022 rows and
    a region's rows in `region_years`. No currency names the region."""
    arcs = _base_arcs()
    arcs["CurrencyKES"] = {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": (_KEN,)}
    rows = [
        _role_row(recipient=_KEN, year="2021", value="100"),
        _role_row(recipient=_KEN, year="2022", value="1"),
        *(_role_row(recipient=_REGION, year=year, value="50") for year in region_years),
    ]
    rates = _SUBJECT_HEADER + "".join(
        _rate_row(stat_var=_EXCHANGE_VAR, entity=entity, year=year, value=value)
        for entity, value in (("CurrencyUSD", "1.0"), ("CurrencyKES", "100.0"))
        for year in ("2021", "2022")
    )
    sdmx, _ = _router_sdmx(
        role_bodies={_BARE_STAT_VAR: _ROLE_HEADER + "".join(rows)},
        subject_bodies={_EXCHANGE_VAR: rates},
    )
    request = _request(
        stat_var=_STAT_VAR,
        unit=FacetBinding(state="request", request=UnitRequest("destination", "current")),
        window=_WINDOW,
        role_bindings={"financingProvider": ["country/FRA"], "financingRecipient": [_KEN, _REGION]},
    )
    common = {
        "ceiling": ceiling,
        "sdmx": sdmx,
        "graph": _graph(arcs),
        "materialiser_build": "test-build",
        "capability_version": "v1",
        "enrichment": EnrichmentLookup(_graph(arcs)),
        "parties": {},
        "rate_cache": RateTableCache(),
        "coverage": _coverage_for([(_BARE_STAT_VAR, _KEN), (_BARE_STAT_VAR, _REGION)]),
    }
    if entry == "group":
        return materialise_group([request], **common)
    return materialise_spec(request, **common)


def _codes(result: MaterialisationResult) -> set[str]:
    return {warning["code"] for warning in result.warnings}


@pytest.mark.parametrize("entry", ["member", "group"])
def test_an_entity_only_in_the_prior_year_raises_no_unmapped_warning(entry: str) -> None:
    result = _minted(entry, region_years=("2021",))

    assert "currency_unmapped_omitted" not in _codes(result)
    (frame,) = result.frame_chunks()
    assert frame.table["time_period"].to_pylist() == ["2022"]
    assert result.coverage.rows == 1
    (totals,) = result.year_totals.values()
    # The share counts the rows read, unconverted, so the region's 2021 row counts too.
    assert totals.series == {2021: 2, 2022: 1}


@pytest.mark.parametrize("entry", ["member", "group"])
def test_control_an_entity_served_in_the_year_raises_the_warning(entry: str) -> None:
    assert "currency_unmapped_omitted" in _codes(_minted(entry, region_years=("2022",)))


@pytest.mark.parametrize("entry", ["member", "group"])
def test_the_row_ceiling_counts_the_served_year_alone(entry: str) -> None:
    """Kenya's one 2022 row is all the frame serves, so a one-row ceiling holds, though the
    read also holds Kenya's 2021 row."""
    result = _minted(entry, region_years=(), ceiling=1)

    assert result.coverage.rows == 1


@pytest.mark.parametrize("entry", ["member", "group"])
def test_share_rows_past_the_ceiling_state_no_share(entry: str) -> None:
    """Kenya's and the region's 2021 rows, read only to count the share, are two rows past a
    one-row ceiling. The read serves what a two-row ceiling serves, and its pair states no
    share."""
    capped = _minted(entry, region_years=("2021",), ceiling=1)
    control = _minted(entry, region_years=("2021",), ceiling=2)

    assert capped.frame_sha256 == control.frame_sha256
    assert capped.coverage == control.coverage
    assert capped.held_runs == control.held_runs
    assert reported_share(list(control.year_totals.values()), year=2022) is not None
    assert reported_share(list(capped.year_totals.values()), year=2022) is None
