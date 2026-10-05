"""Pins `ResolvedSpec.download` against Datasets mints recorded on dev.

`tests/fixtures/download_mints/manifests.json` holds, per case, the query and entry kind
the Spec came from, the mint's `coverage.rows`, `series_count` and window, and the read
inputs the answer planned from: a standard answer's facets at its place, as the Data
Commons observation read returned them, and a role-bearing answer's bound roles and the
donors its conversion left out. `coverage_rows.json.gz` holds the coverage rows of each
role-bearing answer's members under its bound roles (coverage build
`7a081f21b36a988a91ddfa2d9a9df700`), one facet per case, entities and variables indexed
and each row's years as a hex bit mask from 1960.

Each test plans the read the engine plans from those inputs and asserts the download
facts equal the recorded mint. Ethiopia's mint served a head stored before manifests
carried `series_count`, so its series count is not pinned.
"""

from __future__ import annotations

import gzip
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.retrieval.download import download_facts
from search.engine.retrieval.read_plan import PlanSeries, plan_pairs, plan_years, role_years
from search.engine.retrieval.role_resolve import _role_plan
from search.engine.unit.conversion import CountedConversion
from search.engine.unit.conversion_catalogue import RecipeTables
from search.models import RequestedWindow, TimeWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import build_coverage_index

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "download_mints"
_CRS_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_EPOCH_YEAR = 1960


@cache
def _cases() -> dict[str, dict[str, Any]]:
    cases = json.loads((_FIXTURES / "manifests.json").read_text())["cases"]
    return {case["name"]: case for case in cases}


@cache
def _coverage_rows() -> dict[str, dict[str, Any]]:
    return json.loads(gzip.decompress((_FIXTURES / "coverage_rows.json.gz").read_bytes()))


def _date_request(window: str | dict[str, int]) -> DateRequest | None:
    if window == "all":
        return None
    if window == "latest":
        return DateRequest(window=None, latest=True)
    assert isinstance(window, dict)
    return DateRequest(window=TimeWindow(**window), latest=False)


def _requested_window(window: str | dict[str, int]) -> RequestedWindow | None:
    if window == "all":
        return None
    if window == "latest":
        return "latest"
    assert isinstance(window, dict)
    return TimeWindow(**window)


def _coverage_index(name: str) -> CoverageIndex:
    recorded = _coverage_rows()[name]
    variables, entities = recorded["variables"], recorded["entities"]
    rows = [
        (
            variables[variable],
            recorded["facet_id"],
            entities[donor],
            entities[recipient],
            [
                _EPOCH_YEAR + bit
                for bit in range(int(mask, 16).bit_length())
                if int(mask, 16) >> bit & 1
            ],
        )
        for variable, donor, recipient, mask in recorded["rows"]
    ]
    return build_coverage_index(rows, role_names=dict.fromkeys(variables, _CRS_ROLES))


def _assert_matches_mint(case: dict[str, Any], download: Any) -> None:
    mint = case["mint"]
    assert download.rows == mint["rows"]
    assert download.window == TimeWindow(**mint["window"])
    if mint["series_count"] is not None:
        assert download.series == mint["series_count"]


@pytest.mark.parametrize("name", ["pakistan", "latest_pakistan", "ethiopia", "india_gdp"], ids=str)
def test_a_standard_answer_states_the_rows_its_mint_wrote(name: str) -> None:
    """India's nominal GDP holds five facets at the place; the file holds the elected
    one's 65 rows, where the facets' observations sum to 109."""
    case = _cases()[name]
    series = case["series"]
    facets = tuple(Facet(**facet) for facet in series["facets"])
    plan = plan_pairs(
        [
            PlanSeries(
                stat_var=series["stat_var"],
                role_tuple=(("observationAbout", series["place"]),),
                candidates=facets,
            )
        ],
        members=[series["stat_var"]],
        years=plan_years(_date_request(case["requested_window"])),
    )

    download = download_facts(
        plan,
        build_coverage_index([], role_names={}),
        requested_window=_requested_window(case["requested_window"]),
        conversion=None,
        provenance=None,
    )

    _assert_matches_mint(case, download)
    assert download.rows_exact is True


@pytest.mark.parametrize(
    "name",
    ["germany", "germany_to_kenya_eur", "kenya_donor_currency", "latest_france_to_kenya"],
    ids=str,
)
def test_a_role_answer_states_the_rows_its_mint_wrote(name: str) -> None:
    """Germany's 1,626 members serve 152,601 rows across 32,490 series; the latest window
    serves each series' own latest year (308 rows from 1991 to 2024), where a bulk preview
    reads 2024 alone; the donor-currency answer leaves out the 27 donors no currency names."""
    case = _cases()[name]
    bulk = case["bulk"]
    index = _coverage_index(name)
    recorded = _coverage_rows()[name]
    members, entities = recorded["variables"], recorded["entities"]
    # An open role reads every entity holding a row on it.
    recipients = sorted({entities[recipient] for _v, _d, recipient, _m in recorded["rows"]})
    donors = sorted({entities[donor] for _v, donor, _r, _m in recorded["rows"]})
    plan = _role_plan(
        members,
        member_facts={},
        names={"destination": "financingRecipient", "origin": "financingProvider"},
        subject_dcids=recipients,
        origin_dcids=donors,
        bound_roles={role: frozenset(dcids) for role, dcids in bulk["bound"].items()},
        years=role_years(_date_request(case["requested_window"]), index.horizon),
        coverage_index=index,
    ).omitting(
        frozenset(
            (sv, "financingProvider", donor) for sv in members for donor in bulk["omitted_donors"]
        )
    )

    download = download_facts(
        plan,
        index,
        requested_window=_requested_window(case["requested_window"]),
        conversion=_current_price_conversion(donors, bulk) if case["converts"] else None,
        provenance=None,
    )

    _assert_matches_mint(case, download)
    assert download.rows_exact is True


def _current_price_conversion(donors: list[str], bulk: dict[str, Any]) -> CountedConversion:
    """The recorded mints' current-price conversion of US-dollar CRS rows: into euros, or
    into each donor's own currency, which no currency names for the donors the mint left
    out. A current-price conversion reads no deflator, so its tables bound no row here."""
    omitted = set(bulk["omitted_donors"])
    return CountedConversion(
        tables=RecipeTables(rate_years={}, deflator_years={}, base_year_min=0, base_year_max=0),
        recipe=RECIPE_BODIES["ConversionRecipeDAC"],
        natives=frozenset({"USD"}),
        base_year=None,
        deflator_role=None,
        target_role="financingProvider" if omitted else None,
        named=frozenset(donor for donor in donors if donor not in omitted),
    )
