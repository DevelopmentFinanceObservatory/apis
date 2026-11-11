"""A composite Search answer states the rows its Datasets mint writes: `composite_download`,
counted in memory from the coverage snapshot, equals the minted frame's `coverage.rows`,
`series_count` and years for each declared composite.

Each case mints over the fixture index's members with SDMX rows written for these tests, as
`test_mint_composite` does, and counts the same rows from the coverage snapshot that serves
them."""

from __future__ import annotations

from collections.abc import Mapping

import httpx
import pytest

from code_index.declarations import COMPOSITES, CRS_INCLUSIVE, DAC2A_INCLUSIVE
from code_index.shared import get_index
from datasets_api.config import DatasetsBuildSettings
from search.models import BaseIdentity
from search.scope.composite import composite_download, served_components
from search.scope.tables import ScopeFacts
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from .test_definition_door import _client
from .test_mint_composite import (
    _CRS_FACET,
    _CRS_YEARS,
    _CSV_HEADER,
    _DAC2A_FACET,
    _DAC2A_IMPUTED,
    _IMPUTED_FACET,
    _IMPUTED_YEARS,
    _PROVIDER,
    _RECIPIENT,
    _SDMX_PROVENANCE,
    _dac2a_oda_identity,
    _identity,
    _members,
    _mint,
)

_HEADLINE = COMPOSITES[2].provenance
_DAC1 = "provenance/OECDDAC1"
_DAC1_FACET = "facet_dac1"
_NET = "Amount_FinancialFlow_Net_Disbursement_DAC1AidType1010_Nominal"
_GRANT_EQUIVALENT = "GrantEquivalent_FinancialFlow_Disbursement_DAC1AidType1010_Nominal"
_PROVENANCE = {**_SDMX_PROVENANCE, _DAC1_FACET: _DAC1}

type _Populated = Mapping[str, tuple[str, int, tuple[int, ...]]]


def _headline_identity() -> dict:
    """France's total ODA over 2014 to 2020 on the OECD headline: net flows to 2017, grant
    equivalents from 2018."""
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {"property": "dac1AidType", "binding": {"state": "value", "ref": "DAC1AidType1010"}},
            {"property": "financialFlowType", "binding": {"state": "value", "ref": "Disbursement"}},
            {"property": "nettingBasis", "binding": {"state": "value", "ref": "Net"}},
        ],
        "entities": {"origin": [_PROVIDER]},
        "requested_window": {"kind": "range", "start_year": 2014, "end_year": 2020},
        "facets": {"provenance": {"state": "value", "ref": _HEADLINE}},
    }


def _stacked_crs() -> _Populated:
    """`test_mint_composite`'s rows with every CRS member the snapshot must hold: each over
    2019 to 2021, the first two valued 100 and 50, and one imputed member over 2020 and 2021,
    so the imputed table's other members fill zeros in two cells."""
    members = _members()
    bilateral = [member.removeprefix("dcid:") for member in members["bilateral"]]
    imputed = members["imputed"][0].removeprefix("dcid:")
    return {
        **{
            dcid: (_CRS_FACET, 100 if index == 0 else 50, _CRS_YEARS)
            for index, dcid in enumerate(bilateral)
        },
        imputed: (_IMPUTED_FACET, 7, _IMPUTED_YEARS),
    }


def _stacked_dac2a() -> _Populated:
    """Total ODA over 2018 to 2021 and the imputed line over 2020 to 2022: the composite
    keeps 2020 and 2021."""
    return {
        "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal": (
            _DAC2A_FACET,
            100,
            (2018, 2019, 2020, 2021),
        ),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021, 2022)),
    }


def _spliced_headline() -> _Populated:
    """Net flows and grant equivalents over 2012 to 2022: the composite keeps net flows over
    2014 to 2017 and grant equivalents over 2018 to 2020."""
    years = tuple(range(2012, 2023))
    return {_NET: (_DAC1_FACET, 100, years), _GRANT_EQUIVALENT: (_DAC1_FACET, 90, years)}


def _recipient(facet: str) -> str:
    """The recipient a facet's rows name: none for DAC1, which counts a donor's totals."""
    return ABSENT_ENTITY if facet == _DAC1_FACET else _RECIPIENT


def _sdmx_response(populated: _Populated, request: httpx.Request) -> httpx.Response:
    """The rows `populated` holds for the members `request` names, as SDMX CSV, with no
    recipient column for DAC1's rows."""
    named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
    rows = [
        (dcid, facet, value, year)
        for dcid, (facet, value, years) in populated.items()
        if dcid in named
        for year in years
    ]
    donor_only = any(facet == _DAC1_FACET for _dcid, facet, _value, _year in rows)
    header = _CSV_HEADER.replace("financingRecipient,", "") if donor_only else _CSV_HEADER
    recipient = "" if donor_only else f"{_RECIPIENT},"
    body = header + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{dcid},{_PROVIDER},{recipient}USD,GrantEquivalent,"
        f"NotApplicable,{_PROVENANCE[facet]},{year},{value},,{facet}\r\n"
        for dcid, facet, value, year in rows
    )
    return httpx.Response(
        200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
    )


def _coverage(populated: _Populated) -> FakeCoverageService:
    index = build_coverage_index(
        [
            (dcid, facet, _PROVIDER, _recipient(facet), list(years))
            for dcid, (facet, _value, years) in populated.items()
        ],
        role_names={
            dcid: RoleNames(
                donor="financingProvider",
                recipient=None if facet == _DAC1_FACET else "financingRecipient",
            )
            for dcid, (facet, _value, _years) in populated.items()
        },
        facet_metadata={
            facet: FacetMetadata(
                provenance_id=provenance, measurement_method="GrantEquivalent", unit="USD"
            )
            for facet, provenance in _PROVENANCE.items()
        },
    )
    return FakeCoverageService(index)


@pytest.mark.parametrize(
    ("identity", "populated"),
    [
        pytest.param(_identity(), _stacked_crs(), id=CRS_INCLUSIVE),
        pytest.param(_dac2a_oda_identity("Net"), _stacked_dac2a(), id=DAC2A_INCLUSIVE),
        pytest.param(_headline_identity(), _spliced_headline(), id=_HEADLINE),
    ],
)
def test_the_in_memory_count_is_the_minted_frame_s(identity: dict, populated: _Populated):
    coverage = _coverage(populated)
    client = _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        sdmx_transport=httpx.MockTransport(lambda request: _sdmx_response(populated, request)),
        coverage=coverage,
        artifact_store=FakeArtifactStore(),
        settings=DatasetsBuildSettings.from_env(),
    )
    facts = ScopeFacts(index=get_index(), coverage=coverage.snapshot())

    minted = _mint(client, identity=identity)
    counted = composite_download(
        served_components(identity, facts=facts),
        BaseIdentity.model_validate(identity),
        facts=facts,
    )

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    assert manifest["coverage"]["rows"] > 0
    assert counted is not None
    assert counted.model_dump(exclude={"licence"}) == {
        "rows": manifest["coverage"]["rows"],
        "rows_exact": False,
        "series": manifest["series_count"],
        "window": {
            "start_year": manifest["coverage"]["window"]["start_year"],
            "end_year": manifest["coverage"]["window"]["end_year"],
        },
    }


def test_a_member_the_snapshot_does_not_hold_leaves_the_counts_null_and_keeps_the_licence():
    """A CRS member with no row in the snapshot adds rows the count cannot see, so the rows,
    series and window are null, where the same composite counts them with the member held. The
    imputed table's members the snapshot lacks are zero-filled, which the count covers."""
    populated = _stacked_crs()
    missing = next(
        dcid for dcid, (facet, _value, _years) in populated.items() if facet == _CRS_FACET
    )
    partial = {dcid: held for dcid, held in populated.items() if dcid != missing}
    identity = _identity()

    def count(held: _Populated):
        facts = ScopeFacts(index=get_index(), coverage=_coverage(held).snapshot())
        return composite_download(
            served_components(identity, facts=facts),
            BaseIdentity.model_validate(identity),
            facts=facts,
        )

    full, uncovered = count(populated), count(partial)

    assert full is not None
    assert full.rows is not None
    assert uncovered is not None
    assert uncovered.model_dump() == {
        "rows": None,
        "rows_exact": False,
        "series": None,
        "window": None,
        "licence": full.licence,
    }


def test_the_snapshot_settles_a_component_s_years_only_while_it_holds_every_member():
    """A component's years are settled (`YearCounts.settled`) while the snapshot holds a row
    of each StatVar of its family, so an empty count there is one no series holds, and not
    once it lacks one, where an empty count may be the snapshot not knowing."""
    populated = _stacked_crs()
    missing = next(
        dcid for dcid, (facet, _value, _years) in populated.items() if facet == _CRS_FACET
    )
    partial = {dcid: held for dcid, held in populated.items() if dcid != missing}

    def settled(held: _Populated) -> dict[str, bool]:
        facts = ScopeFacts(index=get_index(), coverage=_coverage(held).snapshot())
        return {
            item.component.role: item.years.settled
            for item in served_components(_identity(), facts=facts)
        }

    assert settled(populated)["bilateral"]
    assert not settled(partial)["bilateral"]
