"""A composite Search answer states the rows its Datasets mint writes: `composite_download`,
counted in memory from the coverage snapshot, equals the minted frame's `coverage.rows`,
`series_count` and years for each declared composite.

Each case mints over the fixture index's members with SDMX rows written for these tests, as
`test_mint_composite` does, and counts the same rows from the coverage snapshot that serves
them."""

from __future__ import annotations

import contextvars
from collections.abc import Mapping, Sequence

import httpx
import pytest

from code_index.composites import component_identities
from code_index.declarations import COMPOSITES, CRS_INCLUSIVE, DAC2A_INCLUSIVE
from code_index.shared import get_index
from datasets_api.config import DatasetsBuildSettings
from materialise import composite as composite_module
from origin_catalogue import CatalogueNotBound, current_catalogue, scoped_reads
from search.models import BaseIdentity, ResolvedStatVars
from search.scope.composite import (
    CompositeMint,
    ServedComponent,
    component_years,
    composite_download,
    donor_years,
    served_components,
)
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

_HEADLINE = COMPOSITES[3].provenance
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


_OTHER_DONOR = "country/DEU"
_BILATERAL_206 = "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal"

type _SnapshotRow = tuple[str, str, str, str, list[int]]


def _rows_of(populated: _Populated) -> list[_SnapshotRow]:
    """The snapshot rows of France to Kenya `populated` holds."""
    return [
        (dcid, facet, _PROVIDER, _RECIPIENT, list(years))
        for dcid, (facet, _value, years) in populated.items()
    ]


def _dac2a_case(
    populated: _Populated, snapshot: Sequence[_SnapshotRow]
) -> tuple[httpx.Response, list[ServedComponent]]:
    """France's DAC2a ODA to Kenya including imputed ODA, minted from SDMX rows `populated`
    holds over a coverage snapshot of `snapshot`, and Search's served components over the
    same snapshot."""
    coverage = _dac2a_coverage(snapshot)
    client = _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        sdmx_transport=httpx.MockTransport(lambda request: _sdmx_response(populated, request)),
        coverage=coverage,
        artifact_store=FakeArtifactStore(),
        settings=DatasetsBuildSettings.from_env(),
    )
    identity = _dac2a_oda_identity("Net")
    facts = ScopeFacts(index=get_index(), coverage=coverage.snapshot())
    return _mint(client, identity=identity), served_components(identity, facts=facts)


def _dac2a_coverage(snapshot: Sequence[_SnapshotRow]) -> FakeCoverageService:
    """A coverage snapshot of DAC2a rows `snapshot`."""
    return FakeCoverageService(
        build_coverage_index(
            list(snapshot),
            role_names={
                dcid: RoleNames(donor="financingProvider", recipient="financingRecipient")
                for dcid, *_rest in snapshot
            },
            facet_metadata={
                facet: FacetMetadata(
                    provenance_id=provenance, measurement_method="GrantEquivalent", unit="USD"
                )
                for facet, provenance in _PROVENANCE.items()
            },
        )
    )


def test_a_stacked_composite_serves_the_years_its_components_hold_for_the_frame_s_keys():
    """Another donor's imputed line starts in 2018, so the imputed table covers 2018, while
    France's to Kenya starts in 2020. A composite of France to Kenya serves 2020 and 2021,
    the years both lines hold for its keys: Search states them, and the mint keeps them and
    drops total ODA's 2018 and 2019."""
    populated: _Populated = {
        _BILATERAL_206: (_DAC2A_FACET, 100, (2018, 2019, 2020, 2021)),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021, 2022)),
    }

    minted, served = _dac2a_case(
        populated,
        [
            *_rows_of(populated),
            (_DAC2A_IMPUTED, _DAC2A_FACET, _OTHER_DONOR, _RECIPIENT, [2018, 2019]),
        ],
    )

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    by_role = {entry["role"]: entry for entry in manifest["trace"]["components"]}
    assert by_role["bilateral"]["dropped_years"] == [2018, 2019]
    assert by_role["bilateral"]["periods"] == [[2020, 2021]]
    window = manifest["coverage"]["window"]
    assert (window["start_year"], window["end_year"]) == (2020, 2021)
    assert {item.component.role: item.served for item in served} == {
        "bilateral": frozenset({2020, 2021}),
        "imputed": frozenset({2020, 2021}),
    }


def test_a_component_s_held_years_read_only_the_provenance_it_pins():
    """DAC2a's imputed line from France to Kenya holds 2018 and 2019 only under the CRS's
    provenance, which the imputed component's DAC2a pin does not read. The composite serves
    2020 and 2021, the years the line holds under DAC2a, so the mint drops total ODA's 2018
    and 2019 in place of serving them alone, and Search states the same years and counts
    the imputed line's series in those alone."""
    populated: _Populated = {
        _BILATERAL_206: (_DAC2A_FACET, 100, (2018, 2019, 2020, 2021)),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021)),
    }

    minted, served = _dac2a_case(
        populated,
        [*_rows_of(populated), (_DAC2A_IMPUTED, _CRS_FACET, _PROVIDER, _RECIPIENT, [2018, 2019])],
    )

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    by_role = {entry["role"]: entry for entry in manifest["trace"]["components"]}
    assert by_role["bilateral"]["dropped_years"] == [2018, 2019]
    assert by_role["bilateral"]["periods"] == [[2020, 2021]]
    assert {item.component.role: item.served for item in served} == {
        "bilateral": frozenset({2020, 2021}),
        "imputed": frozenset({2020, 2021}),
    }
    assert {item.component.role: item.years.held for item in served} == {
        "bilateral": frozenset({2018, 2019, 2020, 2021}),
        "imputed": frozenset({2020, 2021}),
    }


def test_a_part_s_source_scope_leaves_a_composite_s_component_reads_alone():
    """A part read under the CRS's source scope builds its scope over the DAC2a composite.
    Each component reads under its own DAC2a pin alone, as the composite answer
    (`region.composite_answer.composite_served`) and the mint read it, so the composite
    serves the years and counts it serves under no source scope."""
    populated: _Populated = {
        _BILATERAL_206: (_DAC2A_FACET, 100, (2018, 2019, 2020, 2021)),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021)),
    }
    coverage = _dac2a_coverage(_rows_of(populated)).snapshot()
    identity = _dac2a_oda_identity("Net")

    with scoped_reads(frozenset({_PROVENANCE[_CRS_FACET]})):
        scoped = served_components(identity, facts=ScopeFacts(index=get_index(), coverage=coverage))
    unscoped = served_components(identity, facts=ScopeFacts(index=get_index(), coverage=coverage))

    assert [(item.served, item.years) for item in scoped] == [
        (item.served, item.years) for item in unscoped
    ]
    assert {item.served for item in unscoped} == {frozenset({2020, 2021})}


def test_a_plain_pinned_table_read_keeps_the_part_s_source_scope():
    """The DAC2a imputed line, read as a plain table pinned to DAC2a, holds 2020 and 2021.
    Under a part's CRS source scope its pin narrows that scope, which admits none of its
    rows, as a plain answer read under that scope admits none."""
    populated: _Populated = {_DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021))}
    facts = ScopeFacts(index=get_index(), coverage=_dac2a_coverage(_rows_of(populated)).snapshot())
    components = component_identities(get_index(), _dac2a_oda_identity("Net"))
    assert components is not None
    table = next(c.identity for c in components if c.role == "imputed")

    with scoped_reads(frozenset({_PROVENANCE[_CRS_FACET]})):
        scoped = component_years(table, facts=facts).held
        scoped_donors = donor_years(table, facts=facts)

    assert component_years(table, facts=facts).held == frozenset({2020, 2021})
    assert donor_years(table, facts=facts) == {_PROVIDER: frozenset({2020, 2021})}
    assert scoped == frozenset()
    assert scoped_donors == {}


def _composite_reads(mint: CompositeMint) -> tuple[object, ...]:
    return (
        mint.headline(None),
        mint.stat_vars(ResolvedStatVars(member_count=0, member_count_exact=True)),
        mint.party_split("destination"),
    )


def test_a_composite_mint_reads_its_counts_under_the_request_s_catalogue():
    """A `CompositeMint` given the request's catalogue reads its headline, StatVar counts and
    parties under it, so a context with no catalogue bound reads what the bound one reads."""
    populated: _Populated = {
        _BILATERAL_206: (_DAC2A_FACET, 100, (2020, 2021)),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021)),
    }
    facts = ScopeFacts(index=get_index(), coverage=_dac2a_coverage(_rows_of(populated)).snapshot())
    identity = _dac2a_oda_identity("Net")

    def mint() -> CompositeMint:
        return CompositeMint(
            identity=BaseIdentity.model_validate(identity),
            spec_id="spec",
            served=tuple(served_components(identity, facts=facts)),
            facts=facts,
            catalogue=current_catalogue(),
        )

    bound = _composite_reads(mint())
    unbound = contextvars.Context().run(_composite_reads, mint())

    assert bound[0] is not None
    assert unbound == bound


def test_a_snapshot_behind_a_component_s_rows_leaves_the_mint_its_rows_years():
    """SDMX serves total ODA from France to Kenya in 2020 and 2021, while the coverage
    snapshot holds it in 2020 alone, as when a source publishes before the snapshot
    rebuilds. The mint reads the bilateral line's years from its rows and serves 2021 too,
    dropping nothing, while Search states 2020 from the snapshot until it rebuilds."""
    populated: _Populated = {
        _BILATERAL_206: (_DAC2A_FACET, 100, (2020, 2021)),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021)),
    }

    minted, served = _dac2a_case(
        populated,
        [
            (_BILATERAL_206, _DAC2A_FACET, _PROVIDER, _RECIPIENT, [2020]),
            (_BILATERAL_206, _DAC2A_FACET, _OTHER_DONOR, _RECIPIENT, [2021]),
            (_DAC2A_IMPUTED, _DAC2A_FACET, _PROVIDER, _RECIPIENT, [2020, 2021]),
        ],
    )

    assert minted.status_code == 200, minted.text
    by_role = {entry["role"]: entry for entry in minted.json()["trace"]["components"]}
    assert by_role["bilateral"]["dropped_years"] == []
    assert by_role["bilateral"]["periods"] == [[2020, 2021]]
    assert {item.served for item in served} == {frozenset({2020})}


def test_a_snapshot_lacking_a_year_the_whole_table_lacks_drops_and_names_it():
    """SDMX serves total ODA from France to Kenya in 2020 and 2021, while the coverage
    snapshot holds the line in 2020 alone and no other donor holds 2021, as when a source
    publishes before the snapshot rebuilds. Table years come from the snapshot, so the mint
    serves 2020, the year Search states, and names 2021 among the bilateral line's dropped
    years."""
    populated: _Populated = {
        _BILATERAL_206: (_DAC2A_FACET, 100, (2020, 2021)),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021)),
    }

    minted, served = _dac2a_case(
        populated,
        [
            (_BILATERAL_206, _DAC2A_FACET, _PROVIDER, _RECIPIENT, [2020]),
            (_DAC2A_IMPUTED, _DAC2A_FACET, _PROVIDER, _RECIPIENT, [2020, 2021]),
        ],
    )

    assert minted.status_code == 200, minted.text
    by_role = {entry["role"]: entry for entry in minted.json()["trace"]["components"]}
    assert by_role["bilateral"]["dropped_years"] == [2021]
    assert by_role["bilateral"]["periods"] == [[2020, 2020]]
    assert {item.served for item in served} == {frozenset({2020})}


def test_a_snapshot_holding_none_of_a_component_s_years_for_the_keys_serves_its_rows():
    """The coverage snapshot holds France's total ODA to Kenya in 2019 alone, and another
    donor's in 2020 and 2021, so the table covers those years while France's line holds
    none of them. SDMX serves France's rows in 2020 and 2021, and the mint serves them,
    while Search states no year from the snapshot."""
    populated: _Populated = {
        _BILATERAL_206: (_DAC2A_FACET, 100, (2020, 2021)),
        _DAC2A_IMPUTED: (_DAC2A_FACET, 7, (2020, 2021)),
    }

    minted, served = _dac2a_case(
        populated,
        [
            (_BILATERAL_206, _DAC2A_FACET, _PROVIDER, _RECIPIENT, [2019]),
            (_BILATERAL_206, _DAC2A_FACET, _OTHER_DONOR, _RECIPIENT, [2020, 2021]),
            (_DAC2A_IMPUTED, _DAC2A_FACET, _PROVIDER, _RECIPIENT, [2020, 2021]),
        ],
    )

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    by_role = {entry["role"]: entry for entry in manifest["trace"]["components"]}
    assert by_role["bilateral"]["periods"] == [[2020, 2021]]
    assert by_role["imputed"]["periods"] == [[2020, 2021]]
    assert manifest["coverage"]["rows"] > 0
    assert {item.served for item in served} == {frozenset()}


def test_the_mint_reads_no_held_years_from_the_snapshot(monkeypatch: pytest.MonkeyPatch):
    """The mint reads a stacked component's held years from its own rows, so neither the
    CRS composite nor the spliced OECD headline reads any from the coverage snapshot."""
    read: list[Mapping[str, object]] = []
    real = composite_module.held_years

    def recording(index, coverage, identity):
        read.append(identity)
        return real(index, coverage, identity)

    monkeypatch.setattr(composite_module, "held_years", recording)
    cases = [
        (_identity(), _stacked_crs()),
        (_headline_identity(), _spliced_headline()),
    ]
    for identity, populated in cases:
        client = _client(
            graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
            sdmx_transport=httpx.MockTransport(
                lambda request, populated=populated: _sdmx_response(populated, request)
            ),
            coverage=_coverage(populated),
            artifact_store=FakeArtifactStore(),
            settings=DatasetsBuildSettings.from_env(),
        )
        assert _mint(client, identity=identity).status_code == 200

    assert read == []


def test_a_pinned_component_s_years_refuse_a_read_with_no_catalogue_bound():
    """The DAC2a composite's imputed line reads its years pinned to DAC2a under the caller's
    catalogue, so a context with none bound refuses."""
    components = component_identities(get_index(), _dac2a_oda_identity("Net"))
    assert components is not None
    component = next(c for c in components if c.role == "imputed")
    coverage = _dac2a_coverage([]).snapshot().index

    with pytest.raises(CatalogueNotBound):
        contextvars.Context().run(
            composite_module.read_component_years, get_index(), coverage, component
        )
