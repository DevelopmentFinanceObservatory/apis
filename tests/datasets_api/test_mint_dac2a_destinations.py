"""An open destination under a DAC2a composite that counts bilateral flows reads bilateral-side
destinations alone. DAC2a aid type 206 also holds a row for each multilateral agency and a
`DACAllRecipients` row spanning both sides, so a mint that reads every destination counts core
contributions beside the countries' rows.

The SDMX double holds one agency, one spanning and one country row of 206 and one country row
of the imputed 106, and honours the recipient filter a request sends, as the endpoint does.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence

import httpx
import pytest

import code_index.composites as composites_module
from code_index.composites import destination_population
from code_index.declarations import COMPOSITES, DAC2A_BILATERAL, DAC2A_INCLUSIVE
from code_index.loader import parse_index
from code_index.scope_conventions import BILATERAL_SOURCES, destination_side
from code_index.shared import get_index
from datasets_api.adapter import composite_components
from dc_sdmx import SdmxClient
from materialise.errors import DestinationUnclassifiedError, FacetNotAvailableError
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.code_index.support import INDEX_FIXTURE
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from .conftest import FakeDatasetStore
from .test_definition_door import _client
from .test_mint_composite import (
    _CSV_HEADER,
    _DAC2A,
    _DAC2A_FACET,
    _DAC2A_IMPUTED,
    _IMPUTED_YEARS,
    _PROVIDER,
    _RECIPIENT,
    _composite_client,
    _csv_rows,
    _dac2a_oda_identity,
    _identity,
    _mint,
)

_BILATERAL = "provenance/ONEDAC2aBilateral"
_TOTAL_ODA = "Amount_FinancialFlow_Net_Disbursement_DAC2aAidType206_Nominal"
_AGENCY = "EuropeanCommission"
_SPANNING = "DACAllRecipients"
# What each (stat var, recipient) holds.
_HELD = {
    (_TOTAL_ODA, _AGENCY): 200,
    (_TOTAL_ODA, _SPANNING): 500,
    (_TOTAL_ODA, _RECIPIENT): 100,
    (_DAC2A_IMPUTED, _RECIPIENT): 7,
}


def _open_destination(provenance: str) -> dict:
    identity = {**_dac2a_oda_identity("Net"), "facets": {"provenance": _ref(provenance)}}
    identity["entities"] = {"origin": [_PROVIDER]}
    identity["open_roles"] = ["destination"]
    return identity


def _ref(provenance: str) -> dict:
    return {"state": "value", "ref": provenance}


_REQUESTS: list[httpx.Request] = []


def _respond(request: httpx.Request) -> httpx.Response:
    """The held rows of the named variables, cut to the recipients the request names."""
    _REQUESTS.append(request)
    named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
    recipients = request.url.params.get("c[financingRecipient]")
    wanted = None if recipients is None else str(recipients).split(",")
    body = _CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{dcid},{_PROVIDER},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,{_DAC2A},{year},{value},,{_DAC2A_FACET}\r\n"
        for (dcid, recipient), value in _HELD.items()
        if dcid in named and (wanted is None or recipient in wanted)
        for year in _IMPUTED_YEARS
    )
    return httpx.Response(
        200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
    )


def _open_client(
    *,
    artifacts: FakeArtifactStore | None = None,
    store: FakeDatasetStore | None = None,
    held_elsewhere: Sequence[str] = (),
):
    """A client over `_HELD`, whose coverage also holds total ODA at each recipient of
    `held_elsewhere`."""
    _REQUESTS.clear()
    held = [*_HELD, *((_TOTAL_ODA, recipient) for recipient in held_elsewhere)]
    coverage = FakeCoverageService(
        build_coverage_index(
            [
                (dcid, _DAC2A_FACET, _PROVIDER, recipient, list(_IMPUTED_YEARS))
                for dcid, recipient in held
            ],
            role_names={
                dcid: RoleNames(donor="financingProvider", recipient="financingRecipient")
                for dcid, _recipient in held
            },
            facet_metadata={
                _DAC2A_FACET: FacetMetadata(
                    provenance_id=_DAC2A, measurement_method="GrantEquivalent", unit="USD"
                )
            },
        )
    )
    return _client(
        graph=FakeMintGraphReader(
            labels={
                _PROVIDER: "France",
                _RECIPIENT: "Kenya",
                _AGENCY: "European Commission",
                _SPANNING: "Developing countries and multilateral agencies",
            }
        ),
        sdmx_transport=httpx.MockTransport(_respond),
        coverage=coverage,
        artifact_store=artifacts,
        store=store,
    )


def _mint_open_manifest(provenance: str, **options: object) -> tuple[dict, list[dict]]:
    """The manifest of a mint of `_open_destination(provenance)` and its CSV rows."""
    artifacts = FakeArtifactStore()
    client = _open_client(artifacts=artifacts)
    minted = _mint(client, identity=_open_destination(provenance), options=options)
    assert minted.status_code == 200, minted.text
    return minted.json(), _csv_rows(artifacts, minted.json())


def _mint_open(provenance: str, **options: object) -> list[dict]:
    return _mint_open_manifest(provenance, **options)[1]


@pytest.mark.parametrize("provenance", [_BILATERAL, DAC2A_INCLUSIVE])
def test_an_open_destination_mints_no_agency_or_spanning_row(provenance: str) -> None:
    rows = _mint_open(provenance)

    assert {row["financingRecipient"] for row in rows} == {_RECIPIENT}


@pytest.mark.parametrize(
    ("provenance", "total"),
    [(_BILATERAL, 100), (DAC2A_INCLUSIVE, 100 + 7)],
)
def test_an_open_destination_sums_the_country_row_alone_and_inclusive_adds_106(
    provenance: str, total: int
) -> None:
    rows = _mint_open(provenance)

    sums: dict[str, float] = {}
    for row in rows:
        sums[row["time_period"]] = sums.get(row["time_period"], 0.0) + float(row["value"])
    assert sums == {"2020": total, "2021": total}


def test_the_bound_recipient_list_fits_one_request_under_the_url_budget() -> None:
    """A bilateral read sends the population's recipients as one filter, 207 dcids on the
    recorded index, about 3,800 bytes of the client's 7,800-byte budget, so no request splits
    the population across requests."""
    _mint_open(_BILATERAL)

    members = set(destination_population(get_index(), _DAC2A).members)
    budget = SdmxClient(base_url="https://example.org/core/api").url_budget_bytes
    carried = [request for request in _REQUESTS if "c[financingRecipient]" in request.url.params]
    assert carried, "no request carried the recipient filter"
    for request in carried:
        assert len(request.url.query) <= budget
        assert set(str(request.url.params["c[financingRecipient]"]).split(",")) == members


@pytest.mark.parametrize("provenance", [_BILATERAL, DAC2A_INCLUSIVE])
def test_an_open_destination_states_no_population_member_as_unserved(provenance: str) -> None:
    """The population an open destination binds is no list the caller named, so the mint
    counts none of its members it holds no rows at below the scope. Kenya is served in every
    year, so nothing else falls below it."""
    manifest, _rows = _mint_open_manifest(provenance)

    below = [w for w in manifest["warnings"] if w["code"] == "coverage_below_scope"]
    assert below == []


@pytest.mark.parametrize("provenance", [_BILATERAL, DAC2A_INCLUSIVE])
def test_a_composite_whose_population_coverage_cannot_place_refuses_before_any_read(
    provenance: str,
) -> None:
    """Coverage holding total ODA at a recipient the code index does not place leaves the
    bilateral population unavailable, since its members would leave that recipient out
    without a word: the mint refuses as a server fault naming it, and reads no rows."""
    client = _open_client(artifacts=FakeArtifactStore(), held_elsewhere=["SomeNewAgency"])

    refused = _mint(client, identity=_open_destination(provenance))

    assert refused.status_code == 500, refused.text
    assert refused.json()["code"] == "internal_error"
    assert "SomeNewAgency" in refused.json()["detail"]
    assert _REQUESTS == []


@pytest.mark.parametrize("provenance", [_BILATERAL, DAC2A_INCLUSIVE])
def test_a_composite_whose_code_index_cannot_place_a_destination_is_a_server_fault(
    monkeypatch: pytest.MonkeyPatch, provenance: str
) -> None:
    """A code index publishing a destination `destination_side` cannot place leaves the population
    unavailable: the components refuse as a server fault naming it, and not as the caller's
    unservable composite."""

    def unplaced(row, *, datasets, index):
        return (
            None
            if row.dcid == _RECIPIENT
            else destination_side(row, datasets=datasets, index=index)
        )

    monkeypatch.setattr(composites_module, "destination_side", unplaced)
    with gzip.open(INDEX_FIXTURE) as handle:
        drifted = parse_index(json.load(handle), source="fixture", composites=COMPOSITES)

    with pytest.raises(DestinationUnclassifiedError) as refused:
        composite_components(_open_destination(provenance), code_index=drifted)

    assert (refused.value.dataset, refused.value.destinations) == (_DAC2A, (_RECIPIENT,))


_SOURCE = next(source for source in BILATERAL_SOURCES if source.dataset == _DAC2A)
_OFF_SIDE = [
    pytest.param(
        DAC2A_INCLUSIVE,
        _SOURCE.spanning,
        "destination",
        _SOURCE.rebound,
        id="inclusive-spanning-total",
    ),
    pytest.param(
        DAC2A_BILATERAL,
        _SOURCE.spanning,
        "destination",
        _SOURCE.rebound,
        id="bilateral-spanning-total",
    ),
    pytest.param(DAC2A_BILATERAL, _AGENCY, "provenance", _DAC2A, id="bilateral-agency"),
    pytest.param(
        DAC2A_INCLUSIVE,
        "DACAllMultilateralRecipients",
        "provenance",
        _DAC2A,
        id="inclusive-multilateral-total",
    ),
]


def _off_side_identity(provenance: str, destination: str) -> dict:
    """France's DAC2a total ODA to `destination` read through the composite `provenance`."""
    identity = {**_dac2a_oda_identity("Net"), "facets": {"provenance": _ref(provenance)}}
    identity["entities"] = {"origin": [_PROVIDER], "destination": [destination]}
    return identity


def _assert_off_side_refusal(
    refused: httpx.Response, *, destination: str, axis: str, fix: str
) -> None:
    """`refused` is a 422 `facet_not_available` whose one issue names `destination` and the
    `fix` to bind on `axis` in its place."""
    assert refused.status_code == 422, refused.text
    problem = refused.json()
    assert problem["code"] == "facet_not_available"
    [issue] = problem["issues"]
    assert issue["member"] == destination
    assert issue["axis"] == axis
    assert [alternative["ref"] for alternative in issue["alternatives"]] == [fix]


@pytest.mark.parametrize(("provenance", "destination", "axis", "fix"), _OFF_SIDE)
def test_the_mint_door_refuses_a_composite_bound_to_a_destination_off_its_side(
    provenance: str, destination: str, axis: str, fix: str
) -> None:
    """A definition binding a DAC2a composite that reads bilateral destinations to one off
    that side is the client's fault: the mint door refuses it 422 before any read and names
    the fix, the total of developing countries for the total of all recipients and the plain
    DAC2a table for an agency or the total of them. Nothing is registered."""
    store = FakeDatasetStore()
    client = _open_client(artifacts=FakeArtifactStore(), store=store)
    identity = _off_side_identity(provenance, destination)

    refused = _mint(client, identity=identity)

    _assert_off_side_refusal(refused, destination=destination, axis=axis, fix=fix)
    assert _REQUESTS == []
    assert LiveSpecReader(store=store).read_definition(recompute_identity(identity).spec_id) is None


@pytest.mark.parametrize(("provenance", "destination", "axis", "fix"), _OFF_SIDE[1:3])
def test_the_definition_door_refuses_a_composite_bound_to_a_destination_off_its_side(
    provenance: str, destination: str, axis: str, fix: str
) -> None:
    """The options route refuses such a definition 422 as the mint does, before it registers
    the definition."""
    store = FakeDatasetStore()
    client = _open_client(artifacts=FakeArtifactStore(), store=store)
    identity = _off_side_identity(provenance, destination)
    spec_id = recompute_identity(identity).spec_id

    refused = client.post(
        f"/api/datasets/{spec_id}/options",
        json={"definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity}},
    )

    _assert_off_side_refusal(refused, destination=destination, axis=axis, fix=fix)
    assert LiveSpecReader(store=store).read_definition(spec_id) is None


@pytest.mark.parametrize(("provenance", "destination", "axis", "fix"), _OFF_SIDE)
def test_the_off_side_refusal_points_where_its_fix_is(
    provenance: str, destination: str, axis: str, fix: str
) -> None:
    """The total spanning both sides is fixed on the destination role, so its refusal points
    there; an agency or the total of them is fixed on the provenance facet."""
    with pytest.raises(FacetNotAvailableError) as refused:
        composite_components(_off_side_identity(provenance, destination), code_index=get_index())

    assert refused.value.facet_pointer == (
        "/entities/destination" if axis == "destination" else "/facets/provenance"
    )


def _mint_scoped(client, *, identity: dict, scope: dict, sidecar: bool) -> httpx.Response:
    """A mint of `identity`'s spec_id under `scope`, posting the sidecar when `sidecar`."""
    spec_id = recompute_identity(identity).spec_id
    definition = {"schema_version": 1, "spec_id": spec_id, "identity": identity}
    return client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": spec_id,
                **({"definition": definition} if sidecar else {}),
                "scope": scope,
            },
            "options": {"bundle": False},
        },
    )


def _sent_recipients() -> set[str]:
    """The recipients the SDMX requests made so far name."""
    return {
        recipient
        for request in _REQUESTS
        for recipient in str(request.url.params.get("c[financingRecipient]", "")).split(",")
        if recipient
    }


@pytest.mark.parametrize("registered", [False, True], ids=["sidecar", "registered"])
@pytest.mark.parametrize("provenance", [_BILATERAL, DAC2A_INCLUSIVE])
def test_a_scope_binding_the_destination_reads_that_destination_alone(
    provenance: str, registered: bool
) -> None:
    """A scope's entities replace the definition's, so a composite with an open destination
    minted under a scope binding Kenya reads Kenya alone, whether the definition is posted
    beside the mint or registered before it."""
    artifacts = FakeArtifactStore()
    client = _open_client(artifacts=artifacts)
    identity = _open_destination(provenance)
    if registered:
        assert _mint(client, identity=identity).status_code == 200
        _REQUESTS.clear()
    scope = {"entities": {"origin": [_PROVIDER], "destination": [_RECIPIENT]}}

    minted = _mint_scoped(client, identity=identity, scope=scope, sidecar=not registered)

    assert minted.status_code == 200, minted.text
    assert _sent_recipients() == {_RECIPIENT}
    assert {row["financingRecipient"] for row in _csv_rows(artifacts, minted.json())} == {
        _RECIPIENT
    }


@pytest.mark.parametrize("registered", [False, True], ids=["sidecar", "registered"])
def test_a_scope_binding_an_agency_refuses_before_any_read(registered: bool) -> None:
    """A scope binding the bilateral composite's destination to an agency is refused 422 with
    the plain table to read in its place, before any read, as the definition itself would
    be."""
    store = FakeDatasetStore()
    client = _open_client(artifacts=FakeArtifactStore(), store=store)
    identity = _open_destination(_BILATERAL)
    if registered:
        assert _mint(client, identity=identity).status_code == 200
        _REQUESTS.clear()
    scope = {"entities": {"origin": [_PROVIDER], "destination": [_AGENCY]}}

    refused = _mint_scoped(client, identity=identity, scope=scope, sidecar=not registered)

    _assert_off_side_refusal(refused, destination=_AGENCY, axis="provenance", fix=_DAC2A)
    assert _REQUESTS == []


_ONE_YEAR = {"window": {"kind": "range", "start_year": 2021, "end_year": 2021}}


@pytest.mark.parametrize("registered", [False, True], ids=["sidecar", "registered"])
@pytest.mark.parametrize("source", ["crs-inclusive", "dac2a-bilateral", "dac2a-inclusive"])
def test_a_scope_narrowing_the_window_reads_that_year_alone(source: str, registered: bool) -> None:
    """A scope's window replaces the definition's, so a composite minted under a scope of one
    year serves that year alone, whether the definition is posted beside the mint or
    registered before it."""
    if source == "crs-inclusive":
        artifacts, client = _composite_client()
        identity = _identity()
    else:
        artifacts = FakeArtifactStore()
        client = _open_client(artifacts=artifacts)
        identity = _open_destination(_BILATERAL if source == "dac2a-bilateral" else DAC2A_INCLUSIVE)
    assert artifacts is not None
    if registered:
        assert _mint(client, identity=identity).status_code == 200

    minted = _mint_scoped(client, identity=identity, scope=_ONE_YEAR, sidecar=not registered)

    assert minted.status_code == 200, minted.text
    rows = _csv_rows(artifacts, minted.json())
    assert rows
    assert {row["time_period"] for row in rows} == {"2021"}
