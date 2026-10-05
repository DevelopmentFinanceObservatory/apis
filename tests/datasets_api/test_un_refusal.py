"""A UN series with no candidate at places it names, which the UN front door refuses, refuses
`upstream_error` naming them, through the real origin client and materialisation. A request
that binds no place covers every place the UN's availability lists, including those the front
door refuses on every read, so it refuses `default_facet_undefined` naming the refused places,
and `upstream_error` when the UN deployment fails. `test_un_doors` drives the same refusals
through the doors."""

from __future__ import annotations

import pytest

from datasets_api.codes_materialise import UnSystemUnavailable
from datasets_api.doors.mint_errors import classify_mint_exception
from datasets_api.origin import origin_reads
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_collection,
    materialise_group,
    materialise_preview,
    probe_collection_facets,
)
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from materialise.place_scope import DefaultPlaceScope
from materialise.revision import RevisionSnapshot
from materialise.window import RequestedWindow
from origin_catalogue.graph import OriginGraphClient
from tests.materialise._scope_availability import ScopeAvailability
from tests.materialise.conftest import FakeRevisionReader
from tests.materialise.test_observation_rows import (
    _sdmx_never_called,
    deployments,  # noqa: F401  (fixture)
    graph,  # noqa: F401  (fixture)
)
from tests.materialise.un_deployment import UN_VARIABLE, UnDeployments

_GRC = "country/GRC"
# Our mixer holds no data for `Count_Person` here, so its default place scope is empty.
_NO_DATA = ScopeAvailability(data=())


def _request(*places: str, stat_var: str = UN_VARIABLE) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{stat_var}",
        role_bindings={"observationAbout": list(places)} if places else {},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def test_a_preview_at_a_refused_place_refuses_upstream_error_naming_it(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    deployments.refused = {_GRC}

    with (
        pytest.raises(UnSystemUnavailable) as refused,
        origin_reads(graph, availability=_NO_DATA, deadline=None),
    ):
        materialise_preview(_request(_GRC), sdmx=_sdmx_never_called(), graph=graph, limit=10)

    assert refused.value.places == (_GRC,)
    assert isinstance(refused.value.__cause__, DefaultFacetUndefinedError)
    spec = classify_mint_exception(refused.value)
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (502, "upstream_error", True)
    assert _GRC in (spec.detail or "")


def test_a_request_binding_no_place_with_no_data_anywhere_stays_default_facet_undefined(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The UN's availability lists Greece and Nigeria, and Nigeria holds no facet. Greece is
    # refused, as it is on every read, so it does not turn a request with no data into a
    # retryable failure. The refusal names it, as the warning on a served request would.
    deployments.available = [_GRC, "country/NGA"]
    deployments.refused = {_GRC}

    with (
        pytest.raises(DefaultFacetUndefinedError) as refused,
        origin_reads(graph, availability=_NO_DATA, deadline=None),
    ):
        materialise_preview(_request(), sdmx=_sdmx_never_called(), graph=graph, limit=10)

    spec = classify_mint_exception(refused.value)
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (422, "default_facet_undefined", False)
    assert (spec.detail or "").endswith(
        f"No UN data was read for {_GRC}, which the UN System Data Commons front door refuses"
    )


def test_a_request_binding_no_place_takes_its_un_countries_when_its_un_availability_fails(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    # The UN's availability read fails with a gateway status, so the UN variable takes the
    # countries where the UN holds its data, for this request only, and says its places are
    # not every place with data.
    deployments.availability_status = 503

    probe = probe_collection_facets(
        [_request(), _request(stat_var="Count_Person")],
        settle=True,
        graph=graph,
        availability=_NO_DATA,
    )

    un_member, ours = probe.members
    assert un_member.scope == DefaultPlaceScope(
        places=("country/GHA", "country/KEN"), every_place=False
    )
    assert ours.places == ()
    assert len(deployments.countries_requests) == 1


def test_a_request_binding_no_place_refuses_upstream_error_when_its_un_read_failed(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    # The UN's facet read fails while the probe also reads one of our variables. The probe runs
    # before `origin_reads` opens, as the mint doors run it, and carries the failed read.
    countries = [_GRC, "country/KEN"]
    deployments.available = countries
    deployments.status = 503
    requests = [_request(), _request(stat_var="Count_Person")]
    probe = probe_collection_facets(requests, settle=True, graph=graph, availability=_NO_DATA)
    assert [failure.variables for failure in probe.un_unserved.failures] == [(UN_VARIABLE,)]
    snapshot = RevisionSnapshot(graph_revision="graph-rev", source_revisions=())

    with (
        pytest.raises(UnSystemUnavailable) as refused,
        origin_reads(graph, availability=_NO_DATA, deadline=None),
    ):
        materialise_collection(
            [(f"spec_{index:032x}", request) for index, request in enumerate(requests)],
            sdmx=_sdmx_never_called(),
            graph=graph,
            revision_reader=FakeRevisionReader([snapshot, snapshot]),
            materialiser_build="build-1",
            capability_version="cap-1",
            enrichment=EnrichmentLookup(graph),
            precomputed_probe=probe,
        )

    assert refused.value.places == tuple(countries)
    spec = classify_mint_exception(refused.value)
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (502, "upstream_error", True)


def _empty_mixed_group(client: OriginGraphClient) -> None:
    """Materialise a group of the UN variable and one of ours, both binding no place, from a
    probe made before `origin_reads` opens, as the mint doors make it."""
    requests = [_request(), _request(stat_var="Count_Person")]
    probe = probe_collection_facets(requests, settle=True, graph=client, availability=_NO_DATA)
    assert probe.candidates == ((), ())
    snapshot = RevisionSnapshot(graph_revision="graph-rev", source_revisions=())

    with origin_reads(client, availability=_NO_DATA, deadline=None):
        materialise_group(
            requests,
            sdmx=_sdmx_never_called(),
            graph=client,
            revision_reader=FakeRevisionReader([snapshot, snapshot]),
            materialiser_build="build-1",
            capability_version="cap-1",
            enrichment=EnrichmentLookup(client),
            precomputed_probe=probe,
        )


def test_an_empty_group_refuses_upstream_error_when_its_un_member_read_failed(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    countries = [_GRC, "country/KEN"]
    deployments.available = countries
    deployments.status = 503

    with pytest.raises(UnSystemUnavailable) as refused:
        _empty_mixed_group(graph)

    assert refused.value.variable == UN_VARIABLE
    assert refused.value.places == tuple(countries)
    assert isinstance(refused.value.__cause__, FacetNotAvailableError)
    spec = classify_mint_exception(refused.value)
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (502, "upstream_error", True)


def test_an_empty_group_with_no_failed_un_read_stays_facet_not_available(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    # The UN's availability lists Greece and Nigeria, neither holds a facet, and every UN read
    # succeeded.
    deployments.available = [_GRC, "country/NGA"]

    with pytest.raises(FacetNotAvailableError) as refused:
        _empty_mixed_group(graph)

    spec = classify_mint_exception(refused.value)
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (422, "facet_not_available", False)


def test_a_place_with_no_data_that_was_not_refused_stays_default_facet_undefined(
    graph: OriginGraphClient,  # noqa: F811
) -> None:
    with (
        pytest.raises(DefaultFacetUndefinedError),
        origin_reads(graph, availability=_NO_DATA, deadline=None),
    ):
        materialise_preview(
            _request("country/NGA"), sdmx=_sdmx_never_called(), graph=graph, limit=10
        )


def test_the_refusal_detail_names_five_places_and_counts_the_rest() -> None:
    # A request binding no place can refuse at every country. The detail stays short and the
    # exception keeps every place.
    places = tuple(f"country/C{index}" for index in range(8))

    refused = UnSystemUnavailable(variable=UN_VARIABLE, places=places)

    assert refused.places == places
    assert refused.detail == (
        f"the UN System Data Commons could not serve observations of {UN_VARIABLE} at "
        f"{', '.join(places[:5])} and 3 more; retry"
    )
