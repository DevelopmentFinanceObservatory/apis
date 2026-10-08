"""A UN series whose facet probe is empty through the real origin client, materialisation and
`prewrite_definition`. At places it names that the UN front door refuses, the mint refuses
`upstream_error` naming them. A request that binds no place covers every place the UN's
availability lists, including those the front door refuses on every read, so it is the empty
result with the `un_system_unavailable` warning naming the refused places, and
`upstream_error` when the UN deployment fails. `test_un_doors` drives the same outcomes
through the doors."""

from __future__ import annotations

import httpx
import pytest

from datasets_api.codes_materialise import UnSystemUnavailable
from datasets_api.doors.mint_errors import classify_mint_exception
from datasets_api.serve import DefinitionPrewrite
from dc_sdmx import SdmxClient
from materialise import (
    CollectionFacetProbe,
    MaterialisationRequest,
    materialise_preview,
    probe_collection_facets,
)
from materialise.errors import DefaultFacetUndefinedError
from materialise.place_scope import DefaultPlaceScope
from materialise.window import RequestedWindow
from origin_catalogue.graph import OriginGraphClient
from tests.materialise._scope_availability import ScopeAvailability, ScopeTransport
from tests.materialise.test_observation_rows import (
    _sdmx_never_called,
    deployments,  # noqa: F401  (fixture)
    graph,  # noqa: F401  (fixture)
)
from tests.materialise.un_deployment import OURS_API, UN_VARIABLE, UnDeployments

from ._prewrite import prewrite

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


def _mint(
    client: OriginGraphClient,
    *requests: MaterialisationRequest,
    probe: CollectionFacetProbe | None = None,
) -> DefinitionPrewrite:
    """The prewrite of a mint of `requests`. Our SDMX service answers the default place scope's
    availability read with no place and refuses a data read."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a UN variable reached the SDMX service: {request.url}")

    sdmx = SdmxClient(
        base_url=OURS_API, transport=ScopeTransport(httpx.MockTransport(refuse), _NO_DATA)
    )
    return prewrite(requests, graph=client, sdmx=sdmx, precomputed_probe=probe)


def _warning_codes(minted: DefinitionPrewrite) -> list[str]:
    return [warning.code for warning in minted.manifest.warnings]


def test_a_mint_at_a_refused_place_refuses_upstream_error_naming_it(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    deployments.refused = {_GRC}

    with pytest.raises(UnSystemUnavailable) as refused:
        _mint(graph, _request(_GRC))

    assert refused.value.places == (_GRC,)
    spec = classify_mint_exception(refused.value)
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (502, "upstream_error", True)
    assert _GRC in (spec.detail or "")


def test_a_request_binding_no_place_with_no_data_anywhere_is_the_empty_result(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    # The UN's availability lists Greece and Nigeria, and Nigeria holds no facet. Greece is
    # refused, as it is on every read, so it does not turn a request with no data into a
    # retryable failure. The warning names it, as it does on a request that served rows.
    deployments.available = [_GRC, "country/NGA"]
    deployments.refused = {_GRC}

    minted = _mint(graph, _request())

    assert minted.result.probe_empty
    assert minted.manifest.coverage.rows == 0
    assert minted.provenance_refs == []
    assert minted.manifest.citation.sources == []
    assert {"empty_result", "un_system_unavailable"} <= set(_warning_codes(minted))
    (unavailable,) = (w for w in minted.manifest.warnings if w.code == "un_system_unavailable")
    assert unavailable.message.endswith(f"No UN data was read for {_GRC}.")


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


def _mixed_group_probe(
    client: OriginGraphClient,
) -> tuple[list[MaterialisationRequest], CollectionFacetProbe]:
    """A group of the UN variable and one of ours, both binding no place, with the probe a
    mint door makes before the prewrite opens its UN record. Neither member holds a candidate."""
    requests = [_request(), _request(stat_var="Count_Person")]
    probe = probe_collection_facets(requests, settle=True, graph=client, availability=_NO_DATA)
    assert probe.candidates == ((), ())
    return requests, probe


def test_an_empty_group_refuses_upstream_error_when_its_un_member_read_failed(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    # The UN's facet read fails while the probe also reads one of our variables. The probe
    # carries the failed read into the record the prewrite opens.
    countries = [_GRC, "country/KEN"]
    deployments.available = countries
    deployments.status = 503
    requests, probe = _mixed_group_probe(graph)
    assert [failure.variables for failure in probe.un_unserved.failures] == [(UN_VARIABLE,)]

    with pytest.raises(UnSystemUnavailable) as refused:
        _mint(graph, *requests, probe=probe)

    assert refused.value.variable == UN_VARIABLE
    assert refused.value.places == tuple(countries)
    spec = classify_mint_exception(refused.value)
    assert spec is not None
    assert (spec.status, spec.code, spec.retryable) == (502, "upstream_error", True)


def test_an_empty_group_with_no_failed_un_read_is_the_empty_result(
    graph: OriginGraphClient,  # noqa: F811
    deployments: UnDeployments,  # noqa: F811
) -> None:
    # The UN's availability lists Greece and Nigeria, neither holds a facet, and every UN read
    # succeeded.
    deployments.available = [_GRC, "country/NGA"]
    requests, probe = _mixed_group_probe(graph)

    minted = _mint(graph, *requests, probe=probe)

    assert minted.result.probe_empty
    assert minted.manifest.coverage.rows == 0
    assert "empty_result" in _warning_codes(minted)


def test_a_place_with_no_data_that_was_not_refused_is_the_empty_result(
    graph: OriginGraphClient,  # noqa: F811
) -> None:
    minted = _mint(graph, _request("country/NGA"))

    assert minted.result.probe_empty
    assert minted.manifest.coverage.rows == 0
    assert "empty_result" in _warning_codes(minted)
    assert "un_system_unavailable" not in _warning_codes(minted)


def test_a_preview_of_a_place_with_no_data_refuses_default_facet_undefined(
    graph: OriginGraphClient,  # noqa: F811
) -> None:
    # The empty result is the mint's. A preview selects facets over the empty probe and
    # refuses, which its callers handle.
    with pytest.raises(DefaultFacetUndefinedError):
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
