"""A preview of a multi-member definition is the preview of the group its mint serves.

The education-aid-to-Kenya identity resolves 257 StatVars in the packaged index. Coverage
holds observations at Kenya for two of them, neither the first in resolution order.
"""

from __future__ import annotations

import threading

import httpx

from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from datasets_api.config import DatasetsBuildSettings
from spec_registry import recompute_identity
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    CRS_MEMBER_COUNT,
    CRS_PROVENANCE,
    education_aid_identity,
)

from ._graph_doubles import FakeMintGraphReader
from .test_definition_door import _client, _coverage_service

_PROVIDER = "country/FRA"
_RECIPIENT = "country/KEN"
_YEARS = (2019, 2020, 2021)

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
    "facetId\r\n"
)


def _family() -> tuple[str, ...]:
    """The bare dcids the mint serves for the identity, in resolution order."""
    graph = FakeMintGraphReader()
    resolved = resolve_definition_members(
        education_aid_identity(),
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=get_index(),
    )
    return tuple(shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes)


def _populated(family: tuple[str, ...]) -> tuple[str, str]:
    """Two members that hold observations at Kenya, neither the first resolved."""
    return family[3], family[40]


def _sdmx_transport(populated: tuple[str, ...]) -> httpx.MockTransport:
    """Answers each fetch with the rows of the populated members it names."""

    def _respond(request: httpx.Request) -> httpx.Response:
        named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
        body = _CSV_HEADER + "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{dcid},{_PROVIDER},{_RECIPIENT},USD,GrantEquivalent,"
            f"NotApplicable,{CRS_PROVENANCE},{year},{100 + position},,facet1\r\n"
            for position, dcid in enumerate(populated)
            if dcid in named
            for year in _YEARS
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(_respond)


def _family_client(
    populated: tuple[str, ...],
    *,
    sdmx_transport: httpx.MockTransport | None = None,
    **client_kwargs,
):
    return _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        sdmx_transport=sdmx_transport if sdmx_transport is not None else _sdmx_transport(populated),
        coverage=_coverage_service(
            stat_vars=populated,
            donor_role="financingProvider",
            donor=_PROVIDER,
            recipient=_RECIPIENT,
            provenance=CRS_PROVENANCE,
            years=_YEARS,
        ),
        **client_kwargs,
    )


def _input() -> dict:
    identity = education_aid_identity()
    spec_id = recompute_identity(identity).spec_id
    return {
        "kind": "spec_id",
        "spec_id": spec_id,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
    }


def test_a_family_whose_first_member_is_empty_at_the_bound_place_previews_as_its_mint_serves():
    family = _family()
    assert len(family) == CRS_MEMBER_COUNT
    populated = _populated(family)
    assert family[0] not in populated
    client = _family_client(populated)
    body = _input()

    previewed = client.post(
        "/api/datasets", json={"schema_version": 1, "input": body, "preview": {"limit": 100}}
    )
    minted = client.post(
        "/api/datasets", json={"schema_version": 1, "input": body, "options": {"bundle": False}}
    )

    assert previewed.status_code == 200, previewed.text
    assert minted.status_code == 200, minted.text
    (entry,) = previewed.json()["members"]
    manifest = minted.json()
    assert entry["stat_var"] == f"dcid:{body['spec_id']}"
    assert entry["content_version"] == manifest["content_version"]
    assert entry["total_rows"] == manifest["coverage"]["rows"] == len(populated) * len(_YEARS)
    assert entry["truncated"] is False
    stat_var_position = entry["sample"]["columns"].index("stat_var")
    assert {row[stat_var_position] for row in entry["sample"]["rows"]} == {
        f"dcid:{dcid}" for dcid in populated
    }


def test_a_truncated_family_preview_keeps_the_latest_rows_and_the_mint_version():
    family = _family()
    populated = _populated(family)
    client = _family_client(populated)
    body = _input()

    previewed = client.post(
        "/api/datasets", json={"schema_version": 1, "input": body, "preview": {"limit": 2}}
    )
    minted = client.post(
        "/api/datasets", json={"schema_version": 1, "input": body, "options": {"bundle": False}}
    )

    assert previewed.status_code == 200, previewed.text
    assert minted.status_code == 200, minted.text
    (entry,) = previewed.json()["members"]
    assert entry["truncated"] is True
    assert entry["total_rows"] == len(populated) * len(_YEARS)
    assert entry["content_version"] == minted.json()["content_version"]
    columns = entry["sample"]["columns"]
    period, stat_var = columns.index("time_period"), columns.index("stat_var")
    # One row per populated member at the latest period, in the group frame's row order.
    assert [(row[stat_var], row[period]) for row in entry["sample"]["rows"]] == [
        (f"dcid:{dcid}", str(max(_YEARS))) for dcid in sorted(populated)
    ]


def test_a_family_preview_waits_for_the_slot_a_mint_holds_and_refuses_as_a_mint_does():
    """While the family's mint holds the only materialisation slot, a preview of the family
    fetches nothing and returns the retryable 503 a second mint returns after the 0.1 s wait
    bound. After release, it previews."""
    populated = _populated(_family())
    answer = _sdmx_transport(populated).handler
    started = threading.Event()
    hold = threading.Event()
    fetches = {"n": 0}

    def _held_first_fetch(request: httpx.Request) -> httpx.Response:
        fetches["n"] += 1
        if fetches["n"] == 1:
            started.set()
            assert hold.wait(timeout=5), "the test never released the mint's fetch"
        return answer(request)  # type: ignore[return-value]

    client = _family_client(
        populated,
        sdmx_transport=httpx.MockTransport(_held_first_fetch),
        settings=DatasetsBuildSettings(materialise_wait_seconds=0.1),
        materialise_max_concurrency=1,
    )
    body = _input()
    preview = {"schema_version": 1, "input": body, "preview": {"limit": 100}}
    mint = {"schema_version": 1, "input": body, "options": {"bundle": False}}
    results: dict[str, int] = {}

    def _mint() -> None:
        results["mint"] = client.post("/api/datasets", json=mint).status_code

    mint_thread = threading.Thread(target=_mint)
    mint_thread.start()
    assert started.wait(timeout=5)

    blocked_preview = client.post("/api/datasets", json=preview)
    blocked_mint = client.post("/api/datasets", json=mint)

    assert not hold.is_set()
    assert blocked_preview.status_code == blocked_mint.status_code == 503, blocked_preview.text
    assert blocked_preview.json()["code"] == blocked_mint.json()["code"] == "service_unavailable"
    assert blocked_preview.json()["retryable"] is True
    assert fetches["n"] == 1

    hold.set()
    mint_thread.join(timeout=10)
    assert not mint_thread.is_alive()
    assert results["mint"] == 200

    previewed = client.post("/api/datasets", json=preview)
    assert previewed.status_code == 200, previewed.text
    assert fetches["n"] > 1
