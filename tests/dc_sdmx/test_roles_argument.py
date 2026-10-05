"""Caller-supplied `roles` skips stream header pre-scans and fixes the returned role order.
Each chunk must declare the same role set, or the read raises `SdmxResponseError`.
"""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx.client import SdmxClient
from dc_sdmx.errors import SdmxResponseError
from dc_sdmx.results import observation_columns

from .test_chunked_reads import (
    _BASE_URL,
    _BUDGET_ONE_PER_CHUNK,
    _ENTITIES,
    _VAR_A,
    _VAR_B,
    _header,
    _row,
    _two_role_header,
    _two_role_row,
)


def _client(transport: httpx.BaseTransport, *, url_budget_bytes: int) -> SdmxClient:
    return SdmxClient(
        base_url=_BASE_URL,
        transport=transport,
        url_budget_bytes=url_budget_bytes,
        chunk_concurrency=1,
    )


def _variable_measured(request: httpx.Request) -> str:
    value = httpx.QueryParams(str(request.url.query, "utf-8"))["c[variableMeasured]"]
    assert value is not None
    return value


# --- roles= skips the pre-scan: one request per chunk, not two -------------


def test_stream_data_with_roles_issues_one_request_per_chunk(sequenced_transport):
    bodies = [
        _header("donor") + _row(_VAR_A, "country/USA"),
        _header("donor") + _row(_VAR_B, "country/DEU"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with client.stream_data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, roles=("donor",)) as (
        roles,
        rows,
    ):
        rows = list(rows)

    assert len(transport.requests) == 2
    assert [_variable_measured(r) for r in transport.requests] == [_VAR_A, _VAR_B]
    assert roles == ("donor",)
    assert [row.variable for row in rows] == [_VAR_A, _VAR_B]


def test_stream_data_batches_with_roles_issues_one_request_per_chunk(sequenced_transport):
    bodies = [
        _header("donor") + _row(_VAR_A, "country/USA"),
        _header("donor") + _row(_VAR_B, "country/DEU"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with client.stream_data_batches(
        variables=[_VAR_A, _VAR_B], entities=_ENTITIES, roles=("donor",)
    ) as (roles, batches):
        row_count = sum(b.num_rows for b in batches)

    assert len(transport.requests) == 2
    assert roles == ("donor",)
    assert row_count == 2


def test_stream_data_without_roles_still_prescans(sequenced_transport):
    # Without `roles`, each chunk still gets a pre-scan and data request.
    bodies = [
        _header("donor"),  # chunk 1 pre-scan
        _header("donor"),  # chunk 2 pre-scan
        _header("donor") + _row(_VAR_A, "country/USA"),
        _header("donor") + _row(_VAR_B, "country/DEU"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with client.stream_data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES) as (roles, rows):
        list(rows)

    assert len(transport.requests) == 4
    assert roles == ("donor",)


# --- roles= is returned verbatim, even for a single chunk -------------------


def test_stream_data_single_chunk_yields_caller_roles_verbatim(sequenced_transport):
    # The response order differs; the returned tuple keeps the caller's order.
    body = _two_role_header("recipient", "donor") + _two_role_row(
        _VAR_A, "country/KEN", "country/USA"
    )
    transport = sequenced_transport([body])
    client = _client(transport, url_budget_bytes=7_800)

    with client.stream_data(
        variables=[_VAR_A], entities=_ENTITIES, roles=("donor", "recipient")
    ) as (roles, rows):
        rows = list(rows)

    assert roles == ("donor", "recipient")
    assert len(transport.requests) == 1
    assert rows[0].role_values == {"donor": "country/USA", "recipient": "country/KEN"}


def test_stream_data_batches_single_chunk_reorders_onto_caller_roles(sequenced_transport):
    # The batch columns follow caller order, reordering the response by name.
    body = _two_role_header("recipient", "donor") + _two_role_row(
        _VAR_A, "country/KEN", "country/USA"
    )
    transport = sequenced_transport([body])
    client = _client(transport, url_budget_bytes=7_800)

    with client.stream_data_batches(
        variables=[_VAR_A], entities=_ENTITIES, roles=("donor", "recipient")
    ) as (roles, batches):
        batch = next(iter(batches))

    assert roles == ("donor", "recipient")
    assert batch.schema.names == list(observation_columns(("donor", "recipient")))
    row = batch.to_pylist()[0]
    assert row["donor"] == "country/USA"
    assert row["recipient"] == "country/KEN"


# --- A chunk declaring a role outside roles= raises -------------------------


def test_stream_data_raises_when_a_chunk_declares_an_unexpected_role(sequenced_transport):
    bodies = [
        _header("donor") + _row(_VAR_A, "country/USA"),
        # The extra role signals a stale or wrong coverage-derived role set.
        _header("recipient") + _row(_VAR_B, "country/KEN"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with (
        pytest.raises(SdmxResponseError, match=r"unexpected role\(s\) \['recipient'\]"),
        client.stream_data(variables=[_VAR_A, _VAR_B], entities=_ENTITIES, roles=("donor",)) as (
            _roles,
            rows,
        ),
    ):
        list(rows)


def test_stream_data_batches_raises_when_a_chunk_declares_an_unexpected_role(sequenced_transport):
    bodies = [
        _header("donor") + _row(_VAR_A, "country/USA"),
        _header("recipient") + _row(_VAR_B, "country/KEN"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with (
        pytest.raises(SdmxResponseError, match=r"unexpected role\(s\) \['recipient'\]"),
        client.stream_data_batches(
            variables=[_VAR_A, _VAR_B], entities=_ENTITIES, roles=("donor",)
        ) as (_roles, batches),
    ):
        for _batch in batches:
            pass


def test_stream_data_single_chunk_raises_when_response_declares_unexpected_role(
    sequenced_transport,
):
    # The check applies to the single-chunk path too, not just a chunked read.
    body = _two_role_header("donor", "recipient") + _two_role_row(
        _VAR_A, "country/USA", "country/KEN"
    )
    transport = sequenced_transport([body])
    client = _client(transport, url_budget_bytes=7_800)

    with (
        pytest.raises(SdmxResponseError, match=r"unexpected role\(s\) \['recipient'\]"),
        client.stream_data(variables=[_VAR_A], entities=_ENTITIES, roles=("donor",)) as (
            _roles,
            rows,
        ),
    ):
        list(rows)


# --- A response naming fewer roles than roles= also raises -------------------


def test_stream_data_single_chunk_raises_when_response_declares_fewer_roles(sequenced_transport):
    # A response missing `recipient` has a stale or wrong coverage-derived role set.
    body = _header("donor") + _row(_VAR_A, "country/USA")
    transport = sequenced_transport([body])
    client = _client(transport, url_budget_bytes=7_800)

    with (
        pytest.raises(SdmxResponseError, match=r"missing role\(s\) \['recipient'\]"),
        client.stream_data(
            variables=[_VAR_A], entities=_ENTITIES, roles=("donor", "recipient")
        ) as (
            _roles,
            rows,
        ),
    ):
        list(rows)


def test_stream_data_batches_single_chunk_raises_when_response_declares_fewer_roles(
    sequenced_transport,
):
    body = _header("donor") + _row(_VAR_A, "country/USA")
    transport = sequenced_transport([body])
    client = _client(transport, url_budget_bytes=7_800)

    with (
        pytest.raises(SdmxResponseError, match=r"missing role\(s\) \['recipient'\]"),
        client.stream_data_batches(
            variables=[_VAR_A], entities=_ENTITIES, roles=("donor", "recipient")
        ) as (_roles, batches),
    ):
        for _batch in batches:
            pass


def test_stream_data_raises_when_a_chunk_declares_fewer_roles(sequenced_transport):
    # Chunk 2 drops `recipient` from the caller's role set.
    bodies = [
        _two_role_header("donor", "recipient")
        + _two_role_row(_VAR_A, "country/USA", "country/KEN"),
        _header("donor") + _row(_VAR_B, "country/DEU"),
    ]
    transport = sequenced_transport(bodies)
    client = _client(transport, url_budget_bytes=_BUDGET_ONE_PER_CHUNK)

    with (
        pytest.raises(SdmxResponseError, match=r"missing role\(s\) \['recipient'\]"),
        client.stream_data(
            variables=[_VAR_A, _VAR_B], entities=_ENTITIES, roles=("donor", "recipient")
        ) as (
            _roles,
            rows,
        ),
    ):
        list(rows)
