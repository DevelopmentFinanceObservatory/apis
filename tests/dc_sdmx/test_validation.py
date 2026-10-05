"""Invalid calls fail before transport; rejection tests assert that no request was sent."""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx.client import ContainedIn, SdmxClient
from dc_sdmx.errors import SdmxQueryError


@pytest.fixture
def client_and_transport():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("no HTTP call should happen for a client-side rejection")

    transport = httpx.MockTransport(handler)
    client = SdmxClient(base_url="https://datacommons.one.org/core/api", transport=transport)
    return client, calls


def test_data_requires_at_least_one_variable(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="at least one variable"):
        client.data(variables=[], entities={"observationAbout": ["country/USA"]})
    assert calls == []


def test_data_requires_at_least_one_entity(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="at least one entity filter"):
        client.data(variables=["Count_Person"], entities={})
    assert calls == []


def test_availability_requires_at_least_one_variable(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="at least one variable"):
        client.availability(variables=[], component="observationAbout")
    assert calls == []


def test_availability_allows_no_filters():
    # `availability()` permits omitted filters; `data()` requires an entity filter.
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"data": {"dataConstraints": []}})

    client = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    client.availability(variables=["Count_Person"], component="observationAbout")
    assert len(calls) == 1


def test_availability_invalid_filter_key_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.availability(
            variables=["Count_Person"],
            component="observationAbout",
            filters={"bad role": ["country/USA"]},
        )
    assert calls == []


def test_availability_filter_value_with_reserved_character_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.availability(
            variables=["Count_Person"],
            component="observationAbout",
            filters={"observationAbout": ["country/USA+extra"]},
        )
    assert calls == []


@pytest.mark.parametrize("bad_value", ["Count+Person", "Count,Person"])
def test_variable_value_with_reserved_character_rejected(client_and_transport, bad_value):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.data(variables=[bad_value], entities={"observationAbout": ["country/USA"]})
    assert calls == []


@pytest.mark.parametrize("bad_value", ["country/USA+extra", "country/USA,country/KEN"])
def test_entity_or_list_value_with_reserved_character_rejected(client_and_transport, bad_value):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.data(variables=["Count_Person"], entities={"observationAbout": [bad_value]})
    assert calls == []


def test_facet_id_with_reserved_character_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            facet_ids=["bad+facet"],
        )
    assert calls == []


def test_contained_in_parent_with_reserved_character_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": ContainedIn(parent="africa,bad", child_type="Country")},
        )
    assert calls == []


def test_empty_entity_role_value_list_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.data(variables=["Count_Person"], entities={"observationAbout": []})
    assert calls == []


def test_invalid_entity_role_name_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.data(variables=["Count_Person"], entities={"bad role": ["country/USA"]})
    assert calls == []


# --- time_period ---------------------------------------------------------


def test_time_period_non_latest_bare_string_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="must be exactly 'LATEST'"):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            time_period="2020",
        )
    assert calls == []


def test_time_period_empty_sequence_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="must not be empty"):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            time_period=[],
        )
    assert calls == []


def test_time_period_latest_mixed_into_sequence_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="must not contain 'LATEST'"):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            time_period=["2020", "LATEST"],
        )
    assert calls == []


@pytest.mark.parametrize("bad_value", ["2020+extra", "2020,2021"])
def test_time_period_value_with_reserved_character_rejected(client_and_transport, bad_value):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            time_period=[bad_value],
        )
    assert calls == []


def test_time_period_bare_bytes_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="pass a list of strings, not a bare string"):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            time_period=b"LATEST",  # type: ignore[arg-type]
        )
    assert calls == []


# `str` is a `Sequence[str]`; without validation, `_or_list` splits it into characters.


def test_data_variables_bare_string_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="pass a list of strings, not a bare string"):
        client.data(
            variables="Count_Person",  # type: ignore[arg-type]
            entities={"observationAbout": ["country/USA"]},
        )
    assert calls == []


def test_availability_variables_bare_string_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="pass a list of strings, not a bare string"):
        client.availability(variables="Count_Person", component="observationAbout")  # type: ignore[arg-type]
    assert calls == []


def test_data_facet_ids_bare_string_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="pass a list of strings, not a bare string"):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": ["country/USA"]},
            facet_ids="facet-1",  # type: ignore[arg-type]
        )
    assert calls == []


def test_entity_role_bare_string_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="pass a list of strings, not a bare string"):
        client.data(
            variables=["Count_Person"],
            entities={"observationAbout": "country/USA"},  # type: ignore[dict-item]
        )
    assert calls == []


def test_data_variables_bare_bytes_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match="pass a list of strings, not a bare string"):
        client.data(
            variables=b"Count_Person",  # type: ignore[arg-type]
            entities={"observationAbout": ["country/USA"]},
        )
    assert calls == []


# Issue #849: repeated variables can duplicate merged rows across chunks.


def test_data_duplicate_variable_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match=r"repeats 'Count_Person' \(2 times\)"):
        client.data(
            variables=["Count_Person", "Count_Person"],
            entities={"observationAbout": ["country/USA"]},
        )
    assert calls == []


def test_availability_duplicate_variable_rejected(client_and_transport):
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match=r"repeats 'Count_Person' \(2 times\)"):
        client.availability(
            variables=["Count_Person", "Count_Person"], component="observationAbout"
        )
    assert calls == []


def test_stream_data_duplicate_variable_rejected(client_and_transport):
    client, calls = client_and_transport
    with (
        pytest.raises(SdmxQueryError, match=r"repeats 'Count_Person' \(2 times\)"),
        client.stream_data(
            variables=["Count_Person", "Count_Person"],
            entities={"observationAbout": ["country/USA"]},
        ),
    ):
        pass
    assert calls == []


def test_stream_data_batches_duplicate_variable_rejected(client_and_transport):
    client, calls = client_and_transport
    with (
        pytest.raises(SdmxQueryError, match=r"repeats 'Count_Person' \(2 times\)"),
        client.stream_data_batches(
            variables=["Count_Person", "Count_Person"],
            entities={"observationAbout": ["country/USA"]},
        ),
    ):
        pass
    assert calls == []


def test_duplicate_variable_error_counts_every_occurrence(client_and_transport):
    # The error count includes all three occurrences.
    client, calls = client_and_transport
    with pytest.raises(SdmxQueryError, match=r"repeats 'Count_Person' \(3 times\)"):
        client.data(
            variables=["Count_Person", "Count_Income", "Count_Person", "Count_Person"],
            entities={"observationAbout": ["country/USA"]},
        )
    assert calls == []
