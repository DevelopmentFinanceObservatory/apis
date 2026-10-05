"""Tests rate-table loading, projection precedence, version checks, and caching."""

from __future__ import annotations

import pyarrow as pa
import pytest

from materialise.errors import UpstreamResponseError, UpstreamVersionChangedError
from materialise.rate_tables import (
    RateTable,
    RateTableCache,
    RateTableInvalidError,
    RateTableKind,
    load_table,
)

from ._rate_table_support import RateTableServer, query_params, rate_csv, rate_row
from .conftest import FakeGraphReader

_STAT_VAR = "ExchangeRate_CurrencyUnit_ConversionRecipeDAC"
_DEFLATOR_VAR = "Deflator_EconomicProduction_ConversionRecipeDAC"
_PROVENANCE_ID = "provenance/ONEConversionTablesDAC"


def _rate_row(
    *,
    entity: str,
    year: str,
    value: str,
    method: str = "NotApplicable",
    observation_period: str = "P1Y",
    facet_id: str = "facet-actual",
    stat_var: str = _STAT_VAR,
    provenance: str = _PROVENANCE_ID,
) -> str:
    return rate_row(
        stat_var=stat_var,
        entity=entity,
        year=year,
        value=value,
        provenance=provenance,
        method=method,
        observation_period=observation_period,
        facet_id=facet_id,
    )


class _SequencedRefreshGraphReader(FakeGraphReader):
    """`FakeGraphReader` subclass (so it stays a structural `GraphReader`)
    whose `node_arcs_many` pops the next value off `versions` on every call.
    `versions` entries are `str | None`; `None` mimics a provenance holding no
    `lastDataRefreshDate` at all.
    """

    def __init__(self, versions: list[str | None]) -> None:
        super().__init__()
        self._versions = list(versions)

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        self._count("node_arcs_many")
        version = self._versions.pop(0)
        values = (version,) if version is not None else ()
        return {dcid: dict.fromkeys(properties, values) for dcid in dcids}


def _graph(version: str | None = "2026-09-01") -> FakeGraphReader:
    values = (version,) if version is not None else ()
    return FakeGraphReader(arcs={_PROVENANCE_ID: {"lastDataRefreshDate": values}})


def _load(
    *,
    server: RateTableServer,
    graph: FakeGraphReader,
    cache: RateTableCache | None = None,
    kind: RateTableKind = "exchange_rate",
    stat_var: str = _STAT_VAR,
    url_budget_bytes: int | None = None,
) -> RateTable:
    return load_table(
        kind=kind,
        stat_var=stat_var,
        provenance_id=_PROVENANCE_ID,
        sdmx=server.client(url_budget_bytes=url_budget_bytes),
        graph=graph,
        cache=cache if cache is not None else RateTableCache(),
    )


def _usd_and_kes(kes: str = "110.5") -> str:
    return rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyKES", year="2020", value=kes),
    )


# ---------------------------------------------------------------------------
# Whole-table load: availability, then one data read over every entity
# ---------------------------------------------------------------------------


def test_load_lists_entities_then_fetches_them_all_in_one_data_read():
    server = RateTableServer({_STAT_VAR: _usd_and_kes()})

    result = _load(server=server, graph=_graph())

    assert len(server.requests) == 2
    availability, data = server.requests
    assert availability in server.availability_requests()
    assert availability.url.path.endswith("/observationAbout")
    assert query_params(availability)["c[variableMeasured]"] == _STAT_VAR
    assert data in server.data_requests()
    params = query_params(data)
    assert params["c[variableMeasured]"] == _STAT_VAR
    assert params["c[observationAbout]"] == "CurrencyUSD,CurrencyKES"
    assert result.table.num_rows == 2


def test_a_stat_var_serving_no_entities_refuses_without_a_data_read():
    server = RateTableServer({_STAT_VAR: _usd_and_kes()}, listed={_STAT_VAR: ()})

    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=server, graph=_graph())

    assert "serves no observationAbout entities" in str(exc_info.value)
    assert server.data_requests() == []


def test_an_entity_list_over_the_url_budget_refuses_without_a_data_read():
    listed = [f"Currency{chr(65 + i // 26)}{chr(65 + i % 26)}X" for i in range(200)]
    server = RateTableServer({_STAT_VAR: _usd_and_kes()}, listed={_STAT_VAR: listed})

    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=server, graph=_graph(), url_budget_bytes=1_000)

    assert exc_info.value.code == "capability_unavailable"
    assert "200 entities" in str(exc_info.value)
    assert len(server.availability_requests()) == 1
    assert server.data_requests() == []


# ---------------------------------------------------------------------------
# Projection union and its precedence
# ---------------------------------------------------------------------------


def test_union_prefers_actual_over_projection_for_the_same_entity_year():
    csv = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyUSD", year="2021", value="1"),
        _rate_row(entity="CurrencyKES", year="2020", value="108.0", facet_id="facet-a"),
        _rate_row(
            entity="CurrencyKES",
            year="2020",
            value="999.0",
            method="IMFProjection",
            facet_id="facet-p",
        ),
        _rate_row(
            entity="CurrencyKES",
            year="2021",
            value="120.0",
            method="IMFProjection",
            facet_id="facet-p",
        ),
    )

    result = _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())

    rows = result.table.to_pylist()
    kes_2020 = next(r for r in rows if r["entity"] == "CurrencyKES" and r["year"] == 2020)
    assert kes_2020["value"] == pytest.approx(108.0)
    assert kes_2020["facet"] == "actual"

    kes_2021 = next(r for r in rows if r["entity"] == "CurrencyKES" and r["year"] == 2021)
    assert kes_2021["value"] == pytest.approx(120.0)
    assert kes_2021["facet"] == "projected"


def test_missing_value_cell_is_a_coverage_gap_not_a_validation_failure():
    result = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes(kes="")}), graph=_graph())

    assert result.table.num_rows == 1
    assert result.table.to_pylist()[0]["entity"] == "CurrencyUSD"


# ---------------------------------------------------------------------------
# Admission validation
# ---------------------------------------------------------------------------


def test_duplicate_actual_entity_year_refuses():
    csv = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyKES", year="2020", value="108.0", facet_id="facet-a"),
        _rate_row(entity="CurrencyKES", year="2020", value="109.0", facet_id="facet-b"),
    )

    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())
    assert exc_info.value.code == "capability_unavailable"
    assert exc_info.value.status == 501
    assert "duplicate entity-year" in str(exc_info.value)


def test_duplicate_from_a_different_provenance_resolves_to_the_load_s_own_provenance():
    # A re-import can leave a stale row from a prior provenance alongside
    # the load's own current row for the same entity-year. The stale row
    # (a different provenance) is dropped in favour of the load's own.
    csv = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyKES", year="2020", value="108.0", facet_id="facet-current"),
        _rate_row(
            entity="CurrencyKES",
            year="2020",
            value="999.0",
            facet_id="facet-stale",
            provenance="provenance/ONEConversionTablesDACStale",
        ),
    )

    result = _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())

    kes_row = next(r for r in result.table.to_pylist() if r["entity"] == "CurrencyKES")
    assert kes_row["value"] == pytest.approx(108.0)


def test_non_annual_period_refuses():
    csv = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyKES", year="2020-Q1", value="108.0", observation_period="P3M"),
    )

    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())
    assert "non-annual" in str(exc_info.value)


def test_non_annual_time_period_without_a_declared_period_also_refuses():
    # TIME_PERIOD's quarter suffix triggers the annual-shape check when observationPeriod is NA.
    csv = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyKES", year="2020-Q1", value="108.0"),
    )

    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())
    assert "non-annual" in str(exc_info.value)


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf"])
def test_non_finite_or_non_positive_rate_refuses(value: str):
    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=RateTableServer({_STAT_VAR: _usd_and_kes(kes=value)}), graph=_graph())
    assert "non-finite or non-positive" in str(exc_info.value)


def test_missing_usd_anchor_refuses_an_exchange_rate_table():
    csv = rate_csv(_rate_row(entity="CurrencyKES", year="2020", value="108.0"))

    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())
    assert "missing USD anchor" in str(exc_info.value)


def test_a_country_keyed_usd_row_is_not_the_anchor():
    csv = rate_csv(
        _rate_row(entity="country/USA", year="2020", value="1"),
        _rate_row(entity="CurrencyKES", year="2020", value="108.0"),
    )

    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())
    assert "no 'CurrencyUSD' row" in str(exc_info.value)


def test_usd_anchor_off_unity_refuses():
    csv = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1.0001"),
        _rate_row(entity="CurrencyKES", year="2020", value="108.0"),
    )

    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())
    assert "not within" in str(exc_info.value)


def test_usd_anchor_within_tolerance_of_unity_admits():
    # Independent rounding can leave WEO's NGDP/NGDPD ratio just below 1.0.
    csv = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="0.9999999999"),
        _rate_row(entity="CurrencyKES", year="2020", value="108.0"),
    )

    result = _load(server=RateTableServer({_STAT_VAR: csv}), graph=_graph())

    usd_row = next(r for r in result.table.to_pylist() if r["entity"] == "CurrencyUSD")
    assert usd_row["value"] == pytest.approx(0.9999999999)


def test_usd_anchor_check_does_not_apply_to_a_deflator_table():
    # Deflator rows are about countries, so the table has no currency anchor.
    csv = rate_csv(
        _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2020", value="102.5")
    )

    result = _load(
        server=RateTableServer({_DEFLATOR_VAR: csv}),
        graph=_graph(),
        kind="deflator",
        stat_var=_DEFLATOR_VAR,
    )

    assert result.table.num_rows == 1


def test_unexpected_role_shape_is_an_upstream_response_error():
    csv = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingRecipient,unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
        "facetId\r\ndataflow,DC:DF_OBS(1.0.0),I,"
        f"{_STAT_VAR},country/KEN,NotApplicable,NotApplicable,P1Y,{_PROVENANCE_ID},2020,108.0,,f1\r\n"
    )
    server = RateTableServer({_STAT_VAR: csv}, listed={_STAT_VAR: ("country/KEN",)})

    with pytest.raises(UpstreamResponseError):
        _load(server=server, graph=_graph())


def test_admitted_table_matches_the_declared_schema():
    result = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes()}), graph=_graph())

    assert result.table.schema == pa.schema(
        [
            ("entity", pa.string()),
            ("year", pa.int32()),
            ("value", pa.float64()),
            ("facet", pa.string()),
        ]
    )


# ---------------------------------------------------------------------------
# Version guard: both reads sit between the two refresh-date reads
# ---------------------------------------------------------------------------


def test_a_moved_refresh_date_retries_both_reads_and_admits_the_second_round():
    first = rate_csv(_rate_row(entity="CurrencyUSD", year="2020", value="1"))
    server = RateTableServer({_STAT_VAR: [first, _usd_and_kes()]})
    # before="v1", after="v2" (mismatch) -> retry: before2="v2", after2="v2" (match).
    graph = _SequencedRefreshGraphReader(["v1", "v2", "v2"])

    result = _load(server=server, graph=graph)

    assert len(server.availability_requests()) == 2
    data = server.data_requests()
    assert len(data) == 2
    # The retry lists entities afresh, so its data read names the new entity.
    assert query_params(data[0])["c[observationAbout]"] == "CurrencyUSD"
    assert query_params(data[1])["c[observationAbout]"] == "CurrencyUSD,CurrencyKES"
    assert result.table.num_rows == 2


def test_refresh_date_still_moving_after_the_retry_refuses():
    server = RateTableServer({_STAT_VAR: _usd_and_kes()})
    # Reads v1/v2 then v2/v3; the retry also detects drift.
    graph = _SequencedRefreshGraphReader(["v1", "v2", "v3"])

    with pytest.raises(UpstreamVersionChangedError):
        _load(server=server, graph=graph)
    assert len(server.availability_requests()) == 2
    assert len(server.data_requests()) == 2


def test_no_refresh_date_at_all_refuses_as_an_invalid_table():
    with pytest.raises(RateTableInvalidError) as exc_info:
        _load(server=RateTableServer({_STAT_VAR: _usd_and_kes()}), graph=_graph(version=None))
    assert "lastDataRefreshDate" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Version: a digest of every row, values included
# ---------------------------------------------------------------------------


def test_the_version_is_a_sha256_hex_digest_not_the_refresh_date():
    result = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes()}), graph=_graph())

    assert len(result.version) == 64
    assert set(result.version) <= set("0123456789abcdef")


def test_identical_rows_under_a_new_refresh_date_keep_the_version():
    server = RateTableServer({_STAT_VAR: _usd_and_kes()})
    cache = RateTableCache()

    first = _load(server=server, graph=_graph(version="v1"), cache=cache)
    second = _load(server=server, graph=_graph(version="v2"), cache=cache)

    assert first is not second
    assert len(server.data_requests()) == 2
    assert first.version == second.version


def test_rows_served_in_another_order_keep_the_version():
    reordered = rate_csv(
        _rate_row(entity="CurrencyKES", year="2020", value="110.5"),
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
    )

    first = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes()}), graph=_graph())
    second = _load(server=RateTableServer({_STAT_VAR: reordered}), graph=_graph())

    assert first.version == second.version


def test_a_changed_value_moves_the_version():
    first = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes("110.5")}), graph=_graph())
    second = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes("110.6")}), graph=_graph())

    assert first.version != second.version


def test_an_actual_turned_projection_moves_the_version():
    projected = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyKES", year="2020", value="110.5", method="IMFProjection"),
    )

    actual = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes()}), graph=_graph())
    projection = _load(server=RateTableServer({_STAT_VAR: projected}), graph=_graph())

    assert actual.version != projection.version


def test_an_added_row_moves_the_version():
    wider = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyKES", year="2020", value="110.5"),
        _rate_row(entity="CurrencyKES", year="2021", value="111.0"),
    )

    narrow = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes()}), graph=_graph())
    wide = _load(server=RateTableServer({_STAT_VAR: wider}), graph=_graph())

    assert narrow.version != wide.version


# ---------------------------------------------------------------------------
# Cache: keyed on StatVar, table provenance, and refresh date
# ---------------------------------------------------------------------------


def test_cache_identity_across_two_loads_at_one_refresh_date():
    server = RateTableServer({_STAT_VAR: _usd_and_kes()})
    cache = RateTableCache()
    graph = _graph(version="v1")

    first = _load(server=server, graph=graph, cache=cache)
    second = _load(server=server, graph=graph, cache=cache)

    assert first is second
    assert len(server.requests) == 2  # one availability read, one data read


def test_a_moved_refresh_date_loads_a_new_table_and_keeps_the_old_one():
    v2_rows = rate_csv(
        _rate_row(entity="CurrencyUSD", year="2020", value="1"),
        _rate_row(entity="CurrencyUSD", year="2021", value="1"),
    )
    server = RateTableServer({_STAT_VAR: [_usd_and_kes(), v2_rows]})
    cache = RateTableCache()

    first = _load(server=server, graph=_graph(version="v1"), cache=cache)
    second = _load(server=server, graph=_graph(version="v2"), cache=cache)

    assert first.version != second.version
    assert len(server.data_requests()) == 2
    assert _load(server=server, graph=_graph(version="v1"), cache=cache) is first
    assert len(server.data_requests()) == 2


def test_two_stat_vars_under_one_provenance_cache_apart():
    deflators = rate_csv(
        _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2020", value="102.5")
    )
    server = RateTableServer({_STAT_VAR: _usd_and_kes(), _DEFLATOR_VAR: deflators})
    cache = RateTableCache()
    graph = _graph(version="v1")

    rates = _load(server=server, graph=graph, cache=cache)
    deflator = _load(
        server=server, graph=graph, cache=cache, kind="deflator", stat_var=_DEFLATOR_VAR
    )

    assert rates is not deflator
    assert deflator.stat_var == _DEFLATOR_VAR
    assert len(server.data_requests()) == 2


def test_a_dcid_prefixed_stat_var_shares_the_bare_stat_var_s_entry():
    server = RateTableServer({_STAT_VAR: _usd_and_kes()})
    cache = RateTableCache()
    graph = _graph(version="v1")

    bare = _load(server=server, graph=graph, cache=cache)
    prefixed = _load(server=server, graph=graph, cache=cache, stat_var=f"dcid:{_STAT_VAR}")

    assert prefixed is bare
    assert len(server.data_requests()) == 1


def test_the_first_admitted_table_wins_a_race_on_one_key():
    cache = RateTableCache()
    first = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes("110.5")}), graph=_graph())
    late = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes("110.6")}), graph=_graph())

    assert cache.admit(first) is first
    assert cache.admit(late) is first
    assert (
        cache.get(stat_var=_STAT_VAR, provenance_id=_PROVENANCE_ID, refresh_date="2026-09-01")
        is first
    )


def test_an_admitted_table_carries_the_refresh_date_it_was_admitted_under():
    table = _load(server=RateTableServer({_STAT_VAR: _usd_and_kes()}), graph=_graph("2026-09-01"))

    assert table.refresh_date == "2026-09-01"
    assert table.version != table.refresh_date


def test_a_request_view_returns_its_pinned_table_after_the_refresh_date_moves():
    server = RateTableServer({_STAT_VAR: [_usd_and_kes("110.5"), _usd_and_kes("120.0")]})
    shared = RateTableCache()
    request = shared.for_request()
    graph = _graph("2026-09-01")

    first = _load(server=server, graph=graph, cache=request)
    graph._arcs[_PROVENANCE_ID]["lastDataRefreshDate"] = ("2026-10-01",)
    reads_before = graph.calls["node_arcs_many"]
    second = _load(server=server, graph=graph, cache=request)

    assert second is first
    assert graph.calls["node_arcs_many"] == reads_before
    assert len(server.data_requests()) == 1
    # The process cache spans requests, so a new request sees the moved table.
    fresh = _load(server=server, graph=graph, cache=shared.for_request())
    assert fresh.version != first.version
    assert fresh.refresh_date == "2026-10-01"


def test_a_request_view_pins_a_process_cache_hit():
    server = RateTableServer({_STAT_VAR: _usd_and_kes()})
    shared = RateTableCache()
    warm = _load(server=server, graph=_graph(), cache=shared)
    request = shared.for_request()

    assert _load(server=server, graph=_graph(), cache=request) is warm
    assert request.pinned(stat_var=_STAT_VAR, provenance_id=_PROVENANCE_ID) is warm
    assert len(server.data_requests()) == 1


def test_a_request_view_of_a_request_view_is_the_same_scope():
    request = RateTableCache().for_request()

    assert request.for_request() is request
