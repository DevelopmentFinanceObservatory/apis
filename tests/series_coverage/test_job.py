"""Pins coverage build and publication with fakes for BigQuery and SDMX."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pydantic
import pytest

from artifact_publication.jobs import Ingestion
from dc_sdmx.results import SdmxDataResult, SdmxObservation, observation_schema
from series_coverage import job, loader
from series_coverage.compatibility import record_serving_index
from series_coverage.io import read_current, read_manifest
from series_coverage.loader import CoverageArtifactError, load_coverage
from series_coverage.schema import (
    BUILD_EPOCH_YEAR,
    COVERAGE_FORMAT_VERSION,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)
from tests.series_coverage.support import code_index_for

_HORIZON = YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=1)

# An index with no StatVar row disagrees with no build, so every publish records it.
_INDEX = code_index_for({})

_CONNECTION = "proj.region.conn"


def year_stats(
    *,
    max_year: int | None,
    unparsed: int = 0,
    early: int = 0,
    outside_rows: int = 0,
    outside_examples: list[str] | None = None,
) -> dict[str, Any]:
    """The year query's one row for a population whose latest year is `max_year`."""
    return {
        "n_outside_rows": outside_rows,
        "outside_examples": outside_examples,
        "n_unparsed": unparsed,
        "n_early": early,
        "max_year": max_year,
    }


class FakeBigQuery:
    """`job.BigQueryReader` fake: `responses` maps a distinctive SQL
    substring to the rows that query should return, checked in insertion
    order (first match wins). A call whose SQL matches nothing raises
    `AssertionError` -- a test wiring up too few canned responses fails
    loudly rather than silently returning nothing.
    """

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self._responses = responses
        self.queries: list[str] = []

    def query(self, sql: str) -> list[dict[str, Any]]:
        self.queries.append(sql)
        for marker, rows in self._responses.items():
            if marker in sql:
                return rows
        raise AssertionError(f"FakeBigQuery: no canned response matches SQL:\n{sql}")

    def stream(self, sql: str) -> Iterator[dict[str, Any]]:
        return iter(self.query(sql))


def _obs(
    variable: str, role_values: dict[str, str], *, facet_id: str = "facet-1"
) -> SdmxObservation:
    return SdmxObservation(
        variable=variable,
        role_values=role_values,
        unit=None,
        measurement_method=None,
        observation_period=None,
        provenance=None,
        time_period="2020",
        value="1.0",
        scaling_factor=None,
        facet_id=facet_id,
    )


class FakeSdmx:
    """`job.CoverageSdmxClient` fake.

    `data_responder(variables, entities, facet_ids)` backs `.data()` --
    returns an `SdmxDataResult`; a fake standing in for a "wrong guess"
    permutation returns one with an empty `rows` tuple (the real mixer's
    own behaviour for a well-formed filter matching nothing), never raises.

    `batches_by_group` backs `.stream_data_batches()`, keyed by the
    `roles` tuple the job passes -- each group's canned rows, as
    `(variable, count)` pairs expanded into that many `variableMeasured`
    cells across a couple of `pa.RecordBatch`es, so a test exercises more
    than one batch per group without hand-building every row.
    """

    url_budget_bytes = 7_800

    def __init__(
        self,
        *,
        data_responder=None,
        batches_by_group: dict[tuple[str, ...], list[tuple[str, int]]] | None = None,
    ) -> None:
        self._data_responder = data_responder
        self._batches_by_group = batches_by_group or {}
        self.stream_data_batches_calls: list[dict[str, Any]] = []

    def data(self, *, variables, entities, facet_ids=None):
        assert self._data_responder is not None
        return self._data_responder(variables, entities, facet_ids)

    @contextmanager
    def stream_data_batches(self, *, variables, entities, roles=None, time_period=None):
        assert roles is not None, "job.reconcile always passes roles="
        self.stream_data_batches_calls.append(
            {"variables": list(variables), "entities": dict(entities), "roles": roles}
        )
        pairs = self._batches_by_group[tuple(roles)]
        schema = observation_schema(roles)

        def _batches():
            for variable, count in pairs:
                if variable not in variables:
                    continue
                row = dict.fromkeys(schema.names)
                row["variableMeasured"] = variable
                yield pa.RecordBatch.from_pylist([row] * count, schema=schema)

        yield roles, _batches()


# --- JobSettings -------------------------------------------------------------


def test_settings_from_env_reads_required_vars(monkeypatch):
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", "file:///tmp/x")
    monkeypatch.setenv("COVERAGE_BQ_CONNECTION", "proj.region.conn")
    monkeypatch.setenv("DATASETS_SDMX_BASE_URL", "https://mixer/core/api")
    monkeypatch.delenv("COVERAGE_BQ_PROJECT", raising=False)
    monkeypatch.delenv("DATASETS_GRAPH_TOKEN", raising=False)

    settings = job.JobSettings.from_env()

    assert settings.artifact_uri == "file:///tmp/x"
    assert settings.bq_connection == "proj.region.conn"
    # bq_project defaults to the connection's own leading segment.
    assert settings.bq_project == "proj"
    assert settings.sdmx_base_url == "https://mixer/core/api"
    assert settings.sdmx_token is None
    assert settings.cycle_seconds == job._DEFAULT_CYCLE_SECONDS
    assert settings.reconcile_max_rows == 250_000


def test_settings_reads_the_reconcile_row_budget(monkeypatch):
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", "file:///tmp/x")
    monkeypatch.setenv("COVERAGE_BQ_CONNECTION", _CONNECTION)
    monkeypatch.setenv("DATASETS_SDMX_BASE_URL", "https://mixer/core/api")
    monkeypatch.setenv("COVERAGE_RECONCILE_MAX_ROWS", "1000")
    assert job.JobSettings.from_env().reconcile_max_rows == 1000

    monkeypatch.setenv("COVERAGE_RECONCILE_MAX_ROWS", "0")
    with pytest.raises(pydantic.ValidationError, match="COVERAGE_RECONCILE_MAX_ROWS"):
        job.JobSettings.from_env()


def test_settings_from_env_missing_required_var_raises(monkeypatch):
    monkeypatch.setenv("COVERAGE_BQ_CONNECTION", "proj.region.conn")
    monkeypatch.delenv("COVERAGE_ARTIFACT_URI", raising=False)
    with pytest.raises(pydantic.ValidationError, match="COVERAGE_ARTIFACT_URI"):
        job.JobSettings.from_env()


def test_settings_from_env_empty_required_var_raises(monkeypatch):
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", "")
    monkeypatch.setenv("COVERAGE_BQ_CONNECTION", "proj.region.conn")
    monkeypatch.setenv("DATASETS_SDMX_BASE_URL", "https://mixer/core/api")
    with pytest.raises(pydantic.ValidationError, match="COVERAGE_ARTIFACT_URI"):
        job.JobSettings.from_env()


def test_settings_by_field_name_construction_still_works():
    settings = job.JobSettings(
        artifact_uri="file:///tmp/x",
        bq_connection=_CONNECTION,
        bq_project="proj",
        sdmx_base_url="https://mixer/core/api",
        sdmx_token=None,
        sdmx_timeout_seconds=90.0,
        cycle_seconds=900.0,
    )
    assert settings.artifact_uri == "file:///tmp/x"
    assert settings.bq_connection == _CONNECTION


def test_settings_bare_unprefixed_variable_has_no_effect(monkeypatch):
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", "file:///real")
    monkeypatch.setenv("COVERAGE_BQ_CONNECTION", _CONNECTION)
    monkeypatch.setenv("DATASETS_SDMX_BASE_URL", "https://mixer/core/api")
    monkeypatch.setenv("ARTIFACT_URI", "file:///bare")
    monkeypatch.setenv("BQ_CONNECTION", "bare-conn")

    settings = job.JobSettings.from_env()

    assert settings.artifact_uri == "file:///real"
    assert settings.bq_connection == _CONNECTION


# --- Fingerprint ---------------------------------------------------------


def test_fingerprint_stamp_is_deterministic():
    fp1 = job.Fingerprint(
        workflow_execution_id="wf-1",
        creation_timestamp="t1",
        row_count=10,
        max_last_update_timestamp="t2",
    )
    fp2 = job.Fingerprint(
        workflow_execution_id="wf-1",
        creation_timestamp="t1",
        row_count=10,
        max_last_update_timestamp="t2",
    )
    fp3 = job.Fingerprint(
        workflow_execution_id="wf-1",
        creation_timestamp="t1",
        row_count=11,
        max_last_update_timestamp="t2",
    )
    assert fp1.stamp == fp2.stamp
    assert fp1.stamp != fp3.stamp


def test_read_fingerprint_combines_history_and_observation_count():
    bq = FakeBigQuery(
        {
            "IngestionHistory": [
                {"WorkflowExecutionID": "wf-9", "CreationTimestamp": "2026-09-24T10:00:00Z"}
            ],
            "COUNT(*) AS row_count": [{"row_count": 6645946, "max_ts": "2026-09-24T09:00:00Z"}],
        }
    )
    fp = job.read_fingerprint(bq, connection=_CONNECTION)
    assert fp.workflow_execution_id == "wf-9"
    assert fp.row_count == 6645946


def test_read_fingerprint_raises_when_no_successful_history_row():
    bq = FakeBigQuery({"IngestionHistory": []})
    with pytest.raises(job.CoverageFingerprintError):
        job.read_fingerprint(bq, connection=_CONNECTION)


def test_lock_held_true_when_lock_owner_set():
    bq = FakeBigQuery({"IngestionLock": [{"LockOwner": "some-workflow"}]})
    assert job.lock_held(bq, connection=_CONNECTION) is True


def test_lock_held_false_when_lock_owner_null():
    bq = FakeBigQuery({"IngestionLock": [{"LockOwner": None}]})
    assert job.lock_held(bq, connection=_CONNECTION) is False


# --- build_coverage_table ----------------------------------------------------

_ROW_V1 = {
    "variable": "V1",
    "facet_id": "facet-1",
    "donor": "country/FRA",
    "recipient": "country/KEN",
    "entities_match_roles": True,
    "n_obs": 3,
    "first_year": 2020,
    "last_year": 2022,
    "year_mask_0": _HORIZON.masks_from_years([2020, 2021, 2022])[0],
}

# An IDS total: one role (`financingRecipient`), so the build puts its
# debtor in `recipient` and leaves `donor` empty.
_IDS_TOTAL = "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_Nominal"
_ROW_IDS_TOTAL = {
    **_ROW_V1,
    "variable": _IDS_TOTAL,
    "donor": "",
    "recipient": "country/KEN",
}
_PROVIDER_TOTAL = "AmountMobilised_FinancialFlow_Gross_Disbursement_DAC1AidType2233_Nominal"
_ROW_PROVIDER_TOTAL = {
    **_ROW_V1,
    "variable": _PROVIDER_TOTAL,
    "donor": "country/FRA",
    "recipient": "",
}


def test_build_coverage_table_matches_schema_and_masks():
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2022)],
            "GROUP BY variable, facet_id, donor, recipient": [_ROW_V1],
        }
    )
    build = job.build_coverage_table(bq, connection=_CONNECTION)
    assert build.horizon == _HORIZON
    table = build.table
    assert table.schema.equals(_HORIZON.schema)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["variable"] == "V1"
    assert row["n_obs"] == 3
    assert row["year_mask_0"] == _HORIZON.masks_from_years([2020, 2021, 2022])[0]


def test_build_coverage_table_remasks_a_negative_signed_year_mask():
    # BigQuery's BIT_OR over a signed INT64 can come back negative once bit
    # 63 is set (2023, the first word's last year); the build must
    # reinterpret it as the equivalent unsigned 64-bit value, not reject or
    # mis-store it.
    row = dict(_ROW_V1)
    row["year_mask_0"] = -(2**63)  # bit 63 alone, signed representation
    row["first_year"] = _HORIZON.max_year
    row["last_year"] = _HORIZON.max_year
    row["n_obs"] = 1
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=_HORIZON.max_year)],
            "GROUP BY variable, facet_id, donor, recipient": [row],
        }
    )
    table = job.build_coverage_table(bq, connection=_CONNECTION).table
    assert table.to_pylist()[0]["year_mask_0"] == 2**63


def test_build_coverage_table_grows_the_horizon_past_the_first_word(tmp_path):
    # A 2040 observation needs a second word (2024-2087): the build sizes the
    # horizon from the year query's max, asks BigQuery for both words, and
    # checks n_obs against their combined bit count.
    horizon = YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=2)
    word_0, word_1 = horizon.masks_from_years([2022, 2023, 2040])
    row = {
        **{k: v for k, v in _ROW_V1.items() if k != "year_mask_0"},
        "n_obs": 3,
        "first_year": 2022,
        "last_year": 2040,
        "year_mask_0": word_0,
        "year_mask_1": word_1,
    }
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2040)],
            "GROUP BY variable, facet_id, donor, recipient": [row],
        }
    )
    build = job.build_coverage_table(bq, connection=_CONNECTION)
    assert build.horizon == horizon
    assert build.table.schema.equals(horizon.schema)
    built = build.table.to_pylist()[0]
    assert (built["year_mask_0"], built["year_mask_1"]) == (word_0, 1 << 16)
    build_sql = bq.queries[-1]
    assert "BETWEEN 1960 AND 2023" in build_sql
    assert "BETWEEN 2024 AND 2087" in build_sql
    assert "AS year_mask_1" in build_sql

    # Published and loaded, the two-word artifact answers a window across
    # the boundary from both words.
    assert job.publish(
        base_uri=f"file://{tmp_path}",
        stamp="stamp-2040",
        build_id="b1",
        table=build.table,
        horizon=build.horizon,
        role_names=_ROLE_NAMES,
        facet_metadata=_FACET_METADATA,
        source="test",
        ingestion=Ingestion(workflow_execution_id="wf-1", creation_timestamp="t1"),
        expected_generation=None,
    )
    loaded = load_coverage(f"file://{tmp_path}")
    assert loaded.manifest.horizon == horizon
    assert loaded.index.horizon == horizon
    [coverage] = loaded.index.filter_candidates(variables=["V1"], first_year=2023, last_year=2040)
    [facet] = coverage.facets
    assert facet.year_counts == {2023: 1, 2040: 1}


def test_build_coverage_table_builds_a_two_word_horizon_from_1960_for_1970_rows():
    # Role-bearing observations run from 1970 to 2032: the epoch is 1960 and the
    # 73-year span needs two words.
    horizon = YearHorizon(epoch_year=1960, mask_words=2)
    word_0, word_1 = horizon.masks_from_years([1970, 2032])
    row = {
        **{k: v for k, v in _ROW_V1.items() if k != "year_mask_0"},
        "n_obs": 2,
        "first_year": 1970,
        "last_year": 2032,
        "year_mask_0": word_0,
        "year_mask_1": word_1,
    }
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2032)],
            "GROUP BY variable, facet_id, donor, recipient": [row],
        }
    )
    build = job.build_coverage_table(bq, connection=_CONNECTION)
    assert build.horizon == horizon
    assert build.table.schema.equals(horizon.schema)
    assert build.table.to_pylist()[0]["year_mask_0"] == 1 << 10


def test_build_coverage_table_refuses_on_unrepresentable_year():
    bq = FakeBigQuery({"AS max_year": [year_stats(max_year=2022, early=2)]})
    with pytest.raises(job.CoverageBuildError, match="year before 1960"):
        job.build_coverage_table(bq, connection=_CONNECTION)
    assert "year < 1960" in bq.queries[0]


def test_build_coverage_table_refuses_an_unparseable_date():
    bq = FakeBigQuery({"AS max_year": [year_stats(max_year=2022, unparsed=2)]})
    with pytest.raises(job.CoverageBuildError, match="2 role-bearing Observation row"):
        job.build_coverage_table(bq, connection=_CONNECTION)


def test_build_coverage_table_refuses_on_year_past_the_clock_derived_ceiling():
    # 9999 parses as a year but exceeds the clock-derived ceiling. The check runs before GROUP BY.
    bq = FakeBigQuery({"AS max_year": [year_stats(max_year=9999)]})
    with pytest.raises(job.CoverageBuildError, match="9999"):
        job.build_coverage_table(bq, connection=_CONNECTION)


def test_build_coverage_table_refuses_on_popcount_mismatch():
    # n_obs=4 but year_mask only sets 3 bits: more than one observation in a
    # year for this series, which the artifact's own invariant forbids.
    row = dict(_ROW_V1)
    row["n_obs"] = 4
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2022)],
            "GROUP BY variable, facet_id, donor, recipient": [row],
        }
    )
    with pytest.raises(job.CoverageBuildError, match="n_obs"):
        job.build_coverage_table(bq, connection=_CONNECTION)


def test_build_coverage_table_selects_by_role_count_and_places_a_lone_entity_on_its_axis():
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2022)],
            "GROUP BY variable, facet_id, donor, recipient": [
                _ROW_V1,
                _ROW_IDS_TOTAL,
                _ROW_PROVIDER_TOTAL,
            ],
        }
    )
    table = job.build_coverage_table(bq, connection=_CONNECTION).table
    assert [(r["variable"], r["donor"], r["recipient"]) for r in table.to_pylist()] == [
        (_PROVIDER_TOTAL, "country/FRA", ""),
        (_IDS_TOTAL, "", "country/KEN"),
        ("V1", "country/FRA", "country/KEN"),
    ]
    year_sql, build_sql = bq.queries
    for sql in (year_sql, build_sql):
        assert "FROM declared_roles WHERE n_roles BETWEEN 1 AND 2" in sql
        assert "extra_entities_id IS NOT NULL" not in sql
    # `lone_origin` holds for a variable whose one role ROLE_AXES puts on the origin axis.
    assert (
        "COUNT(DISTINCT object_id) = 1\n"
        "      AND LOGICAL_AND(object_id IN "
        "('creditor', 'financingProvider', 'provider', 'source')) AS lone_origin"
    ) in build_sql
    assert "JOIN role_bearing AS r ON o.variable_measured = r.variable" in build_sql
    assert "IF(r.n_roles = 1 AND NOT r.lone_origin, '', o.entity1) AS donor" in build_sql
    assert (
        "CASE WHEN r.n_roles = 2 THEN o.extra_entities_id\n"
        "    WHEN r.lone_origin THEN '' ELSE o.entity1 END AS recipient"
    ) in build_sql


def test_build_coverage_table_refuses_entities_that_disagree_with_the_role_count():
    # A two-role variable with an empty extra_entities_id (or a one-role
    # variable with a set one) would put an entity on the wrong axis.
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2022)],
            "GROUP BY variable, facet_id, donor, recipient": [
                _ROW_V1,
                {**_ROW_IDS_TOTAL, "entities_match_roles": False},
            ],
        }
    )
    with pytest.raises(job.CoverageBuildError, match=_IDS_TOTAL):
        job.build_coverage_table(bq, connection=_CONNECTION)


def test_build_coverage_table_refuses_rows_outside_the_role_population():
    # A three-role variable with two-entity rows is outside `role_bearing`,
    # so every later query would drop it. The year scan counts such rows and
    # the build refuses before the build query runs.
    bq = FakeBigQuery(
        {
            "AS max_year": [
                year_stats(max_year=2022, outside_rows=4, outside_examples=["dc/threeRoleFlow"])
            ]
        }
    )
    with pytest.raises(job.CoverageBuildError, match=r"4 Observation row\(s\).*dc/threeRoleFlow"):
        job.build_coverage_table(bq, connection=_CONNECTION)
    [year_sql] = bq.queries
    assert "LEFT JOIN declared_roles AS d ON o.variable_measured = d.variable" in year_sql
    assert "WHERE d.n_roles IS NOT NULL OR IFNULL(o.extra_entities_id, '') != ''" in year_sql
    assert "IFNULL(d.n_roles BETWEEN 1 AND 2, FALSE) AS in_population" in year_sql
    assert "COUNTIF(NOT in_population) AS n_outside_rows" in year_sql


def test_read_fingerprint_counts_role_bearing_observations_by_role_count():
    bq = FakeBigQuery(
        {
            "IngestionHistory": [{"WorkflowExecutionID": "wf-1", "CreationTimestamp": "t1"}],
            "COUNT(*) AS row_count": [{"row_count": 1, "max_ts": "t2"}],
        }
    )
    job.read_fingerprint(bq, connection=_CONNECTION)
    count_sql = bq.queries[-1]
    assert "FROM declared_roles WHERE n_roles BETWEEN 1 AND 2" in count_sql
    assert "WHERE variable_measured IN (SELECT variable FROM role_bearing)" in count_sql


def test_build_coverage_table_refuses_on_empty_result():
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2022)],
            "GROUP BY variable, facet_id, donor, recipient": [],
        }
    )
    with pytest.raises(job.CoverageBuildError, match="zero"):
        job.build_coverage_table(bq, connection=_CONNECTION)


def test_build_coverage_table_sorts_rows_by_variable_then_facet_id():
    # BigQuery's GROUP BY makes no row-order promise; the job must not rely
    # on it, so the rows below arrive deliberately out of (variable, facet_id) order.
    row_v2 = {**_ROW_V1, "variable": "V2", "facet_id": "facet-2"}
    row_v1_facet2 = {**_ROW_V1, "facet_id": "facet-2"}
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2022)],
            "GROUP BY variable, facet_id, donor, recipient": [row_v2, _ROW_V1, row_v1_facet2],
        }
    )
    table = job.build_coverage_table(bq, connection=_CONNECTION).table
    assert list(zip(table["variable"].to_pylist(), table["facet_id"].to_pylist(), strict=True)) == [
        ("V1", "facet-1"),
        ("V1", "facet-2"),
        ("V2", "facet-2"),
    ]


def test_build_coverage_table_output_lets_the_index_skip_reordering_on_load(monkeypatch):
    # Sorted rows let CoverageIndex skip its reorder. Patching `np.argsort`
    # proves the skip path ran.
    from series_coverage import index as index_module

    row_v2 = {**_ROW_V1, "variable": "V2", "facet_id": "facet-2"}
    row_v1_facet2 = {**_ROW_V1, "facet_id": "facet-2"}
    bq = FakeBigQuery(
        {
            "AS max_year": [year_stats(max_year=2022)],
            "GROUP BY variable, facet_id, donor, recipient": [row_v2, _ROW_V1, row_v1_facet2],
        }
    )
    build = job.build_coverage_table(bq, connection=_CONNECTION)

    def _reorder_was_not_supposed_to_run(*_args, **_kwargs):
        raise AssertionError("CoverageIndex reordered an already-sorted table")

    monkeypatch.setattr(index_module.np, "argsort", _reorder_was_not_supposed_to_run)

    role_names = {
        "V1": RoleNames(donor="financingProvider", recipient="financingRecipient"),
        "V2": RoleNames(donor="financingProvider", recipient="financingRecipient"),
    }
    index = index_module.CoverageIndex(build.table, role_names, horizon=build.horizon)
    assert index.num_rows == 3


# --- resolve_role_names --------------------------------------------------


def _coverage_table(*rows: dict[str, Any]) -> pa.Table:
    import pyarrow as pa

    return pa.table(
        {
            "variable": [r["variable"] for r in rows],
            "facet_id": [r["facet_id"] for r in rows],
            "donor": [r["donor"] for r in rows],
            "recipient": [r["recipient"] for r in rows],
            "n_obs": pa.array([r["n_obs"] for r in rows], type=pa.uint32()),
            "first_year": pa.array([r["first_year"] for r in rows], type=pa.uint16()),
            "last_year": pa.array([r["last_year"] for r in rows], type=pa.uint16()),
            "year_mask_0": pa.array([r["year_mask_0"] for r in rows], type=pa.uint64()),
        },
        schema=_HORIZON.schema,
    )


def test_resolve_role_names_matches_donor_recipient_via_live_data():
    table = _coverage_table(_ROW_V1)
    bq = FakeBigQuery(
        {
            "observationProperties": [
                {"variable": "V1", "role": "financingProvider"},
                {"variable": "V1", "role": "financingRecipient"},
            ]
        }
    )

    def _responder(variables, entities, facet_ids):
        # The correct guess names financingProvider for the donor dcid.
        if entities.get("financingProvider") == ["country/FRA"]:
            return SdmxDataResult(
                roles=("financingProvider", "financingRecipient"),
                rows=(
                    _obs(
                        "V1",
                        {"financingProvider": "country/FRA", "financingRecipient": "country/KEN"},
                    ),
                ),
            )
        return SdmxDataResult(roles=(), rows=())

    sdmx = FakeSdmx(data_responder=_responder)
    role_names = job.resolve_role_names(bq, sdmx, connection=_CONNECTION, table=table)
    assert role_names == {
        "V1": RoleNames(donor="financingProvider", recipient="financingRecipient")
    }


def test_resolve_role_names_tries_the_swapped_permutation():
    # Alphabetically, financingProvider < financingRecipient, but here the
    # REAL assignment is reversed -- proves the resolver doesn't just trust
    # alphabetical order.
    table = _coverage_table(_ROW_V1)
    bq = FakeBigQuery(
        {
            "observationProperties": [
                {"variable": "V1", "role": "financingProvider"},
                {"variable": "V1", "role": "financingRecipient"},
            ]
        }
    )

    def _responder(variables, entities, facet_ids):
        if entities.get("financingRecipient") == ["country/FRA"]:
            return SdmxDataResult(
                roles=("financingRecipient", "financingProvider"), rows=(_obs("V1", {}),)
            )
        return SdmxDataResult(roles=(), rows=())

    sdmx = FakeSdmx(data_responder=_responder)
    role_names = job.resolve_role_names(bq, sdmx, connection=_CONNECTION, table=table)
    assert role_names == {
        "V1": RoleNames(donor="financingRecipient", recipient="financingProvider")
    }


def test_resolve_role_names_applies_one_resolution_to_the_whole_group():
    table = _coverage_table(
        _ROW_V1, {**_ROW_V1, "variable": "V2", "donor": "country/DEU", "recipient": "country/ETH"}
    )
    bq = FakeBigQuery(
        {
            "observationProperties": [
                {"variable": "V1", "role": "financingProvider"},
                {"variable": "V1", "role": "financingRecipient"},
                {"variable": "V2", "role": "financingProvider"},
                {"variable": "V2", "role": "financingRecipient"},
            ]
        }
    )
    calls = []

    def _responder(variables, entities, facet_ids):
        calls.append(variables)
        if entities.get("financingProvider"):
            return SdmxDataResult(
                roles=("financingProvider", "financingRecipient"), rows=(_obs("V1", {}),)
            )
        return SdmxDataResult(roles=(), rows=())

    sdmx = FakeSdmx(data_responder=_responder)
    role_names = job.resolve_role_names(bq, sdmx, connection=_CONNECTION, table=table)
    assert role_names["V1"] == role_names["V2"]
    # Only V1 (the first variable in the group) needed a live round trip.
    assert calls == [["V1"]]


@pytest.mark.parametrize(
    ("row", "role", "expected"),
    [
        (
            _ROW_IDS_TOTAL,
            "financingRecipient",
            RoleNames(donor=None, recipient="financingRecipient"),
        ),
        (
            _ROW_PROVIDER_TOTAL,
            "financingProvider",
            RoleNames(donor="financingProvider", recipient=None),
        ),
    ],
)
def test_resolve_role_names_names_a_one_role_variable_by_its_axis_without_an_sdmx_probe(
    row: dict[str, Any], role: str, expected: RoleNames
):
    table = _coverage_table(row)
    bq = FakeBigQuery({"observationProperties": [{"variable": row["variable"], "role": role}]})

    def _no_probe(*_args, **_kwargs):
        raise AssertionError("a one-role variable needs no SDMX probe")

    role_names = job.resolve_role_names(
        bq, FakeSdmx(data_responder=_no_probe), connection=_CONNECTION, table=table
    )
    assert role_names == {row["variable"]: expected}


def test_resolve_role_names_refuses_a_one_role_variable_on_no_axis():
    table = _coverage_table(_ROW_IDS_TOTAL)
    bq = FakeBigQuery(
        {"observationProperties": [{"variable": _IDS_TOTAL, "role": "financingChannel"}]}
    )
    with pytest.raises(job.CoverageRoleNameError, match="no flow axis"):
        job.resolve_role_names(bq, FakeSdmx(), connection=_CONNECTION, table=table)


def test_resolve_role_names_raises_on_three_roles():
    table = _coverage_table(_ROW_V1)
    bq = FakeBigQuery(
        {
            "observationProperties": [
                {"variable": "V1", "role": role}
                for role in ("financingProvider", "financingRecipient", "financingChannel")
            ]
        }
    )
    with pytest.raises(job.CoverageRoleNameError, match="not one or two"):
        job.resolve_role_names(bq, FakeSdmx(), connection=_CONNECTION, table=table)


def test_resolve_role_names_raises_when_both_permutations_fail():
    table = _coverage_table(_ROW_V1)
    bq = FakeBigQuery(
        {
            "observationProperties": [
                {"variable": "V1", "role": "financingProvider"},
                {"variable": "V1", "role": "financingRecipient"},
            ]
        }
    )
    sdmx = FakeSdmx(data_responder=lambda *a, **k: SdmxDataResult(roles=(), rows=()))
    with pytest.raises(job.CoverageRoleNameError, match="could not determine"):
        job.resolve_role_names(bq, sdmx, connection=_CONNECTION, table=table)


# --- resolve_facet_metadata ------------------------------------------------

_FACET_ROW = {
    "facet_id": "facet-1",
    "provenance": "provenance/OECDCreditorReportingSystem",
    "unit": "CurrencyUSD",
    "measurement_method": None,
    "observation_period": "P1Y",
}


def test_resolve_facet_metadata_matches_schema():
    table = _coverage_table(_ROW_V1)  # facet_id "facet-1"
    bq = FakeBigQuery(
        {
            "GROUP BY facet_id, provenance, unit, measurement_method, observation_period": [
                _FACET_ROW
            ]
        }
    )
    metadata = job.resolve_facet_metadata(bq, connection=_CONNECTION, table=table)
    assert metadata == {
        "facet-1": FacetMetadata(
            provenance_id="provenance/OECDCreditorReportingSystem",
            unit="CurrencyUSD",
            scaling_factor=None,
            observation_period="P1Y",
        )
    }


def test_resolve_facet_metadata_reads_observation_period_inside_the_external_query():
    table = _coverage_table(_ROW_V1)
    bq = FakeBigQuery({"GROUP BY facet_id, provenance, unit, measurement_method": [_FACET_ROW]})
    job.resolve_facet_metadata(bq, connection=_CONNECTION, table=table)
    (sql,) = bq.queries
    assert (
        "SELECT facet_id, provenance, unit, measurement_method, observation_period, "
        "variable_measured FROM TimeSeries" in sql
    )
    assert "WHERE variable_measured IN (SELECT variable FROM role_bearing)" in sql
    assert sql.endswith(
        "GROUP BY facet_id, provenance, unit, measurement_method, observation_period"
    )


def test_resolve_facet_metadata_keeps_a_null_observation_period():
    table = _coverage_table(_ROW_V1)
    bq = FakeBigQuery(
        {
            "GROUP BY facet_id, provenance, unit, measurement_method": [
                {**_FACET_ROW, "observation_period": None}
            ]
        }
    )
    metadata = job.resolve_facet_metadata(bq, connection=_CONNECTION, table=table)
    assert metadata["facet-1"].observation_period is None


def test_resolve_facet_metadata_raises_when_a_facet_id_has_two_observation_periods():
    table = _coverage_table(_ROW_V1)
    bq = FakeBigQuery(
        {
            "GROUP BY facet_id, provenance, unit, measurement_method": [
                _FACET_ROW,
                {**_FACET_ROW, "observation_period": None},
            ]
        }
    )
    with pytest.raises(job.CoverageFacetMetadataError, match="observation_period"):
        job.resolve_facet_metadata(bq, connection=_CONNECTION, table=table)


def test_resolve_facet_metadata_raises_when_a_facet_id_maps_to_two_tuples():
    table = _coverage_table(_ROW_V1)
    bq = FakeBigQuery(
        {
            "GROUP BY facet_id, provenance, unit, measurement_method, observation_period": [
                _FACET_ROW,
                {**_FACET_ROW, "unit": "CurrencyEUR"},
            ]
        }
    )
    with pytest.raises(job.CoverageFacetMetadataError, match="more than one"):
        job.resolve_facet_metadata(bq, connection=_CONNECTION, table=table)


def test_resolve_facet_metadata_raises_when_a_table_facet_id_has_no_timeseries_row():
    table = _coverage_table(_ROW_V1)  # facet_id "facet-1"
    bq = FakeBigQuery(
        {
            "GROUP BY facet_id, provenance, unit, measurement_method, observation_period": [
                {**_FACET_ROW, "facet_id": "other"}
            ]
        }
    )
    with pytest.raises(job.CoverageFacetMetadataError, match="facet-1"):
        job.resolve_facet_metadata(bq, connection=_CONNECTION, table=table)


# --- reconcile -----------------------------------------------------------

_ROLE_NAMES = {"V1": RoleNames(donor="financingProvider", recipient="financingRecipient")}
_STAMP = "stamp-1"
_BUDGET = 250_000
_FACET_METADATA = {
    "facet-1": FacetMetadata(
        provenance_id="provenance/OECDCreditorReportingSystem", unit="CurrencyUSD"
    )
}


def test_reconcile_passes_when_counts_agree():
    table = _coverage_table(_ROW_V1)  # n_obs=3
    sdmx = FakeSdmx(batches_by_group={("financingProvider", "financingRecipient"): [("V1", 3)]})
    result = job.reconcile(
        sdmx, table=table, role_names=_ROLE_NAMES, stamp=_STAMP, max_rows=_BUDGET
    )
    assert result.variables_checked == 1
    assert result.request_count >= 1
    # roles= was passed through, skipping the pre-scan.
    assert sdmx.stream_data_batches_calls[0]["roles"] == ("financingProvider", "financingRecipient")


def test_reconcile_raises_on_row_count_mismatch():
    table = _coverage_table(_ROW_V1)  # n_obs=3
    sdmx = FakeSdmx(batches_by_group={("financingProvider", "financingRecipient"): [("V1", 2)]})
    with pytest.raises(job.CoverageReconciliationError, match="V1"):
        job.reconcile(sdmx, table=table, role_names=_ROLE_NAMES, stamp=_STAMP, max_rows=_BUDGET)


def test_reconcile_raises_when_sdmx_side_is_empty():
    table = _coverage_table(_ROW_V1)
    sdmx = FakeSdmx(batches_by_group={("financingProvider", "financingRecipient"): []})
    with pytest.raises(job.CoverageReconciliationError):
        job.reconcile(sdmx, table=table, role_names=_ROLE_NAMES, stamp=_STAMP, max_rows=_BUDGET)


def test_reconcile_raises_on_empty_table():

    empty = _HORIZON.schema.empty_table()
    sdmx = FakeSdmx(batches_by_group={})
    with pytest.raises(job.CoverageReconciliationError, match="no role-bearing rows"):
        job.reconcile(sdmx, table=empty, role_names={}, stamp=_STAMP, max_rows=_BUDGET)


@pytest.mark.parametrize(
    ("row", "column", "names", "dcids"),
    [
        (
            _ROW_IDS_TOTAL,
            "recipient",
            RoleNames(donor=None, recipient="financingRecipient"),
            ("country/KEN", "country/UGA"),
        ),
        (
            _ROW_PROVIDER_TOTAL,
            "donor",
            RoleNames(donor="financingProvider", recipient=None),
            ("country/FRA", "country/UGA"),
        ),
    ],
)
def test_reconcile_filters_a_one_role_group_on_its_own_axis(
    row: dict[str, Any], column: str, names: RoleNames, dcids: tuple[str, ...]
):
    variable = row["variable"]
    table = _coverage_table(_ROW_V1, row, {**row, column: "country/UGA"})
    role_names = {**_ROLE_NAMES, variable: names}
    sdmx = FakeSdmx(
        batches_by_group={
            ("financingProvider", "financingRecipient"): [("V1", 3)],
            names.declared: [(variable, 6)],
        }
    )
    result = job.reconcile(sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET)
    assert result.variables_checked == 2
    one_role_call = next(
        call for call in sdmx.stream_data_batches_calls if call["roles"] == names.declared
    )
    assert one_role_call["entities"] == {names.declared[0]: dcids}


def test_reconcile_samples_a_one_role_group_by_its_smallest_variable():
    # max_rows=1 keeps only each group's smallest variable. The one-role
    # group holds the IDS total (3 rows) and IDS_BIG (6 rows).
    big = {**_ROW_IDS_TOTAL, "variable": "IDS_BIG", "recipient": "country/UGA", "n_obs": 6}
    table = _coverage_table(_ROW_V1, _ROW_IDS_TOTAL, big)
    one_role = RoleNames(donor=None, recipient="financingRecipient")
    role_names = {**_ROLE_NAMES, _IDS_TOTAL: one_role, "IDS_BIG": one_role}
    sdmx = FakeSdmx(
        batches_by_group={
            ("financingProvider", "financingRecipient"): [("V1", 3)],
            ("financingRecipient",): [(_IDS_TOTAL, 3), ("IDS_BIG", 6)],
        }
    )
    result = job.reconcile(sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=1)
    assert result.variables_checked == 2
    assert result.rows_expected == 6
    one_role_call = next(
        call for call in sdmx.stream_data_batches_calls if call["roles"] == ("financingRecipient",)
    )
    assert one_role_call["variables"] == [_IDS_TOTAL]
    assert one_role_call["entities"] == {"financingRecipient": ("country/KEN",)}


def test_reconcile_entities_filter_is_exhaustive_over_the_smaller_axis():
    # Two donors, one recipient among V1's rows -> recipient axis (1
    # distinct dcid) is smaller, so the entities filter binds recipient.
    table = _coverage_table(
        _ROW_V1, {**_ROW_V1, "variable": "V1", "donor": "country/DEU", "facet_id": "facet-2"}
    )
    sdmx = FakeSdmx(batches_by_group={("financingProvider", "financingRecipient"): [("V1", 6)]})
    job.reconcile(sdmx, table=table, role_names=_ROLE_NAMES, stamp=_STAMP, max_rows=_BUDGET)
    entities = sdmx.stream_data_batches_calls[0]["entities"]
    assert entities == {"financingRecipient": ("country/KEN",)}


# Two role groups. Group A holds A00 (4 rows) and A01..A20 (10 rows each);
# group B holds B1 (6 rows) and B2 (500 rows).
_GROUP_A = ("financingProvider", "financingRecipient")
_GROUP_B = ("donor", "recipient")
_SAMPLE_COUNTS = {
    "A00": 4,
    **{f"A{i:02d}": 10 for i in range(1, 21)},
    "B1": 6,
    "B2": 500,
}
_SAMPLE_ROLE_NAMES = {
    variable: RoleNames(
        donor=(_GROUP_A if variable.startswith("A") else _GROUP_B)[0],
        recipient=(_GROUP_A if variable.startswith("A") else _GROUP_B)[1],
    )
    for variable in _SAMPLE_COUNTS
}


def _sample_table(counts: dict[str, int] = _SAMPLE_COUNTS) -> pa.Table:
    # Each variable has its own donor and recipient, so the entity filter
    # binds the donor axis and shows which variables it was computed from.
    return _coverage_table(
        *(
            {
                **_ROW_V1,
                "variable": variable,
                "donor": f"country/D{variable}",
                "recipient": f"country/R{variable}",
                "n_obs": n,
            }
            for variable, n in counts.items()
        )
    )


def _sample_sdmx(counts: dict[str, int] = _SAMPLE_COUNTS) -> FakeSdmx:
    pairs = list(counts.items())
    return FakeSdmx(
        batches_by_group={
            _GROUP_A: [(v, n) for v, n in pairs if v.startswith("A")],
            _GROUP_B: [(v, n) for v, n in pairs if v.startswith("B")],
        }
    )


def _requested(sdmx: FakeSdmx) -> dict[tuple[str, ...], list[str]]:
    return {tuple(call["roles"]): call["variables"] for call in sdmx.stream_data_batches_calls}


def _reconcile_sample(
    *,
    stamp: str = _STAMP,
    max_rows: int,
    counts: dict[str, int] = _SAMPLE_COUNTS,
    sdmx: FakeSdmx | None = None,
) -> tuple[job.ReconciliationResult, FakeSdmx]:
    sdmx = sdmx or _sample_sdmx(counts)
    result = job.reconcile(
        sdmx,
        table=_sample_table(counts),
        role_names=_SAMPLE_ROLE_NAMES,
        stamp=stamp,
        max_rows=max_rows,
    )
    return result, sdmx


def test_reconcile_sample_checks_the_smallest_variable_of_every_group():
    result, sdmx = _reconcile_sample(max_rows=10)
    assert _requested(sdmx) == {_GROUP_A: ["A00"], _GROUP_B: ["B1"]}
    assert result.variables_checked == 2
    assert result.rows_expected == 10


def test_reconcile_sample_stays_within_the_row_budget():
    result, sdmx = _reconcile_sample(max_rows=100)
    requested = [v for variables in _requested(sdmx).values() for v in variables]
    assert result.rows_expected == sum(_SAMPLE_COUNTS[v] for v in requested) <= 100
    assert result.variables_checked == len(requested)
    # Mins (4 + 6) plus nine 10-row variables fill the budget exactly.
    assert len(requested) == 11
    assert "B2" not in requested


def test_reconcile_sample_is_deterministic_per_stamp_and_moves_with_it():
    _, first = _reconcile_sample(stamp="stamp-1", max_rows=50)
    _, again = _reconcile_sample(stamp="stamp-1", max_rows=50)
    _, other = _reconcile_sample(stamp="stamp-2", max_rows=50)
    assert _requested(first) == _requested(again)
    assert _requested(first) != _requested(other)


def test_reconcile_sample_checks_a_group_whose_smallest_variable_exceeds_the_budget():
    # B2 (500) is now the smallest of group B and alone exceeds the budget.
    result, sdmx = _reconcile_sample(max_rows=100, counts={**_SAMPLE_COUNTS, "B1": 900})
    assert _requested(sdmx) == {_GROUP_A: ["A00"], _GROUP_B: ["B2"]}
    assert result.rows_expected == 504


def test_reconcile_sample_raises_on_a_mismatch_in_a_sampled_variable():
    sdmx = _sample_sdmx({**_SAMPLE_COUNTS, "B1": 5})
    with pytest.raises(job.CoverageReconciliationError, match="B1"):
        _reconcile_sample(max_rows=10, sdmx=sdmx)


def test_reconcile_never_requests_variables_outside_the_sample():
    # Mins (4 + 6) leave 40 rows of budget: four 10-row A variables.
    _, sdmx = _reconcile_sample(max_rows=50)
    requested = _requested(sdmx)
    assert requested[_GROUP_B] == ["B1"]
    assert len(requested[_GROUP_A]) == 5
    assert "A00" in requested[_GROUP_A]
    # The entity filter lists only the donors of the sampled variables.
    for call in sdmx.stream_data_batches_calls:
        donors = tuple(sorted(f"country/D{v}" for v in call["variables"]))
        assert call["entities"] == {call["roles"][0]: donors}


class TableBackedSdmx:
    """`job.CoverageSdmxClient` fake that answers `stream_data_batches` from
    a coverage table, honouring the entity filter: each matching row yields
    `n_obs` observations. Records every call's variables and entities."""

    def __init__(
        self, table: pa.Table, role_names: dict[str, RoleNames], *, url_budget_bytes: int = 7_800
    ) -> None:
        self._rows = table.to_pylist()
        self._role_names = role_names
        self.url_budget_bytes = url_budget_bytes
        self.calls: list[tuple[tuple[str, ...], dict[str, tuple[str, ...]]]] = []

    def data(self, *, variables, entities, facet_ids=None):
        raise AssertionError("reconcile never calls data()")

    @contextmanager
    def stream_data_batches(self, *, variables, entities, roles=None, time_period=None):
        assert roles is not None
        self.calls.append((tuple(variables), {k: tuple(v) for k, v in entities.items()}))
        [(role, dcids)] = entities.items()
        schema = observation_schema(roles)

        def _batches():
            for row in self._rows:
                if row["variable"] not in variables:
                    continue
                column = "donor" if role == self._role_names[row["variable"]].donor else "recipient"
                if row[column] in dcids:
                    cells = dict.fromkeys(schema.names)
                    cells["variableMeasured"] = row["variable"]
                    yield pa.RecordBatch.from_pylist([cells] * row["n_obs"], schema=schema)

        yield roles, _batches()


def _expected_rows(table: pa.Table, variables, entities) -> int:
    [(role, dcids)] = entities.items()
    column = "donor" if role == "financingProvider" else "recipient"
    return sum(
        row["n_obs"]
        for row in table.to_pylist()
        if row["variable"] in variables and row[column] in dcids
    )


def test_reconcile_keeps_every_request_within_the_row_budget():
    # V1..V3 expect 3 rows each; a budget of 5 allows one variable per request.
    rows = [{**_ROW_V1, "variable": f"V{i}"} for i in (1, 2, 3)]
    table = _coverage_table(*rows)
    role_names = {f"V{i}": _ROLE_NAMES["V1"] for i in (1, 2, 3)}
    sdmx = TableBackedSdmx(table, role_names)

    result = job.reconcile(
        sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET, row_budget=5
    )

    assert result.variables_checked == 3
    assert result.request_count == len(sdmx.calls) == 3
    assert all(_expected_rows(table, v, e) <= 5 for v, e in sdmx.calls)


def test_reconcile_splits_a_variable_over_the_budget_by_entity_filter():
    # V1 expects 9 rows: one donor, 3 rows on each of three recipients. The
    # donor axis is smaller, but only the recipient axis can split V1.
    recipients = ("country/KEN", "country/TZA", "country/UGA")
    table = _coverage_table(*({**_ROW_V1, "recipient": r} for r in recipients))
    sdmx = TableBackedSdmx(table, _ROLE_NAMES)

    result = job.reconcile(
        sdmx, table=table, role_names=_ROLE_NAMES, stamp=_STAMP, max_rows=_BUDGET, row_budget=6
    )

    assert result.request_count == len(sdmx.calls) == 2
    assert [e for _v, e in sdmx.calls] == [
        {"financingRecipient": ("country/KEN", "country/TZA")},
        {"financingRecipient": ("country/UGA",)},
    ]
    assert all(v == ("V1",) for v, _e in sdmx.calls)


def test_reconcile_packs_within_budget_variables_beside_a_split_one():
    # V1 (9 rows) splits by recipient; V2 (3 rows) packs on the donor axis.
    recipients = ("country/KEN", "country/TZA", "country/UGA")
    table = _coverage_table(
        *({**_ROW_V1, "recipient": r} for r in recipients), {**_ROW_V1, "variable": "V2"}
    )
    role_names = {"V1": _ROLE_NAMES["V1"], "V2": _ROLE_NAMES["V1"]}
    sdmx = TableBackedSdmx(table, role_names)

    result = job.reconcile(
        sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET, row_budget=6
    )

    assert result.variables_checked == 2
    assert sdmx.calls[0] == (("V2",), {"financingProvider": ("country/FRA",)})
    assert all(_expected_rows(table, v, e) <= 6 for v, e in sdmx.calls)


def test_reconcile_refuses_an_entity_whose_rows_alone_pass_the_budget():
    table = _coverage_table(_ROW_V1)  # 3 rows on one donor and one recipient
    sdmx = TableBackedSdmx(table, _ROLE_NAMES)
    with pytest.raises(job.CoverageReconciliationError, match="alone expects 3 rows, over the row"):
        job.reconcile(
            sdmx, table=table, role_names=_ROLE_NAMES, stamp=_STAMP, max_rows=_BUDGET, row_budget=2
        )
    assert sdmx.calls == []


def _query_bytes(variables, entities) -> int:
    """The encoded query length of a `/data` request, as the SDMX client sends it."""
    params = {"c[variableMeasured]": ",".join(variables)}
    params.update({f"c[{role}]": ",".join(dcids) for role, dcids in entities.items()})
    return len(str(httpx.QueryParams(params)))


_URL_BUDGET = 7_800


def _recipient_grid_table(variable: str, *, donors: int, recipients: int, n_obs: int) -> pa.Table:
    return _coverage_table(
        *(
            {
                **_ROW_V1,
                "variable": variable,
                "donor": f"country/D{d}",
                "recipient": f"country/R{r:04d}",
                "n_obs": n_obs,
            }
            for d in range(donors)
            for r in range(recipients)
        )
    )


def test_reconcile_splits_an_oversized_variable_within_the_url_and_row_budgets():
    # 300,000 rows over three donors and 1,000 recipients. The recipient axis
    # splits V1 (300 rows per recipient against 100,000 per donor). Split on
    # rows alone, its first request would list 833 recipients, past the URL
    # budget.
    table = _recipient_grid_table("V1", donors=3, recipients=1_000, n_obs=100)
    sdmx = TableBackedSdmx(table, _ROLE_NAMES)

    result = job.reconcile(
        sdmx, table=table, role_names=_ROLE_NAMES, stamp=_STAMP, max_rows=_BUDGET
    )

    assert all(_query_bytes(v, e) <= _URL_BUDGET for v, e in sdmx.calls)
    assert all(_expected_rows(table, v, e) <= _BUDGET for v, e in sdmx.calls)
    assert result.rows_expected == 300_000
    assert result.request_count == len(sdmx.calls) == 3
    runs = [e["financingRecipient"] for _v, e in sdmx.calls]
    assert [dcid for run in runs for dcid in run] == [f"country/R{r:04d}" for r in range(1_000)]


def test_reconcile_splits_a_within_budget_variable_whose_filter_passes_the_url_budget():
    # An IDS total on 1,000 recipients expects 1,000 rows, well within the
    # row budget, but its one role axis lists too many dcids for one URL.
    recipients = [f"country/R{r:04d}" for r in range(1_000)]
    table = _coverage_table(*({**_ROW_IDS_TOTAL, "recipient": r} for r in recipients))
    role_names = {_IDS_TOTAL: RoleNames(donor=None, recipient="financingRecipient")}
    assert _query_bytes([_IDS_TOTAL], {"financingRecipient": recipients}) > _URL_BUDGET
    sdmx = TableBackedSdmx(table, role_names)

    result = job.reconcile(sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET)

    assert all(_query_bytes(v, e) <= _URL_BUDGET for v, e in sdmx.calls)
    assert result.rows_expected == 3_000
    assert result.request_count == len(sdmx.calls) > 1
    runs = [e["financingRecipient"] for _v, e in sdmx.calls]
    assert [dcid for run in runs for dcid in run] == recipients


def test_reconcile_plans_within_the_clients_configured_url_budget():
    recipients = [f"country/R{r:04d}" for r in range(1_000)]
    table = _coverage_table(*({**_ROW_IDS_TOTAL, "recipient": r} for r in recipients))
    role_names = {_IDS_TOTAL: RoleNames(donor=None, recipient="financingRecipient")}
    default = TableBackedSdmx(table, role_names)
    small = TableBackedSdmx(table, role_names, url_budget_bytes=2_000)

    for sdmx in (default, small):
        job.reconcile(sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET)

    assert len(small.calls) > len(default.calls)
    assert max(_query_bytes(v, e) for v, e in default.calls) > 2_000
    assert all(_query_bytes(v, e) <= 2_000 for v, e in small.calls)


_ONE_ROLE = RoleNames(donor=None, recipient="financingRecipient")
# One dcid that fits beside a variable in 600 bytes but not in half that.
_LONG_RECIPIENT = "country/" + "L" * 300


def _long_recipient_reconcile(url_budget_bytes: int) -> TableBackedSdmx:
    rows = [
        {**_ROW_IDS_TOTAL, "variable": "V1", "recipient": "country/KEN"},
        {**_ROW_IDS_TOTAL, "variable": "V2", "recipient": _LONG_RECIPIENT},
    ]
    table = _coverage_table(*rows)
    role_names = dict.fromkeys(("V1", "V2"), _ONE_ROLE)
    sdmx = TableBackedSdmx(table, role_names, url_budget_bytes=url_budget_bytes)
    job.reconcile(sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET)
    return sdmx


def test_reconcile_takes_the_full_budget_plan_when_half_cannot_fit_a_dcid():
    sdmx = _long_recipient_reconcile(600)

    assert sdmx.calls == [(("V1", "V2"), {"financingRecipient": ("country/KEN", _LONG_RECIPIENT)})]
    assert _query_bytes(*sdmx.calls[0]) <= 600


def test_reconcile_refuses_a_shared_filter_neither_budget_can_carry():
    with pytest.raises(job.CoverageReconciliationError, match="does not split into reconciliation"):
        _long_recipient_reconcile(300)


def test_reconcile_counts_a_row_on_a_dcid_only_another_variable_covers_after_a_split():
    # V1 covers the first 500 recipients and V2 the rest. Their shared
    # filter lists all 1,000 and splits into runs; SDMX also holds a V1 row
    # on V2's last recipient, which only the shared filter can find.
    recipients = [f"country/R{r:04d}" for r in range(1_000)]
    rows = [
        {**_ROW_IDS_TOTAL, "variable": "V1" if r < 500 else "V2", "recipient": recipient}
        for r, recipient in enumerate(recipients)
    ]
    table = _coverage_table(*rows)
    role_names = dict.fromkeys(("V1", "V2"), RoleNames(donor=None, recipient="financingRecipient"))
    extra = {**_ROW_IDS_TOTAL, "variable": "V1", "recipient": recipients[-1], "n_obs": 1}
    sdmx = TableBackedSdmx(_coverage_table(*rows, extra), role_names)

    with pytest.raises(job.CoverageReconciliationError, match=r"\('V1', 1500, 1501\)"):
        job.reconcile(sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET)

    assert len(sdmx.calls) > 1
    assert all(_query_bytes(v, e) <= _URL_BUDGET for v, e in sdmx.calls)


def test_reconcile_counts_a_row_on_a_dcid_only_another_variable_covers_for_a_split_variable():
    # V1 expects 4 rows, over a budget of 3, so it splits on its own; V2
    # covers R3. SDMX also holds a V1 row on R3, which only a split over the
    # group's covered dcids can find.
    rows = [
        {**_ROW_IDS_TOTAL, "variable": "V1", "recipient": "country/R1", "n_obs": 2},
        {**_ROW_IDS_TOTAL, "variable": "V1", "recipient": "country/R2", "n_obs": 2},
        {**_ROW_IDS_TOTAL, "variable": "V2", "recipient": "country/R3", "n_obs": 1},
    ]
    table = _coverage_table(*rows)
    role_names = dict.fromkeys(("V1", "V2"), _ONE_ROLE)
    extra = {**_ROW_IDS_TOTAL, "variable": "V1", "recipient": "country/R3", "n_obs": 1}
    sdmx = TableBackedSdmx(_coverage_table(*rows, extra), role_names)

    with pytest.raises(job.CoverageReconciliationError, match=r"\('V1', 4, 5\)"):
        job.reconcile(
            sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET, row_budget=3
        )

    v1_dcids = [dcid for v, e in sdmx.calls if v == ("V1",) for dcid in e["financingRecipient"]]
    assert v1_dcids == ["country/R1", "country/R2", "country/R3"]
    assert all(_expected_rows(table, v, e) <= 3 for v, e in sdmx.calls)


def test_reconcile_fills_a_split_request_up_to_the_url_budget_exactly():
    # One row per recipient keeps the row budget slack, so the URL budget
    # alone closes each run. The variable name pads a 400-recipient run's
    # query to exactly the budget; a 401st recipient would pass it.
    recipients = [f"country/R{r:04d}" for r in range(1_000)]
    base = _query_bytes(["V"], {"financingRecipient": recipients[:400]})
    variable = "V" + "x" * (_URL_BUDGET - base)
    table = _recipient_grid_table(variable, donors=1, recipients=1_000, n_obs=1)
    role_names = {variable: _ROLE_NAMES["V1"]}
    sdmx = TableBackedSdmx(table, role_names)

    job.reconcile(
        sdmx, table=table, role_names=role_names, stamp=_STAMP, max_rows=_BUDGET, row_budget=999
    )

    first_variables, first_entities = sdmx.calls[0]
    assert first_entities == {"financingRecipient": tuple(recipients[:400])}
    assert _query_bytes(first_variables, first_entities) == _URL_BUDGET
    assert all(_query_bytes(v, e) <= _URL_BUDGET for v, e in sdmx.calls)


# --- publish ---------------------------------------------------------------


def _root(base: Path) -> Path:
    """`base`'s format root for this code's `COVERAGE_FORMAT_VERSION`."""
    return base / f"format-{COVERAGE_FORMAT_VERSION}"


def _build(base: Path, build_id: str) -> Path:
    return _root(base) / "versions" / build_id


def _publish(base_uri: str, *, build_id: str, expected_generation: str | None = None, **kw):
    return job.publish(
        base_uri=base_uri,
        stamp=kw.pop("stamp", "stamp-1"),
        build_id=build_id,
        table=kw.pop("table", _coverage_table(_ROW_V1)),
        horizon=_HORIZON,
        role_names=_ROLE_NAMES,
        facet_metadata=_FACET_METADATA,
        source="test",
        ingestion=Ingestion(workflow_execution_id="wf-1", creation_timestamp="t1"),
        expected_generation=expected_generation,
    )


def _generation(uri: str) -> str | None:
    found = job.read_uri_bytes_and_generation(uri)
    return None if found is None else found[1]


def test_publish_writes_a_new_build_create_only_then_current_json_last(tmp_path, monkeypatch):
    base_uri = f"file://{tmp_path}"
    writes: list[tuple[str, str | None]] = []
    real_write_if_unchanged = job.write_uri_bytes_if_unchanged

    def _spy(uri: str, data: bytes, *, expected_generation):
        writes.append((uri, expected_generation))
        return real_write_if_unchanged(uri, data, expected_generation=expected_generation)

    monkeypatch.setattr(job, "write_uri_bytes_if_unchanged", _spy)

    assert _publish(base_uri, build_id="b1") is True

    build_prefix = f"{base_uri}/format-{COVERAGE_FORMAT_VERSION}/versions/b1/"
    assert [uri.removeprefix(build_prefix) for uri, _ in writes[:-1]] == [
        "coverage.parquet",
        "role_names.json",
        "facets.json",
        "manifest.json",
    ]
    assert all(generation is None for _, generation in writes[:-1])
    assert writes[-1][0] == f"{base_uri}/format-{COVERAGE_FORMAT_VERSION}/current.json"

    loaded = load_coverage(base_uri)
    assert loaded.index.num_rows == 1
    assert loaded.index.role_names == _ROLE_NAMES
    assert loaded.index.facet_metadata == _FACET_METADATA
    manifest = read_manifest(_build(tmp_path, "b1") / "manifest.json")
    assert (manifest.stamp, manifest.build_id) == ("stamp-1", "b1")
    assert manifest.files["facets"] == "facets.json"
    assert manifest.row_counts["rows"] == 1
    assert read_current(_root(tmp_path) / "current.json") == CoveragePointer("stamp-1", "b1")
    assert sorted(p.name for p in tmp_path.iterdir()) == [_root(tmp_path).name]


def test_two_builds_of_one_stamp_land_in_separate_directories(tmp_path):
    """A scheduled run and a deploy run of one fingerprint each publish into
    their own build directory; the second leaves every byte of the first
    as it was, and the pointer moves to the second."""
    base_uri = f"file://{tmp_path}"
    assert _publish(base_uri, build_id="first") is True
    first_files = {p.name: p.read_bytes() for p in _build(tmp_path, "first").iterdir()}

    generation = _generation(f"{base_uri}/format-{COVERAGE_FORMAT_VERSION}/current.json")
    assert _publish(base_uri, build_id="second", expected_generation=generation) is True

    assert {p.name: p.read_bytes() for p in _build(tmp_path, "first").iterdir()} == first_files
    assert sorted(p.name for p in (_root(tmp_path) / "versions").iterdir()) == ["first", "second"]
    assert load_coverage(base_uri).manifest.build_id == "second"


def test_publish_refuses_to_write_into_an_existing_build_directory(tmp_path):
    base_uri = f"file://{tmp_path}"
    assert _publish(base_uri, build_id="b1") is True
    before = {p.name: p.read_bytes() for p in _build(tmp_path, "b1").iterdir()}
    pointer_before = (_root(tmp_path) / "current.json").read_bytes()
    generation = _generation(f"{base_uri}/format-{COVERAGE_FORMAT_VERSION}/current.json")

    with pytest.raises(job.CoverageBuildError, match="written once"):
        _publish(base_uri, build_id="b1", stamp="stamp-2", expected_generation=generation)

    assert {p.name: p.read_bytes() for p in _build(tmp_path, "b1").iterdir()} == before
    assert (_root(tmp_path) / "current.json").read_bytes() == pointer_before


def test_a_reader_holding_the_first_pointer_loads_that_build_while_a_second_publishes(
    tmp_path, monkeypatch
):
    """A reader resolves current.json to build `first`; a whole second
    publish of the same stamp, commit included, lands before the reader
    reads `first`'s files. The reader still loads `first`, intact."""
    base_uri = f"file://{tmp_path}"
    assert _publish(base_uri, build_id="first") is True
    pointer_uri = f"{base_uri}/format-{COVERAGE_FORMAT_VERSION}/current.json"
    real_read = loader.read_uri_bytes
    interleaved: list[bool] = []

    def _read_then_publish_second(uri: str) -> bytes:
        data = real_read(uri)
        if uri == pointer_uri and not interleaved:
            interleaved.append(
                _publish(base_uri, build_id="second", expected_generation=_generation(uri))
            )
        return data

    monkeypatch.setattr(loader, "read_uri_bytes", _read_then_publish_second)

    loaded = load_coverage(base_uri)

    assert interleaved == [True]
    assert loaded.manifest.build_id == "first"
    assert loaded.index.num_rows == 1
    monkeypatch.setattr(loader, "read_uri_bytes", real_read)
    assert load_coverage(base_uri).manifest.build_id == "second"


def test_publish_loses_the_race_when_current_json_moved_since_expected_generation(tmp_path):
    # A concurrent publish moves current.json after this run reads it. The conditional
    # write must lose without changing the winner's pointer.
    pointer = _root(tmp_path) / "current.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(CoveragePointer("concurrent-winner", "winner").to_json())

    # This run observed no current.json at the start of its own run.
    published = _publish(f"file://{tmp_path}", build_id="b1", expected_generation=None)

    assert published is False
    assert read_current(pointer) == CoveragePointer("concurrent-winner", "winner")
    # The build this run wrote stays, unreferenced, for a later prune.
    assert (_build(tmp_path, "b1") / "coverage.parquet").exists()


def test_publish_succeeds_when_current_json_generation_still_matches(tmp_path):
    base_uri = f"file://{tmp_path}"
    pointer = _root(tmp_path) / "current.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(CoveragePointer("previous", "b0").to_json())
    expected_generation = _generation(f"{base_uri}/format-{COVERAGE_FORMAT_VERSION}/current.json")

    published = _publish(base_uri, build_id="b1", expected_generation=expected_generation)

    assert published is True
    assert read_current(pointer) == CoveragePointer("stamp-1", "b1")


def test_new_build_id_is_a_unique_timestamped_path_segment():
    ids = {job.new_build_id() for _ in range(50)}
    assert len(ids) == 50
    for build_id in ids:
        CoveragePointer.from_json(CoveragePointer("s", build_id).to_json())
        stamp, suffix = build_id.split("-")
        assert len(stamp) == len("20260927T142252Z") and stamp.endswith("Z")
        assert len(suffix) == 8


# --- stale-publish state ------------------------------------------------


def test_update_stale_state_no_op_when_caught_up(tmp_path, capsys):
    settings = job.JobSettings(
        artifact_uri=f"file://{tmp_path}",
        bq_connection=_CONNECTION,
        bq_project="proj",
        sdmx_base_url="https://mixer/core/api",
        sdmx_token=None,
        sdmx_timeout_seconds=90.0,
        cycle_seconds=900.0,
    )
    fp = job.Fingerprint("wf", "t", 1, "t2")
    job._update_stale_state(settings, fingerprint=fp, published_stamp=fp.stamp, clock=lambda: 0.0)
    assert capsys.readouterr().out == ""
    assert not _root(tmp_path).exists()


def test_update_stale_state_emits_only_past_the_cycle_threshold(tmp_path, capsys):
    settings = job.JobSettings(
        artifact_uri=f"file://{tmp_path}",
        bq_connection=_CONNECTION,
        bq_project="proj",
        sdmx_base_url="https://mixer/core/api",
        sdmx_token=None,
        sdmx_timeout_seconds=90.0,
        cycle_seconds=100.0,
    )
    fp = job.Fingerprint("wf", "t", 1, "t2")

    # First unpublished observation starts the clock without an alert.
    job._update_stale_state(settings, fingerprint=fp, published_stamp=None, clock=lambda: 0.0)
    assert capsys.readouterr().out == ""
    # Each format root keeps its own clock.
    assert (_root(tmp_path) / "coverage_state.json").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == [_root(tmp_path).name]

    # Still within one cycle: no alert.
    job._update_stale_state(settings, fingerprint=fp, published_stamp=None, clock=lambda: 50.0)
    assert capsys.readouterr().out == ""

    # Past the cycle threshold: alert.
    job._update_stale_state(settings, fingerprint=fp, published_stamp=None, clock=lambda: 150.0)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["event"] == "coverage_publish_stale"
    assert payload["fingerprint_stamp"] == fp.stamp


# --- run() orchestration --------------------------------------------------


def _settings(tmp_path: Path) -> job.JobSettings:
    return job.JobSettings(
        artifact_uri=f"file://{tmp_path}",
        bq_connection=_CONNECTION,
        bq_project="proj",
        sdmx_base_url="https://mixer/core/api",
        sdmx_token=None,
        sdmx_timeout_seconds=90.0,
        cycle_seconds=900.0,
    )


def test_run_skips_when_lock_held(tmp_path, capsys):
    bq = FakeBigQuery({"IngestionLock": [{"LockOwner": "someone"}]})
    sdmx = FakeSdmx()
    job.run(_settings(tmp_path), index=_INDEX, bq=bq, sdmx=sdmx)
    assert json.loads(capsys.readouterr().out)["event"] == "coverage_job_skipped"
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def _fingerprint_responses(row_count: int = 3) -> dict[str, list[dict[str, Any]]]:
    return {
        "IngestionLock": [{"LockOwner": None}],
        "IngestionHistory": [{"WorkflowExecutionID": "wf-1", "CreationTimestamp": "t1"}],
        "COUNT(*) AS row_count": [{"row_count": row_count, "max_ts": "t2"}],
    }


def _build_responses() -> dict[str, list[dict[str, Any]]]:
    return {
        **_fingerprint_responses(row_count=3),
        "AS max_year": [year_stats(max_year=2022)],
        "GROUP BY variable, facet_id, donor, recipient": [dict(_ROW_V1)],
        # Every query joins `role_bearing`, which reads observationProperties;
        # only the role-name read selects `object_id AS role`.
        "object_id AS role": [
            {"variable": "V1", "role": "financingProvider"},
            {"variable": "V1", "role": "financingRecipient"},
        ],
        "GROUP BY facet_id, provenance, unit, measurement_method, observation_period": [_FACET_ROW],
    }


def _build_sdmx() -> FakeSdmx:
    return FakeSdmx(
        data_responder=lambda *a, **k: SdmxDataResult(
            roles=("financingProvider", "financingRecipient"), rows=(_obs("V1", {}),)
        ),
        batches_by_group={("financingProvider", "financingRecipient"): [("V1", 3)]},
    )


def _set_age(path: Path, seconds: float) -> None:
    """Backdates `path`, or every file under it, by `seconds`."""
    when = time.time() - seconds
    for found in [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]:
        os.utime(found, (when, when))


def _write_build(root: Path, build_id: str, *, age: float = 0.0, stamp: str = "s") -> None:
    """A minimal build directory under format root `root`: a manifest and
    one data file, backdated by `age` seconds. Nothing in the no-op or
    prune path reads the data files."""
    build = root / "versions" / build_id
    build.mkdir(parents=True, exist_ok=True)
    (build / "manifest.json").write_text(json.dumps({"stamp": stamp, "build_id": build_id}))
    (build / "coverage.parquet").write_text("x")
    _set_age(build, age)


def _write_pointer(root: Path, *, build_id: str, stamp: str = "s", age: float = 0.0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "current.json").write_text(CoveragePointer(stamp, build_id).to_json())
    _set_age(root / "current.json", age)


def test_run_noops_when_published_stamp_matches_fingerprint(tmp_path, capsys):
    responses = _fingerprint_responses()
    fp = job.read_fingerprint(FakeBigQuery(responses), connection=_CONNECTION)
    _write_build(_root(tmp_path), "b1", stamp=fp.stamp)
    _write_pointer(_root(tmp_path), build_id="b1", stamp=fp.stamp)

    job.run(_settings(tmp_path), index=_INDEX, bq=FakeBigQuery(responses), sdmx=FakeSdmx())

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "coverage_job_noop"


def test_run_ignores_the_next_format_root(tmp_path, capsys):
    """A matching artifact under another format root does not satisfy this version's publish.
    The run leaves that root unchanged."""
    fp = job.read_fingerprint(FakeBigQuery(_build_responses()), connection=_CONNECTION)
    other_root = tmp_path / f"format-{COVERAGE_FORMAT_VERSION + 1}"
    _write_build(other_root, "theirs", stamp=fp.stamp, age=job._MAX_RUN_SECONDS * 10)
    _write_pointer(other_root, build_id="theirs", stamp=fp.stamp, age=job._MAX_RUN_SECONDS * 10)
    _write_build(other_root, "their-old", age=job._MAX_RUN_SECONDS * 10)
    before = {p: p.read_bytes() for p in other_root.rglob("*") if p.is_file()}

    job.run(
        _settings(tmp_path), index=_INDEX, bq=FakeBigQuery(_build_responses()), sdmx=_build_sdmx()
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events[-2:]] == [
        "coverage_job_published",
        "coverage_index_compatible",
    ]
    assert load_coverage(f"file://{tmp_path}").manifest.stamp == fp.stamp
    assert {p: p.read_bytes() for p in other_root.rglob("*") if p.is_file()} == before


def test_run_discards_when_fingerprint_moves_during_build(tmp_path, capsys):
    call_count = {"n": 0}

    class _MovingBigQuery(FakeBigQuery):
        def query(self, sql):
            if "COUNT(*) AS row_count" in sql:
                call_count["n"] += 1
                # First fingerprint read (pre-build) reports row_count=3;
                # the second (post-build, "did anything change?") reports a
                # different count, simulating an ingestion landing mid-run.
                return [{"row_count": 3 if call_count["n"] == 1 else 4, "max_ts": "t2"}]
            return super().query(sql)

    job.run(
        _settings(tmp_path),
        index=_INDEX,
        bq=_MovingBigQuery(_build_responses()),
        sdmx=_build_sdmx(),
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "coverage_job_discarded"
    assert events[-1]["reason"] == "fingerprint_moved_during_build"
    assert not (_root(tmp_path) / "current.json").exists()


def test_run_publishes_end_to_end(tmp_path, capsys):
    job.run(
        _settings(tmp_path), index=_INDEX, bq=FakeBigQuery(_build_responses()), sdmx=_build_sdmx()
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events[-2:]] == [
        "coverage_job_published",
        "coverage_index_compatible",
    ]
    assert events[-2]["rows"] == 1
    pointer = read_current(_root(tmp_path) / "current.json")
    assert pointer.build_id == events[-2]["build_id"] == events[-1]["build_id"]
    manifest = load_coverage(f"file://{tmp_path}").manifest
    assert manifest.build_id == pointer.build_id
    # The raw IngestionHistory row, which the origin catalogue's manifest records too.
    assert manifest.ingestion == Ingestion(workflow_execution_id="wf-1", creation_timestamp="t1")


def test_run_publishes_a_one_role_variable_beside_a_two_role_one(tmp_path, capsys):
    responses = {
        **_build_responses(),
        "GROUP BY variable, facet_id, donor, recipient": [dict(_ROW_V1), dict(_ROW_IDS_TOTAL)],
        "object_id AS role": [
            {"variable": "V1", "role": "financingProvider"},
            {"variable": "V1", "role": "financingRecipient"},
            {"variable": _IDS_TOTAL, "role": "financingRecipient"},
        ],
    }
    sdmx = FakeSdmx(
        data_responder=lambda *a, **k: SdmxDataResult(
            roles=("financingProvider", "financingRecipient"), rows=(_obs("V1", {}),)
        ),
        batches_by_group={
            ("financingProvider", "financingRecipient"): [("V1", 3)],
            ("financingRecipient",): [(_IDS_TOTAL, 3)],
        },
    )

    job.run(_settings(tmp_path), index=_INDEX, bq=FakeBigQuery(responses), sdmx=sdmx)

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    published = next(event for event in events if event["event"] == "coverage_job_published")
    assert published["variables"] == 2
    index = load_coverage(f"file://{tmp_path}").index
    assert index.role_names[_IDS_TOTAL] == RoleNames(donor=None, recipient="financingRecipient")
    [coverage] = index.filter_candidates(
        variables=[_IDS_TOTAL], bound={"recipient": frozenset({"country/KEN"})}
    )
    assert coverage.n_obs == 3


def test_run_republishes_an_unparseable_pointer(tmp_path, capsys):
    (_root(tmp_path)).mkdir(parents=True)
    (_root(tmp_path) / "current.json").write_text('{"stamp": "no build id"}')

    job.run(
        _settings(tmp_path), index=_INDEX, bq=FakeBigQuery(_build_responses()), sdmx=_build_sdmx()
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events[-2:]] == [
        "coverage_job_published",
        "coverage_index_compatible",
    ]
    assert read_current(_root(tmp_path) / "current.json").build_id == events[-2]["build_id"]


def test_run_exits_cleanly_when_it_loses_the_publish_race(tmp_path, capsys, monkeypatch):
    # A second, concurrent invocation publishes its own current.json in the
    # window between this run reading current.json's generation (at the
    # start) and this run's own publish() call (after building) -- this
    # run's conditional write then loses and must exit cleanly, a log line
    # rather than an exception, without touching the winner's current.json.
    real_publish = job.publish
    winner = CoveragePointer("concurrent-winner", "winner")

    def _publish_after_a_concurrent_run_wins(**kwargs):
        _write_pointer(_root(tmp_path), build_id=winner.build_id, stamp=winner.stamp)
        return real_publish(**kwargs)

    monkeypatch.setattr(job, "publish", _publish_after_a_concurrent_run_wins)

    job.run(
        _settings(tmp_path), index=_INDEX, bq=FakeBigQuery(_build_responses()), sdmx=_build_sdmx()
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "coverage_job_lost_publish_race"
    assert read_current(_root(tmp_path) / "current.json") == winner
    assert (_build(tmp_path, events[-1]["build_id"]) / "manifest.json").is_file()


def test_read_published_reraises_a_read_failure_other_than_absence(monkeypatch):
    """`read_uri_bytes_and_generation` returns `None` for "no current.json
    yet" and raises for a failed read; `_read_published` must not flatten
    that raise into the `(None, None)` a first build reads."""

    def _boom(uri: str) -> tuple[bytes, str] | None:
        raise RuntimeError("gcs quota exceeded")

    monkeypatch.setattr(job, "read_uri_bytes_and_generation", _boom)

    with pytest.raises(job.CoveragePointerReadError, match="gcs quota exceeded"):
        job._read_published(f"gs://bucket/coverage/format-{COVERAGE_FORMAT_VERSION}")


def test_main_reports_a_pointer_read_failure_as_coverage_job_failed(tmp_path, capsys, monkeypatch):
    """A `current.json` read that fails for a reason other than "doesn't
    exist yet" must stop the run and reach `main`'s own
    `coverage_job_failed` event, never a bare traceback."""
    bq = FakeBigQuery(_fingerprint_responses())
    sdmx = FakeSdmx()

    monkeypatch.setattr(job.JobSettings, "from_env", classmethod(lambda cls: _settings(tmp_path)))
    monkeypatch.setattr(job, "LiveBigQuery", lambda **_: bq)
    monkeypatch.setattr(job, "SdmxClient", lambda **_: sdmx)

    def _boom(uri: str) -> tuple[bytes, str] | None:
        raise RuntimeError("gcs quota exceeded")

    monkeypatch.setattr(job, "read_uri_bytes_and_generation", _boom)

    exit_code = job.main()

    assert exit_code == 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "coverage_job_failed"
    assert "gcs quota exceeded" in events[-1]["error"]


# --- prune_old_versions ----------------------------------------------------

_OLD = job._MAX_RUN_SECONDS + 600


def _builds(root: Path) -> list[str]:
    versions = root / "versions"
    return sorted(p.name for p in versions.iterdir()) if versions.is_dir() else []


def test_prune_deletes_every_old_build_the_pointer_does_not_name(tmp_path):
    root = _root(tmp_path)
    for build_id in ("b1", "b2", "b3"):
        _write_build(root, build_id, age=_OLD)
    _write_pointer(root, build_id="b3", age=_OLD)

    deleted = job.prune_old_versions(f"file://{tmp_path}")[0]

    assert deleted == ("b1", "b2")
    assert _builds(root) == ["b3"]


def test_prune_never_deletes_the_build_the_pointer_names(tmp_path):
    root = _root(tmp_path)
    _write_build(root, "live", age=_OLD * 100)
    _write_build(root, "orphan", age=_OLD)
    _write_pointer(root, build_id="live", age=_OLD)

    assert job.prune_old_versions(f"file://{tmp_path}")[0] == ("orphan",)
    assert _builds(root) == ["live"]


def test_prune_keeps_a_build_whose_newest_file_is_inside_the_grace_period(tmp_path):
    # Another invocation's publish() in progress: its first file is old
    # enough, but it wrote a file moments ago.
    root = _root(tmp_path)
    _write_build(root, "live", age=_OLD)
    _write_build(root, "in-progress", age=_OLD)
    (root / "versions" / "in-progress" / "role_names.json").write_text("{}")
    _write_pointer(root, build_id="live", age=_OLD)

    assert job.prune_old_versions(f"file://{tmp_path}")[0] == ()
    assert _builds(root) == ["in-progress", "live"]


def test_prune_keeps_the_previous_build_while_the_pointer_is_young(tmp_path):
    """A reader may still load the previous build after the pointer moves. The grace period
    keeps it and same-stamp orphans, then pruning removes both and keeps current."""
    root = _root(tmp_path)
    _write_build(root, "previous", age=_OLD * 50, stamp="s")
    _write_build(root, "orphan", age=_OLD * 2, stamp="s")
    _write_build(root, "current", stamp="s")
    _write_pointer(root, build_id="current", stamp="s")

    assert job.prune_old_versions(f"file://{tmp_path}")[0] == ()
    assert _builds(root) == ["current", "orphan", "previous"]

    later = time.time() + _OLD
    assert job.prune_old_versions(f"file://{tmp_path}", now=lambda: later)[0] == (
        "orphan",
        "previous",
    )
    assert _builds(root) == ["current"]


def test_prune_right_after_a_publish_keeps_the_previous_build(tmp_path):
    """Pruning after publication preserves the previous build for readers holding its pointer."""
    base_uri = f"file://{tmp_path}"
    assert _publish(base_uri, build_id="previous") is True
    _set_age(_root(tmp_path), _OLD * 10)
    generation = _generation(f"{base_uri}/format-{COVERAGE_FORMAT_VERSION}/current.json")
    assert _publish(base_uri, build_id="current", expected_generation=generation) is True

    assert job.prune_old_versions(base_uri)[0] == ()

    assert _builds(_root(tmp_path)) == ["current", "previous"]
    root = _root(tmp_path)
    (root / "current.json").write_text(CoveragePointer("stamp-1", "previous").to_json())
    assert load_coverage(base_uri).manifest.build_id == "previous"


def test_prune_is_a_no_op_with_nothing_published(tmp_path):
    assert job.prune_old_versions(f"file://{tmp_path}")[0] == ()


def test_prune_deletes_nothing_when_the_pointer_is_unreadable(tmp_path):
    """Pruning deletes nothing when `current.json` cannot identify the live build."""
    root = _root(tmp_path)
    for build_id in ("b1", "b2"):
        _write_build(root, build_id, age=_OLD)

    assert job.prune_old_versions(f"file://{tmp_path}")[0] == ()

    (root / "current.json").write_text("not json")
    _set_age(root / "current.json", _OLD)
    assert job.prune_old_versions(f"file://{tmp_path}")[0] == ()
    assert _builds(root) == ["b1", "b2"]


def test_prune_never_touches_another_format_root(tmp_path):
    root = _root(tmp_path)
    _write_build(root, "live", age=_OLD)
    _write_build(root, "old", age=_OLD)
    _write_pointer(root, build_id="live", age=_OLD)
    for other in (tmp_path / "format-0", tmp_path / f"format-{COVERAGE_FORMAT_VERSION + 1}"):
        _write_build(other, "theirs", age=_OLD)
        _write_build(other, "their-old", age=_OLD)
        _write_pointer(other, build_id="theirs", age=_OLD)
    (tmp_path / "unrelated.txt").write_text("x")
    _set_age(tmp_path / "unrelated.txt", _OLD)
    before = {
        p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and root not in p.parents
    }

    assert job.prune_old_versions(f"file://{tmp_path}")[0] == ("old",)

    after = {
        p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and root not in p.parents
    }
    assert after == before


def test_prune_between_a_concurrent_publish_s_files_leaves_it_loadable(tmp_path, monkeypatch):
    """Pruning between a concurrent build's first file and pointer commit leaves it loadable."""
    base_uri = f"file://{tmp_path}"
    root = _root(tmp_path)
    _write_build(root, "ancient", age=_OLD)
    _write_build(root, "live", age=_OLD)
    _write_pointer(root, build_id="live", age=_OLD)
    real_write_if_unchanged = job.write_uri_bytes_if_unchanged
    pruned: list[tuple[str, ...]] = []

    def _write_then_let_job_b_prune(uri: str, data: bytes, *, expected_generation):
        written = real_write_if_unchanged(uri, data, expected_generation=expected_generation)
        if uri.endswith("/job-a/coverage.parquet"):
            pruned.append(job.prune_old_versions(base_uri)[0])
        return written

    monkeypatch.setattr(job, "write_uri_bytes_if_unchanged", _write_then_let_job_b_prune)
    generation = _generation(f"{base_uri}/format-{COVERAGE_FORMAT_VERSION}/current.json")

    assert _publish(base_uri, build_id="job-a", expected_generation=generation) is True

    assert pruned == [("ancient",)]
    assert load_coverage(base_uri).manifest.build_id == "job-a"


def test_run_prunes_old_builds_on_a_no_op_run(tmp_path, capsys):
    """No-op runs also prune old builds."""
    responses = _fingerprint_responses()
    fp = job.read_fingerprint(FakeBigQuery(responses), connection=_CONNECTION)
    root = _root(tmp_path)
    _write_build(root, "live", stamp=fp.stamp, age=_OLD)
    _write_pointer(root, build_id="live", stamp=fp.stamp, age=_OLD)
    _write_build(root, "ancient", age=_OLD * 10)
    _write_build(root, "previous", age=_OLD)

    job.run(_settings(tmp_path), index=_INDEX, bq=FakeBigQuery(responses), sdmx=FakeSdmx())

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "coverage_job_pruned"
    assert events[-1]["builds"] == ["ancient", "previous"]
    assert _builds(root) == ["live"]


# --- compatibility pointers ------------------------------------------------


def _write_compatible_pointer(root: Path, index_hash: str, *, build_id: str, age: float) -> None:
    pointer = root / "compatible" / f"{index_hash}.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(CoveragePointer("s", build_id).to_json())
    _set_age(pointer, age)


def _compatible_pointers(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "compatible").iterdir())


def test_a_compatible_publish_moves_the_index_pointer(tmp_path, capsys):
    job.run(
        _settings(tmp_path), index=_INDEX, bq=FakeBigQuery(_build_responses()), sdmx=_build_sdmx()
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    published = read_current(_root(tmp_path) / "current.json")
    recorded = read_current(_root(tmp_path) / "compatible" / "test-index.json")
    assert recorded == published
    assert events[-1] == {
        "event": "coverage_index_compatible",
        "severity": "INFO",
        "build_id": published.build_id,
        "index_content_hash": "test-index",
    }


def test_an_incompatible_publish_leaves_the_index_pointer_and_reports_an_error(tmp_path, capsys):
    index = code_index_for({"V1": ("provenance/Elsewhere",)}, content_hash="moved-index")
    _write_compatible_pointer(_root(tmp_path), "moved-index", build_id="earlier", age=0)

    job.run(
        _settings(tmp_path), index=index, bq=FakeBigQuery(_build_responses()), sdmx=_build_sdmx()
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    incompatible = events[-1]
    assert (incompatible["event"], incompatible["severity"]) == (
        "coverage_index_incompatible",
        "ERROR",
    )
    assert incompatible["variables"] == ["V1"]
    assert incompatible["index_content_hash"] == "moved-index"
    recorded = read_current(_root(tmp_path) / "compatible" / "moved-index.json")
    assert recorded.build_id == "earlier"


def test_prune_keeps_every_build_a_compatibility_pointer_names(tmp_path):
    root = _root(tmp_path)
    for build_id in ("recorded", "orphan", "live"):
        _write_build(root, build_id, age=_OLD)
    _write_pointer(root, build_id="live", age=_OLD)
    _write_compatible_pointer(root, "test-index", build_id="recorded", age=_OLD)

    assert job.prune_old_versions(f"file://{tmp_path}") == (("orphan",), ())
    assert _builds(root) == ["live", "recorded"]


def test_prune_deletes_nothing_while_a_compatibility_pointer_is_young(tmp_path):
    """A reader may have resolved the build a compatibility pointer named before it moved."""
    root = _root(tmp_path)
    for build_id in ("earlier", "live"):
        _write_build(root, build_id, age=_OLD)
    _write_pointer(root, build_id="live", age=_OLD)
    _write_compatible_pointer(root, "test-index", build_id="live", age=0)

    assert job.prune_old_versions(f"file://{tmp_path}") == ((), ())
    assert _builds(root) == ["earlier", "live"]


def _record_serving(root: Path, index_hash: str, *, age: float) -> None:
    record_serving_index(f"file://{root.parent}", code_index_for({}, content_hash=index_hash))
    _set_age(root / "serving" / "code_index.json", age)


def test_prune_expires_a_retired_indexs_pointer_and_then_its_build(tmp_path):
    root = _root(tmp_path)
    expired_age = job._COMPATIBLE_POINTER_RETENTION_SECONDS + 60
    for build_id in ("retired", "kept", "live"):
        _write_build(root, build_id, age=expired_age)
    _write_pointer(root, build_id="live", age=_OLD)
    _write_compatible_pointer(root, "retired-index", build_id="retired", age=expired_age)
    _write_compatible_pointer(root, "recent-index", build_id="kept", age=expired_age - 600)
    _write_compatible_pointer(root, "test-index", build_id="live", age=expired_age * 10)
    _record_serving(root, "test-index", age=_OLD)

    assert job.prune_old_versions(f"file://{tmp_path}") == ((), ("retired-index",))
    assert _compatible_pointers(root) == ["recent-index", "test-index"]
    assert _builds(root) == ["kept", "live", "retired"]

    assert job.prune_old_versions(f"file://{tmp_path}") == (("retired",), ())
    assert _builds(root) == ["kept", "live"]


def test_prune_keeps_the_serving_indexs_pointer_while_the_job_runs_another_index(tmp_path):
    """A deploy moved the job to a new image and stopped before the services moved. The job
    refreshes only the new index's pointer, while every service still runs the old index and
    needs the build its pointer names."""
    root = _root(tmp_path)
    expired_age = job._COMPATIBLE_POINTER_RETENTION_SECONDS + 60
    for build_id in ("served", "live"):
        _write_build(root, build_id, age=expired_age)
    _write_pointer(root, build_id="live", age=_OLD)
    _write_compatible_pointer(root, "serving-index", build_id="served", age=expired_age)
    _write_compatible_pointer(root, "job-index", build_id="live", age=_OLD)
    _record_serving(root, "serving-index", age=expired_age * 2)

    assert job.prune_old_versions(f"file://{tmp_path}") == ((), ())
    assert _compatible_pointers(root) == ["job-index", "serving-index"]
    assert _builds(root) == ["live", "served"]


def test_prune_keeps_a_pointer_written_after_the_serving_record(tmp_path):
    """A later deploy moved the worker to its index and failed on the apis service. Its
    `verify_index --record` wrote that index's pointer after the last finished deploy's
    record, so the worker's revision still runs it."""
    root = _root(tmp_path)
    expired_age = job._COMPATIBLE_POINTER_RETENTION_SECONDS + 60
    for build_id in ("worker", "live"):
        _write_build(root, build_id, age=expired_age)
    _write_pointer(root, build_id="live", age=_OLD)
    _write_compatible_pointer(root, "worker-index", build_id="worker", age=expired_age)
    _write_compatible_pointer(root, "test-index", build_id="live", age=_OLD)
    _record_serving(root, "test-index", age=expired_age + 600)

    assert job.prune_old_versions(f"file://{tmp_path}") == ((), ())
    assert _compatible_pointers(root) == ["test-index", "worker-index"]


def test_prune_expires_no_pointer_before_a_deploy_records_the_serving_index(tmp_path):
    root = _root(tmp_path)
    expired_age = job._COMPATIBLE_POINTER_RETENTION_SECONDS + 60
    for build_id in ("older", "live"):
        _write_build(root, build_id, age=expired_age)
    _write_pointer(root, build_id="live", age=_OLD)
    _write_compatible_pointer(root, "older-index", build_id="older", age=expired_age * 10)

    assert job.prune_old_versions(f"file://{tmp_path}") == ((), ())
    assert _compatible_pointers(root) == ["older-index"]


def test_prune_refuses_an_unreadable_serving_record(tmp_path):
    root = _root(tmp_path)
    _write_pointer(root, build_id="live", age=_OLD)
    (root / "serving").mkdir()
    (root / "serving" / "code_index.json").write_text("not json")

    with pytest.raises(CoverageArtifactError, match="serving index"):
        job.prune_old_versions(f"file://{tmp_path}")


def test_prune_deletes_nothing_when_a_compatibility_pointer_is_unreadable(tmp_path):
    root = _root(tmp_path)
    for build_id in ("orphan", "live"):
        _write_build(root, build_id, age=_OLD)
    _write_pointer(root, build_id="live", age=_OLD)
    (root / "compatible").mkdir()
    (root / "compatible" / "test-index.json").write_text("not json")
    _set_age(root / "compatible" / "test-index.json", _OLD)

    assert job.prune_old_versions(f"file://{tmp_path}")[0] == ()
    assert _builds(root) == ["live", "orphan"]
