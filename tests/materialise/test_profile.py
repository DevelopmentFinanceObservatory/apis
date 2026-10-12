"""Tests output-profile generation, JSON round trips, and pivot feasibility.

Group fixtures cover two-role and subject-only data; direct tables cover edge cases and pivot
analysis against `analyse_pivot`.
"""

from __future__ import annotations

import json
import random

import httpx

from dc_sdmx import SdmxClient
from materialise import (
    ConcatenatedFrameChunks,
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_group,
)
from materialise.facets import FacetBinding
from materialise.graph import Facet
from materialise.profile import ColumnProfile, OutputProfile, compute_output_profile
from materialise.reshape import PivotAnalysis, Table, analyse_pivot, classify_columns
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from .conftest import FakeGraphReader

_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR = "dcid:MalariaGrantDisbursements"
_BARE_STAT_VAR = "MalariaGrantDisbursements"


_CRS_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,provider,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_OBSERVATION_ABOUT_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

_BOUND_FACETS = {
    "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
    "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
    "unit": FacetBinding(state="bound", ref="USD"),
}

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=6,
    earliest_date="2015",
    latest_date="2016",
)


def _sdmx_client(csv_body: str) -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=csv_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))


def _crs_csv_row(*, provider: str, recipient: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{provider},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


def _coverage_for(entities: list[str], *, facet_id: str = "facet1") -> CoverageIndex:
    """Builds coverage rows for each recipient. The CRS request binds only recipient."""
    role_names = {_BARE_STAT_VAR: RoleNames(donor="provider", recipient="recipient")}
    facet_metadata = {
        facet_id: FacetMetadata(
            provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
        )
    }
    rows = [(_BARE_STAT_VAR, facet_id, "org/DAC1", entity, [2015]) for entity in entities]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _materialise_crs_fixture():
    """Builds a two-role CRS result with missing provider arcs and a label-only Uganda recipient."""
    csv_body = (
        _CRS_CSV_HEADER
        + _crs_csv_row(provider="org/DAC1", recipient="country/KEN", year="2015", value="100")
        + _crs_csv_row(provider="org/DAC1", recipient="country/TGO", year="2016", value="200")
        + _crs_csv_row(provider="org/DAC2", recipient="country/UGA", year="2015", value="300")
        + _crs_csv_row(provider="org/DAC2", recipient="country/KEN", year="2016", value="400")
    )
    sdmx = _sdmx_client(csv_body)
    graph = FakeGraphReader(
        facets={
            (_BARE_STAT_VAR, "country/KEN"): [_FACET],
            (_BARE_STAT_VAR, "country/TGO"): [_FACET],
            (_BARE_STAT_VAR, "country/UGA"): [_FACET],
        },
        arcs={
            "country/KEN": {"countryAlpha3Code": ("KEN",), "name": ("Kenya",)},
            "country/TGO": {"countryAlpha3Code": ("TGO",), "name": ("Togo",)},
            "country/UGA": {"name": ("Uganda",)},
        },
    )
    request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": ["country/KEN", "country/TGO", "country/UGA"]},
        declared_roles=("provider", "recipient"),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )

    coverage = _coverage_for(["country/KEN", "country/TGO", "country/UGA"])
    return materialise_group(
        [request],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )


def _observation_about_csv_row(*, entity: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{entity},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


def _materialise_observation_about_fixture():
    """Builds a complete three-country by two-year `observationAbout` grid."""
    entities = ("country/KEN", "country/TGO", "country/UGA")
    years = ("2015", "2016")
    csv_body = _OBSERVATION_ABOUT_CSV_HEADER + "".join(
        _observation_about_csv_row(entity=entity, year=year, value="10")
        for entity in entities
        for year in years
    )
    sdmx = _sdmx_client(csv_body)
    graph = FakeGraphReader(
        facets={(_BARE_STAT_VAR, entity): [_FACET] for entity in entities},
        arcs={
            "country/KEN": {"countryAlpha3Code": ("KEN",), "name": ("Kenya",)},
            "country/TGO": {"countryAlpha3Code": ("TGO",), "name": ("Togo",)},
            "country/UGA": {"countryAlpha3Code": ("UGA",), "name": ("Uganda",)},
        },
    )
    request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"observationAbout": list(entities)},
        declared_roles=("observationAbout",),
        facet_bindings=_BOUND_FACETS,
        window=RequestedWindow(kind="all"),
    )

    return materialise_group(
        [request],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
    )


class TestComputeOutputProfileEndToEnd:
    def test_crs_two_role_fixture_columns_pivots_and_enrichments(self):
        result = _materialise_crs_fixture()
        profile = result.output_profile

        assert profile.schema_version == 1
        assert profile.rows == 4
        assert {c.name: (c.kind, c.droppable, c.distinct) for c in profile.columns} == {
            "stat_var": ("grain", False, 1),
            "provider": ("role", False, 2),
            "recipient": ("role", False, 3),
            "time_period": ("grain", False, 2),
            "value": ("value", True, None),
            "unit": ("unit", False, 1),
            "measurementMethod": ("facet", True, 1),
            "provenance": ("facet", False, 1),
            "status": ("status", True, None),
        }

        # `stat_var` has one value; the two roles and period form the pivot pairs.
        pivot_pairs = {(p.index, p.columns) for p in profile.pivots}
        assert pivot_pairs == {
            ("provider", "recipient"),
            ("recipient", "provider"),
            ("provider", "time_period"),
            ("time_period", "provider"),
            ("recipient", "time_period"),
            ("time_period", "recipient"),
        }

        # Provider enrichments are omitted because both organisations lack code and label arcs.
        assert {(e.column, e.field) for e in profile.enrichments} == {
            ("recipient", "iso3"),
            ("recipient", "label"),
        }
        by_field = {e.field: e for e in profile.enrichments}
        assert by_field["iso3"].entities == 3
        assert by_field["iso3"].covered == 2
        assert by_field["iso3"].missing == (("country/UGA", "Uganda"),)
        assert by_field["iso3"].truncated is False
        assert by_field["label"].entities == 3
        assert by_field["label"].covered == 3
        assert by_field["label"].missing == ()

    def test_observation_about_single_role_fixture_is_the_common_case(self):
        result = _materialise_observation_about_fixture()
        profile = result.output_profile

        assert profile.rows == 6
        column_kinds = {c.name: c.kind for c in profile.columns}
        assert column_kinds["observationAbout"] == "role"
        assert column_kinds["time_period"] == "grain"

        pivot_pairs = {(p.index, p.columns) for p in profile.pivots}
        assert pivot_pairs == {
            ("observationAbout", "time_period"),
            ("time_period", "observationAbout"),
        }
        for analysis in profile.pivots:
            assert analysis.unique is True

        by_field = {e.field: e for e in profile.enrichments if e.column == "observationAbout"}
        assert by_field["iso3"].entities == 3
        assert by_field["iso3"].covered == 3
        assert by_field["iso3"].missing == ()
        assert by_field["label"].covered == 3


def test_multiple_role_columns_share_one_enrichment_lookup_call():
    # Both role columns resolve in one `node_arcs_many` call.
    columns = (
        "stat_var",
        "provider",
        "recipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    rows = (
        (
            _STAT_VAR,
            "org/DAC1",
            "country/KEN",
            "2015",
            1.0,
            "USD",
            "GrantEquivalent",
            "OECD_DAC2A",
            None,
        ),
        (
            _STAT_VAR,
            "org/DAC2",
            "country/TGO",
            "2016",
            2.0,
            "USD",
            "GrantEquivalent",
            "OECD_DAC2A",
            None,
        ),
    )
    table = Table(columns=columns, rows=rows)
    graph = FakeGraphReader(
        arcs={
            "org/DAC1": {"name": ("DAC One",)},
            "org/DAC2": {"name": ("DAC Two",)},
            "country/KEN": {"countryAlpha3Code": ("KEN",), "name": ("Kenya",)},
            "country/TGO": {"countryAlpha3Code": ("TGO",), "name": ("Togo",)},
        }
    )

    profile = compute_output_profile(
        table,
        role_columns=("provider", "recipient"),
        enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
    )

    assert graph.calls["node_arcs_many"] == 1
    assert {(e.column, e.field) for e in profile.enrichments} == {
        ("provider", "label"),
        ("recipient", "iso3"),
        ("recipient", "label"),
    }


class TestComputeOutputProfileEdgeCases:
    _COLUMNS = (
        "stat_var",
        "recipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )

    def _profile_for(self, rows, *, arcs=None):
        table = Table(columns=self._COLUMNS, rows=rows)
        graph = FakeGraphReader(arcs=arcs or {})
        return compute_output_profile(
            table,
            role_columns=("recipient",),
            enrichment=EnrichmentLookup(graph, type_kinds={}, parties={}),
        )

    def test_zero_row_frame_reports_columns_with_empty_pivots_and_enrichments(self):
        profile = self._profile_for(())

        assert profile.rows == 0
        assert [c.name for c in profile.columns] == list(self._COLUMNS)
        assert profile.pivots == ()
        assert profile.enrichments == ()

    def test_a_single_distinct_column_is_excluded_from_pivot_pairs(self):
        # One distinct recipient and stat var leave no eligible pivot pair.
        rows = (
            (_STAT_VAR, "country/KEN", "2015", 1.0, "USD", "GrantEquivalent", "OECD_DAC2A", None),
            (_STAT_VAR, "country/KEN", "2016", 2.0, "USD", "GrantEquivalent", "OECD_DAC2A", None),
        )
        profile = self._profile_for(rows)

        assert profile.pivots == ()
        recipient_column = next(c for c in profile.columns if c.name == "recipient")
        assert recipient_column.distinct == 1
        time_period_column = next(c for c in profile.columns if c.name == "time_period")
        assert time_period_column.distinct == 2

    def test_missing_names_refs_with_labels_and_truncates_at_the_cap(self):
        refs = [f"country/C{i:02d}" for i in range(12)]
        rows = tuple(
            (_STAT_VAR, ref, "2015", 1.0, "USD", "GrantEquivalent", "OECD_DAC2A", None)
            for ref in refs
        )
        # Eleven refs lack ISO3 codes, exceeding the ten-item cap by one.
        arcs = {ref: {"name": (f"Country {i:02d}",)} for i, ref in enumerate(refs)}
        arcs[refs[0]]["countryAlpha3Code"] = ("C00",)
        profile = self._profile_for(rows, arcs=arcs)

        iso3 = next(e for e in profile.enrichments if e.field == "iso3")
        assert iso3.entities == 12
        assert iso3.covered == 1
        assert len(iso3.missing) == 10
        assert iso3.truncated is True
        assert iso3.missing[0] == ("country/C01", "Country 01")


class TestOutputProfileJson:
    def test_to_json_from_json_round_trips_to_an_equal_profile(self):
        result = _materialise_crs_fixture()
        profile = result.output_profile

        restored = OutputProfile.from_json(profile.to_json())

        assert restored == profile

    def test_round_trip_survives_a_real_json_encode_decode(self):
        result = _materialise_crs_fixture()
        profile = result.output_profile

        wire = json.loads(json.dumps(profile.to_json()))
        restored = OutputProfile.from_json(wire)

        assert restored == profile

    def test_from_json_reconstructs_the_declared_dataclasses(self):
        profile = OutputProfile.from_json(
            {
                "schema_version": 1,
                "rows": 0,
                "columns": [
                    {"name": "stat_var", "kind": "grain", "droppable": False, "distinct": 1}
                ],
                "pivots": [],
                "enrichments": [],
            }
        )

        assert profile.columns == (
            ColumnProfile(name="stat_var", kind="grain", droppable=False, distinct=1),
        )
        assert profile.pivots == ()
        assert profile.enrichments == ()
        assert isinstance(profile.enrichments, tuple)


def test_classify_columns_still_agrees_with_profile_droppable_derivation():
    # A cheap cross-check that `profile.py` reads `classify_columns` for
    # its droppable rule. `provenance` stays non-droppable, and
    # `measurementMethod` is always droppable.
    classification = classify_columns(
        (
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        )
    )
    assert "provenance" in classification.non_droppable_columns
    assert "measurementMethod" not in classification.non_droppable_columns


def _six_facts(analysis: PivotAnalysis) -> tuple[bool, bool, int, int, tuple, bool]:
    return (
        analysis.unique,
        analysis.homogeneous,
        analysis.width,
        analysis.cells,
        analysis.varying,
        bool(analysis.duplicate_keys),
    )


def _assert_pivot_feasibility_agrees_with_analyse_pivot(table: Table) -> None:
    """Compares each reported pivot pair's six feasibility fields with `analyse_pivot`."""
    classification = classify_columns(table.columns)
    profile = compute_output_profile(
        table,
        role_columns=(),
        enrichment=EnrichmentLookup(FakeGraphReader(), type_kinds={}, parties={}),
    )
    assert profile.pivots, "fixture produced no pivot pairs to compare"
    for feasibility in profile.pivots:
        expected = analyse_pivot(
            table,
            index=feasibility.index,
            columns=feasibility.columns,
            classification=classification,
        )
        actual = (
            feasibility.unique,
            feasibility.homogeneous,
            feasibility.width,
            feasibility.cells,
            feasibility.varying,
            feasibility.duplicate_keys,
        )
        assert actual == _six_facts(expected), (feasibility.index, feasibility.columns)


class TestPivotFeasibilityAgreesWithAnalysePivot:
    """Checks pivot feasibility across duplicate, heterogeneous, and unique table shapes."""

    def test_duplicate_keys_with_a_varying_column(self):
        # `("east", "2020")` repeats across two rows whose `group` differs,
        # so the pair is non-unique for the duplicate-key reason, and
        # `group` (differing inside that duplicated group) is `varying`.
        table = Table(
            columns=("region", "period", "group", "value", "status"),
            rows=(
                ("east", "2020", "g1", 1.0, None),
                ("east", "2020", "g2", 2.0, None),
                ("east", "2021", "g1", 3.0, None),
                ("west", "2020", "g1", 4.0, None),
            ),
        )
        _assert_pivot_feasibility_agrees_with_analyse_pivot(table)

    def test_a_heterogeneous_group_with_otherwise_unique_keys(self):
        # `(region, period)` keys stay distinct across rows, but `unit`
        # differs across the two rows sharing `region="east"`, so
        # `homogeneous(region, *)` is `False` while `unique(region,
        # period)` stays `True`.
        table = Table(
            columns=("region", "period", "unit", "value", "status"),
            rows=(
                ("east", "2020", "USD", 1.0, None),
                ("east", "2021", "EUR", 2.0, None),
                ("west", "2020", "USD", 3.0, None),
            ),
        )
        _assert_pivot_feasibility_agrees_with_analyse_pivot(table)

    def test_a_stringification_collision_with_no_duplicate_key(self):
        # `code`'s distinct values `1` and `"1"` are distinct as keys, so
        # `(group, code)` stays unique as a key pair. Both values
        # stringify to `"1"`, so `unique(group, code)` is `False` for the
        # stringification reason alone.
        table = Table(
            columns=("group", "code", "value", "status"),
            rows=(
                ("g1", 1, 10.0, None),
                ("g1", "1", 20.0, None),
                ("g2", 2, 30.0, None),
            ),
        )
        _assert_pivot_feasibility_agrees_with_analyse_pivot(table)

    def test_a_holding_functional_dependency(self):
        # `zone` is constant within each `region` group, so `FD(region ->
        # zone)` holds. `homogeneous(region, period)` and
        # `homogeneous(region, zone)` share that fact, computed once.
        table = Table(
            columns=("region", "period", "zone", "value", "status"),
            rows=(
                ("east", "2020", "Z1", 1.0, None),
                ("east", "2021", "Z1", 2.0, None),
                ("west", "2020", "Z2", 3.0, None),
                ("west", "2021", "Z2", 4.0, None),
            ),
        )
        _assert_pivot_feasibility_agrees_with_analyse_pivot(table)

    def test_a_genuinely_unique_pivot_pair(self):
        # Only two grain columns exist and every `(entity, period)` combo
        # appears exactly once, so both pairs come back fully unique.
        entities = ("e1", "e2", "e3")
        periods = ("2020", "2021")
        table = Table(
            columns=("entity", "period", "value", "status"),
            rows=tuple((entity, period, 1.0, None) for entity in entities for period in periods),
        )
        _assert_pivot_feasibility_agrees_with_analyse_pivot(table)

    def test_a_frame_built_through_the_grammar(self):
        # `_materialise_crs_fixture`'s two-role CRS frame comes from
        # `materialise_group`'s own pipeline and carries its own
        # duplicate keys and varying columns.
        result = _materialise_crs_fixture()
        _assert_pivot_feasibility_agrees_with_analyse_pivot(
            ConcatenatedFrameChunks.from_chunks(result.frame_chunks())
        )

    def test_a_column_jointly_determined_by_index_and_columns_but_neither_alone(self):
        # The `(region, period)` pair determines `group`; either column alone maps to both values.
        # Duplicate keys agree on `group`, so it is absent from `varying`.
        table = Table(
            columns=("region", "period", "group", "value", "status"),
            rows=(
                ("east", "2020", "g1", 1.0, None),
                ("east", "2020", "g1", 2.0, None),
                ("east", "2021", "g2", 3.0, None),
                ("west", "2020", "g2", 4.0, None),
            ),
        )
        _assert_pivot_feasibility_agrees_with_analyse_pivot(table)

    def test_a_seeded_random_table_equivalence_sweep(self):
        # Three hundred nine-column tables cover duplicates, mixed groups, nulls, and
        # stringification collisions. NaN is outside the grammar and breaks reused-cell analysis.
        rng = random.Random(20260903)
        for _ in range(300):
            table = _random_grammar_shaped_table(rng)
            classification = classify_columns(table.columns)
            profile = compute_output_profile(
                table,
                role_columns=(),
                enrichment=EnrichmentLookup(FakeGraphReader(), type_kinds={}, parties={}),
            )
            for feasibility in profile.pivots:
                expected = analyse_pivot(
                    table,
                    index=feasibility.index,
                    columns=feasibility.columns,
                    classification=classification,
                )
                actual = (
                    feasibility.unique,
                    feasibility.homogeneous,
                    feasibility.width,
                    feasibility.cells,
                    feasibility.varying,
                    feasibility.duplicate_keys,
                )
                assert actual == _six_facts(expected), (feasibility.index, feasibility.columns)


def _random_grammar_shaped_table(rng: random.Random) -> Table:
    """A table over the grammar's nine-column shape (`stat_var`, two role
    columns, `time_period`, `value`, `unit`, `measurementMethod`,
    `provenance`, `status`), zero to twelve rows. The role columns and the
    `measurementMethod` facet column can carry `None` cells. The second
    role column mixes `int` and `str` values that stringify alike (`1`
    and `"1"`), and `time_period` can carry the string `"r1"`, equal to
    the first role column's own name. That exercises the cross-column
    stringification case. Row count and per-column value counts are both
    randomised, so duplicate keys and heterogeneous groups occur
    incidentally.
    """
    columns = (
        "stat_var",
        "r1",
        "r2",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    alphabets = {
        "stat_var": ["sv1", "sv2"],
        "r1": ["a", "b", "c", None],
        "r2": ["x", "y", 1, "1", None],
        "time_period": ["2020", "2021", 2020, "r1"],
        "unit": ["USD", "EUR"],
        "measurementMethod": ["m1", "m2", None],
        "provenance": ["p1", "p2"],
    }
    rows = []
    for _ in range(rng.randint(0, 12)):
        row = []
        for column in columns:
            if column == "value":
                row.append(rng.uniform(0, 10))
            elif column == "status":
                row.append(None)
            else:
                choices = alphabets[column]
                width = rng.randint(1, len(choices))
                row.append(rng.choice(choices[:width]))
        rows.append(tuple(row))
    return Table(columns=columns, rows=tuple(rows))
