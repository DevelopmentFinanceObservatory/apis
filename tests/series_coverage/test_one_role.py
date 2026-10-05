"""Pins coverage for a variable that declares one role, shaped like an IDS total.

An IDS total declares only `financingRecipient`, so its rows carry the
debtor in `recipient` and an empty `donor`, and its `RoleNames` has no
donor. The artifact below holds it beside a two-role by-creditor variant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from series_coverage.index import CoverageIndex, ProvenanceShape
from series_coverage.io import (
    write_coverage_table,
    write_current,
    write_facet_metadata,
    write_manifest,
    write_role_names,
)
from series_coverage.loader import CoverageArtifactError, load_coverage
from series_coverage.schema import (
    ABSENT_ENTITY,
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
)
from series_coverage.verify import main as verify_main
from tests.series_coverage.support import build_dir, coverage_table, pointer_path

TOTAL = "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_Nominal"
BY_CREDITOR = (
    "Amount_FinancialPosition_AmountOutstanding_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"
)
FACET = "2052749850102211920"
IDS = "provenance/WorldBankInternationalDebtStatistics"
CHN = "country/CHN"
KEN = "country/KEN"
UGA = "country/UGA"

ROWS = [
    (TOTAL, FACET, ABSENT_ENTITY, KEN, [2020, 2021, 2022]),
    (TOTAL, FACET, ABSENT_ENTITY, UGA, [2021, 2022]),
    (BY_CREDITOR, FACET, CHN, KEN, [2021, 2022]),
]
ROLE_NAMES = {
    TOTAL: RoleNames(donor=None, recipient="financingRecipient"),
    BY_CREDITOR: RoleNames(donor="financingProvider", recipient="financingRecipient"),
}
FACET_METADATA = {FACET: FacetMetadata(provenance_id=IDS, unit="CurrencyUSD")}


def _write_artifact(
    base: Path,
    *,
    rows=ROWS,
    role_names: dict[str, RoleNames] = ROLE_NAMES,
) -> str:
    table, horizon = coverage_table(rows)
    prefix = build_dir(base, "b1")
    prefix.mkdir(parents=True)
    write_coverage_table(table, prefix / "coverage.parquet", horizon=horizon)
    write_role_names(role_names, prefix / "role_names.json")
    write_facet_metadata(FACET_METADATA, prefix / "facets.json")
    write_manifest(
        CoverageManifest(
            stamp="s1",
            build_id="b1",
            built_at="2026-09-30T00:00:00Z",
            source="tests.series_coverage.test_one_role",
            row_counts={"rows": table.num_rows},
            files={
                "coverage": "coverage.parquet",
                "role_names": "role_names.json",
                "facets": "facets.json",
            },
            horizon=horizon,
        ),
        prefix / "manifest.json",
    )
    write_current(CoveragePointer(stamp="s1", build_id="b1"), pointer_path(base))
    return f"file://{base}"


@pytest.fixture
def index(tmp_path: Path) -> CoverageIndex:
    return load_coverage(_write_artifact(tmp_path / "coverage")).index


# --- loader and verify -----------------------------------------------------


def test_role_names_json_stores_a_null_donor_and_loads_it_back(tmp_path: Path) -> None:
    uri = _write_artifact(tmp_path / "coverage")
    raw = json.loads((build_dir(tmp_path / "coverage", "b1") / "role_names.json").read_text())
    assert raw[TOTAL] == {"donor": None, "recipient": "financingRecipient"}
    assert load_coverage(uri).index.role_names == ROLE_NAMES


def test_verify_accepts_an_artifact_holding_a_one_role_variable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert verify_main([_write_artifact(tmp_path / "coverage")]) == 0
    assert "loaded a compatible coverage artifact" in capsys.readouterr().out


def test_load_refuses_an_empty_donor_on_a_variable_with_a_donor_role(tmp_path: Path) -> None:
    role_names = {**ROLE_NAMES, TOTAL: RoleNames(donor="financingProvider", recipient="x")}
    uri = _write_artifact(tmp_path / "coverage", role_names=role_names)
    with pytest.raises(CoverageArtifactError, match=TOTAL):
        load_coverage(uri)


def test_load_refuses_a_null_donor_role_on_rows_that_carry_a_donor(tmp_path: Path) -> None:
    role_names = {**ROLE_NAMES, BY_CREDITOR: RoleNames(donor=None, recipient="x")}
    uri = _write_artifact(tmp_path / "coverage", role_names=role_names)
    with pytest.raises(CoverageArtifactError, match=BY_CREDITOR):
        load_coverage(uri)


def test_load_refuses_an_empty_recipient(tmp_path: Path) -> None:
    rows = [*ROWS, (BY_CREDITOR, FACET, CHN, ABSENT_ENTITY, [2020])]
    uri = _write_artifact(tmp_path / "coverage", rows=rows)
    with pytest.raises(CoverageArtifactError, match="empty recipient"):
        load_coverage(uri)


def test_load_refuses_a_non_string_recipient(tmp_path: Path) -> None:
    uri = _write_artifact(tmp_path / "coverage")
    path = build_dir(tmp_path / "coverage", "b1") / "role_names.json"
    payload = json.loads(path.read_text())
    payload[TOTAL]["recipient"] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(CoverageArtifactError, match=r"role_names\.json"):
        load_coverage(uri)


# --- index lookups -----------------------------------------------------------


def test_a_bound_recipient_confirms_the_one_role_variable(index: CoverageIndex) -> None:
    [coverage] = index.filter_candidates(variables=[TOTAL], bound={"recipient": frozenset({KEN})})
    assert coverage.n_obs == 3


def test_a_bound_donor_matches_no_row_of_the_one_role_variable(index: CoverageIndex) -> None:
    found = index.filter_candidates(
        variables=[TOTAL, BY_CREDITOR], bound={"donor": frozenset({CHN})}
    )
    assert [coverage.variable for coverage in found] == [BY_CREDITOR]


def test_open_axis_never_returns_the_absent_donor(index: CoverageIndex) -> None:
    assert index.open_axis(variables=[TOTAL], axis="donor") == frozenset()
    assert index.open_axis(variables=[TOTAL, BY_CREDITOR], axis="donor") == {CHN}
    assert index.open_axis(variables=[TOTAL], axis="recipient") == {KEN, UGA}


def test_headline_omits_the_donor_count_of_a_one_role_variable(index: CoverageIndex) -> None:
    headline = index.headline(variables=[TOTAL], bound={})
    assert headline is not None
    assert (headline.year, headline.series) == (2022, 2)
    assert headline.role_counts == {"recipient": 2}

    mixed = index.headline(variables=[TOTAL, BY_CREDITOR], bound={})
    assert mixed is not None
    assert mixed.role_counts == {"donor": 1, "recipient": 2}


def test_unbound_estimate_and_dispatch_filter_on_the_recipient_axis(index: CoverageIndex) -> None:
    estimate = index.estimate(variables=[TOTAL, BY_CREDITOR], bound={}, lanes=1)
    assert (estimate.rows, estimate.requests) == (7, 2)
    assert index.exhaustive_single_axis_filter(
        variables=[TOTAL], donor_role=None, recipient_role="financingRecipient"
    ) == {"financingRecipient": (KEN, UGA)}


def test_pair_rows_report_no_donor_for_the_one_role_variable(index: CoverageIndex) -> None:
    rows = index.pair_rows(variables=[TOTAL])
    [group] = rows.groups()
    assert (group.donors, group.recipients) == (frozenset(), {KEN, UGA})
    assert rows.pairs(variable=TOTAL, facet_id=FACET) == [(None, KEN, 3), (None, UGA, 2)]
    assert rows.head_pairs(
        variable=TOTAL,
        facet_id=FACET,
        limit=5,
        member_is_donor=False,
        has_opposite=True,
    ) == [(KEN, None, 3), (UGA, None, 2)]


def test_provenance_shape_lists_the_recipient_axis_alone_for_the_one_role_variable(
    index: CoverageIndex,
) -> None:
    assert index.provenance_shapes[IDS] == ProvenanceShape(
        observation_periods=frozenset({None}),
        role_axes=frozenset({frozenset({"destination"}), frozenset({"origin", "destination"})}),
    )
