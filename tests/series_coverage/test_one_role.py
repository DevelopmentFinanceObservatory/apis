"""Pins coverage for variables that declare one role, on either flow axis.

An IDS total declares only `financingRecipient`, so its rows carry the
debtor in `recipient` and an empty `donor`, and its `RoleNames` has no
donor. A DAC1 aggregate declares only `financingProvider`, so its rows carry
the donor in `donor` and an empty `recipient`, and its `RoleNames` has no
recipient. The artifact below holds both beside a two-role by-creditor
variant.
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
PROVIDER_TOTAL = "AmountMobilised_FinancialFlow_Gross_Disbursement_DAC1AidType2233_Nominal"
FACET = "2052749850102211920"
DAC1_FACET = "3209178722"
IDS = "provenance/WorldBankInternationalDebtStatistics"
DAC1 = "provenance/OECDDAC1"
CHN = "country/CHN"
FRA = "country/FRA"
GBR = "country/GBR"
KEN = "country/KEN"
UGA = "country/UGA"

ROWS = [
    (TOTAL, FACET, ABSENT_ENTITY, KEN, [2020, 2021, 2022]),
    (TOTAL, FACET, ABSENT_ENTITY, UGA, [2021, 2022]),
    (BY_CREDITOR, FACET, CHN, KEN, [2021, 2022]),
    (PROVIDER_TOTAL, DAC1_FACET, FRA, ABSENT_ENTITY, [2019, 2020, 2021, 2022]),
    (PROVIDER_TOTAL, DAC1_FACET, GBR, ABSENT_ENTITY, [2022]),
]
ROLE_NAMES = {
    TOTAL: RoleNames(donor=None, recipient="financingRecipient"),
    BY_CREDITOR: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    PROVIDER_TOTAL: RoleNames(donor="financingProvider", recipient=None),
}
FACET_METADATA = {
    FACET: FacetMetadata(provenance_id=IDS, unit="CurrencyUSD"),
    DAC1_FACET: FacetMetadata(provenance_id=DAC1, unit="CurrencyUSD"),
}


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


def test_role_names_json_stores_the_lacking_role_as_null_and_loads_it_back(
    tmp_path: Path,
) -> None:
    uri = _write_artifact(tmp_path / "coverage")
    raw = json.loads((build_dir(tmp_path / "coverage", "b1") / "role_names.json").read_text())
    assert raw[TOTAL] == {"donor": None, "recipient": "financingRecipient"}
    assert raw[PROVIDER_TOTAL] == {"donor": "financingProvider", "recipient": None}
    assert load_coverage(uri).index.role_names == ROLE_NAMES


def test_verify_accepts_an_artifact_holding_a_one_role_variable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert verify_main([_write_artifact(tmp_path / "coverage")]) == 0
    assert "loaded a compatible coverage artifact" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("variable", "names"),
    [
        (TOTAL, RoleNames(donor="financingProvider", recipient="financingRecipient")),
        (PROVIDER_TOTAL, RoleNames(donor="financingProvider", recipient="financingRecipient")),
        (BY_CREDITOR, RoleNames(donor=None, recipient="financingRecipient")),
        (BY_CREDITOR, RoleNames(donor="financingProvider", recipient=None)),
        (TOTAL, RoleNames(donor="financingProvider", recipient=None)),
        (PROVIDER_TOTAL, RoleNames(donor=None, recipient="financingRecipient")),
    ],
)
def test_load_refuses_role_names_that_disagree_with_the_empty_columns(
    tmp_path: Path, variable: str, names: RoleNames
) -> None:
    uri = _write_artifact(tmp_path / "coverage", role_names={**ROLE_NAMES, variable: names})
    with pytest.raises(CoverageArtifactError, match=variable):
        load_coverage(uri)


@pytest.mark.parametrize(
    "row",
    [
        (BY_CREDITOR, FACET, CHN, ABSENT_ENTITY, [2020]),
        (BY_CREDITOR, FACET, ABSENT_ENTITY, KEN, [2020]),
    ],
)
def test_load_refuses_an_empty_entity_on_a_role_the_variable_declares(
    tmp_path: Path, row: tuple[str, str, str, str, list[int]]
) -> None:
    uri = _write_artifact(tmp_path / "coverage", rows=[*ROWS, row])
    with pytest.raises(CoverageArtifactError, match=BY_CREDITOR):
        load_coverage(uri)


@pytest.mark.parametrize(
    "entry", [{"donor": None, "recipient": None}, {"donor": None, "recipient": 5}]
)
def test_load_refuses_a_role_names_entry_without_a_string_role(
    tmp_path: Path, entry: dict[str, object]
) -> None:
    uri = _write_artifact(tmp_path / "coverage")
    path = build_dir(tmp_path / "coverage", "b1") / "role_names.json"
    payload = json.loads(path.read_text())
    payload[TOTAL] = entry
    path.write_text(json.dumps(payload))
    with pytest.raises(CoverageArtifactError, match=r"role_names\.json"):
        load_coverage(uri)


# --- index lookups -----------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "bound", "n_obs"),
    [
        (TOTAL, {"recipient": frozenset({KEN})}, 3),
        (PROVIDER_TOTAL, {"donor": frozenset({FRA})}, 4),
    ],
)
def test_a_bound_role_confirms_the_one_role_variable_declaring_it(
    index: CoverageIndex, variable: str, bound: dict[str, frozenset[str]], n_obs: int
) -> None:
    [coverage] = index.filter_candidates(variables=[variable], bound=bound)
    assert coverage.n_obs == n_obs


@pytest.mark.parametrize(
    ("bound", "matched"),
    [
        ({"donor": frozenset({CHN})}, [BY_CREDITOR]),
        ({"recipient": frozenset({KEN})}, [TOTAL, BY_CREDITOR]),
    ],
)
def test_a_bound_role_matches_no_row_of_a_one_role_variable_lacking_it(
    index: CoverageIndex, bound: dict[str, frozenset[str]], matched: list[str]
) -> None:
    found = index.filter_candidates(variables=[TOTAL, BY_CREDITOR, PROVIDER_TOTAL], bound=bound)
    assert sorted(coverage.variable for coverage in found) == sorted(matched)


def test_open_axis_never_returns_the_absent_entity(index: CoverageIndex) -> None:
    assert index.open_axis(variables=[TOTAL], axis="donor") == frozenset()
    assert index.open_axis(variables=[TOTAL, BY_CREDITOR], axis="donor") == {CHN}
    assert index.open_axis(variables=[TOTAL], axis="recipient") == {KEN, UGA}
    assert index.open_axis(variables=[PROVIDER_TOTAL], axis="recipient") == frozenset()
    assert index.open_axis(variables=[PROVIDER_TOTAL], axis="donor") == {FRA, GBR}


@pytest.mark.parametrize(
    ("variables", "role_counts"),
    [
        ([TOTAL], {"recipient": 2}),
        ([PROVIDER_TOTAL], {"donor": 2}),
        ([TOTAL, BY_CREDITOR], {"donor": 1, "recipient": 2}),
    ],
)
def test_headline_counts_only_the_roles_a_variable_declares(
    index: CoverageIndex, variables: list[str], role_counts: dict[str, int]
) -> None:
    headline = index.headline(variables=variables, bound={})
    assert headline is not None
    assert headline.year == 2022
    assert headline.role_counts == role_counts


@pytest.mark.parametrize(
    ("variable", "names", "expected"),
    [
        (TOTAL, ROLE_NAMES[TOTAL], {"financingRecipient": (KEN, UGA)}),
        (PROVIDER_TOTAL, ROLE_NAMES[PROVIDER_TOTAL], {"financingProvider": (FRA, GBR)}),
    ],
)
def test_unbound_dispatch_filter_lists_the_one_role_axis(
    index: CoverageIndex, variable: str, names: RoleNames, expected: dict[str, tuple[str, ...]]
) -> None:
    assert (
        index.exhaustive_single_axis_filter(
            variables=[variable], donor_role=names.donor, recipient_role=names.recipient
        )
        == expected
    )


def test_unbound_estimate_prices_each_one_role_group(index: CoverageIndex) -> None:
    estimate = index.estimate(variables=[TOTAL, BY_CREDITOR, PROVIDER_TOTAL], bound={}, lanes=1)
    assert (estimate.rows, estimate.requests) == (12, 3)


def test_pair_rows_report_no_donor_for_the_recipient_only_variable(index: CoverageIndex) -> None:
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


def test_pair_rows_report_no_recipient_for_the_donor_only_variable(index: CoverageIndex) -> None:
    rows = index.pair_rows(variables=[PROVIDER_TOTAL])
    [group] = rows.groups()
    assert (group.donors, group.recipients) == ({FRA, GBR}, frozenset())
    assert rows.pairs(variable=PROVIDER_TOTAL, facet_id=DAC1_FACET) == [
        (FRA, None, 4),
        (GBR, None, 1),
    ]
    assert rows.head_pairs(
        variable=PROVIDER_TOTAL,
        facet_id=DAC1_FACET,
        limit=5,
        member_is_donor=True,
        has_opposite=True,
    ) == [(FRA, None, 4), (GBR, None, 1)]


def test_provenance_shape_lists_the_one_role_axis_alone(index: CoverageIndex) -> None:
    assert index.provenance_shapes[IDS] == ProvenanceShape(
        observation_periods=frozenset({None}),
        role_axes=frozenset({frozenset({"destination"}), frozenset({"origin", "destination"})}),
    )
    assert index.provenance_shapes[DAC1] == ProvenanceShape(
        observation_periods=frozenset({None}),
        role_axes=frozenset({frozenset({"origin"})}),
    )
