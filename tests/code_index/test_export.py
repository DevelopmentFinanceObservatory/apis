"""Offline tests for scripts/export_code_index.py. Drives a small in-memory
fake reader (`_FakeGraph`) implementing only the `GraphReader` surface the
export actually calls -- `node_subjects` and `node_arcs_many`. Nothing here
opens a socket.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_index.loader import (
    IndexFormatError,
    default_index_path,
    load_index,
    load_measure_axes,
    parse_index,
)
from code_index.types import DatasetDeclaration, DeclaredDefault, MeasureAxis, MeasurePartitionLabel
from materialise.graph import (
    UNRESOLVED_NODE,
    GraphNotAdvancingError,
    GraphReadError,
    SubjectPage,
)
from scripts.export_code_index import (
    _MEASURE_VALUE_TEXT,
    CODE_SYSTEMS,
    DATASETS,
    CodeSystemConfig,
    CoverageJoin,
    _MeasureValueText,
    _validate_declared_defaults,
    _verify_axis_coverage,
    _verify_measure_label_named_values,
    _verify_measure_nouns,
    _verify_measure_partition_labels,
    content_hash,
    enumerate_statvar_dcids,
    export_code_system,
    export_graph_blocks,
    export_index,
    export_measure_values,
    export_statvars,
    load_coverage_join,
    stamp_counts,
    strip_system_suffix,
)
from series_coverage.loader import load_coverage

_NOW = datetime(2026, 9, 9, 12, 0, 0, tzinfo=UTC)


@dataclass
class _FakeGraph:
    """Hold graph nodes as dcid-keyed out-arcs and implement filtered reverse reads.

    `page_size` returns a sorted truncated sample like the live graph. `truncate` forces that
    result at any size; `not_advancing` raises on oversized reads, and `read_error` always raises.
    `unreadable` nodes appear in reverse reads but their forward read returns no node.
    `calls` records each reverse read and its filters.
    """

    forward_arcs: dict[str, dict[str, tuple[str, ...]]]
    page_size: int | None = None
    truncate: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    not_advancing: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    read_error: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    unreadable: frozenset[str] = field(default_factory=frozenset)
    calls: list[tuple[str, str, tuple[tuple[str, str], ...] | None]] = field(default_factory=list)

    def node_subjects(self, dcid, *, property, filters=None, max_results):
        key = (dcid, property)
        filter_items = tuple(sorted(filters.items())) if filters else None
        self.calls.append((dcid, property, filter_items))
        if key in self.read_error:
            raise GraphReadError(f"transport failure reading {dcid}/{property}")
        subjects = sorted(
            subject
            for subject, arcs in self.forward_arcs.items()
            if dcid in arcs.get(property, ())
            and all(value in arcs.get(fkey, ()) for fkey, value in (filters or {}).items())
        )
        size_exceeded = self.page_size is not None and len(subjects) > self.page_size
        oversized = key in self.truncate or size_exceeded
        if oversized and key in self.not_advancing:
            raise GraphNotAdvancingError(
                f"{dcid}/{property} not advancing", dcid=dcid, property=property
            )
        if key in self.truncate:
            return SubjectPage(dcids=tuple(subjects[:max_results]), truncated=True)
        if size_exceeded:
            return SubjectPage(dcids=tuple(subjects[: self.page_size]), truncated=True)
        return SubjectPage(dcids=tuple(subjects), truncated=False)

    def node_arcs_many(
        self,
        dcids,
        *,
        properties,
        individual_probe_budget=None,
        deadline: float | None = None,
    ):
        # Collapse duplicate arcs in first-seen order, as the live graph read boundary does.
        result = {}
        for dcid in dcids:
            arcs = None if dcid in self.unreadable else self.forward_arcs.get(dcid)
            if arcs is None:
                result[dcid] = None
            else:
                result[dcid] = {p: tuple(dict.fromkeys(arcs.get(p, ()))) for p in properties}
        return result

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ):
        result = {}
        for dcid in dict.fromkeys(dcids):
            arcs = self.forward_arcs.get(dcid, {})
            result[dcid] = tuple(
                dict.fromkeys(
                    value for prop in ("name", "alternateName") for value in arcs.get(prop, ())
                )
            )
        return result


_TEST_PROVENANCE = "provenance/TestDataset"
_TEST_DATASETS = (
    DatasetDeclaration(
        provenance=_TEST_PROVENANCE,
        label="Test dataset",
        synonyms=(),
        opt_in=False,
        absent_observation="zero",
        rank=0,
    ),
)


def _covered(graph: _FakeGraph, provenance: str = _TEST_PROVENANCE) -> CoverageJoin:
    """A coverage build that holds series under `provenance` for every node of `graph`."""
    return CoverageJoin(
        build_id="test-build", provenances=dict.fromkeys(graph.forward_arcs, (provenance,))
    )


_HIERARCHICAL = CodeSystemConfig(
    code_system="sysHier",
    enum_dcid="SysHierEnum",
    expected_hierarchical=True,
    disjoint_from_children=True,
    summable=True,
    absent_reading="not_declared",
    declared_by_every_statvar_of=(_TEST_PROVENANCE,),
    plain_name="hier",
)
_FLAT = CodeSystemConfig(
    code_system="sysFlat",
    enum_dcid="SysFlatEnum",
    expected_hierarchical=False,
    disjoint_from_children=True,
    summable=True,
    absent_reading="not_declared",
    declared_by_every_statvar_of=(_TEST_PROVENANCE,),
    plain_name="flat",
)


def _two_system_graph() -> _FakeGraph:
    """Build one hierarchical and one flat code system plus four StatVars.

    Two disbursement rows and two commitment rows exercise shape fields. One disbursement carries
    `nettingBasis`. Constraint values are typed nodes. Each StatVar has `typeOf: StatisticalVariable`.
    """
    return _FakeGraph(
        forward_arcs={
            "Disbursement": {"typeOf": ("FinancialFlowTypeEnum",)},
            "Gross": {"typeOf": ("NettingBasisEnum",)},
            "Commitment": {"typeOf": ("FinancialObligationTypeEnum",)},
            "SysHierRoot": {
                "typeOf": ("SysHierEnum",),
                "classificationCode": ("100",),
                "name": ("root sector (CRS Hier)",),
                "alternateName": ("Root Sector",),
            },
            "SysHierChild": {
                "typeOf": ("SysHierEnum",),
                "classificationCode": ("110",),
                "name": ("child sector (CRS Hier)",),
                "alternateName": ("Child Sector", "root sector"),
                "specializationOf": ("SysHierRoot",),
            },
            "SysFlatA": {
                "typeOf": ("SysFlatEnum",),
                "classificationCode": ("10",),
                "name": ("flat alpha (CRS Flat)",),
                "alternateName": (),
            },
            "SysFlatB": {
                "typeOf": ("SysFlatEnum",),
                "classificationCode": ("20",),
                "name": ("flat beta (CRS Flat)",),
                "alternateName": (),
            },
            "SV_Disbursement_Gross": {
                "typeOf": ("StatisticalVariable",),
                "sysHier": ("SysHierChild",),
                "sysFlat": ("SysFlatA",),
                "constraintProperties": ("sysHier", "sysFlat", "financialFlowType", "nettingBasis"),
                "populationType": ("FinancialFlow",),
                "measuredProperty": ("amount",),
                "statType": ("measuredValue",),
                "measurementQualifier": ("Nominal",),
                "observationProperties": ("financingProvider", "financingRecipient"),
                "financialFlowType": ("Disbursement",),
                "nettingBasis": ("Gross",),
            },
            "SV_Disbursement_NoNetting": {
                "typeOf": ("StatisticalVariable",),
                "sysHier": ("SysHierChild",),
                "sysFlat": ("SysFlatA",),
                "constraintProperties": ("sysHier", "sysFlat", "financialFlowType"),
                "populationType": ("FinancialFlow",),
                "measuredProperty": ("amount",),
                "statType": ("measuredValue",),
                "measurementQualifier": ("Nominal",),
                "observationProperties": ("financingProvider", "financingRecipient"),
                "financialFlowType": ("Disbursement",),
            },
            "SV_Commitment_Obligation": {
                "typeOf": ("StatisticalVariable",),
                "sysHier": ("SysHierRoot",),
                "sysFlat": ("SysFlatB",),
                "constraintProperties": ("sysHier", "sysFlat", "financialObligationType"),
                "populationType": ("FinancialObligation",),
                "measuredProperty": ("amount",),
                "statType": ("measuredValue",),
                "measurementQualifier": ("Nominal",),
                "observationProperties": ("financingProvider", "financingRecipient"),
                "financialObligationType": ("Commitment",),
            },
            "SV_Commitment_NoFlow": {
                "typeOf": ("StatisticalVariable",),
                "sysHier": ("SysHierRoot",),
                "sysFlat": ("SysFlatB",),
                "constraintProperties": ("sysHier", "sysFlat", "financialObligationType"),
                "populationType": ("FinancialObligation",),
                "measuredProperty": ("amount",),
                "statType": ("measuredValue",),
                "measurementQualifier": ("RealValue",),
                "observationProperties": ("financingProvider", "financingRecipient"),
                "financialObligationType": ("Commitment",),
            },
        },
    )


# --- Happy path, round-trips through the loader ---


def test_export_index_round_trips_through_load_index(tmp_path: Path):
    graph = _two_system_graph()
    document = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
    )

    out = tmp_path / "index.json"
    out.write_text(json.dumps(document))
    index = load_index(out)

    assert index.stamp.graph_endpoint == "https://fake.example"
    assert index.stamp.code_counts == {"sysHier": 2, "sysFlat": 2}
    assert index.stamp.statvar_shape_counts == {"sysHier": 4, "sysFlat": 4}
    assert {row.dcid for row in index.codes} == {
        "SysHierRoot",
        "SysHierChild",
        "SysFlatA",
        "SysFlatB",
    }
    assert {row.dcid for row in index.statvars} == {
        "SV_Disbursement_Gross",
        "SV_Disbursement_NoNetting",
        "SV_Commitment_Obligation",
        "SV_Commitment_NoFlow",
    }

    child = index.code_by_dcid("SysHierChild")
    assert child is not None
    assert child.label == "child sector"
    assert child.parent == "SysHierRoot"


def test_hierarchical_code_row_carries_stripped_label_and_synonyms():
    graph = _two_system_graph()
    declaration, rows = export_code_system(graph, _HIERARCHICAL)
    by_dcid = {row["dcid"]: row for row in rows}

    assert declaration["hierarchical"] is True
    root = by_dcid["SysHierRoot"]
    assert root["label"] == "root sector"
    assert root["synonyms"] == ["Root Sector"]
    assert root["parent"] is None

    child = by_dcid["SysHierChild"]
    assert child["label"] == "child sector"
    assert child["parent"] == "SysHierRoot"


# --- The flat system's false case ---


def test_flat_system_derives_hierarchical_false_and_every_code_has_no_parent():
    graph = _two_system_graph()
    declaration, rows = export_code_system(graph, _FLAT)

    assert declaration["hierarchical"] is False
    assert all(row["parent"] is None for row in rows)


def test_hierarchical_declared_disagreement_raises_naming_both_values():
    graph = _two_system_graph()
    # sysFlat's own codes carry no specializationOf, so declaring it
    # hierarchical=True disagrees with what the data derives.
    wrong = CodeSystemConfig(
        code_system="sysFlat",
        enum_dcid="SysFlatEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="flat",
    )
    with pytest.raises(SystemExit, match=r"derived hierarchical=False.*expected value True"):
        export_code_system(graph, wrong)


# --- A parent outside the exported code system ---


def _cross_system_parent_graph() -> _FakeGraph:
    """Point `SysHierChild` at a node in the flat system.

    `SysFlatRoot` has no parent of its own.
    """
    graph = _two_system_graph()
    graph.forward_arcs["SysFlatRoot"] = {
        "typeOf": ("SysFlatEnum",),
        "classificationCode": ("1",),
        "name": ("flat root (CRS Flat)",),
        "alternateName": (),
    }
    graph.forward_arcs["SysHierChild"]["specializationOf"] = ("SysFlatRoot",)
    return graph


def test_exporter_refuses_a_parent_outside_the_exported_code_system():
    """A cross-system parent would make hierarchy walks leave their vocabulary (#852)."""
    graph = _cross_system_parent_graph()

    with pytest.raises(SystemExit, match=r"SysHierChild.*sysHier.*SysFlatRoot.*SysHierEnum"):
        export_code_system(graph, _HIERARCHICAL)


def test_exporter_accepts_a_parent_the_exporting_enumeration_returned():
    # A parent in the exported system passes.
    graph = _cross_system_parent_graph()
    graph.forward_arcs["SysHierChild"]["specializationOf"] = ("SysHierRoot",)

    _, rows = export_code_system(graph, _HIERARCHICAL)

    by_dcid = {row["dcid"]: row for row in rows}
    assert by_dcid["SysHierChild"]["parent"] == "SysHierRoot"


# --- Truncation refusals (export_code_system's own typeOf read) ---


def test_truncated_typeof_page_raises_systemexit():
    graph = _two_system_graph()
    graph.truncate = frozenset({("SysHierEnum", "typeOf")})
    with pytest.raises(SystemExit, match="truncated"):
        export_code_system(graph, _HIERARCHICAL, max_results=1)


# --- Recursive partition: never paging a truncated StatVar read ---


def _purpose_config() -> CodeSystemConfig:
    return CodeSystemConfig(
        code_system="sysPurpose",
        enum_dcid="SysPurposeEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="purpose",
    )


def _finance_config() -> CodeSystemConfig:
    return CodeSystemConfig(
        code_system="sysFinance",
        enum_dcid="SysFinanceEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="finance",
    )


def _statvar(*, purpose="P1", finance=None, population_type=None) -> dict[str, tuple[str, ...]]:
    arcs: dict[str, tuple[str, ...]] = {"sysPurpose": (purpose,)}
    if finance is not None:
        arcs["sysFinance"] = (finance,)
    if population_type is not None:
        arcs["populationType"] = (population_type,)
    return arcs


def test_partition_fits_one_response_reads_exactly_once():
    graph = _FakeGraph(
        forward_arcs={f"SV_{i}": _statvar() for i in range(3)},
    )
    purpose = _purpose_config()

    dcids, _ = enumerate_statvar_dcids(
        graph, {"sysPurpose": [{"dcid": "P1"}]}, (purpose,), _TEST_DATASETS
    )

    assert dcids == ("SV_0", "SV_1", "SV_2")
    assert graph.calls == [("P1", "sysPurpose", None)]


def test_partition_splits_once_returns_exact_union():
    # Seven P1 subjects exceed page size 5; F1 and F2 partitions contain 4 and 3.
    forward_arcs = {f"SV_F1_{i}": _statvar(finance="F1") for i in range(4)}
    forward_arcs |= {f"SV_F2_{i}": _statvar(finance="F2") for i in range(3)}
    graph = _FakeGraph(forward_arcs=forward_arcs, page_size=5)
    purpose, finance = _purpose_config(), _finance_config()
    # sysPurpose carries three codes (only P1 has StatVars; P2/P3 pad the
    # count) so it is strictly the larger, chosen system even though
    # sysFinance also carries more than one code.
    code_rows_by_system = {
        "sysPurpose": [{"dcid": "P1"}, {"dcid": "P2"}, {"dcid": "P3"}],
        "sysFinance": [{"dcid": "F1"}, {"dcid": "F2"}],
    }

    dcids, _ = enumerate_statvar_dcids(
        graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS
    )

    assert set(dcids) == set(forward_arcs)
    assert ("P1", "sysPurpose", (("sysFinance", "F1"),)) in graph.calls
    assert ("P1", "sysPurpose", (("sysFinance", "F2"),)) in graph.calls


def test_partition_splits_twice_returns_exact_union():
    # P1's 9 subjects split into F1 (6) and F2 (3) at page size 3. F1's first sample includes
    # both populationType values for the second split.
    f1_arcs = {
        "SV_F1_A": _statvar(finance="F1", population_type="PT1"),
        "SV_F1_B": _statvar(finance="F1", population_type="PT2"),
        "SV_F1_C": _statvar(finance="F1", population_type="PT1"),
        "SV_F1_D": _statvar(finance="F1", population_type="PT2"),
        "SV_F1_E": _statvar(finance="F1", population_type="PT1"),
        "SV_F1_F": _statvar(finance="F1", population_type="PT2"),
    }
    f2_arcs = {f"SV_F2_{i}": _statvar(finance="F2") for i in range(3)}
    graph = _FakeGraph(forward_arcs=f1_arcs | f2_arcs, page_size=3)
    purpose, finance = _purpose_config(), _finance_config()
    # sysPurpose carries three codes (only P1 has StatVars; P2/P3 pad the
    # count) so it is strictly the larger, chosen system even though
    # sysFinance also carries more than one code.
    code_rows_by_system = {
        "sysPurpose": [{"dcid": "P1"}, {"dcid": "P2"}, {"dcid": "P3"}],
        "sysFinance": [{"dcid": "F1"}, {"dcid": "F2"}],
    }

    dcids, _ = enumerate_statvar_dcids(
        graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS
    )

    assert set(dcids) == set(f1_arcs) | set(f2_arcs)
    assert (
        "P1",
        "sysPurpose",
        (("populationType", "PT1"), ("sysFinance", "F1")),
    ) in graph.calls
    assert (
        "P1",
        "sysPurpose",
        (("populationType", "PT2"), ("sysFinance", "F1")),
    ) in graph.calls


def test_partition_axes_exhausted_still_truncated_raises_naming_filter_path():
    # Both F1 rows have PT1, leaving a truncated leaf at page size 1 with no axis to split.
    graph = _FakeGraph(
        forward_arcs={
            "SV_F1_A": _statvar(finance="F1", population_type="PT1"),
            "SV_F1_B": _statvar(finance="F1", population_type="PT1"),
        },
        page_size=1,
    )
    purpose, finance = _purpose_config(), _finance_config()
    code_rows_by_system = {"sysPurpose": [{"dcid": "P1"}], "sysFinance": [{"dcid": "F1"}]}

    with pytest.raises(SystemExit, match="exhausted every partition property") as exc_info:
        enumerate_statvar_dcids(graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS)
    assert "F1" in str(exc_info.value)
    assert "PT1" in str(exc_info.value)


def test_partition_depth_bound_exceeded_raises():
    # A depth limit of 1 blocks the second, populationType split.
    f1_arcs = {
        "SV_F1_A": _statvar(finance="F1", population_type="PT1"),
        "SV_F1_B": _statvar(finance="F1", population_type="PT2"),
        "SV_F1_C": _statvar(finance="F1", population_type="PT1"),
        "SV_F1_D": _statvar(finance="F1", population_type="PT2"),
        "SV_F1_E": _statvar(finance="F1", population_type="PT1"),
        "SV_F1_F": _statvar(finance="F1", population_type="PT2"),
    }
    f2_arcs = {f"SV_F2_{i}": _statvar(finance="F2") for i in range(3)}
    graph = _FakeGraph(forward_arcs=f1_arcs | f2_arcs, page_size=3)
    purpose, finance = _purpose_config(), _finance_config()
    # sysPurpose carries three codes (only P1 has StatVars; P2/P3 pad the
    # count) so it is strictly the larger, chosen system even though
    # sysFinance also carries more than one code.
    code_rows_by_system = {
        "sysPurpose": [{"dcid": "P1"}, {"dcid": "P2"}, {"dcid": "P3"}],
        "sysFinance": [{"dcid": "F1"}, {"dcid": "F2"}],
    }

    with pytest.raises(SystemExit, match="max_depth=1"):
        enumerate_statvar_dcids(
            graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS, max_partition_depth=1
        )


def test_spine_chosen_by_code_count_not_table_order():
    small = CodeSystemConfig(
        code_system="sysSmall",
        enum_dcid="SysSmallEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="small",
    )
    big = CodeSystemConfig(
        code_system="sysBig",
        enum_dcid="SysBigEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="big",
    )
    graph = _FakeGraph(forward_arcs={})
    code_rows_by_system = {
        "sysSmall": [{"dcid": "S1"}],
        "sysBig": [{"dcid": "B1"}, {"dcid": "B2"}, {"dcid": "B3"}],
    }

    # sysSmall is listed first in the table but has fewer codes -- sysBig
    # must be the chosen spine, so its codes are read before sysSmall's.
    enumerate_statvar_dcids(graph, code_rows_by_system, (small, big), _TEST_DATASETS)

    assert graph.calls[:3] == [
        ("B1", "sysBig", None),
        ("B2", "sysBig", None),
        ("B3", "sysBig", None),
    ]
    assert graph.calls[3] == ("S1", "sysSmall", None)


def test_spine_coverage_verification_fires_naming_outside_subject():
    graph = _FakeGraph(
        forward_arcs={
            "SV_1": _statvar(purpose="P1", finance="F1"),
            # Carries sysFinance=F1 but not sysPurpose=P1, so it falls
            # outside the spine-enumerated set.
            "SV_OUTSIDE": {"sysFinance": ("F1",)},
        }
    )
    purpose, finance = _purpose_config(), _finance_config()
    code_rows_by_system = {"sysPurpose": [{"dcid": "P1"}], "sysFinance": [{"dcid": "F1"}]}

    with pytest.raises(SystemExit) as exc_info:
        enumerate_statvar_dcids(graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS)
    assert "SV_OUTSIDE" in str(exc_info.value)
    assert "F1" in str(exc_info.value)


def test_spine_coverage_skips_non_spine_code_too_large_to_read(capsys):
    # P1's 2 subjects fit page size 3; F1 covers 4, so its verification read is skipped.
    graph = _FakeGraph(
        forward_arcs={
            "SV_1": _statvar(purpose="P1", finance="F1"),
            "SV_2": _statvar(purpose="P1", finance="F1"),
            "SV_3": _statvar(purpose="P2", finance="F1"),
            "SV_4": _statvar(purpose="P2", finance="F1"),
        },
        page_size=3,
    )
    purpose, finance = _purpose_config(), _finance_config()
    code_rows_by_system = {"sysPurpose": [{"dcid": "P1"}], "sysFinance": [{"dcid": "F1"}]}

    dcids, _ = enumerate_statvar_dcids(
        graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS
    )

    assert dcids == ("SV_1", "SV_2")
    assert "skipped 1/1" in capsys.readouterr().err


def test_spine_coverage_skips_non_spine_code_whose_read_is_not_advancing(capsys):
    # F1's oversized verification read raises `GraphNotAdvancingError`, which counts as a skip.
    graph = _FakeGraph(
        forward_arcs={"SV_1": _statvar(purpose="P1", finance="F1")},
        truncate=frozenset({("F1", "sysFinance")}),
        not_advancing=frozenset({("F1", "sysFinance")}),
    )
    purpose, finance = _purpose_config(), _finance_config()
    code_rows_by_system = {"sysPurpose": [{"dcid": "P1"}], "sysFinance": [{"dcid": "F1"}]}

    dcids, _ = enumerate_statvar_dcids(
        graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS
    )

    assert dcids == ("SV_1",)
    assert "skipped 1/1" in capsys.readouterr().err


def test_spine_coverage_continues_past_a_skipped_code_to_a_later_one():
    # F1's read is skipped, but F2 must still find `SV_OUTSIDE`.
    # The equal code counts keep sysPurpose as the spine (first in the tuple).
    graph = _FakeGraph(
        forward_arcs={
            "SV_1": _statvar(purpose="P1", finance="F1"),
            # Carries sysFinance=F2 but no sysPurpose arc, so it falls
            # outside the spine-enumerated set.
            "SV_OUTSIDE": {"sysFinance": ("F2",)},
        },
        truncate=frozenset({("F1", "sysFinance")}),
        not_advancing=frozenset({("F1", "sysFinance")}),
    )
    purpose, finance = _purpose_config(), _finance_config()
    code_rows_by_system = {
        "sysPurpose": [{"dcid": "P1"}, {"dcid": "P2"}],
        "sysFinance": [{"dcid": "F1"}, {"dcid": "F2"}],
    }

    with pytest.raises(SystemExit) as exc_info:
        enumerate_statvar_dcids(graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS)
    assert "SV_OUTSIDE" in str(exc_info.value)
    assert "F2" in str(exc_info.value)
    # F1's skipped read happened and was not silently omitted.
    assert ("F1", "sysFinance", None) in graph.calls


def test_spine_coverage_plain_graph_read_error_propagates_uncaught():
    # Transport `GraphReadError` propagates. `GraphNotAdvancingError` alone is skipped.
    graph = _FakeGraph(
        forward_arcs={"SV_1": _statvar(purpose="P1", finance="F1")},
        read_error=frozenset({("F1", "sysFinance")}),
    )
    purpose, finance = _purpose_config(), _finance_config()
    code_rows_by_system = {"sysPurpose": [{"dcid": "P1"}], "sysFinance": [{"dcid": "F1"}]}

    with pytest.raises(GraphReadError) as exc_info:
        enumerate_statvar_dcids(graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS)
    # Neither caught nor skipped -- the plain base type, not the
    # GraphNotAdvancingError subclass the coverage check does catch.
    assert type(exc_info.value) is GraphReadError


def test_non_universal_code_system_excluded_from_spine_and_partition_axis():
    # An empty `declared_by_every_statvar_of` excludes sysModality despite its larger count and first position.
    modality = CodeSystemConfig(
        code_system="sysModality",
        enum_dcid="SysModalityEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(),
        plain_name="modality",
    )
    purpose = _purpose_config()
    forward_arcs = {
        "SV_A": _statvar(population_type="PT1"),
        "SV_B": _statvar(population_type="PT2"),
        "SV_C": _statvar(population_type="PT1"),
        "SV_D": _statvar(population_type="PT2"),
    }
    graph = _FakeGraph(forward_arcs=forward_arcs, page_size=3)
    code_rows_by_system = {
        "sysModality": [{"dcid": "M1"}, {"dcid": "M2"}, {"dcid": "M3"}],
        "sysPurpose": [{"dcid": "P1"}],
    }

    dcids, _ = enumerate_statvar_dcids(
        graph, code_rows_by_system, (modality, purpose), _TEST_DATASETS
    )

    assert set(dcids) == set(forward_arcs)
    assert graph.calls[0] == ("P1", "sysPurpose", None)
    assert any(filters and dict(filters).get("populationType") for _, _, filters in graph.calls)
    assert all("sysModality" not in dict(filters or ()) for _, _, filters in graph.calls)


# --- GraphNotAdvancingError: the same signal as truncated, never GraphReadError ---


def test_not_advancing_leaf_partitions_same_as_truncated_leaf():
    # `GraphNotAdvancingError` on the oversized P1 read triggers the same F1/F2 split as a
    # truncated page.
    forward_arcs = {f"SV_F1_{i}": _statvar(finance="F1") for i in range(4)}
    forward_arcs |= {f"SV_F2_{i}": _statvar(finance="F2") for i in range(3)}
    graph = _FakeGraph(
        forward_arcs=forward_arcs, page_size=5, not_advancing=frozenset({("P1", "sysPurpose")})
    )
    purpose, finance = _purpose_config(), _finance_config()
    code_rows_by_system = {
        "sysPurpose": [{"dcid": "P1"}, {"dcid": "P2"}, {"dcid": "P3"}],
        "sysFinance": [{"dcid": "F1"}, {"dcid": "F2"}],
    }

    dcids, _ = enumerate_statvar_dcids(
        graph, code_rows_by_system, (purpose, finance), _TEST_DATASETS
    )

    assert set(dcids) == set(forward_arcs)
    assert ("P1", "sysPurpose", (("sysFinance", "F1"),)) in graph.calls
    assert ("P1", "sysPurpose", (("sysFinance", "F2"),)) in graph.calls


def test_plain_graph_read_error_propagates_uncaught():
    graph = _FakeGraph(
        forward_arcs={"SV_1": _statvar()},
        read_error=frozenset({("P1", "sysPurpose")}),
    )
    purpose = _purpose_config()
    code_rows_by_system = {"sysPurpose": [{"dcid": "P1"}]}

    with pytest.raises(GraphReadError) as exc_info:
        enumerate_statvar_dcids(graph, code_rows_by_system, (purpose,), _TEST_DATASETS)
    # Neither caught nor partitioned -- the plain base type, not the
    # GraphNotAdvancingError subclass the recursion does catch.
    assert type(exc_info.value) is GraphReadError


def test_not_advancing_leaf_with_no_prior_observation_raises_naming_axis():
    # The only spine read fails, so no earlier leaf or sample can seed populationType.
    graph = _FakeGraph(
        forward_arcs={"SV_1": _statvar(population_type="PT1")},
        truncate=frozenset({("P1", "sysPurpose")}),
        not_advancing=frozenset({("P1", "sysPurpose")}),
    )
    purpose = _purpose_config()
    code_rows_by_system = {"sysPurpose": [{"dcid": "P1"}]}

    with pytest.raises(SystemExit, match=r"could not observe.*populationType"):
        enumerate_statvar_dcids(graph, code_rows_by_system, (purpose,), _TEST_DATASETS)


# --- Observed-axis pooling and the post-hoc coverage check ---


def test_pooled_observation_from_an_earlier_leaf_is_used_by_a_later_one():
    # P1's 2 rows seed PT1 and PT2. P2's 4-row read truncates at 3, all PT1, so pooled PT2
    # is needed to split and recover `SV_P2_4`.
    forward_arcs = {
        "SV_P1_1": _statvar(purpose="P1", population_type="PT1"),
        "SV_P1_2": _statvar(purpose="P1", population_type="PT2"),
        "SV_P2_1": _statvar(purpose="P2", population_type="PT1"),
        "SV_P2_2": _statvar(purpose="P2", population_type="PT1"),
        "SV_P2_3": _statvar(purpose="P2", population_type="PT1"),
        "SV_P2_4": _statvar(purpose="P2", population_type="PT2"),
    }
    graph = _FakeGraph(forward_arcs=forward_arcs, page_size=3)
    purpose = _purpose_config()
    code_rows_by_system = {"sysPurpose": [{"dcid": "P1"}, {"dcid": "P2"}]}

    dcids, observed_axis_usage = enumerate_statvar_dcids(
        graph, code_rows_by_system, (purpose,), _TEST_DATASETS
    )

    assert set(dcids) == set(forward_arcs)
    assert "SV_P2_4" in dcids
    assert observed_axis_usage == {"populationType": ("PT1", "PT2")}


def test_verify_axis_coverage_raises_on_a_value_never_partitioned_on():
    statvars = [{"dcid": "SV_X", "population_type": "PT_UNSEEN"}]
    observed_axis_usage = {"populationType": ("PT1", "PT2")}

    with pytest.raises(SystemExit, match=r"SV_X.*PT_UNSEEN.*populationType"):
        _verify_axis_coverage(statvars, observed_axis_usage)


def test_verify_axis_coverage_passes_silently_when_every_value_was_used():
    statvars = [
        {"dcid": "SV_X", "population_type": "PT1"},
        {"dcid": "SV_Y", "population_type": "PT2"},
    ]
    observed_axis_usage = {"populationType": ("PT1", "PT2")}

    assert _verify_axis_coverage(statvars, observed_axis_usage) is None


def test_verify_axis_coverage_skips_an_axis_never_used_to_partition():
    # No leaf used this axis, so it could not exclude a row.
    statvars = [{"dcid": "SV_X", "population_type": "PT_ANYTHING"}]

    assert _verify_axis_coverage(statvars, {}) is None


# --- Undeclared constraint properties are absent, never a placeholder ---


def test_undeclared_constraint_property_is_absent_not_a_placeholder():
    graph = _two_system_graph()
    rows = export_statvars(graph, ("SV_Disbursement_Gross", "SV_Disbursement_NoNetting"))
    by_dcid = {row["dcid"]: row for row in rows}

    with_netting = by_dcid["SV_Disbursement_Gross"]
    assert with_netting["constraint_properties"]["nettingBasis"] == "Gross"

    without_netting = by_dcid["SV_Disbursement_NoNetting"]
    assert "nettingBasis" not in without_netting["constraint_properties"]
    assert without_netting["constraint_properties"] == {
        "sysHier": "SysHierChild",
        "sysFlat": "SysFlatA",
        "financialFlowType": "Disbursement",
    }


def test_declared_but_unreadable_constraint_property_raises_systemexit():
    # `nettingBasis` is declared, but its forward read has no value.
    graph = _two_system_graph()
    del graph.forward_arcs["SV_Disbursement_Gross"]["nettingBasis"]
    with pytest.raises(SystemExit, match=r"SV_Disbursement_Gross.*'nettingBasis'.*no value"):
        export_statvars(graph, ("SV_Disbursement_Gross",))


# --- a constraint value the export refuses rather than writing as a
# plain dcid indistinguishable from a real one ---


def test_unresolved_constraint_value_raises_systemexit():
    # The arc exists but resolves to `UNRESOLVED_NODE`, so export cannot write a valid dcid.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["nettingBasis"] = (UNRESOLVED_NODE,)
    with pytest.raises(
        SystemExit, match=r"SV_Disbursement_Gross.*'nettingBasis'.*could not resolve"
    ):
        export_statvars(graph, ("SV_Disbursement_Gross",))


def test_provisional_node_typed_constraint_value_raises_systemexit():
    # `DanglingStub` resolves as a node whose only type is `ProvisionalNode`.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["nettingBasis"] = ("DanglingStub",)
    graph.forward_arcs["DanglingStub"] = {"typeOf": ("ProvisionalNode",)}
    with pytest.raises(
        SystemExit,
        match=r"SV_Disbursement_Gross.*'nettingBasis'.*'DanglingStub'.*ProvisionalNode",
    ):
        export_statvars(graph, ("SV_Disbursement_Gross",))


def test_dual_typed_provisional_node_constraint_value_is_not_refused():
    # A real type alongside `ProvisionalNode` keeps the node valid.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["nettingBasis"] = ("NowRealCode",)
    graph.forward_arcs["NowRealCode"] = {"typeOf": ("ProvisionalNode", "SomeRealType")}

    rows = export_statvars(graph, ("SV_Disbursement_Gross",))

    assert rows[0]["constraint_properties"]["nettingBasis"] == "NowRealCode"


def test_provisional_node_check_runs_once_batched_across_every_row():
    # The only dangling value is on the second row.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_NoNetting"]["nettingBasis"] = ("DanglingStub",)
    graph.forward_arcs["SV_Disbursement_NoNetting"]["constraintProperties"] = (
        "sysHier",
        "sysFlat",
        "financialFlowType",
        "nettingBasis",
    )
    graph.forward_arcs["DanglingStub"] = {"typeOf": ("ProvisionalNode",)}

    with pytest.raises(SystemExit, match=r"SV_Disbursement_NoNetting.*ProvisionalNode"):
        export_statvars(graph, ("SV_Disbursement_Gross", "SV_Disbursement_NoNetting"))


def test_constraint_value_absent_from_graph_raises_systemexit():
    # The value dcid is absent from the graph, unlike a held `ProvisionalNode` stub.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["nettingBasis"] = ("TrulyAbsentDcid",)
    with pytest.raises(
        SystemExit,
        match=r"SV_Disbursement_Gross.*'nettingBasis'.*'TrulyAbsentDcid'.*holds no node",
    ):
        export_statvars(graph, ("SV_Disbursement_Gross",))


def test_constraint_value_with_no_typeof_arc_raises_systemexit():
    # The graph holds `UntypedStub`, but its node has no `typeOf` arc.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["nettingBasis"] = ("UntypedStub",)
    graph.forward_arcs["UntypedStub"] = {"classificationCode": ("999",)}
    with pytest.raises(
        SystemExit,
        match=r"SV_Disbursement_Gross.*'nettingBasis'.*'UntypedStub'.*no typeOf arc",
    ):
        export_statvars(graph, ("SV_Disbursement_Gross",))


# StatVar rows must carry their own `StatisticalVariable` type.


def test_statvar_missing_statistical_variable_type_raises_systemexit():
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["typeOf"] = ("SomeOtherType",)
    with pytest.raises(
        SystemExit,
        match=r"SV_Disbursement_Gross.*typeOf arc.*'SomeOtherType'.*does not include "
        "StatisticalVariable",
    ):
        export_statvars(graph, ("SV_Disbursement_Gross",))


def test_statvar_with_no_typeof_arc_at_all_raises_systemexit():
    # The fake returns an empty type set for a node without `typeOf`.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["typeOf"] = ()
    with pytest.raises(SystemExit, match=r"SV_Disbursement_Gross.*typeOf arc.*\(\)"):
        export_statvars(graph, ("SV_Disbursement_Gross",))


def test_statvar_typed_provisional_alongside_statistical_variable_is_not_refused():
    # The real type keeps the StatVar valid alongside `ProvisionalNode`.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["typeOf"] = (
        "StatisticalVariable",
        "ProvisionalNode",
    )

    rows = export_statvars(graph, ("SV_Disbursement_Gross",))

    assert rows[0]["dcid"] == "SV_Disbursement_Gross"


# Shape fields and declared constraints allow one distinct value. The read boundary collapses
# repeated identical arcs before `export_statvars` checks them.


@pytest.mark.parametrize(
    "shape_property",
    ["populationType", "measuredProperty", "statType", "measurementQualifier"],
)
def test_multi_valued_shape_field_raises_systemexit(shape_property: str):
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"][shape_property] = ("ValueA", "ValueB")
    with pytest.raises(
        SystemExit,
        match=f"SV_Disbursement_Gross.*{shape_property!r}.*2 bound values.*'ValueA', 'ValueB'",
    ):
        export_statvars(graph, ("SV_Disbursement_Gross",))


def test_multi_valued_measurement_denominator_raises_systemexit():
    # The base fixture has no `measurementDenominator` value.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["measurementDenominator"] = ("DenomA", "DenomB")
    with pytest.raises(
        SystemExit,
        match=r"SV_Disbursement_Gross.*'measurementDenominator'.*2 bound values",
    ):
        export_statvars(graph, ("SV_Disbursement_Gross",))


def test_duplicate_identical_shape_value_is_not_refused():
    # Identical duplicate arcs collapse to one value at the read boundary.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["populationType"] = (
        "FinancialFlow",
        "FinancialFlow",
    )

    rows = export_statvars(graph, ("SV_Disbursement_Gross",))

    assert rows[0]["population_type"] == "FinancialFlow"


def test_multi_valued_declared_constraint_property_raises_systemexit():
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["nettingBasis"] = ("Gross", "Net")
    graph.forward_arcs["Net"] = {"typeOf": ("NettingBasisEnum",)}
    with pytest.raises(
        SystemExit,
        match=r"SV_Disbursement_Gross.*'nettingBasis'.*2 bound values.*'Gross', 'Net'",
    ):
        export_statvars(graph, ("SV_Disbursement_Gross",))


def test_duplicate_identical_declared_constraint_value_is_not_refused():
    # Duplicate-arc control for a declared constraint value.
    graph = _two_system_graph()
    graph.forward_arcs["SV_Disbursement_Gross"]["nettingBasis"] = ("Gross", "Gross")

    rows = export_statvars(graph, ("SV_Disbursement_Gross",))

    assert rows[0]["constraint_properties"]["nettingBasis"] == "Gross"


# --- Label parenthetical stripping ---


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("standard grant (CRS Finance)", "standard grant"),
        ("imputed multilateral ODA (DAC 2a aid type)", "imputed multilateral ODA"),
        ("basic sanitation", "basic sanitation"),
        ("foo (interior) bar", "foo (interior) bar"),
        ("malaria (severe)", "malaria (severe)"),
        (
            "project-type intervention (group) (CRS Modality)",
            "project-type intervention (group)",
        ),
    ],
)
def test_strip_system_suffix(name: str, expected: str):
    assert strip_system_suffix(name) == expected


def test_alternate_name_equal_to_label_is_dropped_from_synonyms():
    graph = _FakeGraph(
        forward_arcs={
            "SysHierRoot": {
                "typeOf": ("SysHierEnum",),
                "classificationCode": ("100",),
                "name": ("root sector (CRS Hier)",),
                "alternateName": ("root sector", "Root Alias"),
            },
        }
    )
    single_code_config = CodeSystemConfig(
        code_system="sysHier",
        enum_dcid="SysHierEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="hier",
    )
    _, rows = export_code_system(graph, single_code_config)
    assert rows[0]["synonyms"] == ["Root Alias"]


@pytest.mark.parametrize(
    ("dcid", "graph_short_names", "expected"),
    [
        ("CRSTypeOfFlow10", (), "ODA"),
        ("CRSTypeOfFlow10", ("graph ODA",), "graph ODA"),
        ("SysHierRoot", (), None),
    ],
)
def test_code_row_short_name_is_the_graph_arc_else_the_authored_name(
    dcid: str, graph_short_names: tuple[str, ...], expected: str | None
):
    graph = _FakeGraph(
        forward_arcs={
            dcid: {
                "typeOf": ("SysHierEnum",),
                "classificationCode": ("10",),
                "name": ("a flow (CRS Hier)",),
                "shortName": graph_short_names,
            },
        }
    )
    config = CodeSystemConfig(
        code_system="sysHier",
        enum_dcid="SysHierEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="hier",
    )
    _, rows = export_code_system(graph, config)
    assert rows[0]["short_name"] == expected


# --- content_hash stability ---


def test_content_hash_identical_across_runs_and_differs_on_a_row_change():
    graph = _two_system_graph()
    first = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
    )
    later = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=datetime(2026, 9, 9, 13, 0, 0, tzinfo=UTC),
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
    )
    assert first["stamp"]["exported_at"] != later["stamp"]["exported_at"]
    assert first["stamp"]["content_hash"] == later["stamp"]["content_hash"]

    changed_graph = _two_system_graph()
    changed_graph.forward_arcs["SysHierRoot"]["alternateName"] = ("Root Sector", "Extra Alias")
    changed = export_index(
        changed_graph,
        coverage=_covered(changed_graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
    )
    assert changed["stamp"]["content_hash"] != first["stamp"]["content_hash"]


# --- A dcid enumerated but unfetchable ---


def test_dcid_enumerated_via_reverse_arc_but_missing_from_forward_read_raises():
    graph = _two_system_graph()
    # The reverse read finds `DanglingGhostDcid`, but the forward graph has no node for it.
    with pytest.raises(SystemExit, match="holds no node for DanglingGhostDcid"):
        export_statvars(graph, ("SV_Disbursement_Gross", "DanglingGhostDcid"))


# --- Measure axis values ---

_MEASURE_AXES_FOR_TEST = (
    MeasureAxis(
        axis="populationType",
        kind="shape",
        role="free",
        prefer=("FinancialFlow", "FinancialObligation"),
    ),
    MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=("amount",)),
    MeasureAxis(
        axis="measurementQualifier", kind="shape", role="free", prefer=("Nominal", "RealValue")
    ),
    MeasureAxis(axis="financialFlowType", kind="constraint", role="free", prefer=("Disbursement",)),
    MeasureAxis(
        axis="financialObligationType", kind="constraint", role="free", prefer=("Commitment",)
    ),
    MeasureAxis(axis="nettingBasis", kind="constraint", role="free", prefer=("Gross", "Net")),
)

# The commitments label covers both rows, which differ only by qualifier. The disbursement label
# covers the row without `nettingBasis`; the gross label covers the other.
_MEASURE_PARTITION_LABELS_FOR_TEST = (
    MeasurePartitionLabel(
        axis_values=(
            ("financialObligationType", "Commitment"),
            ("measuredProperty", "amount"),
            ("populationType", "FinancialObligation"),
        ),
        label="commitments",
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amount"),
            ("populationType", "FinancialFlow"),
        ),
        label="disbursements",
    ),
    MeasurePartitionLabel(
        axis_values=(
            ("financialFlowType", "Disbursement"),
            ("measuredProperty", "amount"),
            ("nettingBasis", "Gross"),
            ("populationType", "FinancialFlow"),
        ),
        label="gross disbursements",
        named_values=(("sysFlat", "SysFlatA"),),
    ),
)
_MEASURE_NOUNS_FOR_TEST = (
    MeasurePartitionLabel(
        axis_values=(("populationType", "EconomicProduction"), ("measuredProperty", "valueAdded")),
        label="GDP",
    ),
)


def _measure_label_nodes(*, include_net: bool = True) -> dict[str, dict[str, tuple[str, ...]]]:
    """Add labels to observed measure values; these values have no code-system arcs.

    `include_net=False` omits `Net`, which no fixture StatVar observes.
    """
    nodes: dict[str, dict[str, tuple[str, ...]]] = {
        "FinancialFlow": {"name": ("financial flow",)},
        "FinancialObligation": {"name": ("financial obligation",)},
        "amount": {"name": ("amount",)},
        "Nominal": {"name": ("Nominal",)},
        "RealValue": {"name": ("RealValue",)},
        "InflationAdjusted": {"name": ("InflationAdjusted",)},
        "PurchasingPowerParity": {"name": ("PurchasingPowerParity",)},
        "Disbursement": {"name": ("disbursement",)},
        "Gross": {"name": ("gross",)},
        "Commitment": {"name": ("commitment",)},
    }
    if include_net:
        nodes["Net"] = {"name": ("net",), "alternateName": ("net", "settlement basis")}
    return nodes


def _two_system_graph_with_measure_labels(*, include_net: bool = True) -> _FakeGraph:
    """Add labels per node while preserving constraint values' `typeOf` arcs."""
    graph = _two_system_graph()
    for dcid, arcs in _measure_label_nodes(include_net=include_net).items():
        graph.forward_arcs[dcid] = graph.forward_arcs.get(dcid, {}) | arcs
    return graph


def test_export_measure_values_reads_only_the_values_statvars_observe():
    graph = _two_system_graph_with_measure_labels()
    statvar_dcids = (
        "SV_Disbursement_Gross",
        "SV_Disbursement_NoNetting",
        "SV_Commitment_Obligation",
        "SV_Commitment_NoFlow",
    )
    statvars = export_statvars(graph, statvar_dcids)

    rows = export_measure_values(graph, statvars, _MEASURE_AXES_FOR_TEST, declared={})
    by_axis: dict[str, set[str]] = {}
    for row in rows:
        by_axis.setdefault(row["axis"], set()).add(row["dcid"])

    assert by_axis == {
        "populationType": {"FinancialFlow", "FinancialObligation"},
        "measuredProperty": {"amount"},
        "measurementQualifier": {"Nominal", "RealValue"},
        "financialFlowType": {"Disbursement"},
        "financialObligationType": {"Commitment"},
        # No StatVar observes `nettingBasis: Net`.
        "nettingBasis": {"Gross"},
    }


def test_export_measure_values_carries_stripped_label_and_synonyms():
    graph = _two_system_graph_with_measure_labels()
    statvar_dcids = ("SV_Disbursement_Gross",)
    statvars = export_statvars(graph, statvar_dcids)

    rows = export_measure_values(graph, statvars, _MEASURE_AXES_FOR_TEST)
    by_dcid = {row["dcid"]: row for row in rows}

    gross = by_dcid["Gross"]
    assert gross["label"] == "gross"
    assert gross["synonyms"] == []
    assert gross["short_name"] is None


def test_export_measure_values_stamps_matchable_from_the_authored_table():
    graph = _two_system_graph_with_measure_labels()
    statvar_dcids = ("SV_Disbursement_Gross",)
    statvars = export_statvars(graph, statvar_dcids)

    rows = export_measure_values(graph, statvars, _MEASURE_AXES_FOR_TEST)
    by_dcid = {row["dcid"]: row for row in rows}

    # `_MEASURE_VALUE_TEXT` marks `amount` unmatchable; other observed values are matchable.
    assert by_dcid["amount"]["matchable"] is False
    assert by_dcid["Gross"]["matchable"] is True
    assert by_dcid["Nominal"]["matchable"] is True


def test_export_measure_values_never_reads_a_value_no_statvar_carries():
    # The graph has no `Net` node, and no StatVar observes that value.
    graph = _two_system_graph_with_measure_labels(include_net=False)
    statvar_dcids = (
        "SV_Disbursement_Gross",
        "SV_Disbursement_NoNetting",
        "SV_Commitment_Obligation",
        "SV_Commitment_NoFlow",
    )
    statvars = export_statvars(graph, statvar_dcids)

    rows = export_measure_values(graph, statvars, _MEASURE_AXES_FOR_TEST)
    netting_values = {row["dcid"] for row in rows if row["axis"] == "nettingBasis"}
    assert netting_values == {"Gross"}


def test_export_measure_values_exports_declared_price_bases_no_statvar_carries():
    # No fixture StatVar carries PPP or InflationAdjusted. The declared values are exported
    # anyway, with InflationAdjusted filed under RealValue's reading.
    graph = _two_system_graph_with_measure_labels()
    statvars = export_statvars(graph, ("SV_Disbursement_Gross",))

    rows = export_measure_values(graph, statvars, _MEASURE_AXES_FOR_TEST)
    by_dcid = {row["dcid"]: row for row in rows if row["axis"] == "measurementQualifier"}

    assert by_dcid.keys() >= {"InflationAdjusted", "PurchasingPowerParity", "RealValue"}
    assert by_dcid["InflationAdjusted"]["same_reading_as"] == "RealValue"
    assert by_dcid["RealValue"]["same_reading_as"] is None
    assert "PPP" in by_dcid["PurchasingPowerParity"]["synonyms"]


def test_export_index_refuses_an_alias_whose_canonical_value_is_not_exported(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = _two_system_graph_with_measure_labels()
    monkeypatch.setitem(
        _MEASURE_VALUE_TEXT,
        "Gross",
        _MeasureValueText(same_reading_as="NotAMeasureValue"),
    )
    with pytest.raises(SystemExit, match=r"does not load.*NotAMeasureValue"):
        export_index(
            graph,
            coverage=_covered(graph),
            datasets=_TEST_DATASETS,
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(_HIERARCHICAL, _FLAT),
            measure_axes=_MEASURE_AXES_FOR_TEST,
            measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
        )


def test_export_measure_values_raises_on_a_value_with_no_name_arc():
    graph = _two_system_graph_with_measure_labels()
    del graph.forward_arcs["Gross"]["name"]
    statvars = export_statvars(graph, ("SV_Disbursement_Gross",))

    with pytest.raises(SystemExit, match="Gross carries no name arc"):
        export_measure_values(graph, statvars, _MEASURE_AXES_FOR_TEST)


def test_export_measure_values_raises_on_a_dcid_the_graph_holds_no_node_for():
    # `amount` is a measuredProperty value, not a constraint value.
    # Removing its node reaches this check before constraint-type validation.
    graph = _two_system_graph_with_measure_labels()
    del graph.forward_arcs["amount"]
    statvars = export_statvars(graph, ("SV_Disbursement_Gross",))

    with pytest.raises(SystemExit, match="no node for it"):
        export_measure_values(graph, statvars, _MEASURE_AXES_FOR_TEST)


def test_export_index_empty_measure_axes_exports_no_measure_values():
    # Empty `measure_axes` skips measure reads, even when the graph has no measure-value nodes.
    graph = _two_system_graph()
    document = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        measure_partition_labels=(),
    )
    assert document["measure_axes"] == []
    assert document["measure_values"] == []
    assert document["measure_partition_labels"] == []


def test_export_index_wires_measure_axes_into_the_document(tmp_path):
    graph = _two_system_graph_with_measure_labels()
    document = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=_MEASURE_AXES_FOR_TEST,
        measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
        measure_nouns=_MEASURE_NOUNS_FOR_TEST,
    )

    assert document["measure_axes"] == [
        {
            "axis": "populationType",
            "kind": "shape",
            "role": "free",
            "prefer": ["FinancialFlow", "FinancialObligation"],
        },
        {"axis": "measuredProperty", "kind": "shape", "role": "basis", "prefer": ["amount"]},
        {
            "axis": "measurementQualifier",
            "kind": "shape",
            "role": "free",
            "prefer": ["Nominal", "RealValue"],
        },
        {
            "axis": "financialFlowType",
            "kind": "constraint",
            "role": "free",
            "prefer": ["Disbursement"],
        },
        {
            "axis": "financialObligationType",
            "kind": "constraint",
            "role": "free",
            "prefer": ["Commitment"],
        },
        {"axis": "nettingBasis", "kind": "constraint", "role": "free", "prefer": ["Gross", "Net"]},
    ]
    assert {row["dcid"] for row in document["measure_values"]} >= {"FinancialFlow", "Gross"}
    assert document["measure_partition_labels"] == [
        {
            "axis_values": [
                ["financialObligationType", "Commitment"],
                ["measuredProperty", "amount"],
                ["populationType", "FinancialObligation"],
            ],
            "label": "commitments",
            "named_values": [],
        },
        {
            "axis_values": [
                ["financialFlowType", "Disbursement"],
                ["measuredProperty", "amount"],
                ["populationType", "FinancialFlow"],
            ],
            "label": "disbursements",
            "named_values": [],
        },
        {
            "axis_values": [
                ["financialFlowType", "Disbursement"],
                ["measuredProperty", "amount"],
                ["nettingBasis", "Gross"],
                ["populationType", "FinancialFlow"],
            ],
            "label": "gross disbursements",
            "named_values": [["sysFlat", "SysFlatA"]],
        },
    ]
    assert document["measure_nouns"] == [
        {
            "axis_values": [
                ["populationType", "EconomicProduction"],
                ["measuredProperty", "valueAdded"],
            ],
            "label": "GDP",
            "named_values": [],
        }
    ]
    # The loader reads both blocks back into the authored values.
    path = tmp_path / "index.json"
    path.write_text(json.dumps(document))
    measure_index = load_measure_axes(path)
    assert measure_index.partition_labels == _MEASURE_PARTITION_LABELS_FOR_TEST
    assert measure_index.nouns == _MEASURE_NOUNS_FOR_TEST


def test_content_hash_moves_when_only_measure_axes_change():
    # Changing only `measure_axes` changes the content hash.
    graph = _two_system_graph_with_measure_labels()
    with_axes = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=_MEASURE_AXES_FOR_TEST,
        measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
    )
    without_axes = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
    )
    assert with_axes["stamp"]["content_hash"] != without_axes["stamp"]["content_hash"]


# --- The measure-partition-label guard ---

_GUARD_AXES = (
    MeasureAxis(
        axis="financialFlowType", kind="constraint", role="basis", prefer=("Disbursement",)
    ),
    MeasureAxis(axis="nettingBasis", kind="constraint", role="free", prefer=("Gross", "Net")),
)


def _guard_row(*, financial_flow_type: str | None = None, netting_basis: str | None = None) -> dict:
    constraint_properties = {}
    if financial_flow_type is not None:
        constraint_properties["financialFlowType"] = financial_flow_type
    if netting_basis is not None:
        constraint_properties["nettingBasis"] = netting_basis
    return {"dcid": "SV_X", "constraint_properties": constraint_properties}


def test_guard_refuses_when_an_exhibited_partition_matches_no_authored_label():
    rows = [_guard_row(financial_flow_type="Disbursement", netting_basis="Gross")]
    labels = (
        MeasurePartitionLabel(axis_values=(("nettingBasis", "Net"),), label="net disbursements"),
    )

    with pytest.raises(SystemExit, match="Disbursement") as exc_info:
        _verify_measure_partition_labels(rows, _GUARD_AXES, labels)
    assert "Gross" in str(exc_info.value)


def test_guard_refuses_when_two_labels_tie_on_an_exhibited_partition():
    rows = [_guard_row(financial_flow_type="Disbursement", netting_basis="Gross")]
    labels = (
        MeasurePartitionLabel(
            axis_values=(("financialFlowType", "Disbursement"),), label="disbursements"
        ),
        MeasurePartitionLabel(axis_values=(("nettingBasis", "Gross"),), label="gross"),
    )

    with pytest.raises(SystemExit, match="Disbursement"):
        _verify_measure_partition_labels(rows, _GUARD_AXES, labels)


def test_guard_passes_when_every_exhibited_partition_has_exactly_one_label():
    rows = [
        _guard_row(financial_flow_type="Disbursement", netting_basis="Gross"),
        _guard_row(financial_flow_type="Disbursement", netting_basis="Net"),
    ]
    labels = (
        MeasurePartitionLabel(
            axis_values=(("financialFlowType", "Disbursement"), ("nettingBasis", "Gross")),
            label="gross disbursements",
        ),
        MeasurePartitionLabel(
            axis_values=(("financialFlowType", "Disbursement"), ("nettingBasis", "Net")),
            label="net disbursements",
        ),
    )

    _verify_measure_partition_labels(rows, _GUARD_AXES, labels)  # raises nothing


def test_guard_drops_a_none_valued_axis_before_matching_rather_than_refusing_on_it_alone():
    # nettingBasis is one of _GUARD_AXES but this row declares no value for
    # it at all -- the guard must not refuse just because that axis pair is
    # absent, only because the pairs the row does carry go unlabelled.
    rows = [_guard_row(financial_flow_type="Disbursement")]
    labels = (
        MeasurePartitionLabel(
            axis_values=(("financialFlowType", "Disbursement"),), label="disbursements"
        ),
    )

    _verify_measure_partition_labels(rows, _GUARD_AXES, labels)  # raises nothing


def test_guard_names_every_offending_partition_not_just_the_first():
    rows = [
        _guard_row(financial_flow_type="Disbursement", netting_basis="Gross"),
        _guard_row(financial_flow_type="Disbursement", netting_basis="Net"),
    ]

    with pytest.raises(SystemExit) as exc_info:
        _verify_measure_partition_labels(rows, _GUARD_AXES, labels=())
    message = str(exc_info.value)
    assert "Gross" in message
    assert "Net" in message


def test_guard_is_a_noop_when_no_measure_axes_are_declared():
    rows = [_guard_row(financial_flow_type="Disbursement", netting_basis="Gross")]

    _verify_measure_partition_labels(rows, axes=(), labels=())  # raises nothing


def test_export_index_refuses_when_the_guard_fails():
    graph = _two_system_graph_with_measure_labels()

    with pytest.raises(SystemExit, match="resolve to no single authored label"):
        export_index(
            graph,
            coverage=_covered(graph),
            datasets=_TEST_DATASETS,
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(_HIERARCHICAL, _FLAT),
            measure_axes=_MEASURE_AXES_FOR_TEST,
            measure_partition_labels=(),
        )


def test_committed_packaged_index_passes_the_measure_partition_label_guard():
    source = default_index_path()
    payload = json.loads(gzip.decompress(source.read_bytes()))
    measure_index = load_measure_axes(source)

    _verify_measure_partition_labels(
        payload["statvars"], measure_index.axes, measure_index.partition_labels
    )  # raises nothing
    _verify_measure_nouns(payload["statvars"], measure_index.nouns)  # raises nothing
    _verify_measure_label_named_values(
        payload["statvars"], payload["codes"], measure_index.axes, measure_index.partition_labels
    )  # raises nothing


def test_a_statvar_row_matching_a_measure_noun_aborts_the_export():
    """A row under a partition label that a noun also names would read two nouns."""
    row = {
        "dcid": "sv_gdp",
        "population_type": "EconomicProduction",
        "measured_property": "valueAdded",
        "stat_type": "measuredValue",
        "constraint_properties": {},
    }
    with pytest.raises(SystemExit, match="sv_gdp"):
        _verify_measure_nouns([row], _MEASURE_NOUNS_FOR_TEST)
    _verify_measure_nouns([{**row, "population_type": "FinancialFlow"}], _MEASURE_NOUNS_FOR_TEST)


@pytest.mark.parametrize(
    "named_value",
    [
        pytest.param(("sysFlatt", "SysFlatA"), id="misspelt-property"),
        pytest.param(("sysFlat", "SysFlatC"), id="unexported-code"),
        pytest.param(("sysFlat", "SysFlatB"), id="no-row-under-the-partition"),
    ],
)
def test_a_measure_label_named_value_no_partition_row_carries_aborts_the_export(named_value):
    """A named value must be an exported code that a row under the label's partition
    carries; a typo would silence nothing."""
    row = {
        "dcid": "sv_gross",
        "population_type": "FinancialFlow",
        "measured_property": "amount",
        "stat_type": "measuredValue",
        "measurement_qualifier": "Nominal",
        "constraint_properties": {
            "financialFlowType": "Disbursement",
            "nettingBasis": "Gross",
            "sysFlat": "SysFlatA",
        },
    }
    codes = [
        {"dcid": "SysFlatA", "code_system": "sysFlat"},
        {"dcid": "SysFlatB", "code_system": "sysFlat"},
    ]
    (label,) = [
        label
        for label in _MEASURE_PARTITION_LABELS_FOR_TEST
        if label.label == "gross disbursements"
    ]
    _verify_measure_label_named_values([row], codes, _MEASURE_AXES_FOR_TEST, [label])
    wrong = dataclasses.replace(label, named_values=(named_value,))
    with pytest.raises(SystemExit, match=named_value[1]):
        _verify_measure_label_named_values([row], codes, _MEASURE_AXES_FOR_TEST, [wrong])


# --- plain_name flows from the code-system table into the declaration row ---


def test_export_index_stamps_plain_name_from_the_code_system_table():
    # The configured names are `hier` and `flat`.
    graph = _two_system_graph()
    document = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
    )
    by_system = {row["code_system"]: row for row in document["declarations"]}
    assert by_system["sysHier"]["plain_name"] == _HIERARCHICAL.plain_name
    assert by_system["sysFlat"]["plain_name"] == _FLAT.plain_name


# --- CodeSystemDeclaration.default (issue #968) ---

_HIERARCHICAL_WITH_DEFAULT = CodeSystemConfig(
    code_system="sysHier",
    enum_dcid="SysHierEnum",
    expected_hierarchical=True,
    disjoint_from_children=True,
    summable=True,
    absent_reading="not_declared",
    declared_by_every_statvar_of=(_TEST_PROVENANCE,),
    plain_name="hier",
    default=DeclaredDefault(code="SysHierChild", when_bound_under=("SysFlatA",)),
)


def test_export_index_stamps_a_declared_default_and_it_round_trips(tmp_path: Path):
    graph = _two_system_graph()
    document = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL_WITH_DEFAULT, _FLAT),
        measure_axes=(),
    )
    by_system = {row["code_system"]: row for row in document["declarations"]}
    assert by_system["sysHier"]["default"] == {
        "code": "SysHierChild",
        "when_bound_under": ["SysFlatA"],
        "when_named": [],
    }
    # A system with no configured default stamps None, not a missing key.
    assert by_system["sysFlat"]["default"] is None

    out = tmp_path / "with_default.json"
    out.write_text(json.dumps(document))
    index = load_index(out)
    default = index.declarations["sysHier"].default
    assert default is not None
    assert default.code == "SysHierChild"
    assert default.when_bound_under == ("SysFlatA",)
    assert default.when_named == ()
    assert index.declarations["sysFlat"].default is None


_HIERARCHICAL_WITH_NAMED_DEFAULT = CodeSystemConfig(
    code_system="sysHier",
    enum_dcid="SysHierEnum",
    expected_hierarchical=True,
    disjoint_from_children=True,
    summable=True,
    absent_reading="not_declared",
    declared_by_every_statvar_of=(_TEST_PROVENANCE,),
    plain_name="hier",
    default=DeclaredDefault(
        code="SysHierChild", when_bound_under=("SysFlatA",), when_named=("aid",)
    ),
)


def test_export_index_stamps_when_named_and_it_round_trips(tmp_path: Path):
    graph = _two_system_graph()
    document = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL_WITH_NAMED_DEFAULT, _FLAT),
        measure_axes=(),
    )
    by_system = {row["code_system"]: row for row in document["declarations"]}
    assert by_system["sysHier"]["default"]["when_named"] == ["aid"]

    out = tmp_path / "with_named_default.json"
    out.write_text(json.dumps(document))
    index = load_index(out)
    assert index.declarations["sysHier"].default.when_named == ("aid",)


def test_export_index_refuses_a_declared_default_when_named_entry_that_is_not_lowercase():
    graph = _two_system_graph()
    wrong = CodeSystemConfig(
        code_system="sysHier",
        enum_dcid="SysHierEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="hier",
        default=DeclaredDefault(
            code="SysHierChild", when_bound_under=("SysFlatA",), when_named=("Aid",)
        ),
    )

    with pytest.raises(SystemExit, match="when_named entry 'Aid' must be a non-empty lowercase"):
        export_index(
            graph,
            coverage=_covered(graph),
            datasets=_TEST_DATASETS,
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
        )


def test_export_index_refuses_a_declared_default_when_named_entry_that_is_empty():
    graph = _two_system_graph()
    wrong = CodeSystemConfig(
        code_system="sysHier",
        enum_dcid="SysHierEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="hier",
        default=DeclaredDefault(
            code="SysHierChild", when_bound_under=("SysFlatA",), when_named=("",)
        ),
    )

    with pytest.raises(SystemExit, match="when_named entry '' must be a non-empty lowercase"):
        export_index(
            graph,
            coverage=_covered(graph),
            datasets=_TEST_DATASETS,
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
        )


def test_export_index_refuses_a_declared_default_when_named_entry_that_is_null():
    # The validator returns an index so a null entry remains distinguishable from no error.
    graph = _two_system_graph()
    wrong = CodeSystemConfig(
        code_system="sysHier",
        enum_dcid="SysHierEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="hier",
        default=DeclaredDefault(
            code="SysHierChild",
            when_bound_under=("SysFlatA",),
            when_named=(None,),  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(SystemExit, match="when_named entry None must be a non-empty lowercase"):
        export_index(
            graph,
            coverage=_covered(graph),
            datasets=_TEST_DATASETS,
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
        )


def test_export_index_refuses_a_declared_default_code_outside_its_own_system():
    graph = _two_system_graph()
    wrong = CodeSystemConfig(
        code_system="sysHier",
        enum_dcid="SysHierEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="hier",
        # `SysFlatA` belongs to `sysFlat`.
        default=DeclaredDefault(code="SysFlatA", when_bound_under=("SysFlatA",)),
    )

    with pytest.raises(SystemExit, match="declared default code 'SysFlatA' is not a row"):
        export_index(
            graph,
            coverage=_covered(graph),
            datasets=_TEST_DATASETS,
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
        )


def test_export_index_refuses_a_declared_default_scoped_under_an_unknown_dcid():
    graph = _two_system_graph()
    wrong = CodeSystemConfig(
        code_system="sysHier",
        enum_dcid="SysHierEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="hier",
        default=DeclaredDefault(code="SysHierChild", when_bound_under=("NotARealDcid",)),
    )

    with pytest.raises(SystemExit, match=r"NotARealDcid.*not a row of any exported code system"):
        export_index(
            graph,
            coverage=_covered(graph),
            datasets=_TEST_DATASETS,
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
        )


def test_export_index_refuses_a_declared_default_scoped_under_its_own_code_system():
    graph = _two_system_graph()
    wrong = CodeSystemConfig(
        code_system="sysHier",
        enum_dcid="SysHierEnum",
        expected_hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        declared_by_every_statvar_of=(_TEST_PROVENANCE,),
        plain_name="hier",
        # `SysHierRoot` belongs to the default's own system.
        default=DeclaredDefault(code="SysHierChild", when_bound_under=("SysHierRoot",)),
    )

    with pytest.raises(SystemExit, match="belongs to its own code system"):
        export_index(
            graph,
            coverage=_covered(graph),
            datasets=_TEST_DATASETS,
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
        )


def test_validate_declared_defaults_accepts_no_defaults_configured():
    code_rows_by_system = {"sysHier": [{"dcid": "SysHierRoot"}], "sysFlat": [{"dcid": "SysFlatA"}]}
    _validate_declared_defaults((_HIERARCHICAL, _FLAT), code_rows_by_system)  # raises nothing


# --- Datasets, the coverage join, and per-dataset enumeration ---

_OPT_IN_PROVENANCE = "provenance/OptInDataset"
_OPT_IN = DatasetDeclaration(
    provenance=_OPT_IN_PROVENANCE,
    label="Opt-in dataset",
    synonyms=("opt-in", "an opt-in table"),
    opt_in=True,
    absent_observation="unknown",
    rank=10,
)


def _export_two_system_graph(coverage: CoverageJoin, **overrides) -> dict:
    return export_index(
        _two_system_graph(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        coverage=coverage,
        code_systems=overrides.pop("code_systems", (_HIERARCHICAL, _FLAT)),
        measure_axes=(),
        datasets=overrides.pop("datasets", _TEST_DATASETS),
    )


def test_export_index_joins_each_rows_sorted_provenances_and_stamps_the_build():
    coverage = CoverageJoin(
        build_id="20260101T000000Z-0a1b2c3d",
        provenances={
            "SV_Disbursement_Gross": (_TEST_PROVENANCE,),
            "SV_Disbursement_NoNetting": (_TEST_PROVENANCE,),
            "SV_Commitment_Obligation": (_TEST_PROVENANCE,),
            "SV_Not_In_The_Graph": (_TEST_PROVENANCE,),
        },
    )

    document = _export_two_system_graph(coverage)

    provenances = {row["dcid"]: row["provenances"] for row in document["statvars"]}
    assert provenances == {
        "SV_Disbursement_Gross": [_TEST_PROVENANCE],
        "SV_Disbursement_NoNetting": [_TEST_PROVENANCE],
        "SV_Commitment_Obligation": [_TEST_PROVENANCE],
        "SV_Commitment_NoFlow": [],
    }
    assert document["stamp"]["coverage_build"] == "20260101T000000Z-0a1b2c3d"
    assert document["stamp"]["statvars_without_coverage"] == 1


def test_export_index_writes_dataset_declarations_and_they_round_trip(tmp_path: Path):
    graph = _two_system_graph()
    document = export_index(
        graph,
        graph_endpoint="https://fake.example",
        now=_NOW,
        coverage=_covered(graph, _OPT_IN_PROVENANCE),
        code_systems=(
            _HIERARCHICAL,
            dataclasses.replace(
                _FLAT, summable=False, declared_by_every_statvar_of=(_OPT_IN_PROVENANCE,)
            ),
        ),
        measure_axes=(),
        datasets=(_OPT_IN,),
    )

    assert document["datasets"] == [
        {
            "provenance": _OPT_IN_PROVENANCE,
            "label": "Opt-in dataset",
            "synonyms": ["an opt-in table", "opt-in"],
            "opt_in": True,
            "absent_observation": "unknown",
            "rank": 10,
            "invariants": {},
        }
    ]
    out = tmp_path / "with_datasets.json"
    out.write_text(json.dumps(document))
    index = load_index(out)
    assert index.datasets[_OPT_IN_PROVENANCE] == dataclasses.replace(
        _OPT_IN, synonyms=("an opt-in table", "opt-in")
    )
    assert {row.provenances for row in index.statvars} == {(_OPT_IN_PROVENANCE,)}


def test_export_index_refuses_a_row_without_its_datasets_declared_invariant():
    graph = _two_system_graph()
    declared = dataclasses.replace(_TEST_DATASETS[0], invariants={"sysHier": "SysHierRoot"})

    with pytest.raises(SystemExit, match=r"sysHier=.*declares sysHier='SysHierRoot'"):
        _export_two_system_graph(_covered(graph), datasets=(declared,))


def test_content_hash_moves_with_a_dataset_declaration_and_with_a_rows_provenances():
    graph = _two_system_graph()
    base = _export_two_system_graph(_covered(graph))
    relabelled = _export_two_system_graph(
        _covered(graph), datasets=(dataclasses.replace(_TEST_DATASETS[0], label="Renamed"),)
    )
    uncovered_row = _export_two_system_graph(
        CoverageJoin(
            build_id="test-build",
            provenances={
                dcid: (_TEST_PROVENANCE,)
                for dcid in graph.forward_arcs
                if dcid != "SV_Commitment_NoFlow"
            },
        )
    )
    rebuilt = _export_two_system_graph(
        dataclasses.replace(_covered(graph), build_id="another-build")
    )

    assert relabelled["stamp"]["content_hash"] != base["stamp"]["content_hash"]
    assert uncovered_row["stamp"]["content_hash"] != base["stamp"]["content_hash"]
    assert rebuilt["stamp"]["content_hash"] == base["stamp"]["content_hash"]
    assert rebuilt["stamp"]["coverage_build"] == "another-build"


def test_export_graph_blocks_holds_the_documents_blocks_without_provenances_or_a_stamp():
    graph = _two_system_graph()
    blocks = export_graph_blocks(
        graph, code_systems=(_HIERARCHICAL, _FLAT), measure_axes=(), datasets=_TEST_DATASETS
    )
    document = _export_two_system_graph(_covered(graph))

    assert set(blocks) == set(document) - {"stamp"}
    assert all("provenances" not in row for row in blocks["statvars"])
    for block, rows in blocks.items():
        if block != "statvars":
            assert rows == document[block], block
    assert blocks["statvars"] == [
        {key: value for key, value in row.items() if key != "provenances"}
        for row in document["statvars"]
    ]
    with pytest.raises(IndexFormatError):
        parse_index({"stamp": document["stamp"], **blocks})

    counts = stamp_counts(blocks)
    assert counts["code_counts"] == document["stamp"]["code_counts"]
    assert counts["statvar_shape_counts"] == document["stamp"]["statvar_shape_counts"]
    assert content_hash(blocks) != document["stamp"]["content_hash"]
    assert content_hash({key: document[key] for key in blocks}) == document["stamp"]["content_hash"]


def test_export_index_refuses_a_declared_dataset_with_no_row():
    with pytest.raises(SystemExit, match=r"provenance/TestDataset.*no StatVar row"):
        _export_two_system_graph(CoverageJoin(build_id="empty-build", provenances={}))


def test_export_index_refuses_a_provenance_no_dataset_declares():
    graph = _two_system_graph()
    coverage = CoverageJoin(
        build_id="test-build",
        provenances=dict.fromkeys(graph.forward_arcs, ("provenance/ThirdSource", _TEST_PROVENANCE)),
    )

    with pytest.raises(SystemExit, match=r"provenance/ThirdSource.*no dataset declares"):
        _export_two_system_graph(coverage)


def test_export_index_refuses_a_row_lacking_a_system_its_dataset_always_declares():
    graph = _two_system_graph()
    del graph.forward_arcs["SV_Commitment_NoFlow"]["sysFlat"]
    graph.forward_arcs["SV_Commitment_NoFlow"]["constraintProperties"] = (
        "sysHier",
        "financialObligationType",
    )

    with pytest.raises(SystemExit, match=r"SV_Commitment_NoFlow.*does not declare 'sysFlat'"):
        export_index(
            graph,
            graph_endpoint="https://fake.example",
            now=_NOW,
            coverage=_covered(graph),
            code_systems=(_HIERARCHICAL, _FLAT),
            measure_axes=(),
            datasets=_TEST_DATASETS,
        )


def _aid_type_config(*, declared_by_every_statvar_of: tuple[str, ...]) -> CodeSystemConfig:
    return CodeSystemConfig(
        code_system="sysAidType",
        enum_dcid="SysAidTypeEnum",
        expected_hierarchical=False,
        disjoint_from_children=True,
        summable=False,
        absent_reading="not_declared",
        declared_by_every_statvar_of=declared_by_every_statvar_of,
        plain_name="aid type",
    )


def test_each_dataset_is_enumerated_through_its_own_largest_complete_system():
    # P1's read truncates. The first dataset splits it on sysFinance, which the second
    # dataset's rows do not declare, and the second dataset splits it on sysAidType.
    plain = {f"SV_Plain_{i}": _statvar(finance="F1" if i % 2 else "F2") for i in range(4)}
    opt_in = {f"SV_OptIn_{i}": {"sysPurpose": ("P1",), "sysAidType": ("A1",)} for i in range(2)}
    graph = _FakeGraph(forward_arcs=plain | opt_in, page_size=3)
    purpose = dataclasses.replace(
        _purpose_config(), declared_by_every_statvar_of=(_TEST_PROVENANCE, _OPT_IN_PROVENANCE)
    )
    finance = _finance_config()
    aid_type = _aid_type_config(declared_by_every_statvar_of=(_OPT_IN_PROVENANCE,))
    code_rows_by_system = {
        "sysPurpose": [{"dcid": "P1"}, {"dcid": "P2"}, {"dcid": "P3"}],
        "sysFinance": [{"dcid": "F1"}, {"dcid": "F2"}],
        "sysAidType": [{"dcid": "A1"}],
    }

    dcids, _ = enumerate_statvar_dcids(
        graph, code_rows_by_system, (purpose, finance, aid_type), (*_TEST_DATASETS, _OPT_IN)
    )

    assert set(dcids) == set(plain) | set(opt_in)
    assert any(
        base == "sysPurpose" and "sysAidType" in dict(filters or ())
        for _, base, filters in graph.calls
    )

    # Enumerating the first dataset alone drops them, and the coverage check says so.
    with pytest.raises(SystemExit, match=r"sysAidType='A1' carries .*SV_OptIn_0"):
        enumerate_statvar_dcids(
            graph, code_rows_by_system, (purpose, finance, aid_type), _TEST_DATASETS
        )


def test_a_dataset_no_code_system_lists_refuses_enumeration():
    graph = _FakeGraph(forward_arcs={"SV_A": _statvar()})

    with pytest.raises(SystemExit, match="cannot choose an enumeration spine"):
        enumerate_statvar_dcids(
            graph,
            {"sysPurpose": [{"dcid": "P1"}], "sysAidType": [{"dcid": "A1"}]},
            (_purpose_config(), _aid_type_config(declared_by_every_statvar_of=())),
            (_OPT_IN,),
        )


def test_the_packaged_configuration_lists_each_system_under_the_datasets_that_always_declare_it():
    crs, imputed = (dataset.provenance for dataset in DATASETS)
    assert [dataset.opt_in for dataset in DATASETS] == [False, True]

    assert {
        config.code_system: set(config.declared_by_every_statvar_of) for config in CODE_SYSTEMS
    } == {
        "crsPurpose": {crs, imputed},
        "crsTypeOfFinance": {crs},
        "crsCooperationModality": set(),
        "crsTypeOfFlow": {crs, imputed},
        "dac2aAidType": {imputed},
    }


def test_load_coverage_join_reads_the_build_id_and_variable_provenances():
    slice_uri = (Path(__file__).parents[1] / "fixtures" / "coverage_slice").as_uri()
    loaded = load_coverage(slice_uri)

    join = load_coverage_join(slice_uri)

    assert join.build_id == loaded.manifest.build_id
    assert join.provenances == loaded.index.variable_provenances()
    assert join.provenances
    assert all(
        provenances == tuple(sorted(provenances)) for provenances in join.provenances.values()
    )


# --- A StatVar the index would decide must be an exported row ---

_ORPHAN_SHAPE = {
    "typeOf": ("StatisticalVariable",),
    "populationType": ("FinancialFlow",),
    "measuredProperty": ("amount",),
    "statType": ("measuredValue",),
    "measurementQualifier": ("Nominal",),
    "financialFlowType": ("Disbursement",),
}


def _graph_with_unexported(arcs: dict[str, tuple[str, ...]], **knobs) -> _FakeGraph:
    """Add `SV_Unexported` on the disbursement five-tuple, which the `sysHier` spine misses.

    Its `sysHier` value is no enum code, and the `SysFlatA` spine-coverage read truncates, so the
    enumeration neither reaches nor flags it.
    """
    graph = _two_system_graph()
    graph.forward_arcs["SV_Unexported"] = {**_ORPHAN_SHAPE, **arcs}
    return dataclasses.replace(
        graph, truncate=frozenset({("SysFlatA", "sysFlat"), *knobs.pop("truncate", ())}), **knobs
    )


def _export_blocks(graph: _FakeGraph) -> dict:
    return export_graph_blocks(
        graph, code_systems=(_HIERARCHICAL, _FLAT), measure_axes=(), datasets=_TEST_DATASETS
    )


def test_export_refuses_a_statvar_the_index_would_decide_naming_it():
    """Issue #1090: a StatVar outside the spine enumeration whose constraint properties cover
    an exported row of its five-tuple refuses the export."""
    graph = _graph_with_unexported(
        {
            "constraintProperties": ("sysHier", "sysFlat", "financialFlowType"),
            "sysHier": ("SysHierStray",),
            "sysFlat": ("SysFlatA",),
        }
    )

    with pytest.raises(SystemExit, match=r"1 StatVar\(s\).*\['SV_Unexported'\]"):
        _export_blocks(graph)


@pytest.mark.parametrize(
    "arcs",
    [
        pytest.param(
            {"constraintProperties": ("sysFlat", "financialFlowType"), "sysFlat": ("SysFlatA",)},
            id="keys-cover-no-exported-row",
        ),
        pytest.param(
            {
                "constraintProperties": ("sysHier", "sysFlat", "financialFlowType"),
                "sysHier": ("SysHierStray",),
                "sysFlat": ("SysFlatA",),
                "measurementDenominator": ("Count_Person",),
            },
            id="other-denominator",
        ),
    ],
)
def test_export_passes_an_unexported_statvar_the_index_would_not_decide(arcs):
    graph = _graph_with_unexported(arcs)

    blocks = _export_blocks(graph)

    assert "SV_Unexported" not in {row["dcid"] for row in blocks["statvars"]}
    assert ("amount", "measuredProperty") in {(dcid, prop) for dcid, prop, _ in graph.calls}


def test_export_refuses_a_five_tuple_subject_the_graph_holds_no_node_for():
    graph = _graph_with_unexported(
        {"constraintProperties": ("sysFlat", "financialFlowType"), "sysFlat": ("SysFlatA",)},
        unreadable=frozenset({"SV_Unexported"}),
    )

    with pytest.raises(SystemExit, match=r"SV_Unexported is a <-measuredProperty subject"):
        _export_blocks(graph)


def test_export_refuses_a_five_tuple_read_still_truncated_after_every_code_axis():
    graph = _graph_with_unexported({}, truncate={("amount", "measuredProperty")})

    with pytest.raises(SystemExit, match=r"exhausted every partition property.*'amount'"):
        _export_blocks(graph)
