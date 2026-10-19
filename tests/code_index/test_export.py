"""Offline tests for scripts/export_code_index.py. Drives a small in-memory
fake reader (`_FakeGraph`) implementing only the `ExportReader` surface the
export actually calls (`node_subjects`, `node_arcs_many` and `node_arcs_batch`).
Nothing here opens a socket.
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
    parse_measure_axes,
)
from code_index.shared import get_index
from code_index.types import (
    DatasetDeclaration,
    DeclaredDefault,
    MeasureAxis,
    MeasurePartitionLabel,
    PartlyStatedValue,
    PartyKind,
    PartyRow,
    WholeMarker,
    WholeTotalSet,
)
from dfo_platform.roles import RoleAxis
from materialise.graph import (
    UNRESOLVED_NODE,
    GraphNotAdvancingError,
    GraphReadError,
    SubjectPage,
)
from scripts.export_code_index import (
    _CONCEPT_GROUPS,
    _MEASURE_VALUE_TEXT,
    CODE_SYSTEMS,
    DATASETS,
    MEASURE_CLASSES,
    PARTLY_STATED_VALUES,
    PARTY_KIND_OVERRIDES,
    PARTY_TYPE_KINDS,
    PRIVATE_PROVIDERS,
    SHAPE_SLOT_DEFAULTS,
    STAT_TYPE_WORDS,
    TOTAL_GROUPS,
    VARIABLE_ALIASES,
    WHOLE_POPULATION_TOTALS,
    WHOLE_TOTAL_SETS,
    CodeSystemConfig,
    CoverageJoin,
    KindOverride,
    PartyPublication,
    TotalGroup,
    WholePopulationTotal,
    WholeWindow,
    _apply_concept_groups,
    _measure_class_to_dict,
    _MeasureValueText,
    _minimal_system_sets,
    _partly_stated_value_to_dict,
    _shape_slot_default_to_dict,
    _stat_type_word_to_dict,
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
    export_source_codes,
    export_statvars,
    export_value_parents,
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
    `calls` records each reverse read and its filters. `wire_arcs` holds the full forward reads
    `node_arcs_batch` returns, in the graph's wire shape, whose values carry their provenance.
    """

    forward_arcs: dict[str, dict[str, tuple[str, ...]]]
    page_size: int | None = None
    truncate: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    not_advancing: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    read_error: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    unreadable: frozenset[str] = field(default_factory=frozenset)
    calls: list[tuple[str, str, tuple[tuple[str, str], ...] | None]] = field(default_factory=list)
    wire_arcs: dict[str, dict] = field(default_factory=dict)

    def node_arcs_batch(self, dcids):
        return {dcid: self.wire_arcs.get(dcid) for dcid in dcids}

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
        description="A test table.",
        opt_in=False,
        absent_observation="zero",
        account="provider",
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
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
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


@pytest.mark.parametrize(
    ("descriptions", "expected"),
    [(("A flow of official aid.",), "A flow of official aid."), ((), None)],
)
def test_code_row_definition_is_the_graph_description(
    descriptions: tuple[str, ...], expected: str | None
):
    graph = _FakeGraph(
        forward_arcs={
            "CRSTypeOfFlow10": {
                "typeOf": ("SysHierEnum",),
                "classificationCode": ("10",),
                "name": ("a flow (CRS Hier)",),
                "description": descriptions,
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
    assert rows[0]["definition"] == expected


# --- content_hash stability ---


def test_content_hash_identical_across_runs_and_differs_on_a_row_change():
    graph = _two_system_graph()
    first = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
    )
    later = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=datetime(2026, 9, 9, 13, 0, 0, tzinfo=UTC),
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
    )
    assert first["stamp"]["exported_at"] != later["stamp"]["exported_at"]
    assert first["stamp"]["content_hash"] == later["stamp"]["content_hash"]

    changed_graph = _two_system_graph()
    changed_graph.forward_arcs["SysHierRoot"]["alternateName"] = ("Root Sector", "Extra Alias")
    changed = export_index(
        changed_graph,
        coverage=_covered(changed_graph),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
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
    MeasureAxis(
        axis="financialFlowType",
        kind="constraint",
        role="free",
        prefer=("Disbursement",),
        label="flow type",
    ),
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
        definition="The total amount released before any repayments are subtracted.",
    ),
)
_MEASURE_NOUNS_FOR_TEST = (
    MeasurePartitionLabel(
        axis_values=(("populationType", "EconomicProduction"), ("measuredProperty", "valueAdded")),
        label="GDP",
    ),
)


_GROSS_DEFINITION = "An amount counted in full, with nothing offset."


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
        "Gross": {"name": ("gross",), "description": (_GROSS_DEFINITION,)},
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
    assert gross["definition"] == _GROSS_DEFINITION
    assert by_dcid["Disbursement"]["definition"] is None


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
            partly_stated_values=(),
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(_HIERARCHICAL, _FLAT),
            measure_axes=_MEASURE_AXES_FOR_TEST,
            measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
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
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        measure_partition_labels=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
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
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=_MEASURE_AXES_FOR_TEST,
        measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
        measure_nouns=_MEASURE_NOUNS_FOR_TEST,
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
    )

    assert document["measure_axes"] == [
        {
            "axis": "populationType",
            "kind": "shape",
            "role": "free",
            "prefer": ["FinancialFlow", "FinancialObligation"],
            "label": None,
        },
        {
            "axis": "measuredProperty",
            "kind": "shape",
            "role": "basis",
            "prefer": ["amount"],
            "label": None,
        },
        {
            "axis": "measurementQualifier",
            "kind": "shape",
            "role": "free",
            "prefer": ["Nominal", "RealValue"],
            "label": None,
        },
        {
            "axis": "financialFlowType",
            "kind": "constraint",
            "role": "free",
            "prefer": ["Disbursement"],
            "label": "flow type",
        },
        {
            "axis": "financialObligationType",
            "kind": "constraint",
            "role": "free",
            "prefer": ["Commitment"],
            "label": None,
        },
        {
            "axis": "nettingBasis",
            "kind": "constraint",
            "role": "free",
            "prefer": ["Gross", "Net"],
            "label": None,
        },
    ]
    assert parse_measure_axes(document).axes == _MEASURE_AXES_FOR_TEST
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
            "definition": None,
        },
        {
            "axis_values": [
                ["financialFlowType", "Disbursement"],
                ["measuredProperty", "amount"],
                ["populationType", "FinancialFlow"],
            ],
            "label": "disbursements",
            "named_values": [],
            "definition": None,
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
            "definition": "The total amount released before any repayments are subtracted.",
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
            "definition": None,
        }
    ]
    # The loader reads both blocks back into the authored values.
    path = tmp_path / "index.json"
    path.write_text(json.dumps(document))
    measure_index = load_measure_axes(path)
    assert measure_index.partition_labels == _MEASURE_PARTITION_LABELS_FOR_TEST
    assert measure_index.nouns == _MEASURE_NOUNS_FOR_TEST
    assert {row.dcid: row.definition for row in measure_index.values}["Gross"] == (
        _GROSS_DEFINITION
    )


def test_export_strips_the_measure_words_a_code_synonym_would_take():
    """The matcher reads a code text whole before a measure word inside it, so "Gross alpha"
    on a code would leave "gross" unread as the netting basis. The synonym keeps its other
    words. Every SysFlatA series is a disbursement, so "disbursements" stays on it, and every
    SysFlatB series a commitment, so "gross disbursement" has no words left there. `Net` is
    no exported value here, so "net alpha" stays. A query reads "alpha: gross" whatever its
    separators, so it loses "gross", and its remainder "alpha" is already there."""
    graph = _two_system_graph_with_measure_labels()
    graph.forward_arcs["SysFlatA"]["alternateName"] = (
        "Gross alpha",
        "alpha disbursements",
        "Gross alpha disbursements",
        "net alpha",
        "alpha: gross",
        "engrossed alpha",
    )
    graph.forward_arcs["SysFlatB"]["name"] = ("gross beta (CRS Flat)",)
    graph.forward_arcs["SysFlatB"]["alternateName"] = ("gross disbursement",)

    blocks = export_graph_blocks(
        graph,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=_MEASURE_AXES_FOR_TEST,
        measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
        measure_nouns=_MEASURE_NOUNS_FOR_TEST,
        datasets=_TEST_DATASETS,
    )
    by_dcid = {row["dcid"]: row for row in blocks["codes"]}

    assert by_dcid["SysFlatA"]["synonyms"] == [
        "alpha",
        "alpha disbursements",
        "engrossed alpha",
        "net alpha",
    ]
    assert by_dcid["SysFlatB"]["label"] == "gross beta"
    assert by_dcid["SysFlatB"]["synonyms"] == []


def test_content_hash_moves_when_only_measure_axes_change():
    # Changing only `measure_axes` changes the content hash.
    graph = _two_system_graph_with_measure_labels()
    with_axes = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=_MEASURE_AXES_FOR_TEST,
        measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
    )
    without_axes = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        measure_partition_labels=_MEASURE_PARTITION_LABELS_FOR_TEST,
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
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
            partly_stated_values=(),
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(_HIERARCHICAL, _FLAT),
            measure_axes=_MEASURE_AXES_FOR_TEST,
            measure_partition_labels=(),
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
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


def test_committed_packaged_index_carries_the_authored_declarations():
    """The authored tables no graph read checks match the committed export, so an edit to one
    without a re-export fails here rather than in the live drift check."""
    payload = json.loads(gzip.decompress(default_index_path().read_bytes()))

    assert payload["shape_slot_defaults"] == [
        _shape_slot_default_to_dict(d) for d in SHAPE_SLOT_DEFAULTS
    ]
    assert payload["partly_stated_values"] == [
        _partly_stated_value_to_dict(v) for v in PARTLY_STATED_VALUES
    ]
    assert payload["measure_classes"] == [_measure_class_to_dict(d) for d in MEASURE_CLASSES]
    assert payload["variable_aliases"] == dict(VARIABLE_ALIASES)
    assert payload["stat_type_words"] == [_stat_type_word_to_dict(d) for d in STAT_TYPE_WORDS]
    assert {d["code_system"]: d["no_vocabulary_terms"] for d in payload["declarations"]} == {
        config.code_system: list(config.no_vocabulary_terms) for config in CODE_SYSTEMS
    }


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


# --- plain_name and enum_dcid flow from the code-system table into the declaration row ---


def test_export_index_stamps_plain_name_and_enum_dcid_from_the_code_system_table():
    # The configured names are `hier` and `flat`.
    graph = _two_system_graph()
    document = export_index(
        graph,
        coverage=_covered(graph),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
    )
    by_system = {row["code_system"]: row for row in document["declarations"]}
    assert by_system["sysHier"]["plain_name"] == _HIERARCHICAL.plain_name
    assert by_system["sysFlat"]["plain_name"] == _FLAT.plain_name
    assert by_system["sysHier"]["enum_dcid"] == "SysHierEnum"
    assert {
        code_system: declaration.enum_dcid
        for code_system, declaration in parse_index(document).declarations.items()
    } == {"sysHier": "SysHierEnum", "sysFlat": "SysFlatEnum"}


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
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL_WITH_DEFAULT, _FLAT),
        measure_axes=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
    )
    by_system = {row["code_system"]: row for row in document["declarations"]}
    assert by_system["sysHier"]["default"] == {
        "code": "SysHierChild",
        "when_bound_under": ["SysFlatA"],
        "when_named": [],
        "when_open": False,
        "note": None,
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
        partly_stated_values=(),
        graph_endpoint="https://fake.example",
        now=_NOW,
        code_systems=(_HIERARCHICAL_WITH_NAMED_DEFAULT, _FLAT),
        measure_axes=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
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
            partly_stated_values=(),
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
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
            partly_stated_values=(),
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
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
            partly_stated_values=(),
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
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
            partly_stated_values=(),
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
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
            partly_stated_values=(),
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
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
            partly_stated_values=(),
            graph_endpoint="https://fake.example",
            now=_NOW,
            code_systems=(wrong, _FLAT),
            measure_axes=(),
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
        )


def test_validate_declared_defaults_accepts_no_defaults_configured():
    code_rows_by_system = {"sysHier": [{"dcid": "SysHierRoot"}], "sysFlat": [{"dcid": "SysFlatA"}]}
    _validate_declared_defaults((_HIERARCHICAL, _FLAT), code_rows_by_system)  # raises nothing


# --- Datasets, the coverage join, and per-dataset enumeration ---

_OPT_IN_PROVENANCE = "provenance/OptInDataset"
_OPT_IN = DatasetDeclaration(
    provenance=_OPT_IN_PROVENANCE,
    label="Opt-in dataset",
    description="A test table.",
    opt_in=True,
    absent_observation="unknown",
    account="provider",
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
        partly_stated_values=overrides.pop("partly_stated_values", ()),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
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
        partly_stated_values=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
    )

    assert document["datasets"] == [
        {
            "provenance": _OPT_IN_PROVENANCE,
            "label": "Opt-in dataset",
            "description": "A test table.",
            "opt_in": True,
            "absent_observation": "unknown",
            "account": "provider",
            "rank": 10,
            "invariants": {},
        }
    ]
    out = tmp_path / "with_datasets.json"
    out.write_text(json.dumps(document))
    index = load_index(out)
    assert index.datasets[_OPT_IN_PROVENANCE] == _OPT_IN
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


def test_export_graph_blocks_holds_the_documents_blocks_without_provenances_parties_or_stamp():
    graph = _two_system_graph()
    blocks = export_graph_blocks(
        graph,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        whole_total_sets=(),
    )
    document = _export_two_system_graph(_covered(graph))

    assert set(blocks) == set(document) - {"stamp", "parties"}
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
    hashed = {key: document[key] for key in (*blocks, "parties")}
    assert content_hash(hashed) == document["stamp"]["content_hash"]


_NEVER_STATED = (
    PartlyStatedValue(
        provenance=_TEST_PROVENANCE, property_dcid="financialFlowType", value_dcid="Nowhere"
    ),
)


def test_export_graph_blocks_emits_the_partly_stated_values_it_is_given():
    blocks = export_graph_blocks(
        _two_system_graph(),
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        datasets=_TEST_DATASETS,
        partly_stated_values=_NEVER_STATED,
    )

    assert blocks["partly_stated_values"] == [_partly_stated_value_to_dict(_NEVER_STATED[0])]


def test_export_index_refuses_a_partly_stated_value_no_row_bears_out():
    """The finished document must load, so a declaration the rows contradict stops the export."""
    coverage = _covered(_two_system_graph())

    with pytest.raises(SystemExit):
        _export_two_system_graph(coverage, partly_stated_values=_NEVER_STATED)


def test_export_index_refuses_a_declared_dataset_with_no_row():
    with pytest.raises(SystemExit, match=r"provenance/TestDataset.*no StatVar row"):
        _export_two_system_graph(CoverageJoin(build_id="empty-build", provenances={}))


_LISTED_PROVENANCE = "provenance/ListedDataset"


def _code_arcs(*values: tuple[str, str]) -> dict:
    """A full forward read holding the `sourceIndicatorCode` values `values`, each a
    `(provenance, code)` pair, as the graph sends it."""
    return {
        "sourceIndicatorCode": {
            "nodes": [{"provenanceId": provenance, "value": code} for provenance, code in values]
        }
    }


def _graph_with_listed(*extra: str) -> _FakeGraph:
    """The two-system graph plus StatVars on a CRS-like row's five-tuple that declare no code
    system, as IDS series do: `SV_Listed_Disbursement` and each of `extra`, each carrying the
    series code `LST.DISB` under the listed dataset."""
    graph = _two_system_graph()
    for dcid in ("SV_Listed_Disbursement", *extra):
        graph.wire_arcs[dcid] = _code_arcs((_LISTED_PROVENANCE, "LST.DISB"))
        graph.forward_arcs[dcid] = {
            "typeOf": ("StatisticalVariable",),
            "constraintProperties": ("financialFlowType",),
            "populationType": ("FinancialFlow",),
            "measuredProperty": ("amount",),
            "statType": ("measuredValue",),
            "measurementQualifier": ("Nominal",),
            "observationProperties": ("financingRecipient",),
            "financialFlowType": ("Disbursement",),
        }
    return graph


def _export_with_listed(graph: _FakeGraph, provenances: dict[str, tuple[str, ...]]) -> dict:
    return export_index(
        graph,
        graph_endpoint="https://fake.example",
        now=_NOW,
        coverage=CoverageJoin(
            build_id="test-build",
            provenances={**_covered(_two_system_graph()).provenances, **provenances},
        ),
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        total_groups={},
        whole_totals={},
        whole_total_sets=(),
        party_kind_overrides={},
        listed_datasets=(_LISTED_PROVENANCE,),
    )


def test_export_index_lists_the_statvars_of_a_dataset_no_code_system_enumerates():
    """No code read reaches a StatVar that declares no code system, so the coverage build
    lists it, as a listed row apart from the index rows."""
    document = _export_with_listed(
        _graph_with_listed(), {"SV_Listed_Disbursement": (_LISTED_PROVENANCE,)}
    )

    index = parse_index(document)
    assert [row.dcid for row in index.listed_statvars] == ["SV_Listed_Disbursement"]
    assert index.listed_statvars[0].provenances == (_LISTED_PROVENANCE,)
    assert index.listed_statvars[0].constraint_properties == {"financialFlowType": "Disbursement"}
    assert "SV_Listed_Disbursement" not in {row.dcid for row in index.statvars}


def test_export_index_refuses_a_statvar_of_a_listed_five_tuple_it_neither_exports_nor_lists():
    """Members elected from the index stand for every StatVar of a listed row's five-tuple,
    so a StatVar there the coverage build does not list refuses the export, named."""
    with pytest.raises(SystemExit, match=r"1 StatVar.*SV_Unlisted_Disbursement"):
        _export_with_listed(
            _graph_with_listed("SV_Unlisted_Disbursement"),
            {"SV_Listed_Disbursement": (_LISTED_PROVENANCE,)},
        )


def test_export_index_refuses_a_listed_dataset_the_build_holds_no_statvar_of():
    with pytest.raises(SystemExit, match=r"provenance/ListedDataset.*no StatVar"):
        _export_with_listed(_graph_with_listed(), {})


def test_export_index_writes_each_listed_statvar_s_series_codes_upper_cased():
    graph = _graph_with_listed()
    graph.wire_arcs["SV_Listed_Disbursement"] = _code_arcs(
        (_LISTED_PROVENANCE, "lst.disb"), (_LISTED_PROVENANCE, " LST.OTHER ")
    )
    document = _export_with_listed(graph, {"SV_Listed_Disbursement": (_LISTED_PROVENANCE,)})

    assert document["source_codes"] == [
        {"provenance": _LISTED_PROVENANCE, "code": "LST.DISB", "dcids": ["SV_Listed_Disbursement"]},
        {
            "provenance": _LISTED_PROVENANCE,
            "code": "LST.OTHER",
            "dcids": ["SV_Listed_Disbursement"],
        },
    ]
    assert parse_index(document).source_code_members("lst.disb") == {
        _LISTED_PROVENANCE: ("SV_Listed_Disbursement",)
    }


def test_export_index_refuses_a_listed_statvar_with_no_series_code():
    graph = _graph_with_listed()
    graph.wire_arcs["SV_Listed_Disbursement"] = {}
    with pytest.raises(SystemExit, match=r"1 listed StatVar.*SV_Listed_Disbursement"):
        _export_with_listed(graph, {"SV_Listed_Disbursement": (_LISTED_PROVENANCE,)})


def test_export_index_refuses_a_code_whose_arc_provenance_the_coverage_build_does_not_give():
    graph = _graph_with_listed()
    graph.wire_arcs["SV_Listed_Disbursement"] = _code_arcs(("provenance/Elsewhere", "LST.DISB"))
    with pytest.raises(SystemExit, match=r"does not load.*provenance/Elsewhere"):
        _export_with_listed(graph, {"SV_Listed_Disbursement": (_LISTED_PROVENANCE,)})


def test_export_source_codes_keys_a_recorded_ids_read_on_its_arc_provenance():
    """`source_code_arcs.json` holds the `sourceIndicatorCode` arc of the two dev-graph IDS
    StatVars of `DT.TDS.DEGG.CD`, cut from their full forward reads."""
    recorded = json.loads(
        (Path(__file__).parent / "fixtures" / "source_code_arcs.json").read_text()
    )
    graph = _FakeGraph(forward_arcs={}, wire_arcs=recorded)

    assert export_source_codes(graph, sorted(recorded)) == [
        {
            "provenance": "provenance/WorldBankInternationalDebtStatistics",
            "code": "DT.TDS.DEGG.CD",
            "dcids": sorted(recorded),
        }
    ]


def test_export_index_writes_the_parent_of_each_listed_constraint_value_that_has_one():
    graph = _graph_with_listed()
    graph.forward_arcs["Disbursement"] = {
        **graph.forward_arcs["Disbursement"],
        "specializationOf": ("FinancialFlowAny",),
    }
    document = _export_with_listed(graph, {"SV_Listed_Disbursement": (_LISTED_PROVENANCE,)})

    assert document["value_parents"] == [{"dcid": "Disbursement", "parent": "FinancialFlowAny"}]
    assert parse_index(document).value_ancestors("Disbursement") == ("FinancialFlowAny",)


def test_export_value_parents_writes_each_chain_and_refuses_two_parents_or_a_cycle():
    statvars = [{"constraint_properties": {"idsDebtorType": "Child", "guaranteeStatus": "Flat"}}]
    graph = _FakeGraph(
        forward_arcs={
            "Child": {"specializationOf": ("Parent",)},
            "Parent": {"specializationOf": ("Root",)},
            "Root": {},
            "Flat": {},
        }
    )
    assert export_value_parents(graph, statvars) == [
        {"dcid": "Child", "parent": "Parent"},
        {"dcid": "Parent", "parent": "Root"},
    ]

    graph.forward_arcs["Root"] = {"specializationOf": ("Child",)}
    with pytest.raises(SystemExit, match="chain of Child returns to Child"):
        export_value_parents(graph, statvars)

    graph.forward_arcs["Child"] = {"specializationOf": ("Parent", "Other")}
    with pytest.raises(SystemExit, match="Child's 'specializationOf'"):
        export_value_parents(graph, statvars)


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
            partly_stated_values=(),
            total_groups={},
            whole_totals={},
            whole_total_sets=(),
            party_kind_overrides={},
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
    crs, imputed, dac1, dac2a, dac2b = (dataset.provenance for dataset in DATASETS)
    assert [dataset.opt_in for dataset in DATASETS] == [False, True, False, False, False]

    assert {
        config.code_system: set(config.declared_by_every_statvar_of) for config in CODE_SYSTEMS
    } == {
        "crsPurpose": {crs, imputed},
        "crsTypeOfFinance": {crs},
        "crsCooperationModality": set(),
        "crsTypeOfFlow": {crs, imputed},
        "dac2aAidType": {imputed, dac2a},
        "dac1AidType": {dac1},
        "dac2bAidType": {dac2b},
    }


def test_load_coverage_join_reads_the_build_id_and_variable_provenances():
    slice_uri = (Path(__file__).parents[1] / "fixtures" / "coverage_slice").as_uri()
    loaded = load_coverage(slice_uri)

    join = load_coverage_join(slice_uri, datasets={"provenance/OECDCreditorReportingSystem"})

    assert join.build_id == loaded.manifest.build_id
    assert join.provenances == loaded.index.variable_provenances()
    assert join.provenances
    assert all(
        provenances == tuple(sorted(provenances)) for provenances in join.provenances.values()
    )
    # A donor and a recipient of the slice's one CRS facet, each with the dataset publishing
    # it and the axis it is published on, and no empty cell of a one-role row.
    crs = frozenset({"provenance/OECDCreditorReportingSystem"})
    assert join.parties["AdaptationFund"] == PartyPublication(crs, frozenset({"origin"}))
    assert join.parties["DACAfricaUnallocated"] == PartyPublication(crs, frozenset({"destination"}))
    assert "" not in join.parties
    assert load_coverage_join(slice_uri, datasets=()).parties == {}


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


@pytest.mark.parametrize(
    ("key_sets", "expected"),
    [
        pytest.param(
            {frozenset({"sysHier", "sysFlat"}), frozenset({"sysHier", "sysFlat", "sysOther"})},
            [frozenset({"sysHier", "sysFlat"})],
            id="optional-system-folds-into-the-smaller-set",
        ),
        pytest.param(
            {frozenset({"sysHier", "financialFlowType"}), frozenset({"sysFlat", "nettingBasis"})},
            [frozenset({"sysFlat"}), frozenset({"sysHier"})],
            id="two-datasets-with-disjoint-systems-read-apart",
        ),
        pytest.param(
            {frozenset({"financialFlowType"}), frozenset({"sysFlat"})},
            [frozenset()],
            id="a-row-with-no-system-reads-unsplit",
        ),
    ],
)
def test_five_tuple_reads_split_on_each_minimal_system_set(key_sets, expected):
    code_rows_by_system = {system: [] for system in ("sysHier", "sysFlat", "sysOther")}

    assert _minimal_system_sets(key_sets, code_rows_by_system) == expected


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


@pytest.mark.parametrize("group", _CONCEPT_GROUPS, ids=lambda group: "+".join(d for _s, d in group))
def test_every_code_of_a_concept_group_reads_the_same_texts(group):
    """Codes of different tables that name one concept carry one set of texts in the
    packaged index, so a query phrase reads on all of them or none."""
    index = load_index(default_index_path())
    rows = [index.code_by_dcid(dcid) for _system, dcid in group]
    assert all(row is not None for row in rows)
    texts = [
        {text.strip().casefold() for text in (row.label, *row.synonyms)}
        for row in rows
        if row is not None
    ]

    assert all(member == texts[0] for member in texts)


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        pytest.param([(("sysA", "A1"), ("sysA", "A2"))], "one code system", id="one-system"),
        pytest.param(
            [(("sysA", "A1"), ("sysB", "B1")), (("sysA", "A1"), ("sysC", "C1"))],
            "more than one group",
            id="code-in-two-groups",
        ),
    ],
)
def test_a_malformed_concept_group_refuses(groups, message):
    rows = [
        {"dcid": dcid, "code_system": system, "label": dcid.lower(), "synonyms": []}
        for system, dcid in (("sysA", "A1"), ("sysA", "A2"), ("sysB", "B1"), ("sysC", "C1"))
    ]

    with pytest.raises(SystemExit, match=message):
        _apply_concept_groups(rows, groups, {"sysA", "sysB", "sysC"})


# --- Party kinds and the groups totals hold ---


def _export_parties(
    types_by_party: dict[str, tuple[str, ...]],
    *,
    parties: frozenset[str] | None = None,
    total_groups: dict[str, TotalGroup] | None = None,
    kind_overrides: dict[str, KindOverride] | None = None,
    names: dict[str, str] | None = None,
    dac_codes: dict[str, tuple[str, ...]] | None = None,
    private_providers: dict[int, str] | None = None,
    whole_totals: dict[str, WholePopulationTotal] | None = None,
    whole_total_sets: tuple[WholeTotalSet, ...] = (),
    axis: RoleAxis = "destination",
    published_in: frozenset[str] = frozenset({_TEST_PROVENANCE}),
) -> dict:
    """Export the two-system graph with `types_by_party` as graph nodes, named by `names` and
    carrying the `dac_codes`, and `parties` (all of them by default) as the coverage build's
    parties, published on `axis` by the datasets `published_in`."""
    graph = _two_system_graph()
    names = names or {}
    dac_codes = dac_codes or {}
    graph.forward_arcs.update(
        {
            dcid: {
                "typeOf": types,
                "name": (names[dcid],) if dcid in names else (),
                "dacCode": dac_codes.get(dcid, ()),
            }
            for dcid, types in types_by_party.items()
        }
    )
    published = frozenset(types_by_party) if parties is None else parties
    publication = PartyPublication(datasets=published_in, axes=frozenset({axis}))
    coverage = dataclasses.replace(_covered(graph), parties=dict.fromkeys(published, publication))
    return export_index(
        graph,
        graph_endpoint="https://fake.example",
        now=_NOW,
        coverage=coverage,
        code_systems=(_HIERARCHICAL, _FLAT),
        measure_axes=(),
        datasets=_TEST_DATASETS,
        partly_stated_values=(),
        total_groups=total_groups or {},
        party_kind_overrides=kind_overrides or {},
        private_providers=private_providers or {},
        whole_totals=whole_totals or {},
        whole_total_sets=whole_total_sets,
    )


def test_export_index_writes_each_partys_kind_label_and_the_group_a_total_holds(tmp_path: Path):
    document = _export_parties(
        {
            "country/FRA": ("Country",),
            "country/IND": ("Country", "Place"),
            "IDA": ("Organization", "ProvisionalNode"),
            "DACAfrica": ("DACGroup",),
            "DACAllRecipients": ("DACGroup",),
            "DACCountriesTotal": ("DACGroup",),
            "DACAfricaUnallocated": ("DACRecipientUnallocatedByCountry",),
        },
        total_groups={
            "DACAfrica": TotalGroup("africa", ("DACAfricaUnallocated",)),
            "DACCountriesTotal": TotalGroup("groups/OECD.DAC", sums_members=False),
        },
        names={
            "country/FRA": "France",
            "DACAfrica": "Africa (DAC)",
            "DACAfricaUnallocated": "Africa, unallocated by country (DAC)",
        },
    )
    out = tmp_path / "index.json"
    out.write_text(json.dumps(document))
    index = load_index(out)

    published, recipient = (_TEST_PROVENANCE,), ("destination",)
    assert index.parties == {
        "country/FRA": PartyRow(
            dcid="country/FRA", kind="party", label="France", datasets=published, axes=recipient
        ),
        "country/IND": PartyRow(
            dcid="country/IND", kind="party", datasets=published, axes=recipient
        ),
        "IDA": PartyRow(dcid="IDA", kind="party", datasets=published, axes=recipient),
        "DACAfrica": PartyRow(
            dcid="DACAfrica",
            kind="total",
            totals="africa",
            residuals=("DACAfricaUnallocated",),
            label="Africa (DAC)",
            datasets=published,
            axes=recipient,
        ),
        "DACAllRecipients": PartyRow(
            dcid="DACAllRecipients", kind="total", datasets=published, axes=recipient
        ),
        "DACCountriesTotal": PartyRow(
            dcid="DACCountriesTotal",
            kind="total",
            totals="groups/OECD.DAC",
            sums_members=False,
            datasets=published,
            axes=recipient,
        ),
        "DACAfricaUnallocated": PartyRow(
            dcid="DACAfricaUnallocated",
            kind="residual",
            label="Africa, unallocated by country (DAC)",
            datasets=published,
            axes=recipient,
        ),
    }
    assert index.totals_by_group == {
        "africa": index.parties["DACAfrica"],
        "groups/OECD.DAC": index.parties["DACCountriesTotal"],
    }
    assert index.party_types == PARTY_TYPE_KINDS


@pytest.mark.parametrize(
    ("types_by_party", "parties", "total_groups", "message"),
    [
        ({"Mystery": ("MysteryGroup",)}, None, {}, "MysteryGroup"),
        ({"Both": ("Country", "DACGroup")}, None, {}, "no single kind"),
        ({"Earth": ("Place",)}, None, {}, "no single kind"),
        ({}, frozenset({"Unread"}), {}, "no typeOf"),
        (
            {"country/FRA": ("Country",)},
            None,
            {"country/FRA": TotalGroup("europe")},
            "TOTAL_GROUPS names",
        ),
        ({"DACAfrica": ("DACGroup",)}, None, {"DACAsia": TotalGroup("asia")}, "TOTAL_GROUPS names"),
        (
            {"DACAfrica": ("DACGroup",), "country/EGY": ("Country",)},
            None,
            {"DACAfrica": TotalGroup("africa", ("country/EGY",))},
            "as residuals",
        ),
        (
            {"DACAfrica": ("DACGroup",)},
            None,
            {"DACAfrica": TotalGroup("africa", ("DACAfricaUnallocated",))},
            "as residuals",
        ),
        (
            {
                "DACAfrica": ("DACGroup",),
                "DACAfricaUnallocated": ("DACRecipientUnallocatedByCountry",),
            },
            None,
            {"DACAfrica": TotalGroup("africa", ("DACAfricaUnallocated",), sums_members=False)},
            "claims no sum",
        ),
    ],
    ids=[
        "undeclared-type",
        "two-kinds",
        "no-kind",
        "no-node",
        "group-on-a-party",
        "group-on-an-unpublished-total",
        "residual-is-a-party",
        "residual-unpublished",
        "residuals-beside-a-published-total",
    ],
)
def test_export_index_refuses_a_party_it_cannot_declare(
    types_by_party, parties, total_groups, message
):
    with pytest.raises(SystemExit, match=message):
        _export_parties(types_by_party, parties=parties, total_groups=total_groups)


def test_a_kind_override_replaces_the_kind_the_types_give(tmp_path: Path):
    document = _export_parties(
        {
            "WorldBankGroup": ("Organization",),
            "InternationalFinanceCorporation": ("Organization",),
            # Neither type is declared, and the override reads no types.
            "Mystery": ("MysteryGroup",),
        },
        kind_overrides={
            "WorldBankGroup": KindOverride("total", "IFC plus the rest of the group"),
            "Mystery": KindOverride("residual", "published beside the parties it leaves out"),
        },
        total_groups={"WorldBankGroup": TotalGroup("groups/WB")},
    )
    out = tmp_path / "index.json"
    out.write_text(json.dumps(document))

    published, recipient = (_TEST_PROVENANCE,), ("destination",)
    assert load_index(out).parties == {
        "WorldBankGroup": PartyRow(
            dcid="WorldBankGroup",
            kind="total",
            totals="groups/WB",
            datasets=published,
            axes=recipient,
        ),
        "InternationalFinanceCorporation": PartyRow(
            dcid="InternationalFinanceCorporation", kind="party", datasets=published, axes=recipient
        ),
        "Mystery": PartyRow(dcid="Mystery", kind="residual", datasets=published, axes=recipient),
    }


def test_export_index_refuses_a_kind_override_naming_a_party_no_dataset_publishes():
    with pytest.raises(SystemExit, match=r"PARTY_KIND_OVERRIDES names .*Ghost"):
        _export_parties(
            {"country/FRA": ("Country",)},
            kind_overrides={"Ghost": KindOverride("total", "named in error")},
        )


# Each organisation group a donor sum would count beside its members, with the members whose
# gross and net disbursements add to the group's in every year it reports (see the comment on
# `PARTY_KIND_OVERRIDES`).
_REVIEWED_KIND_OVERRIDES = {
    "AfricanDevelopmentBankGroup": ("AfricanDevelopmentBank", "AfricanDevelopmentFund"),
    "InterAmericanDevelopmentBankGroup": ("InterAmericanDevelopmentBank", "IDBInvest"),
    "PostcodeLotteryGroup": (
        "DutchPostcodeLottery",
        "GermanPostcodeLottery",
        "NorwegianPostcodeLottery",
        "PeoplesPostcodeLottery",
        "SwedishPostcodeLottery",
    ),
    "WorldBankGroup": (
        "InternationalDevelopmentAssociation",
        "InternationalBankForReconstructionAndDevelopment",
        "InternationalFinanceCorporation",
    ),
}


def test_party_kind_overrides_holds_only_reviewed_total_rows():
    # A row joins PARTY_KIND_OVERRIDES only after the sum check its comment describes, and is
    # added here with the members it was checked against.
    assert {party: row.kind for party, row in PARTY_KIND_OVERRIDES.items()} == dict.fromkeys(
        _REVIEWED_KIND_OVERRIDES, "total"
    )
    assert all(row.reason for row in PARTY_KIND_OVERRIDES.values())


@pytest.mark.parametrize(("group", "members"), sorted(_REVIEWED_KIND_OVERRIDES.items()))
def test_the_packaged_index_carries_each_reviewed_kind_override(group: str, members: tuple[str]):
    index = get_index()
    assert (index.parties[group].kind, index.parties[group].totals) == ("total", None)
    # The five national lotteries are private providers; every other member is a plain party.
    assert {member: index.parties[member].kind for member in members} == {
        member: "private" if member in PRIVATE_PROVIDERS.values() else "party" for member in members
    }


_AFRICA_RESIDUALS = (
    "DACAfricaUnallocated",
    "DACEasternAfricaUnallocated",
    "DACMiddleAfricaUnallocated",
    "DACNorthOfSaharaUnallocated",
    "DACSouthOfSaharaUnallocated",
    "DACSouthernAfricaUnallocated",
    "DACWesternAfricaUnallocated",
)
_AMERICA_RESIDUALS = (
    "DACAmericaUnallocated",
    "DACCaribbeanAndCentralAmericaUnallocated",
    "DACCaribbeanUnallocated",
    "DACCentralAmericaUnallocated",
    "DACSouthAmericaUnallocated",
)
_OCEANIA_RESIDUALS = (
    "DACMelanesiaUnallocated",
    "DACMicronesiaUnallocated",
    "DACOceaniaUnallocated",
    "DACPolynesiaUnallocated",
)
_ASIA_RESIDUALS = (
    "DACAsiaUnallocated",
    "DACCentralAsiaUnallocated",
    "DACFarEastAsiaUnallocated",
    "DACMekongDeltaProject",
    "DACMiddleEastUnallocated",
    "DACSouthAndCentralAsiaUnallocated",
    "DACSouthAsiaUnallocated",
)
_SOUTH_OF_SAHARA_RESIDUALS = (
    "DACEasternAfricaUnallocated",
    "DACMiddleAfricaUnallocated",
    "DACSouthOfSaharaUnallocated",
    "DACSouthernAfricaUnallocated",
    "DACWesternAfricaUnallocated",
)
_DAC_REGION = "groups/OECD.DAC.Region."
_REVIEWED_TOTAL_GROUPS = {
    "DACAfrica": TotalGroup("africa", _AFRICA_RESIDUALS),
    "DACAmerica": TotalGroup("undata-geo/G00134000", _AMERICA_RESIDUALS),
    "DACAsia": TotalGroup(f"{_DAC_REGION}Asia", _ASIA_RESIDUALS),
    "DACCaribbean": TotalGroup("Caribbean", ("DACCaribbeanUnallocated",)),
    "DACCaribbeanAndCentralAmerica": TotalGroup(
        f"{_DAC_REGION}CaribbeanAndCentralAmerica",
        (
            "DACCaribbeanAndCentralAmericaUnallocated",
            "DACCaribbeanUnallocated",
            "DACCentralAmericaUnallocated",
        ),
    ),
    "DACCentralAmerica": TotalGroup("CentralAmerica", ("DACCentralAmericaUnallocated",)),
    "DACCountriesTotal": TotalGroup("groups/OECD.DAC", sums_members=False),
    "DACEasternAfrica": TotalGroup(f"{_DAC_REGION}EasternAfrica", ("DACEasternAfricaUnallocated",)),
    "DACEurope": TotalGroup(
        f"{_DAC_REGION}Europe", ("DACEuropeUnallocated", "DACExYugoslaviaUnallocated")
    ),
    "DACFarEastAsia": TotalGroup(
        f"{_DAC_REGION}FarEastAsia", ("DACFarEastAsiaUnallocated", "DACMekongDeltaProject")
    ),
    "DACMelanesia": TotalGroup("Melanesia", ("DACMelanesiaUnallocated",)),
    "DACMicronesia": TotalGroup("Micronesia", ("DACMicronesiaUnallocated",)),
    "DACMiddleAfrica": TotalGroup(f"{_DAC_REGION}MiddleAfrica", ("DACMiddleAfricaUnallocated",)),
    "DACMiddleEast": TotalGroup(f"{_DAC_REGION}MiddleEast", ("DACMiddleEastUnallocated",)),
    "DACNonDACCountries": TotalGroup("groups/OECD.NonDAC"),
    "DACNorthOfSahara": TotalGroup(f"{_DAC_REGION}NorthOfSahara", ("DACNorthOfSaharaUnallocated",)),
    "DACOceania": TotalGroup("oceania", _OCEANIA_RESIDUALS),
    "DACPolynesia": TotalGroup("Polynesia", ("DACPolynesiaUnallocated",)),
    "DACSouthAmerica": TotalGroup(f"{_DAC_REGION}SouthAmerica", ("DACSouthAmericaUnallocated",)),
    "DACSouthAndCentralAsia": TotalGroup(
        f"{_DAC_REGION}SouthAndCentralAsia",
        (
            "DACCentralAsiaUnallocated",
            "DACSouthAndCentralAsiaUnallocated",
            "DACSouthAsiaUnallocated",
        ),
    ),
    "DACSouthOfSahara": TotalGroup(f"{_DAC_REGION}SouthOfSahara", _SOUTH_OF_SAHARA_RESIDUALS),
    "DACSouthernAfrica": TotalGroup("SouthernAfrica", ("DACSouthernAfricaUnallocated",)),
    "DACWesternAfrica": TotalGroup(f"{_DAC_REGION}WesternAfrica", ("DACWesternAfricaUnallocated",)),
    "GroupOfSeven": TotalGroup("GroupOfSeven"),
    "UNLandlockedDevelopingCountries": TotalGroup("groups/UN.LLDC"),
    "UNLeastDevelopedCountries": TotalGroup("groups/UN.LDC"),
    "WorldBankHighIncome": TotalGroup("groups/WB.IncomeGroup.High"),
    "WorldBankLowIncome": TotalGroup("groups/WB.IncomeGroup.Low"),
}


def test_total_groups_holds_only_reviewed_rows():
    # A row joins TOTAL_GROUPS only after the sum check its comment describes, and is added here.
    assert dict(TOTAL_GROUPS) == _REVIEWED_TOTAL_GROUPS


@pytest.mark.parametrize(("party", "total"), sorted(_REVIEWED_TOTAL_GROUPS.items()))
def test_the_packaged_index_carries_each_reviewed_total_group(party: str, total: TotalGroup):
    index = get_index()
    row = index.parties[party]
    assert (row.kind, row.totals, row.residuals, row.sums_members) == (
        "total",
        total.group,
        total.residuals,
        total.sums_members,
    )
    assert index.totals_by_group[total.group] == row
    assert {index.parties[residual].kind for residual in total.residuals} <= {"residual"}
    assert all(index.parties[residual].datasets for residual in total.residuals)


@pytest.mark.parametrize(
    ("party", "kind", "axis"),
    [
        ("WorldBankSubSaharanAfricaExcludingHighIncome", "total", "destination"),
        ("WorldBankIDATotal", "total", "destination"),
        ("IDSBondholders", "residual", "origin"),
        ("EUROFIMA", "party", "origin"),
    ],
)
def test_the_packaged_index_declares_the_ids_parties(party: str, kind: str, axis: str):
    """IDS is a listed dataset, outside `DATASETS`, and its debtors and creditors still need
    a kind for a sum across them."""
    row = get_index().parties[party]
    assert (row.kind, row.datasets, row.axes) == (
        kind,
        ("provenance/WorldBankInternationalDebtStatistics",),
        (axis,),
    )


def test_the_packaged_index_declares_the_exporters_party_types():
    index = get_index()
    assert index.party_types == PARTY_TYPE_KINDS
    assert index.parties


# --- Private providers ---


def test_a_listed_private_provider_takes_the_kind_private(tmp_path: Path):
    document = _export_parties(
        {"GatesFoundation": ("Organization",), "WorldBank": ("Organization",)},
        dac_codes={"GatesFoundation": ("1601",), "WorldBank": ("1011",)},
        private_providers={1601: "GatesFoundation"},
        axis="origin",
    )
    out = tmp_path / "index.json"
    out.write_text(json.dumps(document))

    parties = load_index(out).parties
    assert {dcid: row.kind for dcid, row in parties.items()} == {
        "GatesFoundation": "private",
        "WorldBank": "party",
    }


@pytest.mark.parametrize(
    ("types_by_party", "dac_codes", "private_providers", "kind_overrides", "message"),
    [
        pytest.param(
            {"GatesFoundation": ("Organization",), "SomeNewFoundation": ("Organization",)},
            {"GatesFoundation": ("1601",), "SomeNewFoundation": ("1607",)},
            {1601: "GatesFoundation"},
            {},
            r"do(?:es)? not list.*1607",
            id="unlisted-party-with-a-private-code",
        ),
        pytest.param(
            {"GatesFoundation": ("Organization",), "SomeNewFoundation": ("Organization",)},
            {"GatesFoundation": ("1601",), "SomeNewFoundation": ("1607",)},
            {1601: "GatesFoundation", 1607: "GatesFoundation"},
            {},
            r"different party.*1607",
            id="listed-code-names-another-dcid",
        ),
        pytest.param(
            {"GatesFoundation": ("Organization",)},
            {"GatesFoundation": ("1602",)},
            {1601: "GatesFoundation"},
            {},
            r"different party.*1601",
            id="listed-party-does-not-carry-its-code",
        ),
        pytest.param(
            {"GatesFoundation": ("Organization",)},
            {"GatesFoundation": ("1601",)},
            {1601: "GatesFoundation"},
            {"GatesFoundation": KindOverride("total", "named in error")},
            "both name",
            id="listed-party-with-a-kind-override",
        ),
        pytest.param(
            {"GatesFoundation": ("DACGroup",)},
            {"GatesFoundation": ("1601",)},
            {1601: "GatesFoundation"},
            {},
            "only a party can be private",
            id="listed-party-the-types-make-a-total",
        ),
    ],
)
def test_export_index_refuses_a_private_provider_table_the_graph_contradicts(
    types_by_party, dac_codes, private_providers, kind_overrides, message
):
    with pytest.raises(SystemExit, match=message):
        _export_parties(
            types_by_party,
            dac_codes=dac_codes,
            private_providers=private_providers,
            kind_overrides=kind_overrides,
            axis="origin",
        )


def test_a_private_code_on_a_recipient_is_no_donor_to_list():
    document = _export_parties(
        {"SomeRecipient": ("Country",)}, dac_codes={"SomeRecipient": ("1607",)}, axis="destination"
    )
    assert [row["kind"] for row in document["parties"]] == ["party"]


# The OECD donor codes 1601 to 1650 marked `Private donor` in the donor code list, with the party
# each names in the graph. Two codes name no published party: 1645 (no node) and 1649
# (`NovoNordiskFoundation`, a node no dataset publishes).
_REVIEWED_PRIVATE_PROVIDERS = {
    1601: "GatesFoundation",
    1602: "DutchPostcodeLottery",
    1603: "SwedishPostcodeLottery",
    1604: "PeoplesPostcodeLottery",
    1605: "MetLifeFoundation",
    1606: "MastercardFoundation",
    1608: "IKEAFoundation",
    1609: "VanLeerFoundation",
    1610: "MAVAFoundation",
    1611: "OakFoundation",
    1612: "HAndMFoundation",
    1613: "LaudesFoundation",
    1614: "ComicRelief",
    1615: "ChildrensInvestmentFundFoundation",
    1616: "GatsbyFoundation",
    1617: "ConradNHiltonFoundation",
    1618: "DavidAndLucilePackardFoundation",
    1619: "JohnDAndCatherineTMacArthurFoundation",
    1620: "AndrewCarnegieFoundation",
    1621: "MichaelAndSusanDellFoundation",
    1622: "OmidyarNetwork",
    1623: "RockefellerFoundation",
    1624: "WilliamAndFloraHewlettFoundation",
    1625: "ArcusFoundation",
    1626: "GordonAndBettyMooreFoundation",
    1627: "FordFoundation",
    1628: "WellcomeTrust",
    1629: "UBSOptimusFoundation",
    1630: "WorldDiabetesFoundation",
    1631: "McKnightFoundation",
    1632: "CitiFoundation",
    1633: "LEGOFoundation",
    1634: "NorwegianPostcodeLottery",
    1636: "JacobsFoundation",
    1637: "Arcadia",
    1638: "MargaretACargillFoundation",
    1639: "LaCaixaFoundation",
    1640: "BloombergPhilanthropies",
    1641: "SusanThompsonBuffettFoundation",
    1642: "HowardGBuffettFoundation",
    1643: "OpenSocietyFoundations",
    1644: "FondationBotnar",
    1646: "BezosEarthFund",
    1647: "GermanPostcodeLottery",
    1648: "LeonaMAndHarryBHelmsleyCharitableTrust",
    1650: "GoodVenturesFoundation",
}


def test_private_providers_holds_only_reviewed_rows():
    # A code joins PRIVATE_PROVIDERS only after its OECD `Private donor` row is read against the
    # party the graph gives that code, and is added here.
    assert dict(PRIVATE_PROVIDERS) == _REVIEWED_PRIVATE_PROVIDERS
    assert set(PRIVATE_PROVIDERS) <= set(range(1601, 1651))


def test_the_packaged_index_makes_exactly_the_listed_providers_private():
    index = get_index()
    private = {dcid for dcid, row in index.parties.items() if row.kind == "private"}
    assert private == set(_REVIEWED_PRIVATE_PROVIDERS.values())
    assert all(index.parties[dcid].axes == ("origin",) for dcid in private)


# --- Whole-population totals ---

_OFFICIAL_AND_PRIVATE = {
    "DACOfficialDonors": ("DACGroup",),
    "DACPrivateDonors": ("DACGroup",),
}
_OTHER = "provenance/OtherDataset"
_BOTH = frozenset({_TEST_PROVENANCE, _OTHER})


def _window(*holds: PartyKind, years: tuple[int, int] | None = None) -> WholeWindow:
    return WholeWindow(holds, years)


def test_a_whole_total_carries_one_marker_per_named_dataset_and_a_set_joins_two(tmp_path: Path):
    document = _export_parties(
        {**_OFFICIAL_AND_PRIVATE, "DACMultilaterals": ("DACGroup",)},
        whole_totals={
            "DACOfficialDonors": WholePopulationTotal(
                "origin",
                {_TEST_PROVENANCE: _window("party", years=(1960, 2024)), _OTHER: _window("party")},
                "a",
            ),
            "DACPrivateDonors": WholePopulationTotal(
                "origin", {_TEST_PROVENANCE: _window("private")}, "b"
            ),
        },
        whole_total_sets=(
            WholeTotalSet("all donors including private", "origin", tuple(_OFFICIAL_AND_PRIVATE)),
        ),
        axis="origin",
        published_in=_BOTH,
    )
    out = tmp_path / "index.json"
    out.write_text(json.dumps(document))

    index = load_index(out)
    assert {dcid: row.whole for dcid, row in index.parties.items()} == {
        "DACOfficialDonors": (
            WholeMarker(_OTHER, ("party",)),
            WholeMarker(_TEST_PROVENANCE, ("party",), (1960, 2024)),
        ),
        "DACPrivateDonors": (WholeMarker(_TEST_PROVENANCE, ("private",)),),
        "DACMultilaterals": (),
    }
    assert index.whole_total_sets == (
        WholeTotalSet("all donors including private", "origin", tuple(_OFFICIAL_AND_PRIVATE)),
    )
    # A row outside the table carries no key, and a marker without a window carries no `years`.
    rows = {row["dcid"]: row for row in document["parties"]}
    assert "whole" not in rows["DACMultilaterals"]
    assert rows["DACPrivateDonors"]["whole"] == [
        {"dataset": _TEST_PROVENANCE, "holds": ["private"]}
    ]


def test_a_dataset_the_whole_row_leaves_out_reads_the_total_as_ordinary(tmp_path: Path):
    document = _export_parties(
        {"DACAllRecipients": ("DACGroup",)},
        whole_totals={
            "DACAllRecipients": WholePopulationTotal(
                "destination",
                {_TEST_PROVENANCE: _window("party", "residual", years=(1984, 2024))},
                "a",
            )
        },
        published_in=_BOTH,
    )
    out = tmp_path / "index.json"
    out.write_text(json.dumps(document))

    row = load_index(out).parties["DACAllRecipients"]
    assert row.marker(_TEST_PROVENANCE) == WholeMarker(
        _TEST_PROVENANCE, ("party", "residual"), (1984, 2024)
    )
    assert row.marker(_OTHER) is None


_ORIGIN_TOTAL = {_TEST_PROVENANCE: WholeWindow(("party",))}


@pytest.mark.parametrize(
    ("types_by_party", "whole_totals", "axis", "message"),
    [
        pytest.param(
            _OFFICIAL_AND_PRIVATE,
            {
                "DACOfficialDonors": WholePopulationTotal("origin", _ORIGIN_TOTAL, "a"),
                "DACPrivateDonors": WholePopulationTotal(
                    "origin", {_TEST_PROVENANCE: _window("party", "private")}, "b"
                ),
            },
            "origin",
            "both hold",
            id="overlapping-markers",
        ),
        pytest.param(
            {"country/FRA": ("Country",)},
            {"country/FRA": WholePopulationTotal("origin", _ORIGIN_TOTAL, "a")},
            "origin",
            "publishes as a total",
            id="whole-row-on-a-party",
        ),
        pytest.param(
            _OFFICIAL_AND_PRIVATE,
            {"DACOfficialDonors": WholePopulationTotal("origin", _ORIGIN_TOTAL, "a")},
            "destination",
            "publish it on",
            id="whole-row-on-another-axis",
        ),
        pytest.param(
            _OFFICIAL_AND_PRIVATE,
            {"DACOfficialDonors": WholePopulationTotal("origin", {_OTHER: _window("party")}, "a")},
            "origin",
            "do not publish it",
            id="dataset-named-for-a-party-it-does-not-publish",
        ),
        pytest.param(
            _OFFICIAL_AND_PRIVATE,
            {"DACGhost": WholePopulationTotal("origin", _ORIGIN_TOTAL, "a")},
            "origin",
            "publishes as a total",
            id="whole-row-on-an-unpublished-party",
        ),
        pytest.param(
            _OFFICIAL_AND_PRIVATE,
            {"DACOfficialDonors": WholePopulationTotal("origin", _ORIGIN_TOTAL, "")},
            "origin",
            "no reason",
            id="whole-row-without-a-reason",
        ),
        pytest.param(
            _OFFICIAL_AND_PRIVATE,
            {"DACOfficialDonors": WholePopulationTotal("origin", {}, "a")},
            "origin",
            "names no dataset",
            id="whole-row-without-a-dataset",
        ),
        pytest.param(
            _OFFICIAL_AND_PRIVATE,
            {
                "DACOfficialDonors": WholePopulationTotal(
                    "origin", {_TEST_PROVENANCE: _window("total")}, "a"
                )
            },
            "origin",
            "outside",
            id="whole-row-holding-a-total",
        ),
    ],
)
def test_export_index_refuses_a_whole_total_row_it_cannot_stand_behind(
    types_by_party, whole_totals, axis, message
):
    with pytest.raises(SystemExit, match=message):
        _export_parties(types_by_party, whole_totals=whole_totals, axis=axis)


def test_export_index_refuses_a_whole_total_set_naming_a_total_that_holds_no_population():
    with pytest.raises(SystemExit, match="no whole total"):
        _export_parties(
            {**_OFFICIAL_AND_PRIVATE, "DACMultilaterals": ("DACGroup",)},
            whole_totals={"DACOfficialDonors": WholePopulationTotal("origin", _ORIGIN_TOTAL, "a")},
            whole_total_sets=(
                WholeTotalSet("both", "origin", ("DACOfficialDonors", "DACMultilaterals")),
            ),
            axis="origin",
        )


_DAC2A = "provenance/OECDDAC2A"
_DAC2B = "provenance/OECDDAC2B"
# Each whole-population total with the markers its composition checks gave, as reviewed against
# the comment on `WHOLE_POPULATION_TOTALS`: the axis, then each dataset's kinds and window.
_REVIEWED_WHOLE_TOTALS = {
    "DACAllRecipients": ("destination", {_DAC2A: (("party", "residual"), (1984, 2024))}),
    "DACOfficialDonors": ("origin", {_DAC2A: (("party",), None), _DAC2B: (("party",), None)}),
    "DACPrivateDonors": ("origin", {_DAC2A: (("private",), None)}),
}
_REVIEWED_WHOLE_TOTAL_SETS = (
    WholeTotalSet(
        "all donors including private", "origin", ("DACOfficialDonors", "DACPrivateDonors")
    ),
    WholeTotalSet("all donors", "origin", ("DACOfficialDonors", "DACPrivateDonors")),
    WholeTotalSet("every donor", "origin", ("DACOfficialDonors", "DACPrivateDonors")),
)


def test_whole_population_totals_holds_only_reviewed_rows():
    # A row joins WHOLE_POPULATION_TOTALS only after the composition check its comment describes,
    # and is added here. DACAllRecipients names DAC2A alone: DAC2B fails the check.
    assert {
        dcid: (row.axis, {d: (w.holds, w.years) for d, w in row.datasets.items()})
        for dcid, row in WHOLE_POPULATION_TOTALS.items()
    } == _REVIEWED_WHOLE_TOTALS
    assert all(row.reason for row in WHOLE_POPULATION_TOTALS.values())
    assert tuple(WHOLE_TOTAL_SETS) == _REVIEWED_WHOLE_TOTAL_SETS


def test_the_packaged_index_carries_each_reviewed_whole_total():
    index = get_index()
    for dcid, (axis, datasets) in _REVIEWED_WHOLE_TOTALS.items():
        row = index.parties[dcid]
        assert (row.kind, row.axes) == ("total", (axis,))
        assert row.whole == tuple(
            WholeMarker(dataset, holds, years)
            for dataset, (holds, years) in sorted(datasets.items())
        )
    assert {dcid for dcid, row in index.parties.items() if row.whole} == set(_REVIEWED_WHOLE_TOTALS)
    assert index.parties["DACAllRecipients"].marker(_DAC2B) is None
    assert tuple(index.whole_total_sets) == _REVIEWED_WHOLE_TOTAL_SETS
