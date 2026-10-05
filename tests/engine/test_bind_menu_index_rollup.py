"""Index-carried shapes use covering roots from the packaged code index in the bind prompt.
Other shapes keep recall-sourced values unchanged.

The FinancialFlow/amount covering roots are 10 for crsPurpose, 6 for crsTypeOfFinance, 8 for
crsCooperationModality, and 4 for crsTypeOfFlow. They sit below the 20-value cap. Raw counts in this
five-tuple are 286, 42, 24, and 4, respectively. Across all amount rows, finance and modality have
44 and 25 values. The prompt menu uses the five-tuple counts.
"""

from __future__ import annotations

import asyncio

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _build_universal_bind_prompt, _UniversalBindOutput
from search.engine.qualifier_match import QualifierMatch, QualifierTie
from search.engine.region import resolve_variable
from search.engine.region.codes import REDUNDANT_BINDING_DROPPED
from search.engine.retrieval.derive import derive_shapes
from tests.engine._discover_fixtures import PROP_PURPOSE as _DEMO_PROP_PURPOSE
from tests.engine._discover_fixtures import _fake_graph_with_svs, _three_constraint_arcs
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_POP_TYPE = "FinancialFlow"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"

_AMOUNT_MEAS_PROP = "amount"
_AMOUNT_SV = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)

_GRANT_EQ_MEAS_PROP = "grantEquivalent"
_GRANT_EQ_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)

_PROP_PURPOSE = "crsPurpose"
_PROP_FINANCE = "crsTypeOfFinance"
_PROP_MODALITY = "crsCooperationModality"
_PROP_FLOW = "crsTypeOfFlow"


def _sv_arcs_from_real_row(dcid: str, *, meas_prop: str) -> dict:
    """Build graph arcs from the StatVar's packaged-index row."""
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == dcid)
    constraint_props = list(row.constraint_properties)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "measurementQualifier": {"nodes": [{"dcid": _MEAS_QUAL}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraint_props]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def _derive_single_shape(recalled_dcid: str, *, meas_prop: str):
    """Derive a shape from one confirmed CRS member without place scoping."""
    arcs = _sv_arcs_from_real_row(recalled_dcid, meas_prop=meas_prop)
    graph = FakeGraph(
        nodes={
            recalled_dcid: {"label": recalled_dcid, "type": "StatisticalVariable", "arcs": arcs}
        },
        detect={},
        resolve={},
        obs={},
    )
    shapes = derive_shapes(confirmed_svs=[recalled_dcid], graph=graph)
    assert len(shapes) == 1
    return shapes[0]


def test_fixture_matches_the_real_packaged_index():
    """Confirm the fixture StatVars and their amount-only flow value exist in the packaged index."""
    index = get_index()
    amount_row = next(r for r in index.statvars if r.dcid == _AMOUNT_SV)
    grant_row = next(r for r in index.statvars if r.dcid == _GRANT_EQ_SV)
    assert amount_row.measured_property == _AMOUNT_MEAS_PROP
    assert grant_row.measured_property == _GRANT_EQ_MEAS_PROP

    def _group(row):
        return (
            row.population_type or "",
            row.measured_property,
            row.stat_type or "",
            row.measurement_qualifier,
            row.measurement_denominator,
        )

    amount_flow_values = {
        r.constraint_properties[_PROP_FLOW]
        for r in index.statvars
        if _group(r) == _group(amount_row) and _PROP_FLOW in r.constraint_properties
    }
    grant_flow_values = {
        r.constraint_properties[_PROP_FLOW]
        for r in index.statvars
        if _group(r) == _group(grant_row) and _PROP_FLOW in r.constraint_properties
    }
    assert "CRSTypeOfFlow30" in amount_flow_values
    assert "CRSTypeOfFlow30" not in grant_flow_values


# ---------------------------------------------------------------------------
# Compare recall-sourced and index-sourced prompt menus.
# ---------------------------------------------------------------------------


def test_a_non_index_carried_shape_renders_its_raw_recall_values_unchanged():
    """A shape absent from the index keeps its insertion-ordered recall values."""
    sv_a = "CrsShapedFixture/A"
    sv_b = "CrsShapedFixture/B"
    graph = _fake_graph_with_svs(
        {
            sv_a: _three_constraint_arcs("PurposeOne", "SchemeOne", "country/AAA"),
            sv_b: _three_constraint_arcs("PurposeTwo", "SchemeOne", "country/AAA"),
        }
    )
    shapes = derive_shapes(confirmed_svs=[sv_a, sv_b], graph=graph)
    assert len(shapes) == 1
    shape = shapes[0]
    assert shape.slot_taxonomy_index_sourced is False

    prompt = _build_universal_bind_prompt("test variable", shapes)
    purpose_key = f"how:{_DEMO_PROP_PURPOSE}"
    assert shape.slot_taxonomy is not None
    raw_values = shape.slot_taxonomy[purpose_key]
    assert set(raw_values) == {"PurposeOne", "PurposeTwo"}
    assert f"taxonomy: {raw_values}" in prompt


def test_an_index_carried_shapes_menu_carries_the_covering_roots():
    """An indexed CRS shape renders 10/6/8/4 covering roots, below the 20-value cap."""
    shape = _derive_single_shape(_AMOUNT_SV, meas_prop=_AMOUNT_MEAS_PROP)
    assert shape.slot_taxonomy_index_sourced is True
    assert shape.slot_taxonomy is not None

    # Keep full taxonomy values for grounding. Roll up only the prompt display.
    assert len(shape.slot_taxonomy[f"how:{_PROP_PURPOSE}"]) > 100

    prompt = _build_universal_bind_prompt("test variable", [shape])

    expected_counts = {
        _PROP_PURPOSE: 10,
        _PROP_FINANCE: 6,
        _PROP_MODALITY: 8,
        _PROP_FLOW: 4,
    }
    for slot in shape.slot_keys:
        prop = slot.property_dcid
        if prop not in expected_counts:
            continue
        line_prefix = f"axis={slot.axis}, property_dcid={prop}"
        assert line_prefix in prompt, f"expected a prompt line for {prop}"
        # The line immediately after the "axis=..." header is "taxonomy: [...]".
        lines = prompt.splitlines()
        header_idx = next(i for i, ln in enumerate(lines) if line_prefix in ln)
        taxonomy_line = lines[header_idx + 1]
        assert taxonomy_line.strip().startswith("taxonomy: [")
        assert "more)" not in taxonomy_line, (
            f"{prop}'s covering-root menu should sit under the truncation cap: {taxonomy_line}"
        )
        rendered_count = taxonomy_line.count("'") // 2
        assert rendered_count == expected_counts[prop], (
            f"{prop}: expected {expected_counts[prop]} covering roots, prompt line was "
            f"{taxonomy_line!r}"
        )

    # This flat code system has four values and needs no rollup.
    flow_key = f"how:{_PROP_FLOW}"
    assert set(shape.slot_taxonomy[flow_key]) == {
        "CRSTypeOfFlow10",
        "CRSTypeOfFlow21",
        "CRSTypeOfFlow30",
        "CRSTypeOfFlow60",
    }


def test_an_index_carried_shapes_menu_renders_bare_dcids_plus_a_labels_line():
    """Taxonomy values stay bare dcids so verbatim bindings match graph arcs. Labels use a separate
    prompt line to help the binder match terms such as "aid" to a code.
    """
    shape = _derive_single_shape(_AMOUNT_SV, meas_prop=_AMOUNT_MEAS_PROP)
    assert shape.slot_taxonomy_index_sourced is True

    prompt = _build_universal_bind_prompt("test variable", [shape])

    flow_key = f"how:{_PROP_FLOW}"
    line_prefix = f"axis=how, property_dcid={_PROP_FLOW}"
    lines = prompt.splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if line_prefix in ln)
    taxonomy_line = lines[header_idx + 1]
    labels_line = lines[header_idx + 2]

    # The taxonomy line itself is bare dcids, byte-identical in shape to a
    # recall-sourced shape's own taxonomy line.
    for dcid in shape.slot_taxonomy[flow_key]:
        assert f"'{dcid}'" in taxonomy_line
    assert "(" not in taxonomy_line

    # The labels line decodes the same dcids, one line below.
    assert labels_line.strip().startswith("labels: ")
    assert "CRSTypeOfFlow10 = official development assistance" in labels_line
    assert "CRSTypeOfFlow21 = non-export credit official flow" in labels_line
    assert "CRSTypeOfFlow30 = private development finance" in labels_line
    assert "CRSTypeOfFlow60 = private sector instrument" in labels_line


def test_a_tied_slots_labels_line_is_omitted_in_favor_of_its_named_alternatives():
    """A tied purpose slot omits labels so the named alternatives outrank the covering root."""
    shape = _derive_single_shape(_AMOUNT_SV, meas_prop=_AMOUNT_MEAS_PROP)
    assert shape.slot_taxonomy_index_sourced is True
    assert "CRSPurposeCode100" in shape.slot_taxonomy[f"how:{_PROP_PURPOSE}"]

    tie = QualifierTie(
        matched_text="upper secondary education",
        rows=(
            QualifierMatch(
                code_system=_PROP_PURPOSE,
                code_dcid="CRSPurposeCode113",
                label="upper secondary education",
                source="label",
                matched_text="upper secondary education",
            ),
            QualifierMatch(
                code_system=_PROP_PURPOSE,
                code_dcid="CRSPurposeCode11320",
                label="upper secondary schooling",
                source="synonym",
                matched_text="upper secondary education",
            ),
        ),
    )

    prompt = _build_universal_bind_prompt("test variable", [shape], admitted_ties=(tie,))
    lines = prompt.splitlines()

    assert "QUERY-NAMED ALTERNATIVES" in prompt
    assert (
        "property_dcid=crsPurpose, value_dcid=CRSPurposeCode113 (upper secondary education)"
        in prompt
    )
    assert (
        "property_dcid=crsPurpose, value_dcid=CRSPurposeCode11320 (upper secondary schooling)"
        in prompt
    )

    # The tied slot (crsPurpose) keeps its taxonomy line and has no labels line.
    purpose_prefix = f"axis=how, property_dcid={_PROP_PURPOSE}"
    purpose_idx = next(i for i, ln in enumerate(lines) if purpose_prefix in ln)
    assert lines[purpose_idx + 1].strip().startswith("taxonomy: [")
    assert not lines[purpose_idx + 2].strip().startswith("labels:")

    # Without a tie, crsTypeOfFlow keeps its labels line.
    flow_prefix = f"axis=how, property_dcid={_PROP_FLOW}"
    flow_idx = next(i for i, ln in enumerate(lines) if flow_prefix in ln)
    assert lines[flow_idx + 2].strip().startswith("labels: ")
    assert "CRSTypeOfFlow10 = official development assistance" in lines[flow_idx + 2]


def test_a_grant_equivalent_shapes_menu_never_carries_an_amount_only_flow_type():
    """The grantEquivalent menu excludes CRSTypeOfFlow30, which only the amount family realizes."""
    shape = _derive_single_shape(_GRANT_EQ_SV, meas_prop=_GRANT_EQ_MEAS_PROP)
    assert shape.slot_taxonomy_index_sourced is True
    flow_key = f"how:{_PROP_FLOW}"
    assert "CRSTypeOfFlow30" not in shape.slot_taxonomy[flow_key]
    assert shape.slot_taxonomy[flow_key]  # sanity: the axis is not simply empty


# ---------------------------------------------------------------------------
# A synthetic flat index checks the unrolled menu branch.
# ---------------------------------------------------------------------------

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _synthetic_flat_index() -> CodeIndex:
    codes = [
        CodeRow(
            dcid="FlatA", code_system="synthFlat", code="A", label="A", synonyms=(), parent=None
        ),
        CodeRow(
            dcid="FlatB", code_system="synthFlat", code="B", label="B", synonyms=(), parent=None
        ),
    ]
    declarations = [
        CodeSystemDeclaration(
            code_system="synthFlat",
            hierarchical=False,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="axis",
        )
    ]
    return CodeIndex(stamp=_STAMP, declarations=declarations, codes=codes, statvars=())


def test_a_flat_index_sourced_systems_menu_is_its_values_unrolled(monkeypatch):
    from search.engine import bind as bind_module
    from search.engine.shape import ShapeDraft, SlotKeyDraft

    monkeypatch.setattr(bind_module, "get_index", lambda: _synthetic_flat_index())

    shape = ShapeDraft(
        shape_id="synthetic-flat",
        label="synthetic flat shape",
        pop_type_dcid="TestPop",
        meas_prop_dcid="testMeasure",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="synthFlat", label="synth flat"),),
        slot_taxonomy={"how:synthFlat": ["FlatA", "FlatB"]},
        slot_taxonomy_index_sourced=True,
    )

    prompt = _build_universal_bind_prompt("test variable", [shape])
    assert "taxonomy: ['FlatA', 'FlatB']" in prompt
    assert "labels: FlatA = A; FlatB = B" in prompt


# ---------------------------------------------------------------------------
# End-to-end test for repair and `SlotOptions.truncated` on an index-sourced slot.
# ---------------------------------------------------------------------------

_QUERY = "grant equivalent disbursements to Kenya"
_KEN_DCID = "country/KEN"

_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _GRANT_EQ_SV: {
        "label": "test recalled grant equivalent",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs_from_real_row(_GRANT_EQ_SV, meas_prop=_GRANT_EQ_MEAS_PROP),
    },
    # SlotOptions only surfaces an entry for a slot whose option dcids
    # resolve at least one ->name label (region.refs._refs_present) --
    # labelling the flat crsTypeOfFlow axis's two real, index-carried values
    # is enough to observe truncated on that entry, without labelling
    # crsPurpose's own 200+ covering-root dcids.
    "CRSTypeOfFlow10": {"label": "Disbursement flow"},
    "CRSTypeOfFlow60": {"label": "Guarantee flow"},
}
_DETECT = {_QUERY: {"svs": [_GRANT_EQ_SV], "cosine_scores": [0.9]}}
_OBS = {
    f"{_GRANT_EQ_SV}|{_KEN_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}]
}


class _BindPurposeMentionedFlowSoleOption:
    """Bind `crsPurpose` as mentioned and model `crsTypeOfFlow` as a sole option."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_PURPOSE,
                    kind="value",
                    value_dcids=["CRSPurposeCode51010"],
                    evidence="mentioned",
                ),
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_FLOW,
                    kind="value",
                    value_dcids=["CRSTypeOfFlow10"],
                    evidence="sole_option",
                ),
            ],
            also_plausible=[],
            ask=None,
        ), None


def _run():
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs=_OBS)
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_BindPurposeMentionedFlowSoleOption(),
            base_steps=[],
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


def test_a_sole_option_claim_over_an_index_widened_axis_leaves_the_slot_open():
    """The recalled member holds one flow value and the index's family holds two, so the
    slot offers two and the model's sole-option claim is false. The slot stays open, and
    the answer does not narrow to a flow the question never named. Singleton repair has
    nothing to drop.
    """
    result = _run()
    assert result.status == "definite"
    spec = result.specs[0]
    assert not any(w.code == REDUNDANT_BINDING_DROPPED for w in result.warnings)

    flow_slot = next(
        s
        for s in spec.resolved.slots
        if s.key.property is not None and s.key.property.dcid == _PROP_FLOW
    )
    assert flow_slot.binding.kind == "unbound"
    purpose_slot = next(
        s
        for s in spec.resolved.slots
        if s.key.property is not None and s.key.property.dcid == _PROP_PURPOSE
    )
    assert purpose_slot.binding.kind == "value"


def test_slot_options_truncated_is_false_for_the_index_sourced_axis():
    """Index-sourced taxonomy covers the full known set, so `truncated` is False.
    Recall-sourced values come from a capped sample.
    """
    result = _run()
    spec = result.specs[0]
    flow_options = next(
        so
        for so in spec.resolved.slot_options
        if so.key.property is not None and so.key.property.dcid == _PROP_FLOW
    )
    assert flow_options.truncated is False
    assert flow_options.source == "observed"
