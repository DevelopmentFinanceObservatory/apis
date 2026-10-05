"""Checks measure-axis withholding and election results at the bind and repair phase boundaries.

Synthetic axes separate axis-kind behavior from property names; direct `ResolveState` fixtures isolate
phase wiring."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import cast

import search.engine.region.phase_election as phase_election_module
import search.engine.region.phase_rank_bind as phase_rank_bind_module
from code_index import CodeIndex, IndexStamp, StatVarRow
from code_index.types import MeasureAxis, MeasureIndex
from entity_resolution import Admission, EntityResolver
from search.engine.bind import DraftBindingKind, SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.region.phase_election import _phase_repair_bindings
from search.engine.region.phase_rank_bind import _phase_bind
from search.engine.region.state import ResolveState
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._fake_coverage import empty_coverage_snapshot

_FIVE_TUPLE = {
    "pop_type_dcid": "Widget",
    "meas_prop_dcid": "count",
    "stat_type_dcid": "measuredValue",
    "meas_qual_dcid": None,
    "meas_denom_dcid": None,
}

_MATERIAL = MeasureAxis(
    axis="material", kind="constraint", role="free", prefer=("Plastic", "Metal")
)
_FINISH = MeasureAxis(axis="finish", kind="constraint", role="free", prefer=("Matte", "Glossy"))
_MATERIAL_SLOT = SlotKeyDraft(axis="what", property_dcid="material", label="material")
_FINISH_SLOT = SlotKeyDraft(axis="what", property_dcid="finish", label="finish")


def _row(dcid: str, **constraints: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type=_FIVE_TUPLE["pop_type_dcid"],
        measured_property=_FIVE_TUPLE["meas_prop_dcid"],
        stat_type=_FIVE_TUPLE["stat_type_dcid"],
        measurement_qualifier=_FIVE_TUPLE["meas_qual_dcid"],
        measurement_denominator=_FIVE_TUPLE["meas_denom_dcid"],
        observation_properties=(),
        constraint_properties=constraints,
        provenances=("provenance/Test",),
    )


def _index(statvars: list[StatVarRow]) -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test-hash",
        ),
        declarations=(),
        codes=(),
        statvars=statvars,
    )


def _measure_index(*axes: MeasureAxis) -> MeasureIndex:
    return MeasureIndex(axes=axes, values=(), partition_labels=())


def _shape(
    *,
    slot_keys: tuple[SlotKeyDraft, ...],
    place_available: bool | None = None,
    slot_taxonomy: dict[str, list[str]] | None = None,
    sv_arc_facts: dict[str, dict] | None = None,
    value_labels: dict[str, str] | None = None,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id="widget",
        label="widget",
        pop_type_dcid=_FIVE_TUPLE["pop_type_dcid"],
        meas_prop_dcid=_FIVE_TUPLE["meas_prop_dcid"],
        stat_type_dcid=_FIVE_TUPLE["stat_type_dcid"],
        meas_qual_dcid=_FIVE_TUPLE["meas_qual_dcid"],
        meas_denom_dcid=_FIVE_TUPLE["meas_denom_dcid"],
        slot_keys=slot_keys,
        place_available=place_available,
        slot_taxonomy=slot_taxonomy,
        sv_arc_facts=sv_arc_facts,
        value_labels=value_labels,
    )


def _state(*, shape_draft: ShapeDraft, bindings: list[SlotBindingDraft]) -> ResolveState:
    """Build a minimal state for binding repair; service dependencies are unused by these tests."""
    return ResolveState(
        variable="a widget query",
        entities=["Kenya"],
        date_request=None,
        detect_query="a widget query",
        role_query="a widget query",
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        source_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        pipeline_steps=[],
        timing={},
        warnings=[],
        bind_result=_UniversalBindOutput(
            chosen_shape_index=0, bindings=bindings, also_plausible=None, ask=None
        ),
        shape_draft=shape_draft,
        coverage=empty_coverage_snapshot(),
    )


def _run_repair(state: ResolveState) -> list[SlotBindingDraft]:
    result = asyncio.run(_phase_repair_bindings(state))
    assert result is None, "expected _phase_repair_bindings to fall through"
    return state.bindings


def _binding(
    prop: str,
    *,
    kind: DraftBindingKind = "unbound",
    value_dcids: list[str] | None = None,
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="what", property_dcid=prop, kind=kind, value_dcids=value_dcids or []
    )


class _RecordingBindLLM:
    """Records the bind prompt and leaves constraint axes unbound."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=None, ask=None
        ), None


def test_phase_bind_withholds_the_declared_constraint_measure_axes_from_the_prompt(
    monkeypatch,
):
    """`_phase_bind` excludes declared constraint axes from an index-sourced shape's prompt."""
    monkeypatch.setattr(
        phase_rank_bind_module, "get_measure_index", lambda: _measure_index(_MATERIAL)
    )
    shape = dataclasses.replace(
        _shape(slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT)), slot_taxonomy_index_sourced=True
    )
    llm = _RecordingBindLLM()
    state = _state(shape_draft=shape, bindings=[])
    state.capped_shapes = [shape]
    state.llm = llm

    result = asyncio.run(_phase_bind(state))

    assert result is None
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "property_dcid=material" not in prompt
    assert "property_dcid=finish" in prompt


def test_no_measure_binding_forces_the_elected_pair_as_value(monkeypatch):
    monkeypatch.setattr(
        phase_election_module, "get_index", lambda: _index([_row("SV1", finish="Matte")])
    )
    monkeypatch.setattr(phase_election_module, "get_measure_index", lambda: _measure_index(_FINISH))
    shape = _shape(slot_keys=(_FINISH_SLOT,))
    state = _state(shape_draft=shape, bindings=[_binding("finish", kind="unbound")])

    bindings = _run_repair(state)

    assert len(bindings) == 1
    assert bindings[0].property_dcid == "finish"
    assert bindings[0].kind == "value"
    assert bindings[0].value_dcids == ["Matte"]


def test_a_forced_axis_replaces_an_absent_draft_and_only_one_binding_survives(monkeypatch):
    """An absent draft is replaced by one forced value binding."""
    monkeypatch.setattr(
        phase_election_module, "get_index", lambda: _index([_row("SV1", finish="Matte")])
    )
    monkeypatch.setattr(phase_election_module, "get_measure_index", lambda: _measure_index(_FINISH))
    shape = _shape(slot_keys=(_FINISH_SLOT,))
    state = _state(shape_draft=shape, bindings=[_binding("finish", kind="absent")])

    bindings = _run_repair(state)

    finish_bindings = [b for b in bindings if b.property_dcid == "finish"]
    assert len(finish_bindings) == 1
    assert finish_bindings[0].kind == "value"
    assert finish_bindings[0].value_dcids == ["Matte"]


def test_a_query_named_value_is_not_re_elected_but_every_other_open_axis_still_is(
    monkeypatch,
):
    # Binding material=Metal narrows the correlated rows to the Metal/Glossy group.
    monkeypatch.setattr(
        phase_election_module,
        "get_index",
        lambda: _index(
            [
                _row("SV_METAL", material="Metal", finish="Glossy"),
                _row("SV_PLASTIC", material="Plastic", finish="Matte"),
            ]
        ),
    )
    monkeypatch.setattr(
        phase_election_module, "get_measure_index", lambda: _measure_index(_MATERIAL, _FINISH)
    )
    shape = _shape(slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT))
    already_forced = _binding("material", kind="value", value_dcids=["Metal"])
    state = _state(shape_draft=shape, bindings=[already_forced, _binding("finish", kind="unbound")])

    bindings = _run_repair(state)

    by_prop = {b.property_dcid: b for b in bindings}
    assert len(bindings) == 2
    # The query-named material draft survives unchanged.
    assert by_prop["material"] is already_forced
    assert by_prop["material"].value_dcids == ["Metal"]
    assert by_prop["finish"].kind == "value"
    assert by_prop["finish"].value_dcids == ["Glossy"]


def test_a_restored_binding_is_not_overridden_while_a_sibling_axis_is_still_elected(
    monkeypatch,
):
    # Empty taxonomy lets text restore material=Plastic before election; finish then elects Matte.
    monkeypatch.setattr(
        phase_election_module,
        "get_index",
        lambda: _index(
            [
                _row("SV_PLASTIC_MATTE", material="Plastic", finish="Matte"),
                _row("SV_PLASTIC_GLOSSY", material="Plastic", finish="Glossy"),
                _row("SV_METAL", material="Metal"),
            ]
        ),
    )
    monkeypatch.setattr(
        phase_election_module, "get_measure_index", lambda: _measure_index(_MATERIAL, _FINISH)
    )
    # The recall sample holds index rows only, so the family election reads the index.
    shape = _shape(
        slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT),
        place_available=True,
        slot_taxonomy={"what:material": []},
        sv_arc_facts={
            "SV_PLASTIC_MATTE": {
                "constraintProperties": {"nodes": [{"dcid": "material"}]},
                "material": {"nodes": [{"dcid": "Plastic"}]},
            },
            "SV_METAL": {
                "constraintProperties": {"nodes": [{"dcid": "material"}]},
                "material": {"nodes": [{"dcid": "Metal"}]},
            },
        },
        value_labels={"Plastic": "plastic", "Metal": "metal"},
    )
    state = _state(
        shape_draft=shape,
        bindings=[_binding("material", kind="unbound"), _binding("finish", kind="unbound")],
    )
    state.variable = "plastic widgets"

    bindings = _run_repair(state)

    by_prop = {b.property_dcid: b for b in bindings}
    assert len(bindings) == 2
    assert by_prop["material"].kind == "value"
    assert by_prop["material"].value_dcids == ["Plastic"]
    assert by_prop["finish"].kind == "value"
    assert by_prop["finish"].value_dcids == ["Matte"]


def test_a_scope_that_populates_no_partition_forces_nothing(monkeypatch):
    # An empty index produces no election, so the seed binding remains unchanged.
    monkeypatch.setattr(phase_election_module, "get_index", lambda: _index([]))
    monkeypatch.setattr(phase_election_module, "get_measure_index", lambda: _measure_index(_FINISH))
    shape = _shape(slot_keys=(_FINISH_SLOT,))
    seed = _binding("finish", kind="unbound")
    state = _state(shape_draft=shape, bindings=[seed])

    bindings = _run_repair(state)

    assert bindings == [seed]
    # No election means no dropped axes.
    assert state.measure_dropped_axes == ()


_WARRANTY = MeasureAxis(axis="warranty", kind="constraint", role="free", prefer=("Yes", "No"))
_WARRANTY_SLOT = SlotKeyDraft(axis="what", property_dcid="warranty", label="warranty")


def test_dropped_axis_reaches_state_beside_forced(monkeypatch):
    # The winner declares finish; the rows omit warranty, so the election drops it.
    monkeypatch.setattr(
        phase_election_module, "get_index", lambda: _index([_row("SV1", finish="Matte")])
    )
    monkeypatch.setattr(
        phase_election_module, "get_measure_index", lambda: _measure_index(_FINISH, _WARRANTY)
    )
    shape = _shape(slot_keys=(_FINISH_SLOT, _WARRANTY_SLOT))
    state = _state(
        shape_draft=shape,
        bindings=[_binding("finish", kind="unbound"), _binding("warranty", kind="unbound")],
    )

    bindings = _run_repair(state)

    by_prop = {b.property_dcid: b for b in bindings}
    assert by_prop["finish"].value_dcids == ["Matte"]
    assert by_prop["warranty"].kind == "unbound"
    assert state.measure_dropped_axes == ("warranty",)
