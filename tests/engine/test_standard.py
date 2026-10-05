"""Offline tests cover standard resolution, data absence, and unresolved variables.

Standard entities use the subject role and retain their resolved place.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.core import resolve_async
from search.engine.extract import ExtractedEntity, Extraction
from search.models import (
    DefinitePart,
    NoDataPart,
    RawTextInput,
    ResolveRequest,
    ResolveResponse,
)
from tests.engine._harness import EMPTY_SDMX, offline_resolve
from tests.engine._harness import make_request as _req
from tests.fixtures import FakeGraph

# std-01 resolves Count_Person for India.


class TestStd01TotalPopulationIndia:
    """std-01 resolves total population in India with Count_Person."""

    def test_status_is_definite(self):
        result = offline_resolve(_req("total population India"))
        assert result.status == "definite"

    def test_shape_is_person_count(self):
        result = offline_resolve(_req("total population India"))
        inner = result.parts[0]
        assert isinstance(inner, DefinitePart)
        spec = inner.spec
        shape = spec.resolved.shape
        assert shape.population_type and shape.population_type.dcid == "Person"
        assert shape.measured_property and shape.measured_property.dcid == "count"
        assert shape.stat_type and shape.stat_type.dcid == "measuredValue"

    def test_sv_is_count_person(self):
        result = offline_resolve(_req("total population India"))
        inner = result.parts[0]
        assert isinstance(inner, DefinitePart)
        sv_dcids = [sv.ref for sv in inner.spec.resolved.stat_vars.items]
        assert "Count_Person" in sv_dcids

    def test_entity_is_india(self):
        result = offline_resolve(_req("total population India"))
        inner = result.parts[0]
        assert isinstance(inner, DefinitePart)
        entity_dcids = [e.ref for e in inner.spec.resolved.entities]
        assert "country/IND" in entity_dcids

    def test_entity_role_is_subject(self):
        """Standard entities use subject role when the query names no flow direction."""
        result = offline_resolve(_req("total population India"))
        inner = result.parts[0]
        assert isinstance(inner, DefinitePart)
        for entity in inner.spec.resolved.entities:
            if entity.ref == "country/IND":
                assert entity.role == "observationAbout"


# nd-01 has no observations for Nauru's teacher count.


class TestNd01TeacherCountNauru:
    """Count_Teacher matches but has no observations for Nauru, so it appears as `flagged_match`."""

    def test_status_is_no_data(self):
        result = offline_resolve(_req("teacher count in Nauru"))
        assert result.status == "no_data"

    def test_reason_is_no_observations(self):
        result = offline_resolve(_req("teacher count in Nauru"))
        inner = result.parts[0]
        assert isinstance(inner, NoDataPart)
        assert inner.no_data.reason == "no_observations"

    def test_flagged_match_is_populated(self):
        result = offline_resolve(_req("teacher count in Nauru"))
        inner = result.parts[0]
        assert isinstance(inner, NoDataPart)
        flagged = inner.no_data.flagged_match
        assert flagged is not None
        assert flagged.resolved.available_at_place is False
        assert flagged.resolved.confidence == "low"
        assert flagged.resolved.stat_vars.items, (
            "flagged_match must carry the matched-but-unavailable SV"
        )


# nd-03 leaves left-handedness in France unresolved.


class TestNd03LeftHandednessFrance:
    """Recall scores near 0.42 fall below the 0.5 threshold; no left-handedness StatVar exists."""

    def test_status_is_no_data(self):
        result = offline_resolve(_req("left-handedness rate in France"))
        assert result.status == "no_data"

    def test_reason_is_variable_not_resolved(self):
        result = offline_resolve(_req("left-handedness rate in France"))
        inner = result.parts[0]
        assert isinstance(inner, NoDataPart)
        assert inner.no_data.reason == "variable_not_resolved"


# Partial five-tuples resolve when observations confirm them.


class _ExtractOnlyLLM:
    """Returns a fixed extraction and elects the highest-scoring shape without bindings."""

    def __init__(self, variable: str, entities: list[str]):
        self._extraction = Extraction(
            kind="resolution",
            variables=[variable],
            entities=[ExtractedEntity(text=e) for e in entities],
        )

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "Extraction":
            return self._extraction, None
        if schema.__name__ == "_UniversalBindOutput":
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[],
                also_plausible=None,
                ask=None,
            ), None
        raise AssertionError(f"unexpected schema {schema.__name__!r}")


# Custom DC records can omit measuredProperty or another tuple arc.
# Observations decide answerability.

_SPARSE_SV = "Amount_HealthFinancingScheme_Test"
_COMPLETE_SV = "Amount_Econ_Health_Test"
_OOP_QUERY = "out-of-pocket health spending Kenya"
_OOP_VARIABLE = "out-of-pocket health spending"


def _partial_tuple_graph(*, with_obs: bool, with_competitor: bool = False) -> FakeGraph:
    """Build a top-scoring SV without measuredProperty and an optional complete competitor."""
    nodes: dict = {
        _SPARSE_SV: {
            "label": "Out-of-pocket health expenditure",
            "type": "StatisticalVariable",
            "arcs": {
                "populationType": {"nodes": [{"dcid": "currentHealthExpenditure"}]},
                # measuredProperty is absent from this partial five-tuple.
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "constraintProperties": {"nodes": [{"dcid": "healthFinancingScheme"}]},
                "healthFinancingScheme": {"nodes": [{"dcid": "HF_3"}]},
                "name": {"nodes": [{"value": "Out-of-pocket health expenditure"}]},
            },
        },
        "country/KEN": {"label": "Kenya", "type": "Country"},
        "healthFinancingScheme": {"label": "health financing scheme"},
        "HF_3": {"label": "household out-of-pocket payments"},
    }
    svs = [_SPARSE_SV]
    scores = [0.86]
    if with_competitor:
        nodes[_COMPLETE_SV] = {
            "label": "Healthcare expenditure per person",
            "type": "StatisticalVariable",
            "arcs": {
                "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "name": {"nodes": [{"value": "Healthcare expenditure per person"}]},
            },
        }
        svs.append(_COMPLETE_SV)
        scores.append(0.78)
    detect = {
        _OOP_VARIABLE: {
            "svs": svs,
            "cosine_scores": scores,
            "types": dict.fromkeys(svs, "StatisticalVariable"),
            "entities": ["Kenya"],
        }
    }
    obs = {}
    if with_obs:
        obs = {
            f"{sv}|country/KEN": [{"earliestDate": "2000", "latestDate": "2023", "obsCount": 24}]
            for sv in svs
        }
    return FakeGraph(nodes=nodes, obs=obs, detect=detect, resolve={"Kenya": "country/KEN"})


def _resolve_oop(graph: FakeGraph) -> ResolveResponse:
    req = ResolveRequest(input=RawTextInput(query=_OOP_QUERY))
    llm = _ExtractOnlyLLM(variable=_OOP_VARIABLE, entities=["Kenya"])
    return asyncio.run(resolve_async(req, graph=graph, llm=llm, sdmx=EMPTY_SDMX))


class TestPartialFiveTupleResolves:
    """An SV lacking measuredProperty resolves to definite when it has data."""

    def _run(self) -> ResolveResponse:
        return _resolve_oop(_partial_tuple_graph(with_obs=True))

    def test_status_is_definite(self):
        assert self._run().status == "definite"

    def test_sparse_sv_is_returned(self):
        inner = self._run().parts[0]
        assert isinstance(inner, DefinitePart)
        assert [sv.ref for sv in inner.spec.resolved.stat_vars.items] == [_SPARSE_SV]

    def test_measured_property_is_none(self):
        inner = self._run().parts[0]
        assert isinstance(inner, DefinitePart)
        shape = inner.spec.resolved.shape
        assert shape.measured_property is None
        assert shape.population_type is not None  # populationType is grounded from its arc.
        assert shape.population_type.dcid == "currentHealthExpenditure"


class TestPartialFiveTupleBeatsLowerScoredComplete:
    """The who_hf3 0.86 partial tuple beats the 0.78 complete tuple."""

    def _run(self) -> ResolveResponse:
        return _resolve_oop(_partial_tuple_graph(with_obs=True, with_competitor=True))

    def test_status_is_definite(self):
        assert self._run().status == "definite"

    def test_sparse_sv_wins_on_score(self):
        inner = self._run().parts[0]
        assert isinstance(inner, DefinitePart)
        assert [sv.ref for sv in inner.spec.resolved.stat_vars.items] == [_SPARSE_SV]


class TestPartialFiveTupleNoObservations:
    """A partial tuple without observations returns no_observations."""

    def _run(self) -> ResolveResponse:
        return _resolve_oop(_partial_tuple_graph(with_obs=False))

    def test_status_is_no_data(self):
        assert self._run().status == "no_data"

    def test_reason_is_data_absence_not_resolution_failure(self):
        inner = self._run().parts[0]
        assert isinstance(inner, NoDataPart)
        assert inner.no_data.reason == "no_observations"


# ---------------------------------------------------------------------------
# arc-derived slot_values for standard specs
# ---------------------------------------------------------------------------


class TestStd01SlotValuesArcDerived:
    """Count_Person has no constraints, though its shape includes observed constraint slots.

    Those slots are unbound and listed in inapplicable_slots. No inapplicable property may
    appear in a StatVar's slot_values. The assertions use real resolved Count_Person data.
    """

    def _result(self):
        return offline_resolve(_req("total population India"))

    def test_count_person_slot_values_is_empty(self):
        """Count_Person has no constraint axes, so slot_values is empty."""
        result = self._result()
        inner = result.parts[0]
        assert isinstance(inner, DefinitePart)
        for sv in inner.spec.resolved.stat_vars.items:
            if sv.ref == "Count_Person":
                assert sv.slot_values == [], (
                    f"Count_Person must carry no slot_values; got {sv.slot_values!r}"
                )

    def test_inapplicable_slots_is_non_empty_for_count_person(self):
        """Count_Person has at least one inapplicable constraint property."""
        spec = self._result().parts[0].spec
        assert isinstance(spec.resolved.inapplicable_slots, list)
        assert spec.resolved.inapplicable_slots, (
            "expected at least one inapplicable constraint property for Count_Person"
        )

    def test_inapplicable_slots_not_in_stat_var_slot_values(self):
        """Inapplicable properties are absent from every StatVar's slot_values."""
        result = self._result()
        inner = result.parts[0]
        assert isinstance(inner, DefinitePart)
        spec = inner.spec

        inapplicable_props = set(spec.resolved.inapplicable_slots)

        realised_props = {
            sv_val.key.property.dcid
            for sv in spec.resolved.stat_vars.items
            for sv_val in sv.slot_values
            if sv_val.key.property is not None
        }

        # Inapplicable properties and realised slot values form disjoint sets.
        overlap = inapplicable_props & realised_props
        assert not overlap, (
            f"Properties listed inapplicable on spec also appear in stat_var slot_values: {overlap}"
        )

    def test_inapplicable_slots_still_read_unbound_in_slots(self):
        """Inapplicable slots use the unbound wire state and appear in inapplicable_slots."""
        spec = self._result().parts[0].spec
        inapplicable_props = set(spec.resolved.inapplicable_slots)
        assert inapplicable_props, "expected at least one inapplicable property"

        slot_by_property = {
            slot.key.property.dcid: slot
            for slot in spec.resolved.slots
            if slot.key.property is not None
        }
        for prop_dcid in inapplicable_props:
            slot = slot_by_property.get(prop_dcid)
            assert slot is not None, f"expected a slot for inapplicable property {prop_dcid!r}"
            assert slot.binding.kind == "unbound", (
                f"inapplicable property {prop_dcid!r} slot must still read unbound; "
                f"got {slot.binding.kind!r}"
            )
