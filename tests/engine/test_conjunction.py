"""End-to-end tests for multi-variable conjunctions, part ordering, and same-shape merges.
Folded parts share a spec, and later parts carry `binding_folded`.
Refused merges keep both specs.
"""

from __future__ import annotations

import dataclasses

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.config import settings
from search.engine.conjoin import BINDING_FOLDED, VARIABLES_CLAMPED, combine_regions
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError, GraphInfraError
from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT, Extraction
from search.engine.region import RegionResult
from search.engine.region.codes import OBSERVATIONS_PROBE_FAILED, QUERY_QUALIFIER_DROPPED
from search.engine.spec_build import build_spec
from search.models import (
    BindingValue,
    CoverageBare,
    Slot,
    SlotKey,
    SlotValue,
    StatusLiteral,
    StatVar,
    Warning,
)
from tests.engine._builders import (
    conjoin_spec as _spec,
)
from tests.engine._builders import (
    constraint_value_slot as _value_slot,
)
from tests.engine._builders import (
    gref as _gr,
)
from tests.engine._builders import (
    shape_five_tuple as _shape,
)
from tests.engine._harness import PINNED_DATE, make_request, offline_resolve
from tests.fixtures import FakeGraph, FakeLLM, _load_json
from tests.fixtures.keys import fixture_key

# ---------------------------------------------------------------------------
# Extraction key helper for per-test FakeLLM fixtures
# ---------------------------------------------------------------------------

_EXTRACT_SYS = _EXTRACTION_SYSTEM_PROMPT.replace("[[TODAY]]", PINNED_DATE.isoformat())


def _extract_key(query: str) -> str:
    return fixture_key(Extraction, _EXTRACT_SYS, query)


def _rr(
    variable_text: str,
    status: StatusLiteral,
    spec=None,
    *,
    earliest_index: int = 0,
    no_data_reason: str | None = None,
    extra_specs: tuple = (),
    warnings: tuple = (),
    unresolved_entity_names: tuple[str, ...] = (),
) -> RegionResult:
    if status == "definite" and spec is not None:
        specs: tuple = (spec,)
    elif status == "candidates" and spec is not None:
        specs = (spec, *extra_specs)
    else:
        specs = ()
    return RegionResult(
        variable_text=variable_text,
        status=status,
        specs=specs,
        no_data_reason=no_data_reason or "variable_not_resolved",
        warnings=warnings,
        timing_by_step={},
        earliest_index=earliest_index,
        unresolved_entity_names=unresolved_entity_names,
    )


def _combine(regions):
    """Returns just the response -- combine_regions also returns the post-collapse
    region list, which this file's tests don't need."""
    response, _effective = combine_regions(
        regions,
        query="test query",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        max_candidates=10,
    )
    return response


_PERSON_SHAPE = _shape("Person", "count")
_ECON_SHAPE = _shape("EconomicActivity", "amount", mq="Nominal")


# ---------------------------------------------------------------------------
# Single-variable regression (shared fixtures, N=1 path)
# ---------------------------------------------------------------------------


def test_single_variable_regression():
    """N=1: one part, no binding_folded warning."""
    result = offline_resolve(make_request("total population India"))
    assert result.status == "definite"
    assert len(result.parts) == 1
    part = result.parts[0]
    assert part.status == "definite"
    codes = [w.code for w in part.warnings]
    assert BINDING_FOLDED not in codes
    assert part.variable_text == "total population"


# ---------------------------------------------------------------------------
# Cross-shape both-definite (shared fixtures)
# ---------------------------------------------------------------------------


def test_cross_shape_both_definite():
    """Two definite cross-shape regions each become their own part, in query order."""
    result = offline_resolve(make_request("population and fertility rate in Brazil"))
    assert result.status == "definite"
    assert len(result.parts) == 2

    population, fertility = result.parts
    assert population.status == "definite"
    assert population.variable_text == "population"
    assert any(sv.ref.dcid == "Count_Person" for sv in population.spec.resolved.members)
    codes = [w.code for w in population.warnings]
    assert BINDING_FOLDED not in codes

    assert fertility.status == "definite"
    assert fertility.variable_text == "fertility rate"
    assert any(
        sv.ref.dcid == "FertilityRate_Person_Female" for sv in fertility.spec.resolved.members
    )
    assert population.spec.spec_id != fertility.spec.spec_id


# ---------------------------------------------------------------------------
# Same-shape collapse (combine_regions direct)
# ---------------------------------------------------------------------------


def test_same_shape_collapse():
    """Two definite same-shape regions -> both parts share the merged spec;
    the second carries binding_folded, and the merged slot is a BindingSet."""
    spec_a = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv")
    spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv")
    response = _combine(
        [
            _rr("var A", "definite", spec_a, earliest_index=0),
            _rr("var B", "definite", spec_b, earliest_index=1),
        ]
    )
    assert response.status == "definite"
    assert len(response.parts) == 2
    first, second = response.parts
    assert first.status == "definite"
    assert second.status == "definite"
    assert first.spec.spec_id == second.spec.spec_id
    assert first.spec.resolved.slots[0].binding.kind == "set"

    first_codes = [w.code for w in first.warnings]
    assert BINDING_FOLDED not in first_codes
    second_codes = [w.code for w in second.warnings]
    assert BINDING_FOLDED in second_codes


@pytest.mark.parametrize("separate_index", [0, 1])
def test_a_separate_part_never_folds_and_the_others_fold_as_before(separate_index: int):
    """Of three same-shape definite parts, the one marked separate keeps its own spec and
    carries no binding_folded, in query order. The other two fold onto one set spec."""
    regions = [
        dataclasses.replace(
            _rr(
                f"var {i}",
                "definite",
                _spec(_PERSON_SHAPE, f"Val{i}", sv_dcid="sv"),
                earliest_index=i,
            ),
            separate=i == separate_index,
        )
        for i in range(3)
    ]
    response = _combine(regions)

    assert [p.variable_text for p in response.parts] == ["var 0", "var 1", "var 2"]
    separate = response.parts[separate_index]
    folded = [p for i, p in enumerate(response.parts) if i != separate_index]
    assert separate.spec.resolved.slots[0].binding.kind == "value"
    assert BINDING_FOLDED not in [w.code for w in separate.warnings]
    assert folded[0].spec.spec_id == folded[1].spec.spec_id != separate.spec.spec_id
    assert folded[0].spec.resolved.slots[0].binding.kind == "set"
    assert BINDING_FOLDED not in [w.code for w in folded[0].warnings]
    assert BINDING_FOLDED in [w.code for w in folded[1].warnings]


def test_same_shape_collapse_remaps_spec_scoped_warning_to_merged_spec_id():
    """A spec-scoped warning follows its region to the merged spec id."""
    spec_a = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv")
    spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv")
    probe_failed_warning = Warning(
        code=OBSERVATIONS_PROBE_FAILED,
        severity="warn",
        message="observations probe failed",
        spec_id=spec_a.spec_id,
    )
    response = _combine(
        [
            _rr("var A", "definite", spec_a, earliest_index=0, warnings=(probe_failed_warning,)),
            _rr("var B", "definite", spec_b, earliest_index=1),
        ]
    )
    assert response.status == "definite"
    first, second = response.parts
    # The merge produced a NEW spec_id -- neither original survives.
    assert first.spec.spec_id not in (spec_a.spec_id, spec_b.spec_id)
    assert first.spec.spec_id == second.spec.spec_id

    probe_warning = next(w for w in first.warnings if w.code == OBSERVATIONS_PROBE_FAILED)
    assert probe_warning.spec_id == first.spec.spec_id

    # General invariant: every spec-scoped warning on a part names a spec_id
    # actually present on that same part.
    for part in response.parts:
        for w in part.warnings:
            if w.spec_id is not None:
                assert w.spec_id == part.spec.spec_id


def test_same_shape_multislot_refusal_keeps_each_members_own_spec():
    """Same five-tuple, 2+ slots differ: both parts retain their own specs."""
    shape_two_slots = _shape("Person", "count")
    shape_two_slots = shape_two_slots.model_copy(
        update={
            "slot_keys": [
                SlotKey(axis="what", property=_gr("propA"), label="propA"),
                SlotKey(axis="how", property=_gr("propB"), label="propB"),
            ]
        }
    )

    def _two_slot_spec(val_a: str, val_b: str):
        return build_spec(
            open_roles=(),
            measure_elected_properties=frozenset(),
            shape=shape_two_slots,
            slots=[
                _value_slot("propA", val_a),
                Slot(
                    key=SlotKey(axis="how", property=_gr("propB"), label="propB"),
                    binding=BindingValue(value=SlotValue(ref=_gr(val_b), value_kind="enum_value")),
                ),
            ],
            stat_vars=[
                StatVar(
                    ref=_gr(f"sv_{val_a}_{val_b}"),
                    shape_id=shape_two_slots.shape_id,
                    slot_values=[],
                    constraint_properties=[],
                )
            ],
            entities=[],
            coverage=CoverageBare(has_data=True),
        )

    spec_1 = _two_slot_spec("A1", "B1")
    spec_2 = _two_slot_spec("A2", "B2")
    response = _combine(
        [
            _rr("var 1", "definite", spec_1, earliest_index=0),
            _rr("var 2", "definite", spec_2, earliest_index=1),
        ]
    )
    assert response.status == "definite"
    first, second = response.parts
    assert first.spec.spec_id == spec_1.spec_id
    assert second.spec.spec_id == spec_2.spec_id
    assert BINDING_FOLDED not in {w.code for w in first.warnings}
    assert BINDING_FOLDED not in {w.code for w in second.warnings}


# ---------------------------------------------------------------------------
# Every region visible as its own part, whatever its status
# ---------------------------------------------------------------------------


def test_definite_part_alongside_a_genuinely_ambiguous_part():
    """A definite part and a genuinely ambiguous part (a distinct five-tuple
    shape) each surface fully -- no warning stands in for the ambiguous
    part's own payload."""
    spec_person = _spec(_PERSON_SHAPE, "any", sv_dcid="sv")
    spec_cand_a = _spec(_ECON_SHAPE, "any", sv_dcid="sv_cand_a")
    spec_cand_b = _spec(_ECON_SHAPE, "other", sv_dcid="sv_cand_b")
    response = _combine(
        [
            _rr("population", "definite", spec_person, earliest_index=0),
            _rr(
                "malaria",
                "candidates",
                spec_cand_a,
                earliest_index=1,
                extra_specs=(spec_cand_b,),
            ),
        ]
    )
    assert response.status == "mixed"
    assert len(response.parts) == 2
    population, malaria = response.parts
    assert population.status == "definite"
    assert malaria.status == "candidates"
    assert malaria.variable_text == "malaria"
    assert len(malaria.candidates.specs) == 2


def test_definite_part_alongside_a_no_data_part():
    """A definite part and a no_data part each surface fully."""
    spec_person = _spec(_PERSON_SHAPE, "any", sv_dcid="sv")
    response = _combine(
        [
            _rr("population", "definite", spec_person, earliest_index=0),
            _rr("left-handedness", "no_data", earliest_index=1),
        ]
    )
    assert response.status == "mixed"
    population, left_handedness = response.parts
    assert population.status == "definite"
    assert left_handedness.status == "no_data"
    assert left_handedness.variable_text == "left-handedness"


def test_parts_stay_in_query_order_regardless_of_status():
    """A no_data part followed by a definite part keeps that order -- there is
    no promotion of a later definite region ahead of an earlier no_data one."""
    spec_gdp = _spec(_ECON_SHAPE, "any", sv_dcid="sv")
    response = _combine(
        [
            _rr("left-handedness", "no_data", earliest_index=0),
            _rr("GDP", "definite", spec_gdp, earliest_index=1),
        ]
    )
    assert response.status == "mixed"
    first, second = response.parts
    assert first.status == "no_data"
    assert first.variable_text == "left-handedness"
    assert second.status == "definite"
    assert second.variable_text == "GDP"


def test_candidates_part_alongside_a_no_data_part():
    spec_a = _spec(_PERSON_SHAPE, "ValA", sv_dcid="sv")
    spec_b = _spec(_PERSON_SHAPE, "ValB", sv_dcid="sv")
    response = _combine(
        [
            _rr("ambiguous A", "candidates", spec_a, earliest_index=0, extra_specs=(spec_b,)),
            _rr("left-handedness", "no_data", earliest_index=1),
        ]
    )
    assert response.status == "mixed"
    ambiguous, left_handedness = response.parts
    assert ambiguous.status == "candidates"
    assert left_handedness.status == "no_data"


def test_all_parts_no_data():
    """All parts no_data -> the envelope's shared status is no_data too."""
    response = _combine(
        [
            _rr("left-handedness", "no_data", earliest_index=0),
            _rr("flying saucers", "no_data", earliest_index=1),
        ]
    )
    assert response.status == "no_data"
    assert all(part.status == "no_data" for part in response.parts)


def test_each_no_data_part_names_its_own_variable_text_regardless_of_list_order():
    """Each no_data part's own sentence names ITS OWN variable text -- there is
    no shared/primary sentence to promote regardless of which order the
    regions are passed to combine_regions in."""
    response = _combine(
        [
            _rr(
                "flying saucers",
                "no_data",
                earliest_index=1,
                no_data_reason="variable_not_resolved",
            ),
            _rr(
                "carbon unicorn index",
                "no_data",
                earliest_index=0,
                no_data_reason="variable_not_resolved",
            ),
        ],
    )
    # Parts are always sorted by earliest_index, regardless of input order.
    first, second = response.parts
    assert first.variable_text == "carbon unicorn index"
    assert first.rendered_sentence == (
        'Could not match "carbon unicorn index" to a supported indicator. '
        "Try a different wording or a related indicator."
    )
    assert second.variable_text == "flying saucers"
    assert second.rendered_sentence == (
        'Could not match "flying saucers" to a supported indicator. '
        "Try a different wording or a related indicator."
    )


def test_each_no_data_part_names_its_own_unresolved_entity_names():
    """Each no_data part's own sentence names ITS OWN unresolved_entity_names,
    never another part's."""
    response = _combine(
        [
            _rr(
                "population",
                "no_data",
                earliest_index=1,
                no_data_reason="entity_not_resolved",
                unresolved_entity_names=("Mercosur",),
            ),
            _rr(
                "gdp",
                "no_data",
                earliest_index=0,
                no_data_reason="entity_not_resolved",
                unresolved_entity_names=("European Union",),
            ),
        ],
    )
    gdp_part, population_part = response.parts
    assert gdp_part.variable_text == "gdp"
    assert gdp_part.rendered_sentence == (
        'Could not use "European Union" as a place for this query. Try a different place name.'
    )
    assert population_part.variable_text == "population"
    assert population_part.rendered_sentence == (
        'Could not use "Mercosur" as a place for this query. Try a different place name.'
    )


# ---------------------------------------------------------------------------
# Clamp (per-test FakeLLM + FakeGraph via offline_resolve)
# ---------------------------------------------------------------------------

_CLAMP_QUERY = "a and b and c and d and e and f and g in Brazil"


def test_clamp():
    """7 variables → clamped to 6; VARIABLES_CLAMPED emitted on the envelope."""
    fake_llm = FakeLLM(
        responses={
            _extract_key(_CLAMP_QUERY): {
                "kind": "resolution",
                "variables": ["a", "b", "c", "d", "e", "f", "g"],
                "entities": [{"text": "Brazil"}],
                "dates": [],
            }
        }
    )
    fake_graph = FakeGraph(
        detect={},  # empty → all variables no_data
        resolve={"Brazil": "country/BRA"},
    )
    result = offline_resolve(make_request(_CLAMP_QUERY), graph=fake_graph, llm=fake_llm)
    codes = [w.code for w in result.diagnostics.warnings]
    assert VARIABLES_CLAMPED in codes
    w = next(w for w in result.diagnostics.warnings if w.code == VARIABLES_CLAMPED)
    assert "7" in w.message and "6" in w.message


# ---------------------------------------------------------------------------
# Entity cap / entity dedupe (per-test FakeLLM + FakeGraph via offline_resolve)
# ---------------------------------------------------------------------------

# Real country names drawn from the shared graph_nodes.json fixture
# (tests/fixtures/graph_nodes.json, 160 Country nodes), reused rather than
# invented. The cap check itself runs on raw text length before any
# resolution attempt, so these need not be resolvable through
# graph_resolve.json too -- only genuinely real place names, not placeholders.
_REAL_COUNTRIES: list[str] = sorted(
    node["label"]
    for dcid, node in _load_json("graph_nodes.json").items()
    if dcid.startswith("country/") and isinstance(node, dict) and node.get("type") == "Country"
)[: settings().max_entities + 1]
assert len(_REAL_COUNTRIES) == settings().max_entities + 1


def _entity_cap_result(entities: list[str], *, query: str):
    fake_llm = FakeLLM(
        responses={
            _extract_key(query): {
                "kind": "resolution",
                "variables": ["population"],
                "entities": [{"text": e} for e in entities],
                "dates": [],
            }
        }
    )
    fake_graph = FakeGraph(detect={})  # empty → the one variable goes no_data
    return offline_resolve(make_request(query), graph=fake_graph, llm=fake_llm)


def test_entity_cap_refuses_over_limit():
    """settings().max_entities + 1 entities → request_invalid, naming both counts."""
    query = "population in many countries"
    with pytest.raises(EngineInputError) as exc_info:
        _entity_cap_result(_REAL_COUNTRIES, query=query)
    assert exc_info.value.code == "request_invalid"
    assert str(len(_REAL_COUNTRIES)) in str(exc_info.value)
    assert str(settings().max_entities) in str(exc_info.value)


def test_entity_cap_at_boundary_does_not_refuse():
    """Exactly settings().max_entities unique entities → resolves normally."""
    query = "population at exactly the cap"
    entities = _REAL_COUNTRIES[: settings().max_entities]
    assert len(entities) == settings().max_entities
    result = _entity_cap_result(entities, query=query)
    assert result.status == "no_data"  # detect={} -- the cap itself is what's under test


def test_entity_dedup_before_cap():
    """Repeated names count once toward `max_entities`."""
    query = "population in Kenya repeated many times"
    # 17 raw entries, 1 unique text -- would refuse if dedupe ran after the cap
    # check (or not at all), instead of before it.
    entities = ["Kenya"] * (settings().max_entities + 1)
    result = _entity_cap_result(entities, query=query)
    assert result.status == "no_data"


def test_entity_occurrence_cap_refuses_over_limit():
    """`max_entity_occurrences` limits raw repeats separately from distinct names."""
    query = "population in Kenya repeated far too many times"
    entities = ["Kenya"] * (settings().max_entity_occurrences + 1)
    with pytest.raises(EngineInputError) as exc_info:
        _entity_cap_result(entities, query=query)
    assert exc_info.value.code == "request_invalid"
    assert str(len(entities)) in str(exc_info.value)
    assert str(settings().max_entity_occurrences) in str(exc_info.value)


@pytest.mark.parametrize(
    "variables", [["unsupported metric"], ["unsupported metric", "other metric"]]
)
def test_dropped_qualifiers_warn_once_on_direct_and_combined_paths(variables):
    """N=1 and N>=2 retain each nonempty dropped qualifier as an envelope warning."""
    query = "unsupported metrics for top 10 countries by population"
    qualifier = "top 10 countries by population"
    fake_llm = FakeLLM(
        responses={
            _extract_key(query): {
                "kind": "resolution",
                "variables": variables,
                "entities": [],
                "dates": [],
                "dropped_qualifiers": [qualifier, qualifier, ""],
            }
        }
    )
    result = offline_resolve(
        make_request(query),
        graph=FakeGraph(detect={}),
        llm=fake_llm,
    )

    warnings = [
        warning
        for warning in result.diagnostics.warnings
        if warning.code == QUERY_QUALIFIER_DROPPED
    ]
    assert len(warnings) == 1
    assert warnings[0].severity == "degraded"
    assert warnings[0].message == f'The requested qualifier "{qualifier}" was not applied.'


def test_dropped_qualifiers_count_cap_overflow_reported_as_a_count():
    """Overflow emits its count warning plus one warning per retained phrase."""
    n = settings().max_dropped_qualifiers + 5
    query = "a query naming an excessive number of unsupported qualifiers"
    fake_llm = FakeLLM(
        responses={
            _extract_key(query): {
                "kind": "resolution",
                "variables": ["population"],
                "entities": [],
                "dates": [],
                "dropped_qualifiers": [
                    f"unsupported qualifier phrase number {i}" for i in range(n)
                ],
            }
        }
    )
    result = offline_resolve(
        make_request(query),
        graph=FakeGraph(detect={}),
        llm=fake_llm,
    )

    warnings = [w for w in result.diagnostics.warnings if w.code == QUERY_QUALIFIER_DROPPED]
    overflow = n - settings().max_dropped_qualifiers
    overflow_warnings = [w for w in warnings if w.message.startswith(f"{overflow} additional")]
    assert len(overflow_warnings) == 1
    assert overflow_warnings[0].severity == "degraded"
    # One overflow notice plus one per-phrase warning for each of the
    # settings().max_dropped_qualifiers surviving phrases.
    assert len(warnings) == settings().max_dropped_qualifiers + 1


# ---------------------------------------------------------------------------
# Duplicate dedupe (per-test FakeLLM + FakeGraph)
# ---------------------------------------------------------------------------

_DEDUP_QUERY = "population and population in India"


class _DedupLLM:
    """Extraction from a small scripted dict; elects the sole candidate shape with
    no constraint bindings for the universal choose+bind call (Count_Person carries
    none)."""

    def __init__(self, extraction_responses: dict):
        self._fake = FakeLLM(responses=extraction_responses)

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "_UniversalBindOutput":
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[],
                also_plausible=None,
                ask=None,
            ), None
        return await self._fake.generate_structured(prompt=prompt, system=system, schema=schema)


def test_duplicate_dedupe():
    """Duplicate variable names dedupe to one region before the fan-out even
    runs, so this is the plain N=1 path: one part, no binding_folded warning."""
    fake_llm = _DedupLLM(
        {
            _extract_key(_DEDUP_QUERY): {
                "kind": "resolution",
                "variables": ["population", "population"],
                "entities": [{"text": "India"}],
                "dates": [],
            }
        }
    )
    # N=1 after dedup: detect_query = the bare deduped variable ("population")
    fake_graph = FakeGraph(
        detect={"population": {"svs": ["Count_Person"]}},
        resolve={"India": "country/IND"},
        # obs uses shared file (Count_Person|country/IND exists)
    )
    result = offline_resolve(make_request(_DEDUP_QUERY), graph=fake_graph, llm=fake_llm)
    assert result.status == "definite"
    assert len(result.parts) == 1
    codes = [w.code for w in result.parts[0].warnings]
    assert BINDING_FOLDED not in codes


# ---------------------------------------------------------------------------
# Infra error propagation — N≥2 path must not swallow EngineInfraError
# ---------------------------------------------------------------------------

_INFRA_QUERY = "population and GDP in Brazil"


def test_infra_error_reraises_in_multi_variable():
    """A GraphInfraError from one variable leg propagates instead of becoming no_data."""
    fake_llm = FakeLLM(
        responses={
            _extract_key(_INFRA_QUERY): {
                "kind": "resolution",
                "variables": ["population", "GDP"],
                "entities": [{"text": "Brazil"}],
                "dates": [],
            }
        }
    )
    # raise_on_call=True makes every graph method raise GraphInfraError, simulating
    # a transport/timeout failure on any leg of the concurrent resolve.
    fake_graph = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        offline_resolve(make_request(_INFRA_QUERY), graph=fake_graph, llm=fake_llm)
